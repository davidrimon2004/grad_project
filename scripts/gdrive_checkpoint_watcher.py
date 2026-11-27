#!/usr/bin/env python3
"""
gdrive_checkpoint_watcher.py
============================
Automatically uploads Video-LLaVA checkpoints and final weights to Google Drive
using `rclone`, with resumable uploads, atomic completion checks, and state tracking.

Two modes:
  1. Continuous watcher daemon (monitors a checkpoints root dir for any new
     completed `checkpoint-<step>` dirs or final `mm_projector.bin` files):
       python scripts/gdrive_checkpoint_watcher.py \
           --watch /home/aitech/hdd/grad_project_david/grad_project_david/checkpoints \
           --remote gdrive:Video-LLaVA/checkpoints

  2. One-shot upload (called from LLaVATrainer._save_checkpoint or run_pretrain.sh):
       python scripts/gdrive_checkpoint_watcher.py \
           --once /path/to/checkpoint-5000 \
           --remote gdrive:Video-LLaVA/checkpoints/videollava-7b-pretrain/checkpoint-5000
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_REMOTE = os.environ.get("GDRIVE_REMOTE", "gdrive:Video-LLaVA/checkpoints")
DEFAULT_WATCH_DIR = "/home/aitech/hdd/grad_project_david/grad_project_david/checkpoints"
DEFAULT_LOG_FILES = [
    "/home/aitech/pretrain_log_images.txt",
    "/home/aitech/pretrain_status.txt",
    "/home/aitech/precompute_embeddings.log",
    "/home/aitech/boot_resume.log",
    "/home/aitech/finetune_lora_log.txt",
    "/home/aitech/finetune_lora_status.txt",
    "/home/aitech/hdd/pretrain_logs/pretrain_status.txt",
    "/home/aitech/hdd/finetune_logs/finetune_lora_status.txt",
]

# Giant DeepSpeed / optimizer state shards that do not need to be mirrored to Drive
# by default (weight files + configs + adapter/trainer state are sufficient for eval
# and lightweight resume). Override with --include-optimizer or GDRIVE_UPLOAD_OPTIMIZER=1.
DEFAULT_DIR_EXCLUDES = (
    "global_step*/**",
    "optimizer.pt",
    "rng_state*.pth",
    "zero_to_fp32.py",
    "latest",
    "*.tmp",
    "gdrive_upload.log",
)

# Files we consider part of a usable weight checkpoint
WEIGHT_FILENAMES = (
    "mm_projector.bin",
    "non_lora_trainables.bin",
    "adapter_model.bin",
    "adapter_model.safetensors",
    "pytorch_model.bin",
    "model.safetensors",
)
META_FILENAMES = (
    "config.json",
    "trainer_state.json",
    "adapter_config.json",
)


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] [GDriveSync] {msg}", flush=True)


def find_rclone() -> Optional[str]:
    for candidate in (
        shutil.which("rclone"),
        str(Path.home() / ".local" / "bin" / "rclone"),
        "/usr/local/bin/rclone",
        "/usr/bin/rclone",
    ):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def is_remote_configured(rclone_bin: str, remote: str) -> bool:
    remote_bare = remote.split(":", 1)[0]
    remote_name = remote_bare + ":"
    try:
        res = subprocess.run(
            [rclone_bin, "listremotes"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode != 0:
            return False
        remotes = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if remote_name not in remotes:
            return False
        cfg = subprocess.run(
            [rclone_bin, "config", "show", remote_bare],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if cfg.returncode != 0:
            return False
        for line in cfg.stdout.splitlines():
            line = line.strip()
            if (line.startswith("token =") or line.startswith("service_account_file =")) and line.split("=", 1)[1].strip():
                return True
        return False
    except Exception:
        return False


def _file_signature(p: Path) -> Optional[str]:
    try:
        st = p.stat()
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return None


def _is_file_settled(p: Path, settle_secs: float = 5.0) -> bool:
    """Returns True if file exists, is non-empty, and hasn't been modified for settle_secs."""
    try:
        st = p.stat()
        if st.st_size == 0:
            return False
        return (time.time() - st.st_mtime) >= settle_secs
    except OSError:
        return False


def _load_state(state_path: Path) -> Dict[str, str]:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state_path: Path, state: Dict[str, str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(tmp), str(state_path))


def rclone_upload(
    rclone_bin: str,
    src: Path,
    dst_remote: str,
    retries: int = 5,
    bwlimit: Optional[str] = None,
    include_optimizer: bool = False,
    exclude_checkpoints: bool = False,
) -> bool:
    cmd = [
        rclone_bin,
        "copy",
        str(src),
        dst_remote,
        "--transfers", "2",
        "--retries", str(retries),
        "--low-level-retries", "10",
        "--retries-sleep", "10s",
        "--stats-one-line",
        "--stats", "30s",
    ]
    if src.is_dir():
        use_opt = include_optimizer or (os.environ.get("GDRIVE_UPLOAD_OPTIMIZER", "0") == "1")
        if not use_opt:
            for pat in DEFAULT_DIR_EXCLUDES:
                cmd.extend(["--exclude", pat])
        if exclude_checkpoints:
            cmd.extend(["--exclude", "checkpoint-*/**"])
    if bwlimit:
        cmd.extend(["--bwlimit", bwlimit])

    log(f"Uploading {src} -> {dst_remote}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        if proc.returncode == 0:
            log(f"OK uploaded {src.name} in {elapsed:.1f}s -> {dst_remote}")
            return True
        else:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
            log(f"ERR upload failed (rc={proc.returncode}) for {src}: {' | '.join(err)}")
            return False
    except Exception as exc:
        log(f"ERR exception during rclone upload of {src}: {exc}")
        return False


def _checkpoint_step(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint-"):
        tail = name.split("-", 1)[1]
        if tail.isdigit():
            return int(tail)
    return -1


def discover_upload_targets(
    watch_root: Path,
    run_filter: Optional[str] = None,
    settle_secs: float = 5.0,
) -> List[Tuple[str, Path, str, str]]:
    """
    Scans watch_root/<run_name>/ for:
      - Completed checkpoint-<step> dirs (must have a weight file and be settled)
      - Final top-level weight files (<run_name>/mm_projector.bin, etc.)

    Returns list of (state_key, local_path, remote_subpath, signature) sorted so that
    final weights and the NEWEST checkpoints upload first!
    """
    if not watch_root.is_dir():
        return []

    targets: List[Tuple[int, int, str, Path, str, str]] = []

    for run_dir in sorted(watch_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        # Skip tiny test dirs unless explicitly matched
        if run_filter and run_filter not in run_name:
            continue
        if not run_filter and (run_name.startswith("test-") or "archive" in run_name):
            continue
        if run_name in ("downloaded_projector",):
            continue

        # 1. Final weights at <run_dir>/ (mm_projector.bin or LoRA adapter + non_lora_trainables.bin)
        found_final_weight = False
        for fname in WEIGHT_FILENAMES:
            fpath = run_dir / fname
            if fpath.is_file() and _is_file_settled(fpath, settle_secs):
                found_final_weight = True
                sig = _file_signature(fpath)
                if sig:
                    targets.append((0, 0, f"{run_name}/final/{fname}", fpath, f"{run_name}/final", sig))

        if found_final_weight:
            # Attach config.json / trainer_state.json / adapter_config.json ONCE per run_dir
            for mname in META_FILENAMES:
                mpath = run_dir / mname
                if mpath.is_file() and _is_file_settled(mpath, settle_secs):
                    msig = _file_signature(mpath)
                    if msig:
                        targets.append((0, 1, f"{run_name}/final/{mname}", mpath, f"{run_name}/final", msig))

        # 2. Intermediate checkpoint-<step> dirs (newest step first!)
        for child in run_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("checkpoint-"):
                continue
            step = _checkpoint_step(child)
            # Check that at least one weight file exists and is settled
            weight_files = [child / w for w in WEIGHT_FILENAMES if (child / w).is_file()]
            if not weight_files:
                continue
            # Prefer waiting for trainer_state.json (written last by LLaVATrainer._save_checkpoint)
            ts_file = child / "trainer_state.json"
            if ts_file.is_file():
                if not _is_file_settled(ts_file, settle_secs):
                    continue
            if not all(_is_file_settled(wf, settle_secs) for wf in weight_files):
                continue

            # For Stage 2 LoRA checkpoints, wait until non_lora_trainables.bin and config.json
            # are also written & settled (with a 60s legacy fallback based on trainer_state.json).
            if (child / "adapter_config.json").is_file():
                legacy_settled = False
                if ts_file.is_file():
                    try:
                        legacy_settled = (time.time() - ts_file.stat().st_mtime) >= 60.0
                    except OSError:
                        legacy_settled = False
                if not legacy_settled:
                    req_non_lora = child / "non_lora_trainables.bin"
                    req_cfg = child / "config.json"
                    if not (
                        req_non_lora.is_file()
                        and _is_file_settled(req_non_lora, settle_secs)
                        and req_cfg.is_file()
                        and _is_file_settled(req_cfg, settle_secs)
                    ):
                        continue

            # Build composite signature of weight + meta files in this checkpoint dir
            sig_parts = []
            for fn in sorted(set(WEIGHT_FILENAMES + META_FILENAMES)):
                fp = child / fn
                if fp.is_file():
                    s = _file_signature(fp)
                    if s:
                        sig_parts.append(f"{fn}={s}")
            sig = "|".join(sig_parts)
            # priority 1, sort descending by step (-step) so latest checkpoint uploads FIRST
            targets.append((1, -step, f"{run_name}/{child.name}", child, f"{run_name}/{child.name}", sig))

    targets.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(k, p, sub, sig) for _, _, k, p, sub, sig in targets]


def _collect_log_files(watch_root: Optional[Path] = None) -> List[Path]:
    """Collect existing training/status log files across Stage 1, Stage 2, and custom env overrides."""
    candidates: List[Path] = [Path(p) for p in DEFAULT_LOG_FILES]
    extra = os.environ.get("GDRIVE_EXTRA_LOGS", "").strip()
    if extra:
        for item in extra.split(":"):
            if item.strip():
                candidates.append(Path(item.strip()))
    if watch_root is not None and watch_root.is_dir():
        for run_dir in sorted(watch_root.iterdir()):
            if run_dir.is_dir():
                for name in ("finetune_lora_status.txt", "pretrain_status.txt", "train_status.txt"):
                    candidates.append(run_dir / name)

    seen = set()
    existing: List[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen and p.is_file():
            seen.add(key)
            existing.append(p)
    return existing


def _write_live_status_snapshot(out_path: Path, watch_root: Optional[Path] = None) -> None:
    """Writes a compact status snapshot (timestamp, GPU stats, latest training log tail)."""
    lines = [f"Updated: {_ts()}", ""]
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv"],
            capture_output=True, text=True, timeout=10,
        )
        lines.append("=== GPU STATUS ===")
        lines.append((gpu.stdout or gpu.stderr).strip())
    except Exception as exc:
        lines.append(f"nvidia-smi unavailable: {exc}")

    for lp in _collect_log_files(watch_root):
        lines.append(f"\n=== TAIL {lp.parent.name}/{lp.name} ===")
        try:
            content = lp.read_text(encoding="utf-8", errors="replace").splitlines()
            lines.extend(content[-30:])
        except Exception as exc:
            lines.append(f"(read error: {exc})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def watch_loop(
    watch_root: Path,
    remote: str,
    state_path: Path,
    interval: float = 30.0,
    settle_secs: float = 5.0,
    run_filter: Optional[str] = None,
    bwlimit: Optional[str] = None,
    sync_logs: bool = True,
    log_sync_interval: float = 900.0,
    max_existing: int = 0,
    include_optimizer: bool = False,
) -> None:
    rclone_bin = find_rclone()
    if not rclone_bin:
        log("ERR rclone binary not found! Install rclone first.")
        sys.exit(1)

    running = True

    def _handle_stop(signum, frame):
        nonlocal running
        log(f"Received signal {signum}, shutting down cleanly...")
        running = False

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    state = _load_state(state_path)
    last_unauth_warn = 0.0
    last_log_sync = 0.0
    status_snapshot = Path.home() / ".cache" / "live_status.txt"
    remote_base = remote.rstrip("/")

    log(f"Watcher started: dir={watch_root} -> remote={remote_base} (poll={interval}s, log_sync={int(log_sync_interval)}s)")

    # If max_existing > 0 on first startup (when state is empty), pre-mark older checkpoints as skipped
    # so only the latest N existing checkpoints + all future checkpoints get uploaded.
    if max_existing > 0 and not state:
        initial_targets = discover_upload_targets(watch_root, run_filter=run_filter, settle_secs=settle_secs)
        ckpt_targets = [t for t in initial_targets if "/checkpoint-" in t[0]]
        if len(ckpt_targets) > max_existing:
            for key, _, _, sig in ckpt_targets[max_existing:]:
                state[key] = sig
            _save_state(state_path, state)
            log(f"Skipped {len(ckpt_targets) - max_existing} older pre-existing checkpoints; will upload latest {max_existing} + all new ones.")

    while running:
        if not is_remote_configured(rclone_bin, remote_base):
            now = time.time()
            if now - last_unauth_warn > 300:
                remote_name = remote_base.split(":", 1)[0]
                log(
                    f"WAITING: rclone remote '{remote_name}:' is not configured yet. "
                    f"Run `bash scripts/setup_gdrive.sh` (or `rclone config`) to authorize Google Drive. "
                    f"Uploads will begin automatically as soon as it is configured."
                )
                last_unauth_warn = now
            time.sleep(min(interval, 15.0))
            continue

        targets = discover_upload_targets(watch_root, run_filter=run_filter, settle_secs=settle_secs)
        uploaded_any = False
        for key, local_path, remote_subpath, sig in targets:
            if not running:
                break
            if state.get(key) == sig:
                continue
            dst = f"{remote_base}/{remote_subpath}"
            if rclone_upload(rclone_bin, local_path, dst, bwlimit=bwlimit, include_optimizer=include_optimizer):
                state[key] = sig
                _save_state(state_path, state)
                uploaded_any = True

        # Sync live status & training logs after any checkpoint upload OR every log_sync_interval (default 15m)
        now = time.time()
        if sync_logs and running and (uploaded_any or (now - last_log_sync >= log_sync_interval)):
            try:
                _write_live_status_snapshot(status_snapshot, watch_root=watch_root)
                rclone_upload(rclone_bin, status_snapshot, f"{remote_base}/logs", retries=2, bwlimit=bwlimit)
            except Exception as exc:
                log(f"WARN failed to sync live_status.txt: {exc}")
            for lp in _collect_log_files(watch_root):
                rclone_upload(rclone_bin, lp, f"{remote_base}/logs", retries=2, bwlimit=bwlimit)
            last_log_sync = time.time()

        for _ in range(int(max(1, interval))):
            if not running:
                break
            time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-upload Video-LLaVA checkpoints to Google Drive via rclone.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", type=str, help="Watch a checkpoints root directory continuously.")
    mode.add_argument("--once", type=str, help="Upload a single file or directory once and exit.")

    parser.add_argument("--remote", type=str, default=DEFAULT_REMOTE, help=f"rclone destination (default: {DEFAULT_REMOTE})")
    parser.add_argument("--state-file", type=str, default=str(Path.home() / ".cache" / "videollava_gdrive_uploads.json"))
    parser.add_argument("--interval", type=float, default=30.0, help="Polling interval in seconds (default: 30).")
    parser.add_argument("--settle-secs", type=float, default=5.0, help="Min seconds since last file mtime before upload.")
    parser.add_argument("--run-filter", type=str, default=None, help="Only watch run directories matching this substring.")
    parser.add_argument("--bwlimit", type=str, default=None, help="Optional rclone bandwidth limit (e.g. 20M).")
    parser.add_argument("--no-logs", action="store_true", help="Do not upload training log files alongside checkpoints.")
    parser.add_argument(
        "--max-existing",
        type=int,
        default=1,
        help="On first start with empty state, upload at most N newest pre-existing checkpoints (0 = upload all existing; default: 1 = latest + all future).",
    )
    parser.add_argument(
        "--include-optimizer",
        action="store_true",
        help="Also upload DeepSpeed global_step*/optimizer shards (excluded by default to save Drive space/bandwidth).",
    )
    parser.add_argument(
        "--exclude-checkpoints",
        action="store_true",
        help="In --once mode, exclude checkpoint-*/ subdirs (used when uploading a run_dir's top-level final weights).",
    )
    args = parser.parse_args()

    if args.once:
        rclone_bin = find_rclone()
        if not rclone_bin:
            log("ERR rclone not found.")
            sys.exit(1)
        if not is_remote_configured(rclone_bin, args.remote):
            log(f"SKIP: rclone remote '{args.remote}' not configured yet.")
            sys.exit(0)
        src = Path(args.once)
        if not src.exists():
            log(f"ERR path does not exist: {src}")
            sys.exit(1)
        ok = rclone_upload(
            rclone_bin,
            src,
            args.remote,
            bwlimit=args.bwlimit,
            include_optimizer=args.include_optimizer,
            exclude_checkpoints=args.exclude_checkpoints,
        )
        sys.exit(0 if ok else 1)

    watch_loop(
        watch_root=Path(args.watch),
        remote=args.remote,
        state_path=Path(args.state_file),
        interval=args.interval,
        settle_secs=args.settle_secs,
        run_filter=args.run_filter,
        bwlimit=args.bwlimit,
        sync_logs=not args.no_logs,
        max_existing=args.max_existing,
        include_optimizer=args.include_optimizer,
    )


if __name__ == "__main__":
    main()
