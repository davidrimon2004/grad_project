#!/usr/bin/env python3
"""
Video-ChatGPT Tuning Dataset Extractor for Video-LLaVA
======================================================
Extracts the Video-ChatGPT video tuning dataset (~160 GB) from either:
  1. The 5 split volumes (videochatgpt_tune_2.zip.001 ... videochatgpt_tune_2.zip.005)
  2. The assembled master archive (videochatgpt_tune.zip)

Features:
  - Two-phase automated pipeline:
      Phase 1: videochatgpt_tune_2.zip.001..005 -> videochatgpt_tune.zip (if not already extracted)
      Phase 2: videochatgpt_tune.zip -> video files (videochatgpt_tune/*.mp4)
  - Resumable: skips already-extracted video files using 7-Zip (-aos)
  - Auto-flattens nested directory structures (avoids videochatgpt_tune/videochatgpt_tune/...)
  - Auto-cleans intermediate videochatgpt_tune.zip to save ~160 GB of disk space
  - Real-time progress output and disk space safety checks

Usage:
  # Standard extraction (auto-deletes intermediate 160 GB videochatgpt_tune.zip when done):
  python scripts/extract_tune_videos.py --data_root E:\\videollava_data

  # Keep the intermediate master zip:
  python scripts/extract_tune_videos.py --data_root E:\\videollava_data --keep_intermediate

  # Extract and also delete the original 5 split parts (saves ~320 GB total):
  python scripts/extract_tune_videos.py --data_root E:\\videollava_data --clean_splits
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

VIDEO_TUNE_PARTS = [
    "videochatgpt_tune_2.zip.001",
    "videochatgpt_tune_2.zip.002",
    "videochatgpt_tune_2.zip.003",
    "videochatgpt_tune_2.zip.004",
    "videochatgpt_tune_2.zip.005",
]

# Total uncompressed master zip is approximately 160 GB
MASTER_ZIP_NAME = "videochatgpt_tune.zip"
TARGET_FOLDER_NAME = "videochatgpt_tune"


class Colors:
    HEADER  = "\033[95m"
    OKCYAN  = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL    = "\033[91m"
    ENDC    = "\033[0m"
    BOLD    = "\033[1m"


def log_info(msg: str):
    print(f"{Colors.OKCYAN}[INFO]{Colors.ENDC}    {msg}", flush=True)


def log_success(msg: str):
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {msg}", flush=True)


def log_warn(msg: str):
    print(f"{Colors.WARNING}[WARN]{Colors.ENDC}    {msg}", flush=True)


def log_err(msg: str):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC}   {msg}", flush=True)


def log_header(msg: str):
    sep = "=" * 64
    print(f"\n{Colors.HEADER}{Colors.BOLD}{sep}\n  {msg}\n{sep}{Colors.ENDC}", flush=True)


def find_7z() -> Optional[str]:
    """Find 7-Zip executable on the system."""
    for candidate in ("7z", "7za"):
        p = shutil.which(candidate)
        if p:
            return p
    if os.name == "nt":
        for win_path in [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
        ]:
            if os.path.exists(win_path):
                return win_path
    return None


def get_drive_free_gb(path: Path) -> float:
    """Return free disk space in GB for the drive containing path."""
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0


def run_7z_with_progress(cmd: List[str], description: str) -> bool:
    """Run 7-Zip with real-time console output."""
    log_info(f"Starting: {description}")
    log_info(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0
        elapsed_str = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))
        if proc.returncode == 0:
            log_success(f"Completed: {description} in {elapsed_str}")
            return True
        else:
            log_err(f"7-Zip exited with return code {proc.returncode} after {elapsed_str}")
            return False
    except Exception as exc:
        log_err(f"Execution failed: {exc}")
        return False


def flatten_nested_folder(target_dir: Path) -> None:
    """Check if extraction created target/target/ and flatten it to target/."""
    nested_dir = target_dir / target_dir.name
    if nested_dir.exists() and nested_dir.is_dir():
        log_info(f"Found nested directory: {nested_dir}. Flattening to {target_dir}...")
        sub_items = list(nested_dir.iterdir())
        moved = 0
        for item in sub_items:
            target = target_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
                moved += 1
            else:
                log_warn(f"Target already exists: {target.name}, skipping move.")
        try:
            nested_dir.rmdir()
            log_success(f"Flattened {moved} items and removed nested folder {nested_dir.name}/")
        except OSError:
            log_warn(f"Could not remove {nested_dir}; it may still contain some files.")


def count_extracted_videos(video_dir: Path) -> Tuple[int, int]:
    """Count subdirectories and .mp4 files inside video directory."""
    if not video_dir.exists():
        return 0, 0
    subdirs = [d for d in video_dir.iterdir() if d.is_dir() and d.name != video_dir.name]
    total_videos = 0
    # Also check direct mp4 files if no subdirectories
    for f in video_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".mp4":
            total_videos += 1
    for sd in subdirs:
        try:
            for f in sd.iterdir():
                if f.is_file() and f.suffix.lower() == ".mp4":
                    total_videos += 1
        except OSError:
            pass
    return len(subdirs), total_videos


def main():
    parser = argparse.ArgumentParser(
        description="Extract Video-ChatGPT Tuning Dataset from split archives or master zip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="E:\\videollava_data",
        help="Directory containing archives and output folder (default: E:\\videollava_data)",
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep intermediate videochatgpt_tune.zip after extracting videos (default: auto-delete)",
    )
    parser.add_argument(
        "--clean_splits",
        action="store_true",
        help="Delete split volumes (videochatgpt_tune_2.zip.001..005) after successful extraction to save disk space",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Inspect existing files and plan extraction without modifying files",
    )

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    log_header("Video-ChatGPT Video Tuning Extractor")
    log_info(f"Target Data Root : {data_root}")
    free_gb = get_drive_free_gb(data_root)
    log_info(f"Drive Free Space : {free_gb:.1f} GB")

    seven_zip = find_7z()
    if not seven_zip:
        log_err("7-Zip not found!")
        log_err("Please install 7-Zip from https://www.7-zip.org or run:")
        log_err("  winget install 7zip.7zip")
        sys.exit(1)
    log_info(f"7-Zip Executable : {seven_zip}")

    master_zip = data_root / MASTER_ZIP_NAME
    first_split = data_root / VIDEO_TUNE_PARTS[0]
    video_dir = data_root / TARGET_FOLDER_NAME

    # -------------------------------------------------------------
    # Step 1: Check Master videochatgpt_tune.zip Status
    # -------------------------------------------------------------
    has_valid_master_zip = False
    if master_zip.exists():
        actual_size = master_zip.stat().st_size
        size_gb = actual_size / (1024 ** 3)
        if size_gb >= 100:  # Complete or nearly complete
            log_success(f"Found complete master archive: {master_zip.name} ({size_gb:.2f} GB)")
            has_valid_master_zip = True
        else:
            log_warn(
                f"Found partial/incomplete {master_zip.name} ({size_gb:.2f} GB). "
                "This file is incomplete and will be replaced."
            )
            if not args.dry_run:
                log_info(f"Removing partial {master_zip.name} to free space and re-extract clean...")
                master_zip.unlink(missing_ok=True)
                has_valid_master_zip = False

    # -------------------------------------------------------------
    # Step 2: Extract split parts -> master zip (if needed)
    # -------------------------------------------------------------
    if not has_valid_master_zip:
        log_header("Phase 1: Extracting Split Volumes -> videochatgpt_tune.zip")
        if not first_split.exists():
            log_err(f"Cannot extract: First part '{VIDEO_TUNE_PARTS[0]}' not found in {data_root}.")
            sys.exit(1)

        # Check split volumes presence
        missing_parts = [p for p in VIDEO_TUNE_PARTS if not (data_root / p).exists()]
        if missing_parts:
            log_err(f"Missing {len(missing_parts)} split volume(s): {missing_parts}")
            log_err("Please ensure all 5 split volumes are fully downloaded.")
            sys.exit(1)
        log_success("All 5 split volumes verified present.")

        free_gb = get_drive_free_gb(data_root)
        if free_gb < 165:
            log_warn(
                f"Free space ({free_gb:.1f} GB) is close to the ~160 GB needed for videochatgpt_tune.zip. "
                "Extraction will proceed, but monitor disk space."
            )

        cmd_phase1 = [
            seven_zip,
            "x",
            str(first_split),
            f"-o{data_root}",
            "-bsp1",
            "-y",
        ]

        if args.dry_run:
            log_info(f"[DRY-RUN] Would run: {' '.join(cmd_phase1)}")
        else:
            ok = run_7z_with_progress(cmd_phase1, f"Extracting split archives to {MASTER_ZIP_NAME}")
            if not ok:
                log_err("Phase 1 extraction failed.")
                sys.exit(1)

            if not master_zip.exists():
                # Some archives unpack directly into target folder instead of a master zip
                if video_dir.exists() and any(video_dir.iterdir()):
                    log_success(f"Split archives unpacked directly into {video_dir.name}/.")
                else:
                    log_err(f"Expected {master_zip.name} or {video_dir.name}/ was not created by 7-Zip.")
                    sys.exit(1)
            else:
                size_gb = master_zip.stat().st_size / (1024 ** 3)
                log_success(f"Phase 1 complete: {master_zip.name} ready ({size_gb:.2f} GB).")

    # -------------------------------------------------------------
    # Step 3: Extract master_zip -> Videos (videochatgpt_tune/)
    # -------------------------------------------------------------
    if master_zip.exists():
        log_header(f"Phase 2: Extracting {master_zip.name} -> Video Clips")
        video_dir.mkdir(parents=True, exist_ok=True)

        # -o{data_root}: unpacks to {data_root}/videochatgpt_tune/
        # -aos: skip existing files (resumable)
        # -bsp1: progress bar
        # -y: assume yes
        cmd_phase2 = [
            seven_zip,
            "x",
            str(master_zip),
            f"-o{data_root}",
            "-aos",
            "-bsp1",
            "-y",
        ]

        if args.dry_run:
            log_info(f"[DRY-RUN] Would run: {' '.join(cmd_phase2)}")
        else:
            ok = run_7z_with_progress(cmd_phase2, f"Extracting video files from {master_zip.name}")
            if not ok:
                log_err("Phase 2 video extraction encountered an error or was interrupted.")
                log_warn("You can safely re-run this script to resume — existing videos will be skipped.")
                sys.exit(1)

    # -------------------------------------------------------------
    # Step 4: Normalization (Fix nested folders if needed)
    # -------------------------------------------------------------
    if not args.dry_run and video_dir.exists():
        flatten_nested_folder(video_dir)

    # -------------------------------------------------------------
    # Step 5: Verification & Cleanup
    # -------------------------------------------------------------
    log_header("Phase 3: Verification & Cleanup")
    if not args.dry_run:
        num_folders, num_videos = count_extracted_videos(video_dir)
        log_success(f"Extracted Content: {num_folders} subfolders, {num_videos:,} video files (.mp4)")

        # Cleanup intermediate master zip
        if not args.keep_intermediate and master_zip.exists():
            log_info(f"Deleting intermediate {master_zip.name} to reclaim ~160 GB...")
            try:
                master_zip.unlink()
                log_success(f"Deleted {master_zip.name}! Reclaimed intermediate space.")
            except OSError as exc:
                log_warn(f"Could not delete {master_zip.name}: {exc}")
        elif master_zip.exists():
            log_info(f"Retained intermediate {master_zip.name} (--keep_intermediate).")

        # Cleanup split archives if requested
        if args.clean_splits:
            log_info("Cleaning up original split archives (videochatgpt_tune_2.zip.001..005)...")
            reclaimed_bytes = 0
            for part in VIDEO_TUNE_PARTS:
                p = data_root / part
                if p.exists():
                    reclaimed_bytes += p.stat().st_size
                    p.unlink(missing_ok=True)
            log_success(f"Deleted split parts! Reclaimed {reclaimed_bytes / (1024**3):.1f} GB.")
        else:
            log_info("Split archives (videochatgpt_tune_2.zip.001..005) preserved. (Use --clean_splits to delete them).")

    final_free_gb = get_drive_free_gb(data_root)
    log_header("Video Tuning Dataset Ready")
    log_info(f"Extracted folder : {video_dir}")
    log_info(f"Free disk space  : {final_free_gb:.1f} GB")
    sys.exit(0)


if __name__ == "__main__":
    main()
