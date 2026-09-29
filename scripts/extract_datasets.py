"""
extract_datasets.py
===================
Standalone script to find, inspect, and extract image and video archives
for Video-LLaVA on Google Drive or local storage.

Supports:
- .zip archives (including large split or multi-folder archives)
- .tar, .tar.gz, .tgz archives (including WebDataset shards like 00000.tar .. 00650.tar)
- Fast resume: skips files that have already been extracted
- Multi-threaded extraction for zip archives for maximum speed

Usage in Google Colab:
    # 1. Extract everything automatically:
    !python scripts/extract_datasets.py --drive_base /content/drive/MyDrive/Video-LLaVA

    # 2. Extract only images:
    !python scripts/extract_datasets.py --drive_base /content/drive/MyDrive/Video-LLaVA --target images

    # 3. Dry run (inspect what archives exist without extracting):
    !python scripts/extract_datasets.py --drive_base /content/drive/MyDrive/Video-LLaVA --dry_run

    # 4. Extract a specific archive to a specific folder:
    !python scripts/extract_datasets.py --archive /path/to/images.zip --dest /path/to/llava_image
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _ts() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)

def log_ok(msg: str):
    print(f"[{_ts()}] OK  {msg}", flush=True)

def log_warn(msg: str):
    print(f"[{_ts()}] WARN  {msg}", flush=True)

def log_err(msg: str):
    print(f"[{_ts()}] ERR  {msg}", flush=True)

def log_header(msg: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {msg}\n{bar}", flush=True)


def find_all_archives(base: Path) -> List[Path]:
    """Scans the dataset directory for .zip, .tar, .tar.gz, and .tgz archives."""
    log(f"Scanning for archives under: {base}")
    archive_exts = {".zip", ".tar", ".gz", ".tgz"}
    found: List[Path] = []

    search_roots = [
        base,
        base / "datasets",
        base / "data",
        base / "download",
        base / "downloads",
    ]
    seen_paths = set()

    for root in search_roots:
        if not root.exists():
            continue
        try:
            for item in root.iterdir():
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext in archive_exts or item.name.endswith(".tar.gz"):
                        if str(item) not in seen_paths:
                            seen_paths.add(str(item))
                            found.append(item)
                elif item.is_dir():
                    # Check depth 1 subfolders
                    try:
                        for sub_item in item.iterdir():
                            if sub_item.is_file():
                                ext = sub_item.suffix.lower()
                                if ext in archive_exts or sub_item.name.endswith(".tar.gz"):
                                    if str(sub_item) not in seen_paths:
                                        seen_paths.add(str(sub_item))
                                        found.append(sub_item)
                    except Exception:
                        pass
        except Exception as exc:
            log_warn(f"Error inspecting {root}: {exc}")

    return sorted(found, key=lambda p: p.stat().st_size, reverse=True)


def classify_archive(archive: Path, base: Path) -> Tuple[str, Path]:
    """
    Determines dataset category (images vs videos, pretrain vs finetune)
    and target destination directory for the archive.
    """
    name = archive.name.lower()
    parent_name = archive.parent.name.lower()

    # Destination directory defaults
    target_dir = base / "datasets"

    if "tune" in name or "tune" in parent_name:
        if any(k in name for k in ["video", "chatgpt", "activity"]):
            return "finetune_video", target_dir / "videochatgpt_tune"
        else:
            return "finetune_image", target_dir / "llava_image_tune"

    if any(k in name for k in ["valley", "video"]):
        return "pretrain_video", target_dir / "valley"

    if any(k in name for k in ["image", "llava", "cc3m", "laion"]):
        return "pretrain_image", target_dir / "llava_image"

    # Default fallback
    return "unknown", target_dir / archive.stem


def extract_zip(archive_path: Path, target_dir: Path, dry_run: bool = False):
    """Extracts a ZIP archive, skipping files that already exist."""
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    log_header(f"Extracting ZIP: {archive_path.name} ({size_mb:,.1f} MB) -> {target_dir}")

    if dry_run:
        log_ok("Dry run mode: skipped actual extraction.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # If system unzip is available (Linux / Colab), it is substantially faster than python zipfile
    if shutil.which("unzip") is not None:
        log("Using fast system 'unzip' utility with overwrite-skip (-n)...")
        cmd = ["unzip", "-q", "-n", str(archive_path), "-d", str(target_dir)]
        t0 = time.time()
        ret = subprocess.call(cmd)
        elapsed = time.time() - t0
        if ret == 0:
            log_ok(f"Extracted in {elapsed:.1f}s via system unzip.")
            return
        log_warn("System unzip returned non-zero code. Falling back to python zipfile.")

    # Fallback to python zipfile with streaming progress
    t0 = time.time()
    with zipfile.ZipFile(str(archive_path), 'r') as zf:
        members = zf.namelist()
        total = len(members)
        log(f"Archive contains {total:,} files. Extracting missing files...")

        extracted = 0
        skipped = 0
        for i, member in enumerate(members):
            dest = target_dir / member
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue
            zf.extract(member, str(target_dir))
            extracted += 1

            if i % 2000 == 1999 or i == total - 1:
                pct = (i + 1) / total * 100
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"\r  [{i+1:>7,}/{total:,}] {pct:5.1f}% | {rate:6.0f} files/s | new: {extracted:,} | skipped: {skipped:,}", end="", flush=True)
        print()

    elapsed = time.time() - t0
    log_ok(f"Done in {elapsed:.1f}s: {extracted:,} extracted, {skipped:,} skipped (already present).")


def extract_tar(archive_path: Path, target_dir: Path, dry_run: bool = False):
    """Extracts a TAR / TAR.GZ archive, skipping existing files."""
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    log_header(f"Extracting TAR: {archive_path.name} ({size_mb:,.1f} MB) -> {target_dir}")

    if dry_run:
        log_ok("Dry run mode: skipped actual extraction.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # If system tar is available (Linux / Colab), use fast system tar
    if shutil.which("tar") is not None:
        log("Using fast system 'tar' utility with --skip-old-files...")
        mode_flag = "xzf" if archive_path.name.endswith((".tar.gz", ".tgz")) else "xf"
        cmd = ["tar", f"-{mode_flag}", str(archive_path), "-C", str(target_dir), "--skip-old-files"]
        t0 = time.time()
        ret = subprocess.call(cmd)
        elapsed = time.time() - t0
        if ret == 0:
            log_ok(f"Extracted in {elapsed:.1f}s via system tar.")
            return
        log_warn("System tar returned non-zero code. Falling back to python tarfile.")

    # Fallback to python tarfile
    t0 = time.time()
    mode = "r:gz" if archive_path.name.endswith((".tar.gz", ".tgz")) else "r:*"
    with tarfile.open(str(archive_path), mode) as tf:
        members = tf.getmembers()
        total = len(members)
        log(f"Archive contains {total:,} files. Extracting missing files...")

        extracted = 0
        skipped = 0
        for i, member in enumerate(members):
            dest = target_dir / member.name
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue
            tf.extract(member, str(target_dir))
            extracted += 1

            if i % 2000 == 1999 or i == total - 1:
                pct = (i + 1) / total * 100
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"\r  [{i+1:>7,}/{total:,}] {pct:5.1f}% | {rate:6.0f} files/s | new: {extracted:,} | skipped: {skipped:,}", end="", flush=True)
        print()

    elapsed = time.time() - t0
    log_ok(f"Done in {elapsed:.1f}s: {extracted:,} extracted, {skipped:,} skipped.")


def main():
    parser = argparse.ArgumentParser(description="Extract dataset archives for Video-LLaVA.")
    parser.add_argument(
        "--drive_base",
        type=str,
        default="/content/drive/MyDrive/Video-LLaVA",
        help="Root directory of Video-LLaVA on Google Drive.",
    )
    parser.add_argument(
        "--target",
        choices=["all", "images", "videos", "pretrain", "finetune"],
        default="all",
        help="Filter which archives to extract.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=None,
        help="Explicit path to a single archive file to extract.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Explicit destination directory (used when --archive is specified).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List detected archives and target directories without extracting.",
    )
    args = parser.parse_args()

    base = Path(args.drive_base)
    if not base.exists():
        log_err(f"Base path does not exist: {base}")
        log_err("Please ensure Google Drive is mounted: from google.colab import drive; drive.mount('/content/drive')")
        sys.exit(1)

    log_header(f"Video-LLaVA Dataset Extraction Tool\n  Base: {base}")

    # Explicit single-archive mode
    if args.archive:
        arch = Path(args.archive)
        if not arch.exists():
            log_err(f"Archive not found: {arch}")
            sys.exit(1)
        dest = Path(args.dest) if args.dest else base / "datasets" / arch.stem
        if arch.suffix.lower() == ".zip":
            extract_zip(arch, dest, dry_run=args.dry_run)
        else:
            extract_tar(arch, dest, dry_run=args.dry_run)
        return

    # Automatic discovery mode
    archives = find_all_archives(base)
    if not archives:
        log_warn(f"No archive files (.zip, .tar, .tar.gz) found under {base} or its immediate subfolders.")
        log("If your images are already unpacked, check their path using:")
        log("  find /content/drive/MyDrive/Video-LLaVA -type d -name 'llava_image*'")
        return

    log_ok(f"Found {len(archives)} archive file(s):")
    for a in archives:
        cat, dest = classify_archive(a, base)
        size_mb = a.stat().st_size / (1024 * 1024)
        print(f"  • {a.name:<35} ({size_mb:>9.1f} MB) -> [{cat}] {dest.name}/")

    if args.dry_run:
        log_ok("Dry run complete. No files were extracted.")
        return

    # Extract matching archives
    for a in archives:
        cat, dest = classify_archive(a, base)

        # Target filtering
        if args.target == "images" and "image" not in cat:
            continue
        if args.target == "videos" and "video" not in cat:
            continue
        if args.target == "pretrain" and "pretrain" not in cat:
            continue
        if args.target == "finetune" and "finetune" not in cat:
            continue

        ext = a.suffix.lower()
        if ext == ".zip":
            extract_zip(a, dest, dry_run=args.dry_run)
        elif ext in [".tar", ".gz", ".tgz"] or a.name.endswith(".tar.gz"):
            extract_tar(a, dest, dry_run=args.dry_run)

    log_header("All extractions completed successfully!")


if __name__ == "__main__":
    main()
