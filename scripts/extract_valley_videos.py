#!/usr/bin/env python3
"""
Valley Video Dataset Extractor for Video-LLaVA
==============================================
Extracts the Valley video pretraining dataset (~432 GB) from either:
  1. The 12 split volumes (valley_2.zip.001 ... valley_2.zip.012)
  2. The assembled master archive (valley.zip)

Features:
  - Two-phase automated pipeline:
      Phase 1: valley_2.zip.001..012 -> valley.zip (if not already extracted)
      Phase 2: valley.zip -> video files (valley/*.mp4)
  - Resumable: skips already-extracted video files using 7-Zip (-aos)
  - Auto-flattens nested directory structures (avoids valley/valley/...)
  - Auto-cleans intermediate valley.zip to save ~432 GB of disk space
  - Real-time progress output and disk space safety checks

Usage:
  # Standard extraction (auto-deletes intermediate 432 GB valley.zip when done):
  python scripts/extract_valley_videos.py --data_root E:\\videollava_data

  # Keep the intermediate valley.zip:
  python scripts/extract_valley_videos.py --data_root E:\\videollava_data --keep_intermediate

  # Extract and also delete the original 12 split parts (saves ~864 GB total):
  python scripts/extract_valley_videos.py --data_root E:\\videollava_data --clean_splits
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

VALLEY_ZIP_EXPECTED_BYTES = 463_720_864_722  # ~431.87 GiB
VALLEY_FIRST_PART = "valley_2.zip.001"
VALLEY_ALL_PARTS = [f"valley_2.zip.{i:03d}" for i in range(1, 13)]


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
        # Pass stdout/stderr directly through to the terminal so 7-Zip's interactive
        # progress percentages and status lines display natively
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


def flatten_nested_valley(data_root: Path) -> None:
    """Check if extraction created valley/valley/ and flatten it to valley/."""
    valley_dir = data_root / "valley"
    nested_dir = valley_dir / "valley"
    if nested_dir.exists() and nested_dir.is_dir():
        log_info(f"Found nested directory: {nested_dir}. Flattening to {valley_dir}...")
        sub_items = list(nested_dir.iterdir())
        moved = 0
        for item in sub_items:
            target = valley_dir / item.name
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


def count_extracted_videos(valley_dir: Path) -> Tuple[int, int]:
    """Count subdirectories and .mp4 files inside valley directory."""
    if not valley_dir.exists():
        return 0, 0
    subdirs = [d for d in valley_dir.iterdir() if d.is_dir() and d.name != "valley"]
    total_videos = 0
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
        description="Extract Valley Video Dataset from split archives or master zip.",
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
        help="Keep the 432 GB intermediate valley.zip after extracting videos (default: auto-delete)",
    )
    parser.add_argument(
        "--clean_splits",
        action="store_true",
        help="Delete the 12 split volumes (valley_2.zip.001..012) after successful extraction to save another 432 GB",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Inspect existing files and plan extraction without modifying files",
    )

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    log_header("Valley Video Dataset Extractor")
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

    valley_zip = data_root / "valley.zip"
    first_split = data_root / VALLEY_FIRST_PART
    valley_dir = data_root / "valley"

    # -------------------------------------------------------------
    # Step 1: Check Master valley.zip Status
    # -------------------------------------------------------------
    has_valid_valley_zip = False
    if valley_zip.exists():
        actual_size = valley_zip.stat().st_size
        size_gb = actual_size / (1024 ** 3)
        expected_gb = VALLEY_ZIP_EXPECTED_BYTES / (1024 ** 3)
        if actual_size >= (VALLEY_ZIP_EXPECTED_BYTES - 100 * 1024 * 1024):
            # Within 100 MB of expected (complete)
            log_success(f"Found complete master archive: {valley_zip.name} ({size_gb:.2f} GB)")
            has_valid_valley_zip = True
        else:
            log_warn(
                f"Found partial/incomplete {valley_zip.name} ({size_gb:.2f} GB / {expected_gb:.2f} GB). "
                "This file is incomplete and must be replaced."
            )
            if not args.dry_run:
                log_info(f"Removing partial {valley_zip.name} to free space and re-extract clean...")
                valley_zip.unlink(missing_ok=True)
                has_valid_valley_zip = False

    # -------------------------------------------------------------
    # Step 2: Extract split parts -> valley.zip (if needed)
    # -------------------------------------------------------------
    if not has_valid_valley_zip:
        log_header("Phase 1: Extracting Split Volumes -> valley.zip")
        if not first_split.exists():
            log_err(f"Cannot extract: First part '{VALLEY_FIRST_PART}' not found in {data_root}.")
            sys.exit(1)

        # Check split volumes presence
        missing_parts = [p for p in VALLEY_ALL_PARTS if not (data_root / p).exists()]
        if missing_parts:
            log_err(f"Missing {len(missing_parts)} split volume(s): {missing_parts[:3]}...")
            log_err("Please ensure all 12 split volumes are fully downloaded.")
            sys.exit(1)
        log_success("All 12 split volumes verified present.")

        free_gb = get_drive_free_gb(data_root)
        if free_gb < 450:
            log_warn(
                f"Free space ({free_gb:.1f} GB) is close to the ~432 GB needed for valley.zip. "
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
            ok = run_7z_with_progress(cmd_phase1, "Extracting valley_2 split archives to valley.zip")
            if not ok:
                log_err("Phase 1 extraction failed.")
                sys.exit(1)

            if not valley_zip.exists():
                log_err(f"Expected {valley_zip.name} was not created by 7-Zip.")
                sys.exit(1)

            size_gb = valley_zip.stat().st_size / (1024 ** 3)
            log_success(f"Phase 1 complete: {valley_zip.name} ready ({size_gb:.2f} GB).")

    # -------------------------------------------------------------
    # Step 3: Extract valley.zip -> Videos (valley/)
    # -------------------------------------------------------------
    log_header("Phase 2: Extracting valley.zip -> Video Files")
    if not valley_zip.exists() and not args.dry_run:
        log_err(f"Cannot proceed: {valley_zip.name} is missing.")
        sys.exit(1)

    # 7-Zip extract with:
    #   -o{data_root} : valley.zip contains root 'valley/', so this unpacks to {data_root}/valley/
    #   -aos          : skip existing files (resumes without re-extracting already unpacked videos!)
    #   -bsp1         : show real-time percentage progress
    #   -y            : assume Yes on all queries
    cmd_phase2 = [
        seven_zip,
        "x",
        str(valley_zip),
        f"-o{data_root}",
        "-aos",
        "-bsp1",
        "-y",
    ]

    if args.dry_run:
        log_info(f"[DRY-RUN] Would run: {' '.join(cmd_phase2)}")
    else:
        ok = run_7z_with_progress(cmd_phase2, "Extracting video files from valley.zip into valley/")
        if not ok:
            log_err("Phase 2 video extraction encountered an error or was interrupted.")
            log_warn("You can safely re-run this script to resume — existing videos will be skipped.")
            sys.exit(1)

    # -------------------------------------------------------------
    # Step 4: Normalization (Fix nested valley/valley if needed)
    # -------------------------------------------------------------
    if not args.dry_run:
        flatten_nested_valley(data_root)

    # -------------------------------------------------------------
    # Step 5: Verification & Cleanup
    # -------------------------------------------------------------
    log_header("Phase 3: Verification & Cleanup")
    if not args.dry_run:
        num_folders, num_videos = count_extracted_videos(valley_dir)
        log_success(f"Valley Dataset Content: {num_folders} subfolders, {num_videos:,} video clips (.mp4)")

        # Cleanup intermediate master zip
        if not args.keep_intermediate:
            if valley_zip.exists():
                log_info(f"Deleting intermediate {valley_zip.name} to reclaim ~432 GB...")
                try:
                    valley_zip.unlink()
                    log_success(f"Deleted {valley_zip.name}! Reclaimed ~432 GB.")
                except OSError as exc:
                    log_warn(f"Could not delete {valley_zip.name}: {exc}")
        else:
            log_info(f"Retained intermediate {valley_zip.name} (--keep_intermediate).")

        # Cleanup split archives if requested
        if args.clean_splits:
            log_info("Cleaning up original split archives (valley_2.zip.001..012)...")
            reclaimed_bytes = 0
            for part in VALLEY_ALL_PARTS:
                p = data_root / part
                if p.exists():
                    reclaimed_bytes += p.stat().st_size
                    p.unlink(missing_ok=True)
            log_success(f"Deleted split parts! Reclaimed {reclaimed_bytes / (1024**3):.1f} GB.")
        else:
            log_info("Split archives (valley_2.zip.001..012) preserved. (Use --clean_splits to delete them).")

    final_free_gb = get_drive_free_gb(data_root)
    log_header("Valley Video Dataset Ready")
    log_info(f"Extracted folder : {valley_dir}")
    log_info(f"Free disk space  : {final_free_gb:.1f} GB")


if __name__ == "__main__":
    main()
