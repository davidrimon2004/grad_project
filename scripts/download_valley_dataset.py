#!/usr/bin/env python3
"""
Valley Video Pretraining Dataset Downloader for Video-LLaVA
============================================================
Downloads the Valley dataset (Stage 1 Video Pretraining, ~432 GB)
from Hugging Face (LanguageBind/Video-LLaVA).

The dataset is partitioned into 12 split volumes:
  valley_2.zip.001 through valley_2.zip.011 (39.06 GB each)
  valley_2.zip.012 (2.19 GB)
Total size: ~431.87 GB (463,720,864,992 bytes)

Usage:
  # 1. Download all parts to external drive:
  python scripts/download_valley_dataset.py --data_root E:\\videollava_data

  # 2. Download and extract automatically with 7-Zip:
  python scripts/download_valley_dataset.py --data_root E:\\videollava_data --extract

  # 3. Download, extract, and auto-delete archives to save 432 GB:
  python scripts/download_valley_dataset.py --data_root E:\\videollava_data --extract --clean

  # 4. Download specific parts (e.g. parts 1 to 3):
  python scripts/download_valley_dataset.py --data_root E:\\videollava_data --parts 1 2 3
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional

HF_BASE_URL = "https://huggingface.co/datasets/LanguageBind/Video-LLaVA/resolve/main"

# Exact byte sizes on Hugging Face
VALLEY_PARTS_EXACT_BYTES: Dict[str, int] = {
    f"valley_2.zip.{i:03d}": 41_943_040_000 for i in range(1, 12)
}
VALLEY_PARTS_EXACT_BYTES["valley_2.zip.012"] = 2_347_424_992

VALLEY_ALL_PARTS = [f"valley_2.zip.{i:03d}" for i in range(1, 13)]
TOTAL_DATASET_BYTES = sum(VALLEY_PARTS_EXACT_BYTES.values())


class Colors:
    HEADER  = "\033[95m"
    OKBLUE  = "\033[94m"
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
    sep = "=" * 60
    print(f"\n{Colors.HEADER}{Colors.BOLD}{sep}\n{msg}\n{sep}{Colors.ENDC}", flush=True)


def _find_7z() -> Optional[str]:
    """Return path to 7z.exe or None."""
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


ZIP_MAGIC = b'PK\x03\x04'


def _is_valid_zip_part(path: Path, is_first_part: bool) -> bool:
    """Return True if the file looks like a valid zip/split-zip part.

    Only the very first part of a split archive starts with the PK\x03\x04
    magic.  Subsequent parts are raw continuation data and cannot be validated
    by magic alone, so we skip the check for them.
    """
    if not is_first_part:
        return True  # Cannot validate continuation parts by magic
    try:
        with open(path, 'rb') as f:
            header = f.read(4)
        if header == ZIP_MAGIC:
            return True
        log_warn(
            f"{path.name}: first 4 bytes are {header.hex()!r} "
            f"(expected {ZIP_MAGIC.hex()!r} for a valid zip). "
            "File appears corrupt — will re-download."
        )
        return False
    except OSError as exc:
        log_warn(f"Cannot read {path.name} for validation: {exc}")
        return False


def download_part(url: str, output_path: Path, expected_bytes: int,
                  use_aria2: bool = True, max_retries: int = 10,
                  is_first_part: bool = False) -> bool:
    """Download a single part with resume, auto-retry, and magic-byte validation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already complete AND valid
    if output_path.exists():
        actual = output_path.stat().st_size
        if actual >= expected_bytes:
            if _is_valid_zip_part(output_path, is_first_part):
                log_success(f"Already complete: {output_path.name} ({actual / (1024**3):.2f} GB)")
                return True
            else:
                log_warn(f"Deleting corrupt file and re-downloading: {output_path.name}")
                output_path.unlink()
        else:
            log_info(f"Partial file found: {output_path.name} ({actual / (1024**3):.2f} / {expected_bytes / (1024**3):.2f} GB). Resuming...")

    # Fast aria2c if available
    if use_aria2 and shutil.which("aria2c"):
        cmd = [
            "aria2c",
            "-x", "16",
            "-s", "16",
            "-j", "16",
            "-k", "1M",
            "--continue=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--max-tries=10",
            "--retry-wait=5",
            "-d", str(output_path.parent),
            "-o", output_path.name,
            url,
        ]
        log_info(f"Downloading {output_path.name} via aria2c (16 connections)...")
        ret = subprocess.run(cmd)
        if ret.returncode == 0 and output_path.exists():
            if output_path.stat().st_size >= expected_bytes:
                log_success(f"Downloaded: {output_path.name} ({output_path.stat().st_size / (1024**3):.2f} GB)")
                return True
        log_warn("aria2c failed or incomplete. Falling back to HTTP streaming.")

    # HTTP streaming fallback
    for attempt in range(1, max_retries + 1):
        existing = output_path.stat().st_size if output_path.exists() else 0
        if existing >= expected_bytes:
            log_success(f"Already complete: {output_path.name} ({existing / (1024**3):.2f} GB)")
            return True

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if existing > 0:
            req.add_header("Range", f"bytes={existing}-")

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                content_length = int(resp.headers.get("Content-Length", 0))
                total = content_length + existing
                mode = "ab" if existing > 0 else "wb"
                chunk_size = 8 * 1024 * 1024  # 8 MB
                downloaded = existing
                t0 = time.time()
                last_log = t0

                with open(output_path, mode) as f:
                    while True:
                        buf = resp.read(chunk_size)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        now = time.time()
                        if now - last_log >= 20:
                            elapsed = max(now - t0, 0.1)
                            speed = (downloaded - existing) / (1024 ** 2) / elapsed
                            pct = (downloaded / total * 100) if total > 0 else 0.0
                            remaining_mb = (total - downloaded) / (1024 ** 2)
                            eta_sec = (remaining_mb / speed) if speed > 0 else 0
                            eta_str = time.strftime("%Hh %Mm", time.gmtime(eta_sec))
                            log_info(
                                f"{output_path.name}: "
                                f"{downloaded / (1024**3):.2f} / {total / (1024**3):.2f} GB "
                                f"({pct:.1f}%) [{speed:.1f} MB/s, ETA: {eta_str}]"
                            )
                            last_log = now

            final_sz = output_path.stat().st_size
            log_success(f"Downloaded: {output_path.name} ({final_sz / (1024**3):.2f} GB)")
            return True

        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                log_success(f"Already complete (remote EOF): {output_path.name} ({existing / (1024**3):.2f} GB)")
                return True
            wait = min(30 * attempt, 300)
            log_warn(f"Attempt {attempt}/{max_retries} failed for {output_path.name}: {exc}. Resuming in {wait}s...")
            time.sleep(wait)

        except Exception as exc:
            wait = min(30 * attempt, 300)
            log_warn(f"Attempt {attempt}/{max_retries} failed for {output_path.name}: {exc}. Resuming in {wait}s...")
            time.sleep(wait)

    log_err(f"All {max_retries} attempts failed for {output_path.name}.")
    return False


def extract_valley(data_root: Path, clean: bool = False) -> bool:
    """Extract valley_2.zip.001 using 7-Zip."""
    first_part = data_root / "valley_2.zip.001"
    if not first_part.exists():
        log_err(f"Cannot extract: {first_part.name} not found.")
        return False

    seven_zip = _find_7z()
    if not seven_zip:
        log_err("7-Zip not found. Please install 7-Zip (winget install 7zip.7zip) to extract split archive.")
        return False

    log_header("Extracting Valley Video Dataset with 7-Zip")
    log_info(f"Target directory: {data_root}")
    cmd = [seven_zip, "x", str(first_part), f"-o{data_root}", "-y"]
    ret = subprocess.run(cmd)

    if ret.returncode == 0:
        log_success("Extracted Valley dataset successfully!")
        if clean:
            log_info("Cleaning up Valley split archives to save ~432 GB...")
            for i in range(1, 13):
                p = data_root / f"valley_2.zip.{i:03d}"
                if p.exists():
                    p.unlink()
                    log_info(f"Deleted {p.name}")
            log_success("All archive files deleted. Reclaimed ~432 GB!")
        return True
    else:
        log_err(f"7-Zip extraction returned error code {ret.returncode}")
        return False


def verify_parts(data_root: Path, parts: list) -> None:
    """Check size and magic bytes for all downloaded parts and report status."""
    log_header("Verifying Downloaded Valley Parts")
    all_ok = True
    for i, part_name in enumerate(parts):
        path = data_root / part_name
        expected = VALLEY_PARTS_EXACT_BYTES[part_name]
        is_first = (i == 0)
        if not path.exists():
            log_warn(f"MISSING  : {part_name}")
            all_ok = False
        else:
            actual = path.stat().st_size
            size_ok = actual >= expected
            magic_ok = _is_valid_zip_part(path, is_first)
            status = "OK" if (size_ok and magic_ok) else "CORRUPT"
            sym = "[OK]" if status == "OK" else "[XX]"
            msg = f"{sym} {part_name}  {actual / (1024**3):.2f}/{expected / (1024**3):.2f} GB  [{status}]"
            if status == "OK":
                log_success(msg)
            else:
                log_err(msg)
                all_ok = False
    if all_ok:
        log_success("All parts verified successfully.")
    else:
        log_warn("Some parts are missing or corrupt. Re-run the downloader to fix them.")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract the Valley video pretraining dataset (~432 GB).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_root", type=str, default="E:\\videollava_data",
        help="Target folder on your external HDD (default: E:\\videollava_data)",
    )
    parser.add_argument(
        "--parts", type=int, nargs="+", default=None,
        help="Optional: list of specific part numbers to download (e.g. --parts 1 2 3)",
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Automatically extract the dataset with 7-Zip after all parts download",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete split zip archives after successful extraction to save 432 GB",
    )
    parser.add_argument(
        "--no_aria2", action="store_true",
        help="Force plain HTTP stream instead of aria2c",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Check size and magic bytes of all downloaded parts without downloading",
    )

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    use_aria2 = not args.no_aria2

    # Determine which parts to process
    if args.parts:
        selected_parts = [f"valley_2.zip.{i:03d}" for i in args.parts if 1 <= i <= 12]
    else:
        selected_parts = VALLEY_ALL_PARTS

    # --verify: check existing files and exit
    if args.verify:
        verify_parts(data_root, selected_parts)
        return

    log_header(f"Valley Video Pretraining Dataset Downloader ({len(selected_parts)} parts)")
    log_info(f"Target Directory : {data_root}")
    log_info(f"Total Size       : {TOTAL_DATASET_BYTES / (1024**3):.2f} GiB (~432 GB)")
    log_info(f"Auto-extract     : {args.extract}")
    log_info(f"Auto-clean       : {args.clean}")

    all_ok = True
    for i, part_name in enumerate(selected_parts):
        url = f"{HF_BASE_URL}/{part_name}"
        dest = data_root / part_name
        expected_bytes = VALLEY_PARTS_EXACT_BYTES[part_name]
        ok = download_part(url, dest, expected_bytes, use_aria2=use_aria2,
                           is_first_part=(i == 0 and part_name == VALLEY_ALL_PARTS[0]))
        if not ok:
            all_ok = False
            log_err(f"Download failed for {part_name}. Stopping batch.")
            break

    if all_ok:
        log_success("All requested Valley parts downloaded successfully!")
        if args.extract:
            extract_valley(data_root, clean=args.clean)
    else:
        log_warn("Download was interrupted or encountered errors. Re-run to resume safely.")


if __name__ == "__main__":
    main()
