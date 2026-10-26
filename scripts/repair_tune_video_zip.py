#!/usr/bin/env python3
"""
In-Place Repair Script for Video-ChatGPT Tuning Dataset Split Archives
======================================================================
Repairs corrupted/zeroed byte regions in videochatgpt_tune_2.zip.00N using
HTTP Range requests to Hugging Face, avoiding a full 40 GB re-download per part.

The script:
  1. Scans the target file for consecutive zero-byte regions (>= 1 MB)
  2. Downloads only those regions from HuggingFace via HTTP Range requests
  3. Patches them in-place on disk
  4. Re-scans to verify the repair

Usage:
  # Repair part 1:
  python scripts/repair_tune_video_zip.py --part 1

  # Repair all 5 parts on custom drive:
  python scripts/repair_tune_video_zip.py --all --data_root E:\\videollava_data

  # Only scan without patching:
  python scripts/repair_tune_video_zip.py --all --scan_only
"""

import os
import sys
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple

HF_BASE_URL = "https://huggingface.co/datasets/LanguageBind/Video-LLaVA/resolve/main"

VIDEO_TUNE_PARTS = [
    "videochatgpt_tune_2.zip.001",
    "videochatgpt_tune_2.zip.002",
    "videochatgpt_tune_2.zip.003",
    "videochatgpt_tune_2.zip.004",
    "videochatgpt_tune_2.zip.005",
]

EXACT_BYTES: Dict[str, int] = {
    "videochatgpt_tune_2.zip.001": 41_943_040_000,
    "videochatgpt_tune_2.zip.002": 41_943_040_000,
    "videochatgpt_tune_2.zip.003": 41_943_040_000,
    "videochatgpt_tune_2.zip.004": 41_943_040_000,
    "videochatgpt_tune_2.zip.005": 4_103_350_672,
}

# Minimum consecutive zero bytes to be treated as a corrupt region (1 MB)
MIN_ZERO_REGION_BYTES = 1 * 1024 * 1024

# Scan chunk size (8 MB)
SCAN_CHUNK = 8 * 1024 * 1024

# Download chunk size for patching (16 MB)
PATCH_CHUNK = 16 * 1024 * 1024

ZIP_MAGIC = b'PK\x03\x04'


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


def scan_zero_regions(file_path: Path) -> List[Tuple[int, int]]:
    """
    Scan the file for contiguous zero-byte regions >= MIN_ZERO_REGION_BYTES.
    Returns a list of (start_offset, end_offset) tuples (end is exclusive).
    """
    file_size = file_path.stat().st_size
    log_info(f"Scanning {file_path.name} ({file_size / (1024**3):.2f} GB) for corrupt regions...")

    zero_regions: List[Tuple[int, int]] = []
    pos = 0
    in_zero_run = False
    run_start = 0
    last_progress_report = 0

    with open(file_path, "rb") as f:
        while pos < file_size:
            read_len = min(SCAN_CHUNK, file_size - pos)
            chunk = f.read(read_len)
            if not chunk:
                break

            if chunk == b"\x00" * len(chunk):
                # Entire chunk is zeros
                if not in_zero_run:
                    in_zero_run = True
                    run_start = pos
            else:
                if in_zero_run:
                    # Find exact byte where zeros end within this chunk
                    for i, byte in enumerate(chunk):
                        if byte != 0:
                            region_len = (pos + i) - run_start
                            if region_len >= MIN_ZERO_REGION_BYTES:
                                zero_regions.append((run_start, pos + i))
                            in_zero_run = False
                            break
                else:
                    # Detect zero runs starting inside a non-all-zero chunk
                    i = 0
                    while i < len(chunk):
                        if chunk[i] == 0:
                            zero_start_in_chunk = i
                            while i < len(chunk) and chunk[i] == 0:
                                i += 1
                            zero_end_in_chunk = i
                            run_len = zero_end_in_chunk - zero_start_in_chunk
                            if zero_end_in_chunk == len(chunk):
                                # Run continues into next chunk
                                in_zero_run = True
                                run_start = pos + zero_start_in_chunk
                            elif run_len >= MIN_ZERO_REGION_BYTES:
                                zero_regions.append(
                                    (pos + zero_start_in_chunk, pos + zero_end_in_chunk)
                                )
                        else:
                            i += 1

            pos += len(chunk)

            # Progress every 2 GB
            if pos - last_progress_report >= 2 * 1024 ** 3:
                log_info(f"  Scanned {pos / (1024**3):.1f} / {file_size / (1024**3):.1f} GB ...")
                last_progress_report = pos

    # Close any open run at EOF
    if in_zero_run:
        region_len = file_size - run_start
        if region_len >= MIN_ZERO_REGION_BYTES:
            zero_regions.append((run_start, file_size))

    return zero_regions


def download_range(url: str, start: int, end: int, retries: int = 8) -> bytes:
    """Download bytes [start, end) from URL via HTTP Range request."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Range": f"bytes={start}-{end - 1}",
        },
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {resp.status}")
                data = resp.read()
                expected_len = end - start
                if len(data) != expected_len:
                    raise RuntimeError(
                        f"Received {len(data):,} bytes, expected {expected_len:,}"
                    )
                return data
        except Exception as exc:
            wait = min(15 * attempt, 120)
            if attempt == retries:
                raise
            log_warn(f"Attempt {attempt}/{retries} failed ({exc}). Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Download failed after all retries.")


def patch_region(file_path: Path, url: str, start: int, end: int, idx: int, total: int):
    """Download bytes [start, end) and write them in-place to file_path."""
    length = end - start
    log_info(
        f"Patching region {idx}/{total}: offsets {start:,} - {end:,} "
        f"({length / (1024**2):.1f} MB)..."
    )

    downloaded = 0
    t0 = time.time()

    with open(file_path, "r+b") as f:
        f.seek(start)
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + PATCH_CHUNK, end)
            chunk = download_range(url, cursor, chunk_end)
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = max(time.time() - t0, 0.001)
            speed = downloaded / (1024**2) / elapsed
            pct = downloaded / length * 100
            log_info(
                f"  {downloaded / (1024**2):.1f} / {length / (1024**2):.1f} MB "
                f"({pct:.1f}%) [{speed:.1f} MB/s]"
            )
            cursor = chunk_end

        f.flush()
        os.fsync(f.fileno())

    log_success(f"Region {idx}/{total} patched and flushed to disk.")


def repair_part(file_path: Path, part_name: str, scan_only: bool = False) -> bool:
    """Scan and optionally repair a single split archive part."""
    log_header(f"Inspecting & Repairing {part_name}")

    if not file_path.exists():
        log_err(f"File not found: {file_path}")
        return False

    actual_size = file_path.stat().st_size
    expected_size = EXACT_BYTES.get(part_name, 0)
    url = f"{HF_BASE_URL}/{part_name}"

    # Check magic bytes on first part
    if part_name.endswith(".001"):
        try:
            with open(file_path, "rb") as f:
                hdr = f.read(4)
            if hdr != ZIP_MAGIC:
                log_err(f"{part_name}: Magic header is {hdr.hex()!r} (expected {ZIP_MAGIC.hex()!r}).")
                if not scan_only:
                    log_info("Patching first 64 KB of part 1 to fix zip header...")
                    hdr_data = download_range(url, 0, 65536)
                    with open(file_path, "r+b") as f:
                        f.seek(0)
                        f.write(hdr_data)
                        f.flush()
                        os.fsync(f.fileno())
                    log_success("Patched zip header.")
        except Exception as exc:
            log_warn(f"Could not verify header for {part_name}: {exc}")

    # Check if file size is significantly smaller than expected
    if expected_size > 0 and actual_size < expected_size:
        diff_bytes = expected_size - actual_size
        diff_mb = diff_bytes / (1024 ** 2)
        log_warn(
            f"{part_name} is undersized: {actual_size / (1024**3):.2f} / "
            f"{expected_size / (1024**3):.2f} GB (missing {diff_mb:.1f} MB at tail)."
        )
        if not scan_only and diff_bytes <= 200 * 1024 * 1024:
            # If missing <= 200 MB, patch the missing tail directly
            log_info(f"Missing tail is small ({diff_mb:.1f} MB). Appending tail via Range request...")
            try:
                tail_data = download_range(url, actual_size, expected_size)
                with open(file_path, "ab") as f:
                    f.write(tail_data)
                    f.flush()
                    os.fsync(f.fileno())
                actual_size = file_path.stat().st_size
                log_success(f"Appended tail: {part_name} now at {actual_size / (1024**3):.2f} GB.")
            except Exception as e:
                log_err(f"Failed to append tail for {part_name}: {e}")
                return False
        elif actual_size < expected_size:
            log_err(f"{part_name} is incomplete. Please run downloader to complete this part.")
            return False

    # Scan for interior zero blocks
    zero_regions = scan_zero_regions(file_path)

    if not zero_regions:
        log_success(f"No corrupt/zeroed regions found in {part_name}. File is healthy!")
        return True

    total_corrupt_bytes = sum(e - s for s, e in zero_regions)
    log_warn(f"Found {len(zero_regions)} corrupt zero-byte region(s) in {part_name}:")
    for s, e in zero_regions:
        log_warn(f"  -> Offset {s:,} to {e:,} ({(e - s) / (1024**2):.1f} MB)")
    log_info(
        f"Total corrupt data: {total_corrupt_bytes / (1024**2):.1f} MB "
        f"(vs {file_path.stat().st_size / (1024**3):.2f} GB full re-download)"
    )

    if scan_only:
        log_info("--scan_only mode: skipping patch.")
        return False

    for idx, (start, end) in enumerate(zero_regions, 1):
        patch_region(file_path, url, start, end, idx, len(zero_regions))

    # Verify: re-scan to confirm zeros are gone
    log_info("Re-scanning to verify repair...")
    remaining = scan_zero_regions(file_path)
    if not remaining:
        log_success(f"{part_name} repaired successfully! All corrupt blocks resolved.")
        return True
    else:
        log_err(
            f"{part_name} still has {len(remaining)} corrupt region(s) after patch. "
            "Try re-running or check network connection."
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="In-place repair of corrupt Video-ChatGPT fine-tuning dataset split archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_root", type=str, default=r"E:\videollava_data",
        help="Directory containing the videochatgpt_tune_2.zip.00N files (default: E:\\videollava_data)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--part", type=int, metavar="N",
        help="Part number to repair (1-5), e.g. --part 1",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Scan and repair all 5 parts",
    )
    parser.add_argument(
        "--scan_only", action="store_true",
        help="Only scan for corrupt regions, do not patch",
    )

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()

    if args.all:
        parts_to_check = VIDEO_TUNE_PARTS
    else:
        if not (1 <= args.part <= 5):
            log_err("--part must be between 1 and 5.")
            sys.exit(1)
        parts_to_check = [f"videochatgpt_tune_2.zip.{args.part:03d}"]

    overall_ok = True
    for part_name in parts_to_check:
        file_path = data_root / part_name
        ok = repair_part(file_path, part_name, scan_only=args.scan_only)
        if not ok and not args.scan_only:
            overall_ok = False

    if overall_ok and not args.scan_only:
        log_success("All requested video tuning parts are verified healthy and ready for extraction.")
        sys.exit(0)
    elif args.scan_only:
        log_info("Scan complete.")
        sys.exit(0)
    else:
        log_err("One or more parts failed verification/repair.")
        sys.exit(1)


if __name__ == "__main__":
    main()
