#!/usr/bin/env python3
"""
Quick In-Place Repair Script for llava_image.zip
=================================================
Repairs corrupted/zeroed byte regions in llava_image.zip using HTTP Range
requests to Hugging Face, avoiding a full ~27 GB re-download.
"""

import os
import sys
import time
import zipfile
import argparse
import urllib.request
from pathlib import Path

HF_URL = "https://huggingface.co/datasets/LanguageBind/Video-LLaVA/resolve/main/llava_image.zip"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    """Download bytes [start, end) inclusive from URL via HTTP Range request."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Range": f"bytes={start}-{end - 1}",
        },
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {resp.status}")
                data = resp.read()
                expected_len = end - start
                if len(data) != expected_len:
                    raise RuntimeError(f"Received {len(data)} bytes, expected {expected_len}")
                return data
        except Exception as exc:
            if attempt == retries:
                raise
            log(f"Attempt {attempt}/{retries} failed ({exc}). Retrying in 5s...")
            time.sleep(5)
    raise RuntimeError("Download failed after all retries.")


def repair_zero_blocks(zip_path: Path):
    if not zip_path.exists():
        log(f"Error: {zip_path} does not exist.")
        sys.exit(1)

    file_size = zip_path.stat().st_size
    log(f"Target archive: {zip_path} ({file_size / (1024**3):.2f} GB)")

    # 1. Scan for zeroed blocks (blocks of at least 1MB of consecutive zeros)
    log("Scanning for zeroed / corrupt byte blocks...")
    zero_chunk_size = 4 * 1024 * 1024  # 4 MB probe
    zero_regions = []

    with open(zip_path, "rb") as f:
        pos = 0
        in_zero = False
        block_start = 0

        while pos < file_size:
            read_len = min(zero_chunk_size, file_size - pos)
            chunk = f.read(read_len)
            if not chunk:
                break

            # Check if chunk is all zeros
            is_all_zero = (chunk == b"\x00" * len(chunk))
            if is_all_zero:
                if not in_zero:
                    in_zero = True
                    block_start = pos
            else:
                if in_zero:
                    # Scan within chunk to find exact end of zeroes
                    for i in range(len(chunk)):
                        if chunk[i] != 0:
                            zero_regions.append((block_start, pos + i))
                            in_zero = False
                            break
            pos += len(chunk)

        if in_zero:
            zero_regions.append((block_start, file_size))

    if not zero_regions:
        log("No zeroed regions detected.")
    else:
        log(f"Found {len(zero_regions)} zeroed region(s):")
        total_zero_bytes = 0
        for start, end in zero_regions:
            sz_mb = (end - start) / (1024 ** 2)
            total_zero_bytes += (end - start)
            log(f"  -> Offset {start:,} to {end:,} ({sz_mb:.2f} MB)")

        log(f"Total corrupted data to download: {total_zero_bytes / (1024**2):.2f} MB")
        log(f"(Instead of re-downloading {file_size / (1024**3):.2f} GB!)")

        # 2. Patch the corrupt blocks in place
        with open(zip_path, "r+b") as f:
            for idx, (start, end) in enumerate(zero_regions, 1):
                length = end - start
                log(f"Patching block {idx}/{len(zero_regions)}: {length / (1024**2):.2f} MB...")
                t0 = time.time()
                data = download_range(HF_URL, start, end)
                elapsed = max(time.time() - t0, 0.1)
                speed = len(data) / (1024 ** 2) / elapsed
                log(f"Downloaded {len(data) / (1024**2):.2f} MB in {elapsed:.1f}s [{speed:.1f} MB/s]")

                f.seek(start)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
                log(f"Wrote block {idx} successfully to disk.")

    # 3. Test integrity of the patched ZIP
    log("Verifying archive integrity with Python zipfile...")
    try:
        zf = zipfile.ZipFile(str(zip_path))
        # Test the previously broken file
        sample_check = "llava_image/00128/001286118.jpg"
        log(f"Testing extraction of previously broken file: {sample_check}...")
        data = zf.read(sample_check)
        log(f"SUCCESS! Extracted {sample_check} ({len(data):,} bytes)")
        log("Archive repair complete! All header errors fixed.")
    except Exception as exc:
        log(f"Verification error: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Repair corrupt blocks in llava_image.zip in-place.")
    parser.add_argument("--zip_path", type=str, default=r"E:\videollava_data\llava_image.zip",
                        help="Path to llava_image.zip")
    args = parser.parse_args()
    repair_zero_blocks(Path(args.zip_path))


if __name__ == "__main__":
    main()
