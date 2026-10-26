#!/usr/bin/env python3
"""
Image Dataset Downloader for Video-LLaVA / MuLER

Downloads the image datasets required for both pretraining (Stage 1) and
fine-tuning (Stage 2) from HuggingFace (LanguageBind/Video-LLaVA).

Datasets handled:
  1. Image Pretraining (~27.4 GB):
       llava_image.zip  ->  extracted to <data_root>/llava_image/
  2. Image Tuning (~71.4 GB, split archive):
       llava_image_tune_2.zip.001  (40.0 GB)
       llava_image_tune_2.zip.002  (28.4 GB)
       -> extracted to <data_root>/llava_image_tune/

Expected final directory layout:
  <data_root>/
  +-llava_image/          (COCO + GQA + TextVQA + VisualGenome images for Stage-1)
  +-llava_image_tune/     (LLaVA-Instruct images for Stage-2)

Usage:
  # Download and extract everything:
  python scripts/download_image_datasets.py --data_root ./data --extract

  # Download pretraining images only:
  python scripts/download_image_datasets.py --data_root ./data --components pretrain --extract

  # Download tuning images only (no extraction):
  python scripts/download_image_datasets.py --data_root ./data --components tune

  # Skip aria2c (use plain HTTP):
  python scripts/download_image_datasets.py --data_root ./data --no_aria2 --extract

  # Delete zip files after extraction:
  python scripts/download_image_datasets.py --data_root ./data --extract --clean
"""

import os
import sys
import time
import shutil
import zipfile
import argparse
import subprocess
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_BASE_URL = "https://huggingface.co/datasets/LanguageBind/Video-LLaVA/resolve/main"

# Single-file zip for image pretraining (~25.5 GB)
IMAGE_PRETRAIN_FILE = "llava_image.zip"
IMAGE_PRETRAIN_MIN_BYTES = 27_000_000_000  # actual is 27,363,981,620

# Split archive for image tuning (~67.4 GB total)
IMAGE_TUNE_PARTS = [
    "llava_image_tune_2.zip.001",
    "llava_image_tune_2.zip.002",
]
IMAGE_TUNE_PART_MIN_BYTES = {
    "llava_image_tune_2.zip.001": 41_000_000_000,  # actual is 41,943,040,000 (39.06 GiB)
    "llava_image_tune_2.zip.002": 30_000_000_000,  # actual is 30,468,547,028 (28.38 GiB)
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Download helper (resume-capable)
# ---------------------------------------------------------------------------

def download_file(url: str, output_path: Path, min_bytes: int = 0,
                  use_aria2: bool = True, max_retries: int = 10) -> bool:
    """
    Download *url* to *output_path* with resume support and automatic retries.

    Tries aria2c first (16 parallel connections) then falls back to a plain
    HTTP streaming download with Range-request resume.  On timeout or
    connection errors the HTTP fallback automatically resumes from the
    partially-downloaded file with exponential back-off (up to max_retries).

    Returns True on success, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- skip if already complete -------------------------------------------
    if output_path.exists():
        actual = output_path.stat().st_size
        if min_bytes > 0 and actual >= min_bytes:
            log_success(
                f"Already complete: {output_path.name} ({actual / (1024**3):.2f} GB)"
            )
            return True
        log_info(f"Partial file found ({actual / (1024**3):.2f} GB). Resuming...")

    # --- aria2c (fastest, handles retries internally) -----------------------
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
            sz = output_path.stat().st_size
            if min_bytes == 0 or sz >= min_bytes:
                log_success(
                    f"Downloaded: {output_path.name} ({sz / (1024**3):.2f} GB)"
                )
                return True
        log_warn("aria2c failed or produced an incomplete file. Falling back to HTTP.")

    # --- plain HTTP streaming with resume + auto-retry ----------------------
    log_info(f"Downloading {output_path.name} via HTTP stream (timeout=300s, retries={max_retries})...")

    for attempt in range(1, max_retries + 1):
        existing = output_path.stat().st_size if output_path.exists() else 0
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
                        if now - last_log >= 15:
                            elapsed = max(now - t0, 0.1)
                            speed = (downloaded - existing) / (1024 ** 2) / elapsed
                            pct = (downloaded / total * 100) if total > 0 else 0.0
                            log_info(
                                f"{output_path.name}: "
                                f"{downloaded / (1024**3):.2f} GB"
                                f" / {total / (1024**3):.2f} GB"
                                f"  ({pct:.1f}%)  [{speed:.1f} MB/s]"
                            )
                            last_log = now

            final_sz = output_path.stat().st_size
            log_success(
                f"Downloaded: {output_path.name} ({final_sz / (1024**3):.2f} GB)"
            )
            return True

        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                # 416 Range Not Satisfiable means requested range is at or past EOF.
                # File is already 100% downloaded!
                log_success(
                    f"Already complete (remote EOF reached): {output_path.name} ({existing / (1024**3):.2f} GB)"
                )
                return True
            wait = min(30 * attempt, 300)
            partial = output_path.stat().st_size if output_path.exists() else 0
            log_warn(
                f"Attempt {attempt}/{max_retries} failed for {output_path.name}: {exc}\n"
                f"  Resuming from {partial / (1024**3):.2f} GB in {wait}s..."
            )
            time.sleep(wait)

        except Exception as exc:
            wait = min(30 * attempt, 300)  # 30s, 60s, 90s … capped at 5 min
            partial = output_path.stat().st_size if output_path.exists() else 0
            log_warn(
                f"Attempt {attempt}/{max_retries} failed for {output_path.name}: {exc}\n"
                f"  Resuming from {partial / (1024**3):.2f} GB in {wait}s..."
            )
            time.sleep(wait)

    log_err(f"All {max_retries} attempts failed for {output_path.name}.")
    return False


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _find_7z():
    """Return the path to 7z/7za, or None if not found."""
    for candidate in ("7z", "7za"):
        path = shutil.which(candidate)
        if path:
            return path
    if os.name == "nt":
        for win_path in [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
        ]:
            if os.path.exists(win_path):
                return win_path
    return None


def _remove_parts(first_part: Path):
    """Delete all .zip.NNN siblings of a split archive."""
    base = str(first_part).rsplit(".", 1)[0]  # strip the numeric suffix
    for p in sorted(first_part.parent.glob(f"{Path(base).name}.*")):
        p.unlink(missing_ok=True)
        log_info(f"Deleted {p.name}")


def extract_single_zip(zip_path: Path, output_dir: Path, clean: bool = False) -> bool:
    """Extract a standard (non-split) zip archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Extracting {zip_path.name} -> {output_dir} ...")

    seven_zip = _find_7z()
    if seven_zip:
        cmd = [seven_zip, "x", str(zip_path), f"-o{output_dir}", "-y"]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            log_success(f"Extracted {zip_path.name} -> {output_dir}")
            if clean:
                zip_path.unlink(missing_ok=True)
                log_info(f"Deleted {zip_path.name} to save disk space.")
            return True
        log_warn(f"7z returned code {ret.returncode}. Falling back to Python zipfile.")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        log_success(f"Extracted {zip_path.name} -> {output_dir}")
        if clean:
            zip_path.unlink(missing_ok=True)
            log_info(f"Deleted {zip_path.name} to save disk space.")
        return True
    except Exception as exc:
        log_err(f"Failed to extract {zip_path.name}: {exc}")
        return False


def extract_split_archive(first_part: Path, output_dir: Path, clean: bool = False) -> bool:
    """
    Extract a multi-part split archive (.zip.001, .zip.002, ...) using 7z,
    or by concatenating the parts and unzipping on Linux.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Extracting split archive {first_part.name} -> {output_dir} ...")

    seven_zip = _find_7z()
    if seven_zip:
        cmd = [seven_zip, "x", str(first_part), f"-o{output_dir}", "-y"]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            log_success(f"Extracted split archive -> {output_dir}")
            if clean:
                _remove_parts(first_part)
            return True
        log_warn(f"7z returned code {ret.returncode}. Trying unzip fallback.")

    # Linux fallback: concatenate all parts, then unzip
    if shutil.which("unzip"):
        base = str(first_part).rsplit(".", 1)[0]
        parts = sorted(first_part.parent.glob(f"{Path(base).name}.*"))
        merged = first_part.parent / f"{Path(base).name}_merged.zip"
        log_info(f"Concatenating {len(parts)} part(s) -> {merged.name} ...")
        with open(merged, "wb") as out_f:
            for p in parts:
                log_info(f"  merging {p.name}")
                with open(p, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=64 * 1024 * 1024)
        cmd = ["unzip", "-q", "-o", str(merged), "-d", str(output_dir)]
        ret = subprocess.run(cmd)
        merged.unlink(missing_ok=True)
        if ret.returncode == 0:
            log_success(f"Extracted via unzip -> {output_dir}")
            if clean:
                _remove_parts(first_part)
            return True

    log_err(
        "Cannot extract split archive: neither 7z/7za nor unzip was found.\n"
        "  Windows : install 7-Zip from https://www.7-zip.org/\n"
        "  Linux   : sudo apt install p7zip-full   OR   sudo apt install unzip"
    )
    return False


# ---------------------------------------------------------------------------
# Per-dataset functions
# ---------------------------------------------------------------------------

def download_pretrain_images(data_root: Path, extract: bool,
                              clean: bool, use_aria2: bool) -> bool:
    """Download (and optionally extract) the image pretraining dataset."""
    log_header("Image Pretraining Dataset  (llava_image.zip  ~27.4 GB)")

    zip_path = data_root / IMAGE_PRETRAIN_FILE
    out_dir  = data_root / "llava_image"

    # Check if already complete by checking an expected marker or file count
    if out_dir.exists() and not zip_path.exists():
        log_success(f"Output directory already populated: {out_dir}")
        return True

    url = f"{HF_BASE_URL}/{IMAGE_PRETRAIN_FILE}"
    ok  = download_file(url, zip_path, min_bytes=IMAGE_PRETRAIN_MIN_BYTES,
                        use_aria2=use_aria2)
    if not ok:
        log_err("Failed to download image pretraining zip.")
        return False

    if extract:
        # Note: llava_image.zip internally has root folder 'llava_image/'
        # Extracting to data_root avoids double nesting (llava_image/llava_image)
        return extract_single_zip(zip_path, data_root, clean=clean)

    log_info("Skipping extraction (pass --extract to unzip automatically).")
    return True


def download_tune_images(data_root: Path, extract: bool,
                         clean: bool, use_aria2: bool) -> bool:
    """Download (and optionally extract) the image tuning dataset."""
    log_header("Image Tuning Dataset  (llava_image_tune_2.zip.001-002  ~71.4 GB)")

    out_dir = data_root / "llava_image_tune"
    first_part = data_root / IMAGE_TUNE_PARTS[0]

    if out_dir.exists() and not first_part.exists():
        log_success(f"Output directory already populated: {out_dir}")
        return True

    all_ok = True
    for part_name in IMAGE_TUNE_PARTS:
        url   = f"{HF_BASE_URL}/{part_name}"
        dest  = data_root / part_name
        min_b = IMAGE_TUNE_PART_MIN_BYTES.get(part_name, 0)
        ok    = download_file(url, dest, min_bytes=min_b, use_aria2=use_aria2)
        if not ok:
            all_ok = False
            log_err(f"Failed to download {part_name}. Aborting tune dataset download.")
            break

    if not all_ok:
        return False

    if extract:
        inner_zip = data_root / "llava_image_tune.zip"
        if inner_zip.exists():
            log_info(f"Found inner archive {inner_zip.name}. Extracting images to {data_root} ...")
            ok = extract_single_zip(inner_zip, data_root, clean=clean)
            if ok and clean:
                _remove_parts(first_part)
            return ok

        # 7-Zip handles split archives (.001, .002, ...) directly
        ok = extract_split_archive(first_part, data_root, clean=clean)
        if ok and inner_zip.exists():
            log_info(f"Extracting inner archive {inner_zip.name} -> {data_root} ...")
            ok = extract_single_zip(inner_zip, data_root, clean=clean)
            if ok and clean:
                _remove_parts(first_part)
        return ok

    log_info("Skipping extraction (pass --extract to unzip automatically).")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download image datasets for Video-LLaVA (pretraining & tuning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_root", type=str, default="./data",
        help="Root directory to store downloaded/extracted datasets (default: ./data)",
    )
    parser.add_argument(
        "--components", type=str, default="all",
        choices=["all", "pretrain", "tune"],
        help=(
            "Which image dataset to download:\n"
            "  all     – pretraining + tuning (default)\n"
            "  pretrain – only llava_image  (Stage-1)\n"
            "  tune     – only llava_image_tune  (Stage-2)"
        ),
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Automatically extract archives after downloading",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete zip/part files after successful extraction",
    )
    parser.add_argument(
        "--no_aria2", action="store_true",
        help="Disable aria2c and use plain HTTP streaming instead",
    )

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    use_aria2 = not args.no_aria2

    log_header("Video-LLaVA  --  Image Dataset Downloader")
    log_info(f"Data root  : {data_root}")
    log_info(f"Components : {args.components}")
    log_info(f"Extract    : {args.extract}")
    log_info(f"Clean zips : {args.clean}")
    log_info(f"Use aria2c : {use_aria2}")

    results = {}

    if args.components in ("all", "pretrain"):
        results["pretrain"] = download_pretrain_images(
            data_root, extract=args.extract, clean=args.clean, use_aria2=use_aria2
        )

    if args.components in ("all", "tune"):
        results["tune"] = download_tune_images(
            data_root, extract=args.extract, clean=args.clean, use_aria2=use_aria2
        )

    # --- summary ------------------------------------------------------------
    log_header("Summary")
    all_passed = True
    for name, ok in results.items():
        status = (
            f"{Colors.OKGREEN}OK{Colors.ENDC}"
            if ok else
            f"{Colors.FAIL}FAILED{Colors.ENDC}"
        )
        print(f"  {name:12s}: {status}")
        if not ok:
            all_passed = False

    if args.extract:
        print()
        log_info("Expected directory layout after extraction:")
        log_info(f"  {data_root}/")
        if "pretrain" in results:
            log_info("  +-llava_image/          (COCO / GQA / TextVQA / VisualGenome)")
        if "tune" in results:
            log_info("  +-llava_image_tune/     (LLaVA-Instruct fine-tuning images)")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
