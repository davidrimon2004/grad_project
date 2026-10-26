#!/usr/bin/env python3
"""
High-Performance Downloader & Extractor for Video-LLaVA / MuLER Fine-Tuning Datasets (Stage 2)
=============================================================================================

Datasets handled:
  1. Annotations (Google Drive ~471MB):
     - llava_image_tune_.json
     - videochatgpt_tune_.json
     - nlp_tune.json
  2. Image Tuning Dataset (HuggingFace ~67.4 GB):
     - llava_image_tune_2.zip.001 (39.06 GB)
     - llava_image_tune_2.zip.002 (28.38 GB)
  3. Video Tuning Dataset (HuggingFace ~160.1 GB):
     - videochatgpt_tune_2.zip.001 (39.06 GB)
     - videochatgpt_tune_2.zip.002 (39.06 GB)
     - videochatgpt_tune_2.zip.003 (39.06 GB)
     - videochatgpt_tune_2.zip.004 (39.06 GB)
     - videochatgpt_tune_2.zip.005 (3.82 GB)

Usage:
  # 1. Download video tuning dataset only (recommended):
  python scripts/download_finetune_data.py --data_root E:\\videollava_data --components video

  # 2. Download and automatically extract:
  python scripts/download_finetune_data.py --data_root E:\\videollava_data --components video --extract

  # 3. Verify downloaded parts without downloading:
  python scripts/download_finetune_data.py --data_root E:\\videollava_data --components video --verify
"""

import os
import sys
import time
import json
import shutil
import zipfile
import argparse
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional

HF_BASE_URL = "https://huggingface.co/datasets/LanguageBind/Video-LLaVA/resolve/main"
GDRIVE_ANNOTATIONS_URL = "https://drive.usercontent.google.com/download?id=1zGRyVSUMoczGq6cjQFmT0prH67bu2wXD&export=download&confirm=t"

IMAGE_TUNE_PARTS = [
    "llava_image_tune_2.zip.001",
    "llava_image_tune_2.zip.002",
]

VIDEO_TUNE_PARTS = [
    "videochatgpt_tune_2.zip.001",
    "videochatgpt_tune_2.zip.002",
    "videochatgpt_tune_2.zip.003",
    "videochatgpt_tune_2.zip.004",
    "videochatgpt_tune_2.zip.005",
]

EXACT_BYTES: Dict[str, int] = {
    "llava_image_tune_2.zip.001": 41_943_040_000,
    "llava_image_tune_2.zip.002": 30_468_547_028,
    "videochatgpt_tune_2.zip.001": 41_943_040_000,
    "videochatgpt_tune_2.zip.002": 41_943_040_000,
    "videochatgpt_tune_2.zip.003": 41_943_040_000,
    "videochatgpt_tune_2.zip.004": 41_943_040_000,
    "videochatgpt_tune_2.zip.005": 4_103_350_672,
}

FT_ANNOTATION_FILES = [
    "llava_image_tune_.json",
    "videochatgpt_tune_.json",
    "nlp_tune.json",
]

ZIP_MAGIC = b'PK\x03\x04'


class Colors:
    HEADER  = '\033[95m'
    OKCYAN  = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    ENDC    = '\033[0m'
    BOLD    = '\033[1m'


def log_info(msg: str):
    print(f"{Colors.OKCYAN}[INFO]{Colors.ENDC}    {msg}", flush=True)


def log_success(msg: str):
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {msg}", flush=True)


def log_warn(msg: str):
    print(f"{Colors.WARNING}[WARN]{Colors.ENDC}    {msg}", flush=True)


def log_err(msg: str):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC}   {msg}", flush=True)


def log_header(msg: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*64}\n  {msg}\n{'='*64}{Colors.ENDC}", flush=True)


def find_7z() -> Optional[str]:
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


def is_valid_zip_part(path: Path, is_first_part: bool) -> bool:
    """Return True if the file looks like a valid zip part."""
    if not is_first_part:
        return True  # Continuation parts don't have PK header
    try:
        with open(path, 'rb') as f:
            header = f.read(4)
        if header == ZIP_MAGIC:
            return True
        log_warn(
            f"{path.name}: header is {header.hex()!r} (expected {ZIP_MAGIC.hex()!r}). "
            "File appears corrupt."
        )
        return False
    except OSError as exc:
        log_warn(f"Cannot read {path.name} for validation: {exc}")
        return False


def download_file_with_resume(url: str, output_path: Path, expected_bytes: int = 0,
                              use_aria2: bool = True, max_retries: int = 10,
                              is_first_part: bool = False) -> bool:
    """Downloads a file with resume support, auto-retry, and zip magic validation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already complete and valid
    if output_path.exists():
        actual_size = output_path.stat().st_size
        if expected_bytes > 0 and actual_size >= expected_bytes:
            if is_valid_zip_part(output_path, is_first_part):
                log_success(f"Already complete: {output_path.name} ({actual_size / (1024**3):.2f} GB)")
                return True
            else:
                log_warn(f"Existing file corrupt, deleting and restarting: {output_path.name}")
                output_path.unlink()
        else:
            log_info(
                f"Existing partial found: {output_path.name} "
                f"({actual_size / (1024**3):.2f} / {expected_bytes / (1024**3):.2f} GB). Resuming..."
            )

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
            url
        ]
        log_info(f"Downloading {output_path.name} via aria2c (16 parallel connections)...")
        ret = subprocess.run(cmd)
        if ret.returncode == 0 and output_path.exists():
            if expected_bytes == 0 or output_path.stat().st_size >= expected_bytes:
                log_success(f"Downloaded: {output_path.name} ({output_path.stat().st_size / (1024**3):.2f} GB)")
                return True
        log_warn("aria2c incomplete or not found. Falling back to HTTP streaming.")

    # HTTP streaming fallback with automatic retry and resume
    for attempt in range(1, max_retries + 1):
        existing_bytes = output_path.stat().st_size if output_path.exists() else 0
        if expected_bytes > 0 and existing_bytes >= expected_bytes:
            log_success(f"Already complete: {output_path.name} ({existing_bytes / (1024**3):.2f} GB)")
            return True

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if existing_bytes > 0:
            req.add_header("Range", f"bytes={existing_bytes}-")

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                content_len = int(resp.headers.get("Content-Length", 0))
                total_size = content_len + existing_bytes
                open_mode = "ab" if existing_bytes > 0 else "wb"
                chunk_size = 8 * 1024 * 1024  # 8 MB
                downloaded = existing_bytes
                start_time = time.time()
                last_log = start_time

                with open(output_path, open_mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_log >= 15:
                            speed = (downloaded - existing_bytes) / (1024 ** 2) / max(now - start_time, 0.1)
                            pct = (downloaded / total_size * 100) if total_size > 0 else 0
                            remaining_mb = (total_size - downloaded) / (1024 ** 2)
                            eta_sec = (remaining_mb / speed) if speed > 0 else 0
                            eta_str = time.strftime("%Hh %Mm", time.gmtime(eta_sec))
                            log_info(
                                f"{output_path.name}: {downloaded / (1024**3):.2f} / {total_size / (1024**3):.2f} GB "
                                f"({pct:.1f}%) [{speed:.1f} MB/s, ETA: {eta_str}]"
                            )
                            last_log = now

            final_sz = output_path.stat().st_size
            log_success(f"Downloaded: {output_path.name} ({final_sz / (1024**3):.2f} GB)")
            return True

        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                log_success(f"Already complete (remote EOF): {output_path.name}")
                return True
            wait = min(20 * attempt, 180)
            log_warn(f"Attempt {attempt}/{max_retries} HTTP error for {output_path.name}: {exc}. Resuming in {wait}s...")
            time.sleep(wait)

        except Exception as exc:
            wait = min(20 * attempt, 180)
            log_warn(f"Attempt {attempt}/{max_retries} error for {output_path.name}: {exc}. Resuming in {wait}s...")
            time.sleep(wait)

    log_err(f"All {max_retries} download attempts failed for {output_path.name}.")
    return False


def verify_parts(data_root: Path, parts: List[str]) -> bool:
    """Check size and magic bytes for downloaded split parts."""
    log_header("Verifying Downloaded Parts")
    all_ok = True
    for i, part_name in enumerate(parts):
        path = data_root / part_name
        expected = EXACT_BYTES.get(part_name, 0)
        is_first = (i == 0)
        if not path.exists():
            log_warn(f"[MISSING] {part_name}")
            all_ok = False
        else:
            actual = path.stat().st_size
            size_ok = actual >= expected if expected > 0 else True
            magic_ok = is_valid_zip_part(path, is_first)
            status = "OK" if (size_ok and magic_ok) else "CORRUPT"
            sym = "[OK]" if status == "OK" else "[XX]"
            msg = f"{sym} {part_name}  {actual / (1024**3):.2f}/{expected / (1024**3):.2f} GB  [{status}]"
            if status == "OK":
                log_success(msg)
            else:
                log_err(msg)
                all_ok = False
    return all_ok


def download_and_extract_annotations(data_root: Path, ft_json_dir: Path) -> bool:
    """Downloads official annotations zip from Google Drive and extracts fine-tuning JSONs."""
    log_header("Fine-Tuning Annotations (Stage 2)")
    ft_json_dir.mkdir(parents=True, exist_ok=True)

    all_exist = True
    for fn in FT_ANNOTATION_FILES:
        target = ft_json_dir / fn
        if not target.exists() or target.stat().st_size == 0:
            all_exist = False
            break

    if all_exist:
        log_success(f"All fine-tuning annotation JSONs already exist in {ft_json_dir}")
        return True

    zip_tmp = data_root / "train_json_annotations.zip"
    log_info("Downloading annotations archive (~471 MB) from Google Drive...")
    ok = download_file_with_resume(GDRIVE_ANNOTATIONS_URL, zip_tmp, expected_bytes=400 * 1024 * 1024, use_aria2=False)
    if not ok or not zip_tmp.exists():
        log_err("Failed to download annotations zip from Google Drive.")
        return False

    log_info(f"Extracting fine-tuning annotations into {ft_json_dir}...")
    try:
        with zipfile.ZipFile(zip_tmp, "r") as zf:
            for member in zf.namelist():
                base_name = os.path.basename(member)
                if base_name in FT_ANNOTATION_FILES or member.endswith(".json"):
                    dest_file = ft_json_dir / base_name
                    with zf.open(member) as src, open(dest_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    log_success(f"Extracted: {base_name} ({dest_file.stat().st_size / (1024**2):.1f} MB)")

        zip_tmp.unlink(missing_ok=True)
        return True
    except Exception as e:
        log_err(f"Failed to extract annotations zip: {e}")
        return False


def extract_video_tuning(data_root: Path, clean_archives: bool = False) -> bool:
    """Extracts Video-ChatGPT video tuning dataset (.zip.001..005 -> videochatgpt_tune.zip -> videos)."""
    seven_zip = find_7z()
    if not seven_zip:
        log_err("7-Zip not found. Please install 7-Zip (winget install 7zip.7zip).")
        return False

    first_part = data_root / VIDEO_TUNE_PARTS[0]
    if not first_part.exists():
        log_err(f"Cannot extract: {first_part.name} not found.")
        return False

    log_header("Phase 1: Extracting Split Parts -> videochatgpt_tune.zip")
    cmd1 = [seven_zip, "x", str(first_part), f"-o{data_root}", "-bsp1", "-y"]
    ret1 = subprocess.run(cmd1)
    if ret1.returncode != 0:
        log_err("Phase 1 split archive extraction failed.")
        return False

    master_zip = data_root / "videochatgpt_tune.zip"
    video_dir = data_root / "videochatgpt_tune"
    video_dir.mkdir(parents=True, exist_ok=True)

    if master_zip.exists():
        log_header("Phase 2: Extracting videochatgpt_tune.zip -> Videos")
        cmd2 = [seven_zip, "x", str(master_zip), f"-o{data_root}", "-aos", "-bsp1", "-y"]
        ret2 = subprocess.run(cmd2)
        if ret2.returncode == 0:
            log_success("Extracted video tuning clips successfully!")
            log_info(f"Deleting intermediate {master_zip.name} to save ~160 GB...")
            master_zip.unlink(missing_ok=True)
            log_success("Cleaned intermediate master zip.")
            if clean_archives:
                log_info("Cleaning split archives...")
                for p in VIDEO_TUNE_PARTS:
                    (data_root / p).unlink(missing_ok=True)
                log_success("Cleaned split archives.")
            return True
        else:
            log_err(f"Phase 2 extraction failed with code {ret2.returncode}")
            return False
    else:
        # Some archives extract directly into the target folder
        log_success(f"Extraction completed into {data_root}.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract Video-LLaVA Stage 2 Fine-Tuning Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data_root", type=str, default="E:\\videollava_data", help="Root directory to store datasets")
    parser.add_argument("--components", type=str, default="video", choices=["all", "annotations", "image", "video"],
                        help="Which components to download (default: video)")
    parser.add_argument("--extract", action="store_true", help="Automatically extract split archives after download")
    parser.add_argument("--clean_archives", action="store_true", help="Delete downloaded split zip files after extraction")
    parser.add_argument("--no_aria2", action="store_true", help="Disable aria2 multi-connection download accelerator")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded parts without downloading")

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    use_aria2 = not args.no_aria2

    ft_json_dir = data_root / "ft_json"

    # --verify mode
    if args.verify:
        if args.components in ["all", "video"]:
            verify_parts(data_root, VIDEO_TUNE_PARTS)
        if args.components in ["all", "image"]:
            verify_parts(data_root, IMAGE_TUNE_PARTS)
        return

    log_header("Video-LLaVA Fine-Tuning Dataset Downloader")
    log_info(f"Target Data Root : {data_root}")
    log_info(f"Components       : {args.components}")
    log_info(f"Auto-extract     : {args.extract}")

    # 1. Annotations
    if args.components in ["all", "annotations"]:
        download_and_extract_annotations(data_root, ft_json_dir)

    # 2. Image Tuning Dataset (~67.4 GB)
    if args.components in ["all", "image"]:
        log_header("Image Tuning Dataset (LLaVA-Instruct ~67.4 GB)")
        all_image_ok = True
        for i, part in enumerate(IMAGE_TUNE_PARTS):
            url = f"{HF_BASE_URL}/{part}"
            dest = data_root / part
            exp_b = EXACT_BYTES.get(part, 0)
            ok = download_file_with_resume(url, dest, expected_bytes=exp_b, use_aria2=use_aria2, is_first_part=(i == 0))
            if not ok:
                all_image_ok = False
                break

    # 3. Video Tuning Dataset (~160.1 GB)
    if args.components in ["all", "video"]:
        log_header("Video Tuning Dataset (Video-ChatGPT ~160.1 GB)")
        all_video_ok = True
        for i, part in enumerate(VIDEO_TUNE_PARTS):
            url = f"{HF_BASE_URL}/{part}"
            dest = data_root / part
            exp_b = EXACT_BYTES.get(part, 0)
            ok = download_file_with_resume(url, dest, expected_bytes=exp_b, use_aria2=use_aria2, is_first_part=(i == 0))
            if not ok:
                all_video_ok = False
                break

        if all_video_ok:
            log_success("All video tuning parts downloaded successfully!")
            if args.extract:
                extract_video_tuning(data_root, clean_archives=args.clean_archives)
        else:
            log_warn("Download was interrupted. Re-run to resume seamlessly.")

    log_header("Fine-Tuning Dataset Preparation Complete")
    log_info(f"Data Root: {data_root}")


if __name__ == "__main__":
    main()
