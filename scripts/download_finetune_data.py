#!/usr/bin/env python3
"""
High-Performance Downloader & Extractor for Video-LLaVA / MuLER Fine-Tuning Datasets (Stage 2)

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
  # Download everything:
  python scripts/download_finetune_data.py --data_root ./data --components all --extract

  # Download annotations only:
  python scripts/download_finetune_data.py --data_root ./data --components annotations

  # Download image tuning data only:
  python scripts/download_finetune_data.py --data_root ./data --components image --extract

  # Download video tuning data only:
  python scripts/download_finetune_data.py --data_root ./data --components video --extract
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

IMAGE_PART_MIN_BYTES = {
    "llava_image_tune_2.zip.001": int(38.5 * (1024 ** 3)),
    "llava_image_tune_2.zip.002": int(27.5 * (1024 ** 3)),
}

VIDEO_PART_MIN_BYTES = {
    "videochatgpt_tune_2.zip.001": int(38.5 * (1024 ** 3)),
    "videochatgpt_tune_2.zip.002": int(38.5 * (1024 ** 3)),
    "videochatgpt_tune_2.zip.003": int(38.5 * (1024 ** 3)),
    "videochatgpt_tune_2.zip.004": int(38.5 * (1024 ** 3)),
    "videochatgpt_tune_2.zip.005": int(3.5 * (1024 ** 3)),
}

FT_ANNOTATION_FILES = [
    "llava_image_tune_.json",
    "videochatgpt_tune_.json",
    "nlp_tune.json",
]


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def log_info(msg: str):
    print(f"{Colors.OKCYAN}[INFO]{Colors.ENDC} {msg}", flush=True)


def log_success(msg: str):
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {msg}", flush=True)


def log_warn(msg: str):
    print(f"{Colors.WARNING}[WARN]{Colors.ENDC} {msg}", flush=True)


def log_err(msg: str):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {msg}", flush=True)


def log_header(msg: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}\n{msg}\n{'='*60}{Colors.ENDC}", flush=True)


def download_file_with_resume(url: str, output_path: Path, min_bytes: int = 0, use_aria2: bool = True) -> bool:
    """Downloads a file with resume support using aria2c or requests/urllib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        actual_size = output_path.stat().st_size
        if min_bytes > 0 and actual_size >= min_bytes:
            log_success(f"Already complete: {output_path.name} ({actual_size / (1024**3):.2f} GB)")
            return True
        log_info(f"Existing partial found ({actual_size / (1024**3):.2f} GB). Resuming download...")

    # Try aria2c if available
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
            "-d", str(output_path.parent),
            "-o", output_path.name,
            url
        ]
        log_info(f"Downloading {output_path.name} via aria2c (16 parallel connections)...")
        ret = subprocess.run(cmd)
        if ret.returncode == 0 and output_path.exists():
            if min_bytes == 0 or output_path.stat().st_size >= min_bytes:
                log_success(f"Downloaded: {output_path.name} ({output_path.stat().st_size / (1024**3):.2f} GB)")
                return True

    # Fallback to streaming urllib with resume
    log_info(f"Downloading {output_path.name} via HTTP stream...")
    existing_bytes = output_path.stat().st_size if output_path.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if existing_bytes > 0:
        req.add_header("Range", f"bytes={existing_bytes}-")

    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get("Content-Length", 0)) + existing_bytes
            open_mode = "ab" if existing_bytes > 0 else "wb"
            chunk_size = 8 * 1024 * 1024  # 8MB
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
                    if now - last_log >= 10:
                        speed = (downloaded - existing_bytes) / (1024 ** 2) / max(now - start_time, 0.1)
                        pct = (downloaded / total_size * 100) if total_size > 0 else 0
                        log_info(f"Progress {output_path.name}: {downloaded/(1024**3):.2f} GB / {total_size/(1024**3):.2f} GB ({pct:.1f}%) [{speed:.1f} MB/s]")
                        last_log = now

        log_success(f"Successfully downloaded {output_path.name} ({output_path.stat().st_size / (1024**3):.2f} GB)")
        return True
    except Exception as e:
        log_err(f"Download failed for {output_path.name}: {e}")
        return False


def download_and_extract_annotations(data_root: Path, ft_json_dir: Path) -> bool:
    """Downloads official annotations zip from Google Drive and extracts fine-tuning JSONs."""
    log_header("Downloading Fine-Tuning Annotations")
    ft_json_dir.mkdir(parents=True, exist_ok=True)

    # Check if all required fine-tuning json files exist
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
    log_info("Downloading official annotations archive (~471 MB) from Google Drive...")
    ok = download_file_with_resume(GDRIVE_ANNOTATIONS_URL, zip_tmp, min_bytes=400 * 1024 * 1024, use_aria2=False)
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

        # Remove temp zip
        zip_tmp.unlink(missing_ok=True)
        return True
    except Exception as e:
        log_err(f"Failed to extract annotations zip: {e}")
        return False


def extract_split_archive(first_part_path: Path, output_folder: Path, clean_archives: bool = False) -> bool:
    """Extracts a multi-part split archive (.zip.001, .zip.002, ...) using 7z or python fallback."""
    output_folder.mkdir(parents=True, exist_ok=True)
    log_info(f"Extracting {first_part_path.name} into {output_folder}...")

    # Check if 7z is available
    seven_zip = shutil.which("7z") or shutil.which("7za")
    if not seven_zip and os.name == "nt":
        # Check standard Windows 7-Zip installation paths
        for candidate in [r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"]:
            if os.path.exists(candidate):
                seven_zip = candidate
                break

    if seven_zip:
        log_info(f"Using 7-Zip ({seven_zip}) for high-speed multi-part extraction...")
        cmd = [seven_zip, "x", str(first_part_path), f"-o{output_folder}", "-y"]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            log_success(f"Extracted archive into {output_folder}")
            if clean_archives:
                pattern = str(first_part_path).replace(".001", ".*")
                for p in first_part_path.parent.glob(Path(pattern).name):
                    p.unlink(missing_ok=True)
                log_info("Cleaned up split archive parts to save disk space.")
            return True
        else:
            log_err(f"7-Zip extraction returned non-zero code {ret.returncode}")

    # Linux fallback: check if unzip / p7zip is available
    if shutil.which("unzip"):
        log_info("7z not found. Concatenating split parts to single archive for unzip...")
        base_name = first_part_path.stem  # e.g. llava_image_tune_2.zip
        merged_zip = first_part_path.parent / f"{base_name}_merged.zip"
        part_prefix = str(first_part_path).rsplit(".", 1)[0]  # e.g. .../llava_image_tune_2.zip
        parts = sorted(first_part_path.parent.glob(f"{Path(part_prefix).name}.*"))

        with open(merged_zip, "wb") as out_f:
            for p in parts:
                log_info(f"Merging part {p.name}...")
                with open(p, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=64 * 1024 * 1024)

        log_info(f"Extracting merged {merged_zip.name} with unzip...")
        cmd = ["unzip", "-q", "-o", str(merged_zip), "-d", str(output_folder)]
        ret = subprocess.run(cmd)
        merged_zip.unlink(missing_ok=True)
        if ret.returncode == 0:
            log_success(f"Extracted into {output_folder}")
            if clean_archives:
                for p in parts:
                    p.unlink(missing_ok=True)
            return True

    log_err("Could not find 7z or unzip to extract split archives. Please install 7-zip (or p7zip-full).")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download and extract Video-LLaVA Stage 2 Fine-Tuning Datasets")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory to store datasets")
    parser.add_argument("--components", type=str, default="all", choices=["all", "annotations", "image", "video"],
                        help="Which components to download (annotations, image, video, or all)")
    parser.add_argument("--extract", action="store_true", help="Automatically extract split archives after download")
    parser.add_argument("--clean_archives", action="store_true", help="Delete downloaded split zip files after extraction")
    parser.add_argument("--no_aria2", action="store_true", help="Disable aria2 multi-connection download accelerator")

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    use_aria2 = not args.no_aria2

    ft_json_dir = data_root / "ft_json"
    image_dir = data_root / "llava_image_tune"
    video_dir = data_root / "videochatgpt_tune"

    log_header("Video-LLaVA Fine-Tuning Dataset Downloader")
    log_info(f"Target Data Root: {data_root}")
    log_info(f"Components: {args.components}")
    log_info(f"Auto-extract: {args.extract}")

    # 1. Annotations
    if args.components in ["all", "annotations"]:
        download_and_extract_annotations(data_root, ft_json_dir)

    # 2. Image Tuning Dataset (~67.4 GB)
    if args.components in ["all", "image"]:
        log_header("Downloading Image Tuning Dataset (LLaVA-Instruct ~67.4 GB)")
        all_image_ok = True
        for part in IMAGE_TUNE_PARTS:
            url = f"{HF_BASE_URL}/{part}"
            dest = data_root / part
            min_b = IMAGE_PART_MIN_BYTES.get(part, 0)
            ok = download_file_with_resume(url, dest, min_bytes=min_b, use_aria2=use_aria2)
            if not ok:
                all_image_ok = False
                break

        if all_image_ok and args.extract:
            first_part = data_root / IMAGE_TUNE_PARTS[0]
            extract_split_archive(first_part, image_dir, clean_archives=args.clean_archives)

    # 3. Video Tuning Dataset (~160.1 GB)
    if args.components in ["all", "video"]:
        log_header("Downloading Video Tuning Dataset (Video-ChatGPT ~160.1 GB)")
        all_video_ok = True
        for part in VIDEO_TUNE_PARTS:
            url = f"{HF_BASE_URL}/{part}"
            dest = data_root / part
            min_b = VIDEO_PART_MIN_BYTES.get(part, 0)
            ok = download_file_with_resume(url, dest, min_bytes=min_b, use_aria2=use_aria2)
            if not ok:
                all_video_ok = False
                break

        if all_video_ok and args.extract:
            first_part = data_root / VIDEO_TUNE_PARTS[0]
            extract_split_archive(first_part, video_dir, clean_archives=args.clean_archives)

    log_header("Fine-Tuning Dataset Preparation Complete")
    log_info("Expected Directory Layout:")
    log_info(f"  {data_root}/")
    log_info(f"  ├── ft_json/ (llava_image_tune_.json, videochatgpt_tune_.json, nlp_tune.json)")
    log_info(f"  ├── llava_image_tune/ (COCO, GQA, TextVQA, VisualGenome images)")
    log_info(f"  └── videochatgpt_tune/ (Video-ChatGPT video mp4 files)")


if __name__ == "__main__":
    main()
