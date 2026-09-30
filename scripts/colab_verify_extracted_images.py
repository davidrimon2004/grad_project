#!/usr/bin/env python3
"""
colab_verify_extracted_images.py
================================
High-speed verification tool for Google Colab to verify that all images
required by Video-LLaVA (Pretrain LLaVA-558K & Finetune LLaVA-665K) are
completely extracted, uncorrupted, and accessible on Google Drive or local SSD.

Key Features:
  1. Multi-threaded existence verification (32-64 parallel workers) to bypass
     Google Drive FUSE latency and scan 500K+ images in seconds.
  2. Image integrity verification: checks file sizes > 0 and samples PIL image
     decompression to detect truncated or corrupted files.
  3. Archive check: identifies if any raw .zip / .tar / split .zip.001 archives
     are still unextracted.
  4. Auto-Fix recommendations: outputs exact extraction commands if files are missing.

Usage in Google Colab:
---------------------
  # 1. Verify all image datasets (Pretrain + Finetune):
  !python scripts/colab_verify_extracted_images.py --split all

  # 2. Verify only Pretrain images (LLaVA-558K):
  !python scripts/colab_verify_extracted_images.py --split pretrain

  # 3. Verify only Finetune images (LLaVA-665K):
  !python scripts/colab_verify_extracted_images.py --split finetune

  # 4. Thorough check (verify PIL image decompression on 1,000 samples):
  !python scripts/colab_verify_extracted_images.py --split all --sample_verify 1000
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# ==============================================================================
# Console Formatting & Colors
# ==============================================================================

class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log_info(msg: str):
    print(f"[{_ts()}] {Colors.OKCYAN}[INFO]{Colors.ENDC} {msg}", flush=True)


def log_success(msg: str):
    print(f"[{_ts()}] {Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {msg}", flush=True)


def log_warn(msg: str):
    print(f"[{_ts()}] {Colors.WARNING}[WARN]{Colors.ENDC} {msg}", flush=True)


def log_err(msg: str):
    print(f"[{_ts()}] {Colors.FAIL}[ERROR]{Colors.ENDC} {msg}", flush=True)


def log_header(msg: str):
    bar = "=" * 65
    print(f"\n{Colors.HEADER}{Colors.BOLD}{bar}\n  {msg}\n{bar}{Colors.ENDC}", flush=True)


# ==============================================================================
# Colab & Drive Helpers
# ==============================================================================

def is_colab_environment() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return (
            "COLAB_GPU" in os.environ
            or "COLAB_RELEASE_TAG" in os.environ
            or (os.name != "nt" and os.path.exists("/content"))
        )


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    if not is_colab_environment():
        return True

    drive_root = Path(mount_point)
    if (drive_root / "MyDrive").exists() or (drive_root / "My Drive").exists():
        log_success(f"Google Drive is mounted and ready at {mount_point}")
        return True

    try:
        log_info(f"Mounting Google Drive to {mount_point}...")
        from google.colab import drive
        drive.mount(mount_point, force_remount=False)
        return True
    except Exception as exc:
        log_err(f"Failed to mount Google Drive: {exc}")
        return False


def load_json_cached(jp: Path) -> List[Dict[str, Any]]:
    """Loads JSON by caching to local SSD if on Drive FUSE to prevent stalls."""
    p_str = str(jp)
    is_fuse = "/drive/" in p_str or "/MyDrive/" in p_str or p_str.startswith("/content/drive")

    if not is_fuse or not jp.exists():
        with open(str(jp), "r", encoding="utf-8") as f:
            return json.load(f)

    cache_dir = Path("/content/cache_jsons") if Path("/content").exists() else Path("/tmp/cache_jsons")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_cached = cache_dir / jp.name

    need_copy = True
    if local_cached.exists():
        try:
            if local_cached.stat().st_size == os.path.getsize(str(jp)):
                need_copy = False
        except Exception:
            pass

    if need_copy:
        log_info(f"Caching annotation {jp.name} to local SSD for fast parsing...")
        t0 = time.time()
        with open(str(jp), "rb") as src, open(str(local_cached), "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        log_success(f"Cached {jp.name} in {time.time() - t0:.1f}s.")

    with open(str(local_cached), "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# Path & Prefix Resolution
# ==============================================================================

def extract_image_files(json_paths: List[Path]) -> List[str]:
    seen = {}
    for jp in json_paths:
        try:
            data = load_json_cached(jp)
            for item in data:
                if "image" in item and "video" not in item:
                    imgs = item["image"] if isinstance(item["image"], list) else [item["image"]]
                    for img_path in imgs:
                        if img_path and img_path not in seen:
                            seen[img_path] = len(seen)
        except Exception as exc:
            log_warn(f"Failed to read {jp.name}: {exc}")
    return sorted(seen, key=seen.__getitem__)


def safe_is_file(p: Path) -> bool:
    try:
        return os.path.isfile(str(p))
    except (OSError, Exception):
        return False


def safe_is_dir(p: Path) -> bool:
    try:
        return os.path.isdir(str(p))
    except (OSError, Exception):
        return False


def safe_exists(p: Path) -> bool:
    try:
        return os.path.exists(str(p))
    except (OSError, Exception):
        return False


def safe_getsize(p: Path) -> int:
    try:
        return os.path.getsize(str(p))
    except (OSError, Exception):
        return 0


def resolve_image_folder_and_prefix(
    base: Path,
    folder_name: str,
    probe_files: List[str],
) -> Tuple[Optional[Path], Optional[str]]:
    """Finds image folder and determines any relative prefix differences."""
    candidate_folders = [
        base / "datasets" / folder_name,
        base / "datasets" / folder_name / folder_name,
        base / "datasets",
        base / folder_name,
        base / folder_name / folder_name,
        base / "data" / folder_name,
        Path("/content/datasets") / folder_name,
        Path("/content") / folder_name,
        base,
    ]

    for c in candidate_folders:
        if not safe_is_dir(c):
            continue
        for probe in probe_files[:20]:
            bare = Path(probe).name

            # Direct match
            if safe_is_file(c / probe):
                return c, None

            # Flat match
            if safe_is_file(c / bare):
                pfx = probe[:-len(bare)] if len(probe) > len(bare) else None
                return c, pfx

            # Subdirectory match
            if "/" in probe:
                parts = probe.split("/")
                for i in range(1, len(parts)):
                    subpath = "/".join(parts[i:])
                    if safe_is_file(c / subpath):
                        pfx = "/".join(parts[:i]) + "/"
                        return c, pfx

            for sname in [folder_name, "images", "data", "train2017", "coco", "gqa", "vg"]:
                sub = c / sname
                if safe_is_dir(sub) and safe_is_file(sub / bare):
                    pfx = probe[:-len(bare)] if len(probe) > len(bare) else None
                    return sub, pfx

    # Fallback to standard datasets folder
    fallback = base / "datasets" / folder_name
    return fallback, None


def resolve_annotation_json(base: Path, candidate_names: List[str]) -> Optional[Path]:
    search_dirs = [
        base / "datasets" / "pt_json",
        base / "datasets" / "ft_json",
        base / "datasets" / "annotations",
        base / "datasets" / "train_json",
        base / "datasets",
        base / "pt_json",
        base / "ft_json",
        base / "annotations",
        base / "train_json",
        base / "data",
        Path("/content/datasets"),
        Path("/content"),
        base,
    ]
    for cand in candidate_names:
        for sdir in search_dirs:
            if not safe_is_dir(sdir):
                continue
            target = sdir / cand
            if safe_is_file(target) and safe_getsize(target) > 0:
                return target
    return None


# ==============================================================================
# Multi-threaded Image Verification
# ==============================================================================

def check_single_image(
    rel_path: str,
    image_folder: Path,
    strip_prefix: Optional[str] = None,
) -> Tuple[str, bool, int]:
    """Checks if a single image exists and is non-empty (>0 bytes)."""
    actual_rel = (
        rel_path[len(strip_prefix):]
        if strip_prefix and rel_path.startswith(strip_prefix)
        else rel_path
    )
    full_path = image_folder / actual_rel
    try:
        if safe_is_file(full_path):
            size = safe_getsize(full_path)
            return rel_path, size > 0, size
        return rel_path, False, 0
    except (OSError, Exception):
        return rel_path, False, 0


def verify_split_images(
    split_name: str,
    json_path: Path,
    image_folder: Path,
    strip_prefix: Optional[str],
    num_threads: int = 32,
    sample_verify_count: int = 200,
) -> Dict[str, Any]:
    """Verifies all images in a split using multi-threading for maximum speed."""
    log_header(f"Verifying Split: {split_name.upper()}")
    log_info(f"Annotation JSON: {json_path}")
    log_info(f"Images Directory: {image_folder}")
    if strip_prefix:
        log_info(f"Path prefix adjustment: '{strip_prefix}'")

    img_files = extract_image_files([json_path])
    n_total = len(img_files)
    if n_total == 0:
        log_warn(f"No images found in annotation {json_path.name}")
        return {"total": 0, "found": 0, "missing": 0, "missing_samples": []}

    log_info(f"Checking existence of {n_total:,} images with {num_threads} parallel threads...")

    found_count = 0
    empty_count = 0
    missing_samples = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(check_single_image, rel, image_folder, strip_prefix): rel
            for rel in img_files
        }

        for i, fut in enumerate(as_completed(futures)):
            rel_path, exists, size = fut.result()
            if exists:
                found_count += 1
            elif size == 0 and (image_folder / rel_path).exists():
                empty_count += 1
                if len(missing_samples) < 10:
                    missing_samples.append((rel_path, "0 bytes (empty file)"))
            else:
                if len(missing_samples) < 10:
                    missing_samples.append((rel_path, "file not found"))

            if (i + 1) % 25000 == 0 or (i + 1) == n_total:
                pct = (i + 1) / n_total * 100
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"\r  Verified [{i+1:>7,}/{n_total:,}] {pct:5.1f}% | "
                    f"{rate:6.0f} checks/s | Found: {found_count:,} | Missing: {n_total - found_count:,}",
                    end="",
                    flush=True,
                )

    print()
    elapsed = time.time() - t0
    missing_count = n_total - found_count
    completion_pct = (found_count / n_total) * 100 if n_total > 0 else 0

    log_info(f"Existence check completed in {elapsed:.1f}s ({n_total/elapsed:,.0f} files/sec).")

    # Sample PIL Integrity Check
    corrupted_count = 0
    if sample_verify_count > 0 and found_count > 0:
        sample_size = min(sample_verify_count, found_count)
        log_info(f"Running PIL decompression integrity test on {sample_size:,} random images...")
        sample_paths = random.sample(img_files, sample_size)
        for rel in sample_paths:
            actual_rel = rel[len(strip_prefix):] if strip_prefix and rel.startswith(strip_prefix) else rel
            fp = image_folder / actual_rel
            if not fp.is_file():
                continue
            try:
                with Image.open(str(fp)) as img:
                    img.verify()
            except Exception as exc:
                corrupted_count += 1
                log_warn(f"Corrupted image detected: {rel} ({exc})")

        if corrupted_count == 0:
            log_success(f"Integrity check passed! 0 of {sample_size:,} sampled images are corrupted.")
        else:
            log_err(f"Integrity alert: {corrupted_count} / {sample_size} sampled images are corrupt or truncated.")

    # Status summary
    if missing_count == 0 and corrupted_count == 0:
        log_success(f"100% COMPLETE! All {n_total:,} images exist and are ready for precomputation.")
    else:
        log_warn(f"Extraction incomplete: {found_count:,}/{n_total:,} ({completion_pct:.2f}%) present. {missing_count:,} missing.")

    return {
        "total": n_total,
        "found": found_count,
        "missing": missing_count,
        "empty": empty_count,
        "completion_pct": completion_pct,
        "missing_samples": missing_samples,
    }


# ==============================================================================
# Archive Inspection Helper
# ==============================================================================

def check_unextracted_archives(base: Path) -> List[Tuple[Path, float]]:
    """Finds any remaining archive files (.zip, .tar, .zip.001) under base."""
    archive_exts = {".zip", ".tar", ".gz", ".tgz", ".001", ".002"}
    unextracted = []
    search_dirs = [
        base / "download",
        base / "downloads",
        base / "datasets",
        base,
    ]
    for sdir in search_dirs:
        if not sdir.exists():
            continue
        try:
            for item in sdir.iterdir():
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext in archive_exts or ".zip." in item.name:
                        size_gb = item.stat().st_size / 1e9
                        unextracted.append((item, size_gb))
        except Exception:
            pass
    return unextracted


# ==============================================================================
# Main CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify that all image dataset files are fully extracted before precomputing embeddings."
    )
    parser.add_argument(
        "--split",
        choices=["pretrain", "finetune", "all"],
        default="all",
        help="Dataset split to verify: 'pretrain' (LLaVA-558K), 'finetune' (LLaVA-665K), or 'all'.",
    )
    parser.add_argument(
        "--drive_base",
        type=str,
        default="/content/drive/MyDrive/Video-LLaVA",
        help="Root path of Video-LLaVA on Google Drive.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Explicit image directory override.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Explicit annotation JSON override.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of concurrent threads for scanning Google Drive (default: 32).",
    )
    parser.add_argument(
        "--sample_verify",
        type=int,
        default=200,
        help="Number of images to test for PIL image decompression integrity (default: 200).",
    )

    args = parser.parse_args()

    mount_google_drive("/content/drive")
    base = Path(args.drive_base)

    if not base.exists():
        log_err(f"Base path does not exist: {base}")
        log_err("Please ensure Google Drive is mounted: from google.colab import drive; drive.mount('/content/drive')")
        sys.exit(1)

    splits_to_run = ["pretrain", "finetune"] if args.split == "all" else [args.split]
    results = {}

    for split in splits_to_run:
        # Resolve annotation JSON
        if args.json_path:
            json_file = Path(args.json_path)
        elif split == "pretrain":
            json_file = resolve_annotation_json(base, ["llava_image_.json", "llava_image.json"])
        else:
            json_file = resolve_annotation_json(base, ["llava_image_tune_.json", "llava_image_tune.json"])

        if json_file is None or not json_file.exists():
            log_err(f"Could not find annotation JSON for split '{split}' under {base}")
            continue

        # Extract a few probe names to resolve image directory
        probe_names = extract_image_files([json_file])[:30]

        img_dir = None
        strip_pfx = None
        if args.image_folder:
            img_dir = Path(args.image_folder)
        else:
            folder_tag = "llava_image" if split == "pretrain" else "llava_image_tune"
            img_dir, strip_pfx = resolve_image_folder_and_prefix(base, folder_tag, probe_names)

        if img_dir is None or not img_dir.exists():
            log_err(f"Could not locate image directory for split '{split}' under {base}")
            log_info(f"Expected folder like: {base / 'datasets' / ('llava_image' if split == 'pretrain' else 'llava_image_tune')}")
            continue

        res = verify_split_images(
            split_name=split,
            json_path=json_file,
            image_folder=img_dir,
            strip_prefix=strip_pfx,
            num_threads=args.threads,
            sample_verify_count=args.sample_verify,
        )
        results[split] = res

    # Check unextracted archive files
    archives = check_unextracted_archives(base)

    # Final Dashboard Summary
    log_header("IMAGE EXTRACTION VERIFICATION SUMMARY")
    all_complete = True

    for s, res in results.items():
        status_color = Colors.OKGREEN if res["missing"] == 0 else Colors.FAIL
        status_text = "READY (100%)" if res["missing"] == 0 else f"INCOMPLETE ({res['completion_pct']:.1f}%)"
        print(f"• Split [{s.upper()}]: {status_color}{status_text}{Colors.ENDC}")
        print(f"    - Total Expected:  {res['total']:,}")
        print(f"    - Found on Disk:   {res['found']:,}")
        print(f"    - Missing / Empty: {res['missing']:,}")

        if res["missing"] > 0:
            all_complete = False
            print(f"    - Sample missing files:")
            for s_file, reason in res["missing_samples"][:5]:
                print(f"        * {s_file} ({reason})")

    if archives:
        print(f"\n• Archive files detected on Google Drive:")
        for arch, size_gb in archives:
            print(f"    - {arch.name:<40} ({size_gb:5.1f} GB)")

    print()
    if all_complete and results:
        log_success("All datasets are fully extracted and verified! You can proceed to precomputing embeddings:")
        print("  !python scripts/colab_precompute_images.py --split all --batch_size 64")
    else:
        log_warn("Some images are missing. To extract remaining archives, run:")
        print("  !python scripts/extract_datasets.py --drive_base /content/drive/MyDrive/Video-LLaVA --target images")
        print("  or for 7z split archives (e.g. llava_image_tune_2.zip.001):")
        print("  !7z x /content/drive/MyDrive/Video-LLaVA/download/llava_image_tune_2.zip.001 -o/content/drive/MyDrive/Video-LLaVA/datasets/llava_image_tune")


if __name__ == "__main__":
    main()
