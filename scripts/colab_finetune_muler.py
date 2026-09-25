#!/usr/bin/env python3
"""
Video-LLaVA Stage 2 Fine-Tuning & Data Muling Pipeline for Google Colab & Google Drive.

Fine-Tuning Datasets:
  - Annotations: Google Drive zip (llava_image_tune_.json, videochatgpt_tune_.json, nlp_tune.json)
  - Image Tuning: LLaVA-Instruct 665K (llava_image_tune_2.zip.001 - .002, ~67.4 GB)
  - Video Tuning: Video-ChatGPT 100K (videochatgpt_tune_2.zip.001 - .005, ~160.1 GB)

Drive-First Architecture:
  1. Downloads archives directly to Google Drive via 16-connection SSD staging + hole punching
     or direct HTTP streaming (Zero SSD exhaustion).
  2. Extracts multi-part archives directly into Google Drive datasets folder using 7-Zip.
  3. Automatically handles Colab DriveFS cache pruning and FUSE buffer flushing.
  4. Manages LoRA or Full Fine-Tuning execution with automated checkpoint muling to Google Drive.

Usage:
  # Check storage & dataset status:
  python scripts/colab_finetune_muler.py --action status

  # Download all fine-tuning archives to Google Drive:
  python scripts/colab_finetune_muler.py --action download

  # Extract archives directly on Google Drive:
  python scripts/colab_finetune_muler.py --action extract

  # Launch LoRA Fine-Tuning (Stage 2):
  python scripts/colab_finetune_muler.py --action train --lora True

  # Run the full pipeline (download -> extract -> train):
  python scripts/colab_finetune_muler.py --action all --lora True
"""

import os
import sys
import time
import json
import shutil
import glob
import subprocess
import argparse
import threading
import signal
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict


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


def is_colab_environment() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ or (os.name != "nt" and os.path.exists("/content"))


def mount_google_drive(mount_point: str = "/content/drive", force: bool = False) -> bool:
    if not is_colab_environment():
        log_info("Not running in Google Colab. Using local directory paths.")
        return True

    # Check if actually mounted in /proc/mounts
    is_mounted = False
    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                if mount_point in line:
                    is_mounted = True
                    break
    except Exception:
        is_mounted = os.path.ismount(mount_point)

    has_drive_folder = os.path.exists(os.path.join(mount_point, "MyDrive")) or os.path.exists(os.path.join(mount_point, "My Drive"))

    if is_mounted and has_drive_folder and not force:
        _ensure_mydrive_symlink(mount_point)
        log_success(f"Google Drive already mounted at {mount_point}")
        return True

    try:
        log_info(f"Mounting Google Drive to {mount_point}...")
        from google.colab import drive
        if force:
            try:
                drive.flush_and_unmount()
            except Exception:
                pass
        drive.mount(mount_point)
        _ensure_mydrive_symlink(mount_point)
        log_success(f"Google Drive successfully mounted at {mount_point}")
        return True
    except Exception as e:
        log_err(f"Failed to mount Google Drive: {e}")
        return False


def _ensure_mydrive_symlink(mount_point: str = "/content/drive"):
    """Ensure /content/drive/MyDrive exists as a symlink to /content/drive/My Drive if needed."""
    my_drive_spaced = os.path.join(mount_point, "My Drive")
    my_drive_nospace = os.path.join(mount_point, "MyDrive")
    try:
        if os.path.exists(my_drive_spaced) and not os.path.exists(my_drive_nospace):
            os.symlink(my_drive_spaced, my_drive_nospace)
            log_info(f"Created symlink: {my_drive_nospace} -> {my_drive_spaced}")
    except Exception:
        pass


def get_actual_drive_path(path: Path) -> Path:
    """Resolve Google Drive paths robustly across 'MyDrive' vs 'My Drive'."""
    p_str = str(path)
    if os.path.exists(p_str):
        return Path(p_str)

    if "MyDrive" in p_str:
        alt = p_str.replace("MyDrive", "My Drive")
        if os.path.exists(alt) or os.path.exists(os.path.dirname(alt)):
            return Path(alt)
    elif "My Drive" in p_str:
        alt = p_str.replace("My Drive", "MyDrive")
        if os.path.exists(alt) or os.path.exists(os.path.dirname(alt)):
            return Path(alt)

    return Path(p_str)


# ==============================================================================
# Inbound Fine-Tuning Data Muler (HuggingFace/GDrive -> Google Drive)
# ==============================================================================

class InboundFineTuneMuler:
    HF_BASE_URL = "https://huggingface.co/datasets/LanguageBind/Video-LLaVA/resolve/main"
    ANNOTATIONS_GDRIVE_ID = "1zGRyVSUMoczGq6cjQFmT0prH67bu2wXD"

    IMAGE_TUNE_PARTS = [
        "llava_image_tune_2.zip.001",
        "llava_image_tune_2.zip.002",
    ]

    VIDEO_TUNE_PARTS = [
        f"videochatgpt_tune_2.zip.{i:03d}" for i in range(1, 6)
    ]

    IMAGE_PART_MIN_BYTES = {
        "llava_image_tune_2.zip.001": 41_000_000_000,  # Full: 41,943,040,000 bytes (39.06 GB)
        "llava_image_tune_2.zip.002": 30_000_000_000,  # Full: 30,468,547,028 bytes (28.38 GB)
    }

    VIDEO_PART_MIN_BYTES = {
        f"videochatgpt_tune_2.zip.{i:03d}": 41_000_000_000 for i in range(1, 5)  # Full: 41,943,040,000 bytes each
    }
    VIDEO_PART_MIN_BYTES["videochatgpt_tune_2.zip.005"] = 4_000_000_000  # Full: 4,103,350,672 bytes (3.82 GB)

    def __init__(self, drive_root: str, local_scratch_dir: str = "/content/data"):
        self.drive_root = get_actual_drive_path(Path(drive_root))
        self.drive_data_dir = self.drive_root / "datasets"
        self.local_scratch_dir = Path(local_scratch_dir)

        # Drive Dataset Paths
        self.drive_ft_json_dir = self.drive_data_dir / "ft_json"
        self.drive_image_folder = self.drive_data_dir / "llava_image_tune"
        self.drive_video_folder = self.drive_data_dir / "videochatgpt_tune"

        # Key Annotation JSONs
        self.drive_image_json = self.drive_ft_json_dir / "llava_image_tune_.json"
        self.drive_video_json = self.drive_ft_json_dir / "videochatgpt_tune_.json"
        self.drive_nlp_json = self.drive_ft_json_dir / "nlp_tune.json"

        # Local Fast SSD Paths (for fast reads)
        self.local_ft_json_dir = self.local_scratch_dir / "ft_json"

        # Create base dirs
        os.makedirs(str(self.drive_data_dir), exist_ok=True)
        os.makedirs(str(self.drive_ft_json_dir), exist_ok=True)
        os.makedirs(str(self.local_scratch_dir), exist_ok=True)
        os.makedirs(str(self.local_ft_json_dir), exist_ok=True)

    def check_storage(self):
        log_header("Storage Diagnostic & Capacity Check")
        targets = [
            ("Google Drive Data Dir", self.drive_data_dir),
            ("Colab Local SSD (/content)", Path("/content")),
            ("System Temp (/tmp)", Path("/tmp")),
        ]
        seen = set()
        for label, path in targets:
            try:
                resolved = path.resolve() if path.exists() else path
                if str(resolved) in seen or not path.exists():
                    continue
                seen.add(str(resolved))
                usage = shutil.disk_usage(path)
                free_gb = usage.free / (1024 ** 3)
                total_gb = usage.total / (1024 ** 3)
                pct = (usage.used / usage.total) * 100 if usage.total > 0 else 0
                log_info(f"  • {label:28s}: {free_gb:6.1f} GB free / {total_gb:6.1f} GB total ({pct:5.1f}% used) [{path}]")
            except Exception as e:
                log_warn(f"  • {label:28s}: unable to read disk usage ({e})")

    def _prune_caches(self):
        """Prune Colab SSD caches to prevent running out of local disk space."""
        for item in Path("/tmp").glob("*"):
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
            except Exception:
                pass

        shutil.rmtree("/root/.cache/pip", ignore_errors=True)

        drivefs_cache = Path("/root/.config/Google/DriveFS")
        if drivefs_cache.exists():
            shutil.rmtree(str(drivefs_cache), ignore_errors=True)

    def _flush_drive_fuse(self):
        try:
            from google.colab import drive
            log_info("Flushing Google Drive FUSE write buffers...")
            drive.flush_and_unmount()
            drive.mount('/content/drive')
            _ensure_mydrive_symlink('/content/drive')
            log_success("Google Drive remounted cleanly.")
        except Exception:
            pass

    def _ensure_aria2(self) -> bool:
        if shutil.which("aria2c"):
            return True
        log_info("Installing aria2 for 16-channel accelerated downloading...")
        try:
            subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"], check=False)
        except Exception:
            pass
        return shutil.which("aria2c") is not None

    def _stream_copy_to_drive(self, local_path: Path, drive_path: Path, chunk_size: int = 64 * 1024 * 1024):
        size_gb = local_path.stat().st_size / (1024 ** 3)
        log_info(f"Transferring {local_path.name} to Google Drive ({size_gb:.2f} GB)...")

        drive_path = get_actual_drive_path(drive_path)
        os.makedirs(str(drive_path.parent), exist_ok=True)

        if not drive_path.parent.is_dir():
            log_warn(f"Drive path {drive_path.parent} not visible. Re-mounting Drive...")
            mount_google_drive(force=True)
            drive_path = get_actual_drive_path(drive_path)
            os.makedirs(str(drive_path.parent), exist_ok=True)

        if drive_path.exists():
            drive_path.unlink(missing_ok=True)
            time.sleep(0.5)

        start_t = time.time()
        fdst = None
        for attempt in range(1, 4):
            try:
                os.makedirs(str(drive_path.parent), exist_ok=True)
                fdst = open(drive_path, "wb")
                break
            except (FileNotFoundError, OSError) as e:
                log_warn(f"Attempt {attempt}/3 to open {drive_path} failed: {e}")
                if attempt < 3:
                    mount_google_drive(force=True)
                    drive_path = get_actual_drive_path(drive_path)
                    os.makedirs(str(drive_path.parent), exist_ok=True)
                    time.sleep(2)
                else:
                    raise

        try:
            with open(local_path, "rb") as fsrc:
                offset = 0
                last_log = time.time()
                total_bytes = local_path.stat().st_size
                while True:
                    buf = fsrc.read(chunk_size)
                    if not buf:
                        break
                    fdst.write(buf)
                    offset += len(buf)
                    if time.time() - last_log >= 15:
                        pct = (offset / total_bytes) * 100 if total_bytes > 0 else 0
                        speed = (offset / (1024 ** 2)) / max(time.time() - start_t, 1)
                        log_info(f"Drive transfer: {offset / (1024**3):.2f} / {size_gb:.1f} GB ({pct:.1f}%) [{speed:.1f} MB/s]")
                        last_log = time.time()
        finally:
            if fdst:
                fdst.close()

        # Verify Drive target size matches before unlinking staged SSD copy
        if drive_path.exists() and drive_path.stat().st_size >= int(size_gb * 0.99 * (1024**3)):
            local_path.unlink(missing_ok=True)
            self._prune_caches()
            self._flush_drive_fuse()
            log_success(f"✓ Transferred {drive_path.name} to Drive!")
            return True
        else:
            log_warn(f"Drive file size verification pending. Preserving {local_path.name} on SSD.")
            return True

    def _direct_stream_from_url_to_drive(self, url: str, drive_target: Path, min_bytes: int) -> bool:
        drive_target = get_actual_drive_path(drive_target)
        os.makedirs(str(drive_target.parent), exist_ok=True)

        # On Google Drive FUSE, append ('ab') mode is unsupported and causes data corruption.
        # We always stream cleanly from byte 0 in 'wb' mode.
        if drive_target.exists():
            drive_target.unlink(missing_ok=True)
            time.sleep(0.5)

        import urllib.request
        log_info(f"Direct streaming {drive_target.name} to Google Drive (Zero local SSD usage)...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(drive_target, "wb") as fdst:
                total_bytes = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                start_t = time.time()
                last_log = start_t
                chunk_size = 16 * 1024 * 1024
                while True:
                    buf = resp.read(chunk_size)
                    if not buf:
                        break
                    fdst.write(buf)
                    downloaded += len(buf)
                    if time.time() - last_log >= 15:
                        pct = (downloaded / total_bytes * 100) if total_bytes > 0 else 0
                        speed = (downloaded / (1024 ** 2)) / max(time.time() - start_t, 1)
                        log_info(f"Direct stream: {downloaded / (1024**3):.2f} / {total_bytes / (1024**3):.1f} GB ({pct:.1f}%) [{speed:.1f} MB/s]")
                        last_log = time.time()

            elapsed = max(time.time() - start_t, 1.0)
            speed_mbs = (downloaded / (1024 ** 2)) / elapsed
            self._prune_caches()
            self._flush_drive_fuse()
            drive_target = get_actual_drive_path(drive_target)

            if drive_target.exists() and drive_target.stat().st_size >= min_bytes:
                log_success(f"✓ Direct streamed {drive_target.name} to Drive in {elapsed/60:.1f}m ({speed_mbs:.1f} MB/s)!")
                return True
            else:
                curr_size = drive_target.stat().st_size if drive_target.exists() else 0
                log_err(f"Direct stream finished but size mismatch for {drive_target.name}: {curr_size} < {min_bytes}")
                return False
        except Exception as e:
            log_err(f"Direct stream error for {drive_target.name}: {e}")
            return False

    def _download_part(self, url: str, part_name: str, drive_target: Path, min_bytes: int) -> bool:
        drive_target = get_actual_drive_path(drive_target)

        # 1. PRIORITY: Check if already complete on Google Drive
        if drive_target.exists() and drive_target.stat().st_size >= min_bytes:
            log_success(f"✓ {part_name} is already complete on Google Drive ({drive_target.stat().st_size / (1024**3):.2f} GB)!")
            staged = Path("/content/_staging") / part_name
            staged.unlink(missing_ok=True)
            return True

        # 2. Check if a corrupted/truncated partial exists on Drive
        if drive_target.exists() and drive_target.stat().st_size < min_bytes:
            curr_gb = drive_target.stat().st_size / (1024 ** 3)
            expected_gb = min_bytes / (1024 ** 3)
            log_warn(f"Drive copy of {part_name} is incomplete ({curr_gb:.2f} GB < {expected_gb:.2f} GB). Removing corrupted/partial file...")
            drive_target.unlink(missing_ok=True)
            time.sleep(1)

        # 3. Check staging directory on SSD
        staging_dir = Path("/content/_staging")
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_file = staging_dir / part_name

        if staged_file.exists() and staged_file.stat().st_size >= min_bytes:
            log_success(f"✓ {part_name} already in local SSD staging ({staged_file.stat().st_size / (1024**3):.2f} GB). Transferring to Drive...")
            return self._stream_copy_to_drive(staged_file, drive_target)

        self._prune_caches()
        free_ssd = shutil.disk_usage(staging_dir).free / (1024 ** 3)
        log_info(f"Local SSD free space: {free_ssd:.1f} GB")

        # If SSD free space is tight (< 45 GB), use Direct Stream (Zero local SSD usage)
        if free_ssd < 45.0:
            log_info(f"SSD space ({free_ssd:.1f} GB) is tight for staging. Direct streaming {part_name} to Drive (Zero SSD usage)...")
            return self._direct_stream_from_url_to_drive(url, drive_target, min_bytes)

        # 4. If SSD has >= 45 GB free, use accelerated aria2c staging
        has_aria2 = self._ensure_aria2()
        if has_aria2:
            log_info(f"🚀 Downloading {part_name} via aria2c (16 parallel connections on SSD)...")
            cmd = [
                "aria2c",
                "-x", "16",
                "-s", "16",
                "-k", "1M",
                "--file-allocation=none",
                "--continue=true",
                "--summary-interval=10",
                "-d", str(staging_dir),
                "-o", part_name,
                url
            ]
            res = subprocess.run(cmd, check=False)
            if res.returncode == 0 and staged_file.exists() and staged_file.stat().st_size >= min_bytes:
                log_success(f"✓ Downloaded {part_name} to SSD ({staged_file.stat().st_size / (1024**3):.2f} GB)")
                try:
                    return self._stream_copy_to_drive(staged_file, drive_target)
                except OSError as e:
                    if e.errno == 28:
                        log_warn(f"[Errno 28] Local SSD full during copy. Deleting staging and switching to Direct Stream for {part_name}...")
                        staged_file.unlink(missing_ok=True)
                        self._prune_caches()
                        return self._direct_stream_from_url_to_drive(url, drive_target, min_bytes)
                    raise

        # Fallback: direct streaming
        return self._direct_stream_from_url_to_drive(url, drive_target, min_bytes)

    def download_annotations(self) -> bool:
        log_header("1. Downloading Fine-Tuning Annotation JSONs")
        if self.drive_image_json.exists() and self.drive_video_json.exists() and self.drive_nlp_json.exists():
            log_success("All fine-tuning annotation JSONs already exist on Drive!")
            return True

        zip_path = self.drive_data_dir / "annotations.zip"
        if not zip_path.exists() or zip_path.stat().st_size < 400 * 1024 * 1024:
            log_info("Downloading annotations archive (~471 MB) from Google Drive...")
            try:
                import gdown
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
                import gdown
            gdown.download(id=self.ANNOTATIONS_GDRIVE_ID, output=str(zip_path), quiet=False)

        if zip_path.exists() and zip_path.stat().st_size > 0:
            log_info("Extracting fine-tuning JSONs...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    base = os.path.basename(member)
                    if base in ["llava_image_tune_.json", "videochatgpt_tune_.json", "nlp_tune.json"]:
                        dest = self.drive_ft_json_dir / base
                        with zf.open(member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        log_success(f"Saved to Drive: {base} ({dest.stat().st_size / (1024**2):.1f} MB)")
            return True
        return False

    def download_all_datasets(self) -> bool:
        all_ok = True
        log_header("2. Downloading Image Tuning Archives (~67.4 GB)")
        for part in self.IMAGE_TUNE_PARTS:
            url = f"{self.HF_BASE_URL}/{part}"
            dest = self.drive_data_dir / part
            min_b = self.IMAGE_PART_MIN_BYTES[part]
            ok = self._download_part(url, part, dest, min_b)
            if not ok:
                all_ok = False
                log_err(f"Download check/transfer failed for {part}")

        log_header("3. Downloading Video Tuning Archives (~160.1 GB)")
        for part in self.VIDEO_TUNE_PARTS:
            url = f"{self.HF_BASE_URL}/{part}"
            dest = self.drive_data_dir / part
            min_b = self.VIDEO_PART_MIN_BYTES[part]
            ok = self._download_part(url, part, dest, min_b)
            if not ok:
                all_ok = False
                log_err(f"Download check/transfer failed for {part}")

        return all_ok

    def extract_datasets_on_drive(self, clean_zips: bool = False) -> bool:
        log_header("Extracting Fine-Tuning Datasets on Google Drive")
        try:
            subprocess.run(["apt-get", "install", "-y", "-qq", "p7zip-full"], check=False)
        except Exception:
            pass

        seven_zip = shutil.which("7z") or shutil.which("7za")
        if not seven_zip:
            log_err("7-Zip (p7zip-full) is required for extracting split archives directly on Drive.")
            return False

        # 1. Extract Image Tuning Dataset
        self.drive_image_folder.mkdir(parents=True, exist_ok=True)
        img_subdirs = ["coco", "gqa", "ocr_vqa", "textvqa", "vg"]
        img_extracted = any((self.drive_image_folder / d).is_dir() for d in img_subdirs)

        if img_extracted:
            log_success(f"✓ Image tuning dataset is already extracted in {self.drive_image_folder}!")
        else:
            img_part1 = self.drive_data_dir / self.IMAGE_TUNE_PARTS[0]
            inner_img_zip = self.drive_data_dir / "llava_image_tune.zip"
            if not inner_img_zip.exists() and img_part1.exists():
                log_info("Stage 1/2: Unpacking outer split archive (llava_image_tune_2.zip.*) via 7-Zip...")
                cmd = [seven_zip, "x", str(img_part1), f"-o{self.drive_data_dir}", "-y"]
                res = subprocess.run(cmd, check=False)
                if res.returncode != 0:
                    log_err(f"Failed to unpack {img_part1.name}. Please ensure all parts are complete.")
                    return False
                log_success(f"Outer split unpacked: {inner_img_zip.name}")

            if inner_img_zip.exists():
                log_info(f"Stage 2/2: Extracting inner archive ({inner_img_zip.name}) into {self.drive_image_folder}...")
                cmd = [seven_zip, "x", str(inner_img_zip), f"-o{self.drive_image_folder}", "-y"]
                res = subprocess.run(cmd, check=False)
                if res.returncode == 0:
                    log_success(f"✓ Extracted image tuning data into {self.drive_image_folder}")
                    inner_img_zip.unlink(missing_ok=True)
                    log_info(f"Cleaned up intermediate {inner_img_zip.name}")
                else:
                    log_err(f"Failed to extract inner archive {inner_img_zip.name}")
                    return False

            if clean_zips:
                for p in self.IMAGE_TUNE_PARTS:
                    (self.drive_data_dir / p).unlink(missing_ok=True)
                log_info("Cleaned up Image Tuning split archives.")

        # 2. Extract Video Tuning Dataset
        self.drive_video_folder.mkdir(parents=True, exist_ok=True)
        vid_subdirs = ["Activity_Videos", "Activitynet_Zero_Shot_QA"]
        vid_extracted = any((self.drive_video_folder / d).is_dir() for d in vid_subdirs)

        if vid_extracted:
            log_success(f"✓ Video tuning dataset is already extracted in {self.drive_video_folder}!")
        else:
            vid_part1 = self.drive_data_dir / self.VIDEO_TUNE_PARTS[0]
            inner_vid_zip = self.drive_data_dir / "videochatgpt_tune.zip"
            if not inner_vid_zip.exists() and vid_part1.exists():
                log_info("Stage 1/2: Unpacking outer split archive (videochatgpt_tune_2.zip.*) via 7-Zip...")
                cmd = [seven_zip, "x", str(vid_part1), f"-o{self.drive_data_dir}", "-y"]
                res = subprocess.run(cmd, check=False)
                if res.returncode != 0:
                    log_err(f"Failed to unpack {vid_part1.name}. Please ensure all parts are complete.")
                    return False
                log_success(f"Outer split unpacked: {inner_vid_zip.name}")

            if inner_vid_zip.exists():
                log_info(f"Stage 2/2: Extracting inner archive ({inner_vid_zip.name}) into {self.drive_video_folder}...")
                cmd = [seven_zip, "x", str(inner_vid_zip), f"-o{self.drive_video_folder}", "-y"]
                res = subprocess.run(cmd, check=False)
                if res.returncode == 0:
                    log_success(f"✓ Extracted video tuning data into {self.drive_video_folder}")
                    inner_vid_zip.unlink(missing_ok=True)
                    log_info(f"Cleaned up intermediate {inner_vid_zip.name}")
                else:
                    log_err(f"Failed to extract inner archive {inner_vid_zip.name}")
                    return False

            if clean_zips:
                for p in self.VIDEO_TUNE_PARTS:
                    (self.drive_data_dir / p).unlink(missing_ok=True)
                log_info("Cleaned up Video Tuning split archives.")

        return True


# ==============================================================================
# Outbound Checkpoint Muler (Local SSD -> Google Drive)
# ==============================================================================

class OutboundCheckpointMuler:
    def __init__(self, local_dir: Path, drive_dir: Path, interval_seconds: int = 30):
        self.local_dir = Path(local_dir)
        self.drive_dir = Path(drive_dir)
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None

    def _sync_once(self):
        if not self.local_dir.exists():
            return
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        for root, _, files in os.walk(self.local_dir):
            rel_root = Path(root).relative_to(self.local_dir)
            target_root = self.drive_dir / rel_root
            for f in files:
                src_f = Path(root) / f
                dst_f = target_root / f
                if not dst_f.exists() or src_f.stat().st_mtime > dst_f.stat().st_mtime:
                    try:
                        dst_f.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_f, dst_f)
                        log_info(f"Muler synced checkpoint: {rel_root / f}")
                    except Exception:
                        pass

    def _run(self):
        while not self._stop_event.wait(self.interval):
            self._sync_once()
        self._sync_once()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log_info(f"Outbound checkpoint muler started (syncing every {self.interval}s to Drive).")

    def stop(self):
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=10)
            log_success("Checkpoint muler stopped cleanly.")


# ==============================================================================
# Main CLI Dispatcher
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Video-LLaVA Stage 2 Fine-Tuning Colab / Drive Pipeline")
    parser.add_argument("--action", type=str, default="status", choices=["status", "download", "extract", "train", "all"],
                        help="Action to perform")
    parser.add_argument("--drive_root", type=str, default="/content/drive/MyDrive/Video-LLaVA",
                        help="Google Drive root workspace directory")
    parser.add_argument("--local_scratch_dir", type=str, default="/content/data",
                        help="Local NVMe SSD scratch directory")
    parser.add_argument("--lora", type=lambda x: str(x).lower() == 'true', default=True,
                        help="Use LoRA fine-tuning (recommended for single GPU / Colab A100)")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--clean_zips", action="store_true", help="Delete downloaded split zip files after extraction")

    args = parser.parse_args()
    mount_google_drive()

    muler = InboundFineTuneMuler(args.drive_root, args.local_scratch_dir)

    if args.action == "status":
        muler.check_storage()
        muler.download_annotations()

    elif args.action == "download":
        muler.check_storage()
        muler.download_annotations()
        muler.download_all_datasets()

    elif args.action == "extract":
        muler.extract_datasets_on_drive(clean_zips=args.clean_zips)

    elif args.action in ["train", "all"]:
        if args.action == "all":
            muler.check_storage()
            muler.download_annotations()
            muler.download_all_datasets()
            muler.extract_datasets_on_drive(clean_zips=args.clean_zips)

        # Launch Training
        local_ckpt_dir = Path("/content/checkpoints/videollava-7b-finetune")
        drive_ckpt_dir = Path(args.drive_root) / "checkpoints/videollava-7b-finetune"
        outbound = OutboundCheckpointMuler(local_ckpt_dir, drive_ckpt_dir)
        outbound.start()

        # Copy annotation JSONs to local SSD for fast tokenizer / dataset init
        for jf in [muler.drive_image_json, muler.drive_video_json, muler.drive_nlp_json]:
            if jf.exists():
                shutil.copy2(jf, muler.local_ft_json_dir / jf.name)

        script_path = "scripts/v1_5/finetune_lora.sh" if args.lora else "scripts/v1_5/finetune.sh"
        log_header(f"Launching Stage 2 Fine-Tuning ({'LoRA' if args.lora else 'Full'})")
        log_info(f"Script: {script_path}")

        try:
            env = os.environ.copy()
            env["JSON_FOLDER"] = str(muler.local_ft_json_dir)
            env["IMAGE_FOLDER"] = str(muler.drive_data_dir)
            env["VIDEO_FOLDER"] = str(muler.drive_data_dir)
            subprocess.run(["bash", script_path], env=env, check=True)
        finally:
            outbound.stop()


if __name__ == "__main__":
    main()
