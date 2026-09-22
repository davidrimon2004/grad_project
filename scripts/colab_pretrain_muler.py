#!/usr/bin/env python3
"""
Video-LLaVA Stage 1 Pretraining & Data Muling Pipeline for Google Colab & Google Drive.

Original Video-LLaVA Datasets:
  - Image Pretrain: LLaVA-558K (images in `llava_image/`, annotations `llava_image_.json`)
  - Video Pretrain: Valley Video 100K-702K (videos in `valley/`, annotations `valley_.json`)

Architecture (Drive-First):
  Since Colab's ephemeral SSD (~80 GB) cannot hold the full datasets (images: 27 GB,
  videos: 460+ GB), we use Google Drive (5 TB) as the primary data store:

  1. Download archives from HuggingFace directly to Google Drive using wget (no cache overhead).
  2. Extract archives directly on Google Drive (data persists across sessions).
  3. Download annotation JSONs from the official Google Drive zip.
  4. Train reading image/video data from Google Drive paths.
  5. Only checkpoints use the local SSD for fast I/O, and get synced back to Drive.
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
from pathlib import Path
from typing import Optional, List, Dict

# Color helpers for rich Colab terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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


# ==============================================================================
# Google Drive & Environment Utilities
# ==============================================================================

def is_colab_environment() -> bool:
    """Check if the script is executing inside Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ or (os.name != "nt" and os.path.exists("/content"))

def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive if running in Google Colab."""
    if not is_colab_environment():
        log_info("Not running in Google Colab environment. Using local directory paths.")
        return True

    if os.path.ismount(mount_point) or os.path.exists(os.path.join(mount_point, "MyDrive")):
        log_success(f"Google Drive already mounted at {mount_point}")
        return True

    try:
        log_info(f"Mounting Google Drive to {mount_point}...")
        from google.colab import drive
        drive.mount(mount_point)
        log_success(f"Google Drive successfully mounted at {mount_point}")
        return True
    except Exception as e:
        log_err(f"Failed to mount Google Drive: {e}")
        return False


# ==============================================================================
# Inbound Data Muler: HuggingFace -> Google Drive (Drive-First Architecture)
# ==============================================================================

class InboundDataMuler:
    """
    Manages downloading and extracting the original Video-LLaVA pretraining
    datasets directly on Google Drive (since Colab SSD is too small for the
    full dataset). Training reads data from Drive paths.
    """

    HF_DATASET_REPO = "LanguageBind/Video-LLaVA"
    HF_BASE_URL = "https://huggingface.co/datasets/LanguageBind/Video-LLaVA/resolve/main"

    # Official annotations zip from Video-LLaVA authors (Google Drive file ID)
    ANNOTATIONS_GDRIVE_ID = "1zGRyVSUMoczGq6cjQFmT0prH67bu2wXD"

    # Official archives for Video-LLaVA Pretraining Stage 1
    # 1. LLaVA-558K Image archive: llava_image.zip
    # 2. Valley Video archive: valley_2.zip.001 to valley_2.zip.012
    VALLEY_PARTS = [f"valley_2.zip.{i:03d}" for i in range(1, 13)]

    # Expected minimum size in bytes for each part to be considered complete.
    # Parts 001-011: ~41.9 GB each (39.06 GiB = 41,943,040,000 bytes)
    # Part 012: ~2.35 GB (2.19 GiB = 2,351,600,000 bytes)
    VALLEY_PART_MIN_BYTES = {
        f"valley_2.zip.{i:03d}": int(38.5 * (1024 ** 3)) for i in range(1, 12)
    }
    VALLEY_PART_MIN_BYTES["valley_2.zip.012"] = int(2.0 * (1024 ** 3))

    def __init__(self, drive_data_dir: str, local_scratch_dir: str):
        self.drive_data_dir = Path(drive_data_dir)
        self.local_scratch_dir = Path(local_scratch_dir)

        # Data lives on Google Drive (persistent, 5 TB)
        self.drive_image_folder = self.drive_data_dir / "llava_image"
        self.drive_video_folder = self.drive_data_dir / "valley"
        self.drive_json_folder = self.drive_data_dir / "annotations"

        # Local SSD paths (only for annotations & demo data)
        self.local_json_folder = self.local_scratch_dir / "pt_json"
        self.local_image_folder = self.local_scratch_dir / "llava_image"
        self.local_video_folder = self.local_scratch_dir / "valley"

        # Annotation file paths (on Drive after extraction)
        self.drive_image_json = self.drive_json_folder / "llava_image_.json"
        self.drive_video_json = self.drive_json_folder / "valley_.json"

        # Local annotation paths (tiny files, copied to SSD for fast reads)
        self.local_image_json = self.local_json_folder / "llava_image_.json"
        self.local_video_json = self.local_json_folder / "valley_.json"

        # Create directories
        self.drive_data_dir.mkdir(parents=True, exist_ok=True)
        self.local_scratch_dir.mkdir(parents=True, exist_ok=True)
        self.local_json_folder.mkdir(parents=True, exist_ok=True)

    # Minimum file counts to distinguish real datasets from demo stubs
    MIN_REAL_IMAGE_FILES = 10
    MIN_REAL_VIDEO_FILES = 5

    def check_storage_space(self):
        """Diagnose and display storage space across Google Drive, Colab SSD, and /tmp."""
        log_header("Storage Diagnostic & Capacity Check")
        targets = [
            ("Google Drive Data Dir", self.drive_data_dir),
            ("Google Drive Root", Path("/content/drive/MyDrive")),
            ("Colab Local SSD (/content)", Path("/content")),
            ("System Temp (/tmp)", Path("/tmp")),
        ]
        info = {}
        seen_paths = set()
        for label, path in targets:
            try:
                resolved = path.resolve() if path.exists() else path
                if str(resolved) in seen_paths or not path.exists():
                    continue
                seen_paths.add(str(resolved))
                usage = shutil.disk_usage(path)
                free_gb = usage.free / (1024 ** 3)
                total_gb = usage.total / (1024 ** 3)
                used_gb = usage.used / (1024 ** 3)
                pct = (usage.used / usage.total) * 100 if usage.total > 0 else 0
                log_info(f"  • {label:28s}: {free_gb:6.1f} GB free / {total_gb:6.1f} GB total ({pct:5.1f}% used) [{path}]")
                info[label] = {"free_gb": free_gb, "total_gb": total_gb, "used_gb": used_gb}
            except Exception as e:
                log_warn(f"  • {label:28s}: could not read disk usage ({e})")
        return info

    def get_valley_parts_status(self):
        """Inspect all 12 Valley multi-part archives and return status dictionary."""
        status = {}
        total_downloaded = 0
        total_expected = 0
        incomplete_parts = []
        complete_parts = []

        for part_name in self.VALLEY_PARTS:
            part_path = self.drive_data_dir / part_name
            min_expected = self.VALLEY_PART_MIN_BYTES[part_name]
            total_expected += min_expected
            if part_path.exists():
                actual_bytes = part_path.stat().st_size
                total_downloaded += actual_bytes
                is_complete = actual_bytes >= min_expected
            else:
                actual_bytes = 0
                is_complete = False

            actual_gb = actual_bytes / (1024 ** 3)
            expected_gb = min_expected / (1024 ** 3)
            part_info = {
                "name": part_name,
                "path": part_path,
                "actual_gb": actual_gb,
                "expected_gb": expected_gb,
                "is_complete": is_complete,
                "missing_gb": max(0.0, expected_gb - actual_gb),
            }
            status[part_name] = part_info
            if is_complete:
                complete_parts.append(part_name)
            else:
                incomplete_parts.append(part_info)

        return {
            "parts": status,
            "total_downloaded_gb": total_downloaded / (1024 ** 3),
            "total_expected_gb": total_expected / (1024 ** 3),
            "complete_count": len(complete_parts),
            "incomplete_count": len(incomplete_parts),
            "complete_parts": complete_parts,
            "incomplete_parts": incomplete_parts,
        }

    @staticmethod
    def _has_min_files(folder: Path, min_count: int = 10) -> bool:
        """Fast O(1) check: returns True as soon as min_count files exist.
        Never scans all 558,000 files, preventing Drive FUSE timeouts and hangs!"""
        if not folder or not folder.is_dir():
            return False
        count = 0
        try:
            for _ in folder.iterdir():
                count += 1
                if count >= min_count:
                    return True
        except Exception:
            return False
        return count >= min_count

    def verify_drive_dataset(self) -> bool:
        """Check if REAL datasets are already extracted and ready on Google Drive."""
        image_json_ok = self.drive_image_json.is_file() and self.drive_image_json.stat().st_size > 0
        video_json_ok = self.drive_video_json.is_file() and self.drive_video_json.stat().st_size > 0

        image_dir_ok = self._has_min_files(self.drive_image_folder, min_count=10)
        video_dir_ok = self._has_min_files(self.drive_video_folder, min_count=10)

        return image_json_ok and video_json_ok and image_dir_ok and video_dir_ok

    def _ensure_aria2(self) -> bool:
        """Ensure aria2 is installed for 16x parallel multi-connection acceleration."""
        if shutil.which("aria2c"):
            return True
        log_info("Installing aria2 for multi-threaded parallel download acceleration (16 connections)...")
        try:
            subprocess.run(["apt-get", "update", "-qq"], check=False)
        except Exception as e:
            log_warn(f"apt-get install aria2 failed: {e}")
        return shutil.which("aria2c") is not None

    def _prune_local_disk_caches(self):
        """Aggressively prune local caches to maximize SSD space for staging and FUSE buffering."""
        try:
            # 1. Clean /tmp
            for item in Path("/tmp").glob("*"):
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass

            # 2. Clean pip and huggingface temp caches
            shutil.rmtree("/root/.cache/pip", ignore_errors=True)
            for p in Path("/root/.cache").glob("**/tmp*"):
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass

            # 3. Clean apt cache
            try:
                subprocess.run(["apt-get", "clean"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

            # 4. CRITICAL: Clear stale Google DriveFS local upload cache.
            # When Drive FUSE crashes or flushes uncleanly, it leaves its entire local
            # upload buffer (up to 40+ GB per part) under /root/.config/Google/DriveFS.
            # This directory is safe to delete — all uploaded data is already on Drive's
            # servers. Clearing it reclaims SSD space for the next part's staging download.
            drivefs_cache = Path("/root/.config/Google/DriveFS")
            if drivefs_cache.exists():
                cache_size_gb = sum(
                    f.stat().st_size for f in drivefs_cache.rglob("*") if f.is_file()
                ) / (1024 ** 3)
                if cache_size_gb > 1.0:  # Only log if it's actually significant
                    log_info(f"Clearing stale DriveFS local cache ({cache_size_gb:.1f} GB) to reclaim SSD space...")
                shutil.rmtree(str(drivefs_cache), ignore_errors=True)

            # 5. Force OS buffer sync and garbage collection
            import gc
            gc.collect()
            try:
                os.sync()
            except Exception:
                pass
        except Exception as e:
            log_warn(f"Cache pruning warning: {e}")


    def _flush_drive_fuse_cache(self):
        """Flushes and remounts Google Drive FUSE to release local SSD write cache."""
        try:
            from google.colab import drive
            log_info("Flushing Google Drive FUSE write buffers to release local SSD cache...")
            drive.flush_and_unmount()
            drive.mount('/content/drive')
            log_success("Google Drive remounted cleanly.")
        except Exception as e:
            log_warn(f"Drive FUSE flush/remount skipped ({e})")

    def _ensure_ssd_staging_space(self, min_gb_needed: float = 41.0) -> Path:
        """Ensure Colab local NVMe SSD has sufficient space to stage a 39GB part."""
        staging_dir = Path("/content/_staging")
        staging_dir.mkdir(parents=True, exist_ok=True)
        try:
            free_gb = shutil.disk_usage(staging_dir).free / (1024 ** 3)
            if free_gb < min_gb_needed:
                log_info(f"Local SSD free space ({free_gb:.1f} GB) is tight for 39 GB staging. Pruning local caches...")
                self._prune_local_disk_caches()
                free_gb = shutil.disk_usage(staging_dir).free / (1024 ** 3)
                log_info(f"Local SSD free space after cache cleanup: {free_gb:.1f} GB")
                if free_gb < min_gb_needed:
                    self._flush_drive_fuse_cache()
                    free_gb = shutil.disk_usage(staging_dir).free / (1024 ** 3)
                    log_info(f"Local SSD free space after Drive remount: {free_gb:.1f} GB")
        except Exception:
            pass
        return staging_dir

    def _stream_copy_to_drive(self, local_path: Path, drive_path: Path, chunk_size: int = 64 * 1024 * 1024):
        """Stream copy from local NVMe SSD to Google Drive in 64MB blocks to eliminate FUSE latency."""
        size_gb = local_path.stat().st_size / (1024 ** 3)
        log_info(f"Transferring {local_path.name} to Google Drive ({size_gb:.2f} GB) using 64MB streaming buffer...")
        drive_path.parent.mkdir(parents=True, exist_ok=True)

        # If an incomplete/old partial exists on Drive, remove it first to:
        # 1. Avoid FUSE trying to hold/truncate both versions simultaneously (which demands double space)
        # 2. Reclaim old partial file space on Google Drive
        if drive_path.exists():
            old_gb = drive_path.stat().st_size / (1024 ** 3)
            log_info(f"Removing old partial on Drive ({drive_path.name}, {old_gb:.2f} GB) before transfer...")
            drive_path.unlink(missing_ok=True)

        start_t = time.time()
        try:
            # Open with r+b to allow fallocate hole-punching if supported by filesystem
            open_mode = "r+b" if os.access(local_path, os.W_OK) else "rb"
            with open(local_path, open_mode) as fsrc, open(drive_path, "wb") as fdst:
                offset = 0
                last_log = time.time()
                total_bytes = local_path.stat().st_size
                while True:
                    buf = fsrc.read(chunk_size)
                    if not buf:
                        break
                    fdst.write(buf)
                    # Real-time block deallocation: punch holes in source file on local SSD
                    # as blocks are read. This instantly returns disk blocks to local VM SSD,
                    # completely preventing Drive FUSE write buffers from filling the disk!
                    try:
                        if open_mode == "r+b" and hasattr(os, "fallocate"):
                            # 0x03 = FALLOC_FL_PUNCH_HOLE (0x02) | FALLOC_FL_KEEP_SIZE (0x01)
                            os.fallocate(fsrc.fileno(), 0x03, offset, len(buf))
                    except Exception:
                        pass
                    offset += len(buf)
                    if time.time() - last_log >= 15:
                        pct = (offset / total_bytes) * 100 if total_bytes > 0 else 0
                        speed = (offset / (1024 ** 2)) / max(time.time() - start_t, 1)
                        log_info(f"Drive transfer: {offset / (1024**3):.2f} / {size_gb:.1f} GB ({pct:.1f}%) [{speed:.1f} MB/s]")
                        last_log = time.time()

            elapsed = max(time.time() - start_t, 1.0)
            speed_mbs = (size_gb * 1024) / elapsed
            log_success(f"✓ Transferred {local_path.name} to Drive in {elapsed:.0f}s ({speed_mbs:.1f} MB/s)!")
            # Delete local staged file immediately to free the 39 GB for the next part!
            local_path.unlink(missing_ok=True)
            self._prune_local_disk_caches()
            return True
        except OSError as e:
            log_err(f"Failed to transfer {local_path.name} to Google Drive: {e}")
            if e.errno == 28:
                log_warn("=" * 65)
                log_warn("[Errno 28] No space left on device during Drive transfer!")
                log_warn("Action items to resolve:")
                log_warn("1. If local SSD space is low, Colab Drive FUSE cannot allocate buffer space.")
                log_warn("   Run 'drive.flush_and_unmount()' then 'drive.mount(\"/content/drive\")' to wipe FUSE cache.")
                log_warn("2. Empty Google Drive Trash (https://drive.google.com/drive/trash) to reclaim cloud quota.")
                log_warn("3. Note: Staged file on SSD is PRESERVED and will not be re-downloaded.")
                log_warn("=" * 65)
            raise

    def _direct_stream_from_url_to_drive(self, url: str, drive_target: Path, min_bytes: int, chunk_size: int = 16 * 1024 * 1024) -> bool:
        """
        Directly stream from HuggingFace HTTP to Google Drive with ZERO local SSD usage.
        Memory buffer only (16 MB chunks). Protects Colab SSD from ever filling up.
        """
        import urllib.request
        log_info(f"Direct streaming {drive_target.name} to Google Drive (Zero local SSD usage)...")
        drive_target.parent.mkdir(parents=True, exist_ok=True)

        initial_bytes = 0
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        if drive_target.exists():
            initial_bytes = drive_target.stat().st_size
            if initial_bytes >= min_bytes:
                log_success(f"✓ {drive_target.name} is already complete on Google Drive ({initial_bytes / (1024**3):.2f} GB)!")
                return True
            if initial_bytes > 0:
                headers["Range"] = f"bytes={initial_bytes}-"
                log_info(f"Resuming {drive_target.name} on Drive from byte {initial_bytes} ({initial_bytes / (1024**3):.2f} GB)...")

        mode = "ab" if initial_bytes > 0 else "wb"
        start_t = time.time()
        downloaded = initial_bytes

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(drive_target, mode) as fdst:
                total_bytes = int(resp.headers.get("Content-Length", 0)) + initial_bytes
                total_gb = total_bytes / (1024 ** 3) if total_bytes > 0 else 39.0
                last_log = time.time()
                while True:
                    buf = resp.read(chunk_size)
                    if not buf:
                        break
                    fdst.write(buf)
                    downloaded += len(buf)
                    if time.time() - last_log >= 15:
                        pct = (downloaded / total_bytes) * 100 if total_bytes > 0 else 0
                        speed = (downloaded - initial_bytes) / (1024 ** 2) / max(time.time() - start_t, 1)
                        log_info(f"Direct stream: {downloaded / (1024**3):.2f} / {total_gb:.1f} GB ({pct:.1f}%) [{speed:.1f} MB/s]")
                        last_log = time.time()

            elapsed = max(time.time() - start_t, 1.0)
            speed_mbs = ((downloaded - initial_bytes) / (1024 ** 2)) / elapsed
            log_success(f"✓ Direct streamed {drive_target.name} to Drive in {elapsed/60:.1f}m ({speed_mbs:.1f} MB/s)!")
            return drive_target.exists() and drive_target.stat().st_size >= min_bytes
        except Exception as e:
            log_err(f"Direct stream encountered an error: {e}")
            return False

    def _download_part_accelerated(self, url: str, part_name: str, drive_target: Path, min_bytes: int) -> bool:
        """
        Accelerated download:
        1. If already staged on local SSD (complete), skips download and streams to Drive immediately!
        2. Downloads to local NVMe SSD (/content/_staging) via aria2c (16 parallel connections).
        3. Streams completed file to Google Drive using 64MB chunks with real-time hole punching.
        4. Instantly deletes local copy and prunes cache to free SSD space for subsequent parts.
        """
        staging_dir = Path("/content/_staging")
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_file = staging_dir / part_name

        # PRIORITY CHECK: If file is already fully staged on local SSD, DO NOT re-download!
        if staged_file.exists() and staged_file.stat().st_size >= min_bytes:
            log_success(f"✓ {part_name} is ALREADY fully downloaded in local SSD staging ({staged_file.stat().st_size / (1024**3):.2f} GB)!")
            self._stream_copy_to_drive(staged_file, drive_target)
            return drive_target.exists() and drive_target.stat().st_size >= min_bytes

        # RESUME CHECK: If Drive already has a substantial partial (>100 MB), skip aria2c entirely
        # and use HTTP Range resume to append the remaining bytes. This preserves downloaded work
        # after Colab disconnects and avoids re-downloading e.g. 38 GB of a 39 GB part!
        try:
            drive_partial_bytes = drive_target.stat().st_size if drive_target.exists() else 0
        except Exception:
            drive_partial_bytes = 0
        if 0 < drive_partial_bytes < min_bytes and drive_partial_bytes > 100 * 1024 * 1024:
            log_info(f"Partial found on Drive for {part_name} ({drive_partial_bytes / (1024**3):.2f} GB). Resuming via HTTP Range...")
            return self._direct_stream_from_url_to_drive(url, drive_target, min_bytes)
        if drive_partial_bytes >= min_bytes:
            log_success(f"✓ {part_name} already complete on Drive ({drive_partial_bytes / (1024**3):.2f} GB)!")
            return True

        staging_dir = self._ensure_ssd_staging_space(min_gb_needed=41.0)
        has_aria2 = self._ensure_aria2()
        ssd_free_gb = shutil.disk_usage(staging_dir).free / (1024 ** 3)
        log_info(f"Local SSD staging space available: {ssd_free_gb:.1f} GB")

        if has_aria2 and ssd_free_gb >= 40.0:
            log_info(f"🚀 Launching multi-threaded download for {part_name} (16 parallel connections on SSD)...")
            cmd = [
                "aria2c",
                "-x", "16",
                "-s", "16",
                "-k", "1M",
                "--file-allocation=none",
                "--continue=true",
                "--console-log-level=warn",   # Suppress verbose progress flood
                "--summary-interval=0",        # Disable built-in summary (we print our own)
                "-d", str(staging_dir),
                "-o", part_name,
                url
            ]

            # Launch a heartbeat thread to print clean progress every 30s
            # (prevents Colab output from freezing due to rapid log lines)
            import threading
            _stop_heartbeat = threading.Event()
            def _heartbeat(path, total_bytes, stop_event):
                t0 = time.time()
                while not stop_event.wait(30):
                    try:
                        done = path.stat().st_size if path.exists() else 0
                        pct = (done / total_bytes * 100) if total_bytes > 0 else 0
                        speed = done / (1024**2) / max(time.time() - t0, 1)
                        log_info(f"aria2c: {done/(1024**3):.2f} / {total_bytes/(1024**3):.1f} GB ({pct:.1f}%) [{speed:.1f} MB/s]")
                    except Exception:
                        pass
            hb = threading.Thread(target=_heartbeat, args=(staged_file, min_bytes, _stop_heartbeat), daemon=True)
            hb.start()
            res = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _stop_heartbeat.set()
            hb.join(timeout=2)

            if res.returncode == 0 and staged_file.exists() and staged_file.stat().st_size >= min_bytes:
                log_success(f"✓ {part_name} fully downloaded to local SSD ({staged_file.stat().st_size / (1024**3):.2f} GB)!")
                self._stream_copy_to_drive(staged_file, drive_target)
                return True
            else:
                log_warn(f"aria2c returned code {res.returncode}. Staged size: {staged_file.stat().st_size if staged_file.exists() else 0} bytes.")

        # Fallback to direct HTTP stream (Zero local SSD usage) if SSD space was tight
        log_info(f"Using direct stream fallback for {part_name} (Zero local SSD usage)...")
        return self._direct_stream_from_url_to_drive(url, drive_target, min_bytes)

    def _wget_download(self, url: str, output_path: str, desc: str = ""):
        """Download a file using wget with auto-resume (-c), retry resilience, and direct write."""
        log_info(f"Downloading {desc or url}...")
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "wget", "-c",  # -c enables resume of partial downloads
            "--progress=bar:force:noscroll",
            "--tries=10",
            "--retry-connrefused",
            "--waitretry=5",
            "--timeout=30",
            "-O", str(out_p),
            url
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            log_err(f"wget failed for {desc or url} (exit code {result.returncode})")
            return False
        log_success(f"Downloaded/resumed {desc}")
        return True

    def download_annotations(self):
        """Download annotation JSONs from the official Google Drive zip."""
        if self.drive_image_json.exists() and self.drive_video_json.exists():
            log_success("Annotation JSONs already present on Drive.")
            return True

        self.drive_json_folder.mkdir(parents=True, exist_ok=True)
        log_info("Downloading annotation JSONs from official Video-LLaVA Google Drive zip...")

        # Install gdown if needed
        try:
            import gdown
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            import gdown

        # Download the annotations zip
        annot_zip = self.drive_data_dir / "annotations.zip"
        if not annot_zip.exists():
            try:
                gdown.download(
                    id=self.ANNOTATIONS_GDRIVE_ID,
                    output=str(annot_zip),
                    quiet=False
                )
            except Exception as e:
                log_err(f"Failed to download annotations zip: {e}")
                log_info("Trying alternative download with gdown fuzzy mode...")
                try:
                    url = f"https://drive.google.com/uc?id={self.ANNOTATIONS_GDRIVE_ID}"
                    gdown.download(url, str(annot_zip), quiet=False, fuzzy=True)
                except Exception as e2:
                    log_err(f"Alternative download also failed: {e2}")
                    return False

        if annot_zip.exists() and annot_zip.stat().st_size > 0:
            log_info("Extracting annotations zip...")
            extract_dir = self.drive_data_dir / "_annot_extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["unzip", "-q", "-o", str(annot_zip), "-d", str(extract_dir)], check=True)

            found_any = False
            for json_name in ["llava_image_.json", "valley_.json", "chat.json"]:
                matches = list(extract_dir.rglob(json_name))
                if matches:
                    shutil.copy2(matches[0], self.drive_json_folder / json_name)
                    log_success(f"Found and saved {json_name}")
                    found_any = True
                else:
                    log_warn(f"Could not find {json_name} in annotations zip")

            if not found_any:
                log_info("Searching for any annotation JSONs in the zip...")
                for jf in extract_dir.rglob("*.json"):
                    dest = self.drive_json_folder / jf.name
                    shutil.copy2(jf, dest)
                    log_info(f"  Extracted: {jf.name} ({jf.stat().st_size / 1024:.0f} KB)")

            shutil.rmtree(extract_dir, ignore_errors=True)
            return True
        else:
            log_err("Annotations zip download produced empty file")
            return False

    def download_image_archive(self):
        """Download llava_image.zip directly to Google Drive using wget."""
        image_archive = self.drive_data_dir / "llava_image.zip"

        if self._has_min_files(self.drive_image_folder, min_count=10):
            log_success(f"Image dataset already extracted on Drive ({self.drive_image_folder})")
            return True

        if image_archive.exists() and image_archive.stat().st_size > 25 * (1024 ** 3):
            log_success(f"llava_image.zip already on Drive ({image_archive.stat().st_size / (1024**3):.2f} GB)")
        else:
            url = f"{self.HF_BASE_URL}/llava_image.zip"
            success = self._wget_download(url, str(image_archive), "llava_image.zip (~27 GB)")
            if not success:
                return False

        return True

    def extract_image_archive(self):
        """Extract llava_image.zip directly on Google Drive."""
        if self._has_min_files(self.drive_image_folder, min_count=10):
            log_success("Image dataset already extracted on Drive.")
            return True

        image_archive = self.drive_data_dir / "llava_image.zip"
        if not image_archive.exists():
            log_err("llava_image.zip not found on Drive. Download it first.")
            return False

        log_info(f"Extracting llava_image.zip on Google Drive ({image_archive.stat().st_size / (1024**3):.2f} GB)...")
        self.drive_image_folder.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["unzip", "-q", "-o", str(image_archive), "-d", str(self.drive_data_dir)],
            check=False
        )
        if result.returncode != 0:
            log_err(f"Image extraction failed (exit code {result.returncode})")
            return False

        self._fixup_nested_directory(self.drive_image_folder, "llava_image")
        log_success(f"Extracted image files verified in {self.drive_image_folder}")
        return True

    def cleanup_image_archive(self):
        """Reclaims ~27 GB of Google Drive space by removing llava_image.zip if already extracted."""
        image_archive = self.drive_data_dir / "llava_image.zip"
        if not image_archive.exists():
            log_info("llava_image.zip does not exist on Drive (already cleaned up or not downloaded).")
            return
        if self._has_min_files(self.drive_image_folder, min_count=10):
            size_gb = image_archive.stat().st_size / (1024 ** 3)
            image_archive.unlink()
            log_success(f"Removed llava_image.zip to reclaim {size_gb:.1f} GB of Google Drive space (extracted images intact).")
        else:
            log_warn("Cannot remove llava_image.zip because extracted image folder is empty or incomplete.")

    def download_video_archives(self, parts_filter=None):
        """Download or resume valley_2.zip.* parts directly to Google Drive using wget."""
        # Check if already extracted
        if self._has_min_files(self.drive_video_folder, min_count=10):
            log_success(f"Video dataset already extracted on Drive ({self.drive_video_folder})")
            return True

        status = self.get_valley_parts_status()
        if status["incomplete_count"] == 0:
            log_success(f"All 12 Valley video archives already downloaded and complete ({status['total_downloaded_gb']:.1f} GB)!")
            return True

        log_info(f"Valley video archives progress: {status['complete_count']}/12 complete ({status['total_downloaded_gb']:.1f} GB downloaded).")
        needed_total_gb = status['total_expected_gb'] - status['total_downloaded_gb']
        log_info(f"Remaining download needed across all incomplete parts: ~{needed_total_gb:.1f} GB")

        # Proactively reclaim Drive space by removing llava_image.zip if images are extracted
        self.cleanup_image_archive()

        # Normalize parts_filter if specified (e.g. ['4', '5'] or ['004', '005'] or ['valley_2.zip.004'])
        allowed_parts = None
        if parts_filter:
            allowed_parts = set()
            for p in parts_filter:
                p_clean = p.strip()
                if p_clean.isdigit():
                    allowed_parts.add(f"valley_2.zip.{int(p_clean):03d}")
                elif not p_clean.startswith("valley_2.zip."):
                    allowed_parts.add(f"valley_2.zip.{p_clean}")
                else:
                    allowed_parts.add(p_clean)
            log_info(f"Downloading filtered parts only: {sorted(allowed_parts)}")

        log_info("Pipeline: 16x parallel connection SSD staging with 64MB streaming to Google Drive.")
        log_info("Bypasses FUSE latency bottlenecks and speeds download from ~700 KB/s up to 80-120 MB/s.")

        for p in status["incomplete_parts"]:
            part_name = p["name"]
            if allowed_parts and part_name not in allowed_parts:
                continue

            part_file = p["path"]
            actual_gb = p["actual_gb"]
            expected_gb = p["expected_gb"]

            log_info(f"Processing {part_name} (target ~{expected_gb:.1f} GB)...")
            url = f"{self.HF_BASE_URL}/{part_name}"
            min_bytes = self.VALLEY_PART_MIN_BYTES[part_name]
            success = self._download_part_accelerated(url, part_name, part_file, min_bytes)
            if not success:
                log_warn(f"Download of {part_name} encountered an issue.")

            # Check size on Drive after download & transfer
            if part_file.exists():
                new_size_gb = part_file.stat().st_size / (1024 ** 3)
                if part_file.stat().st_size >= min_bytes:
                    log_success(f"✓ {part_name} is now COMPLETE on Google Drive ({new_size_gb:.2f} GB)!")
                    # Flush Drive FUSE write buffers to keep SSD space clean for next parts
                    self._flush_drive_fuse_cache()
                else:
                    log_warn(f"⚠ {part_name} is at {new_size_gb:.2f} GB / ~{expected_gb:.1f} GB.")

        new_status = self.get_valley_parts_status()
        if new_status["incomplete_count"] == 0:
            log_success("All 12 Valley video parts are now fully downloaded and ready for extraction!")
            return True
        else:
            log_warn(f"{new_status['incomplete_count']}/12 parts remain incomplete ({new_status['total_downloaded_gb']:.1f} GB / ~{new_status['total_expected_gb']:.1f} GB).")
            return False

    def extract_video_archives(self):
        """Extract valley multi-part zip archives directly on Google Drive."""
        if self._has_min_files(self.drive_video_folder, min_count=10):
            log_success("Video dataset already extracted on Drive.")
            return True

        status = self.get_valley_parts_status()
        if status["incomplete_count"] > 0:
            log_err(f"Cannot extract archives: {status['incomplete_count']}/12 part(s) are incomplete or missing!")
            for p in status["incomplete_parts"]:
                log_err(f"  • {p['name']}: {p['actual_gb']:.2f} GB / ~{p['expected_gb']:.1f} GB (missing {p['missing_gb']:.2f} GB)")
            log_err("All 12 parts must be fully downloaded before 7-Zip can unpack the multi-part archive.")
            log_err("Please run with '--action download' to finish downloading the remaining parts.")
            return False

        first_part = self.drive_data_dir / "valley_2.zip.001"
        self.drive_video_folder.mkdir(parents=True, exist_ok=True)

        log_info("Extracting multi-part Valley video archives on Google Drive...")
        log_info("(All 12 parts verified complete. Extracting ~460 GB of videos directly on Drive)")
        log_info("Redirecting temporary extraction buffers to Google Drive to protect Colab VM disk.")

        # Create working directory on Drive so 7-Zip NEVER writes temporary files to Colab /tmp (which has only ~50 GB)
        drive_work_dir = self.drive_data_dir / "_7z_work"
        drive_work_dir.mkdir(parents=True, exist_ok=True)

        has_7z = shutil.which("7z") or shutil.which("7za")
        if has_7z:
            seven_z = "7z" if shutil.which("7z") else "7za"
            # -w switch redirects 7-Zip working files to Google Drive instead of Colab /tmp
            cmd = [seven_z, "x", str(first_part), f"-o{self.drive_data_dir}", f"-w{drive_work_dir}", "-y"]
            log_info(f"Running: {' '.join(cmd)}")
            env = os.environ.copy()
            env["TMPDIR"] = str(drive_work_dir)
            result = subprocess.run(cmd, env=env, check=False)
            shutil.rmtree(drive_work_dir, ignore_errors=True)
        else:
            log_err("7z / 7za not installed. Please install with: apt-get install -y p7zip-full")
            return False

        if result.returncode != 0:
            log_err(f"Video extraction failed (exit code {result.returncode})")
            return False

        # Handle nested folder naming (e.g. if extracted as 'valley' or 'valley_2')
        valley_2_dir = self.drive_data_dir / "valley_2"
        if valley_2_dir.is_dir() and not self.drive_video_folder.is_dir():
            valley_2_dir.rename(self.drive_video_folder)
        elif valley_2_dir.is_dir() and self.drive_video_folder.is_dir():
            for item in valley_2_dir.iterdir():
                dest = self.drive_video_folder / item.name
                if not dest.exists():
                    shutil.move(str(item), str(self.drive_video_folder))
            shutil.rmtree(valley_2_dir, ignore_errors=True)

        self._fixup_nested_directory(self.drive_video_folder, "valley")
        log_success(f"Extracted video files verified in {self.drive_video_folder}")
        return True

    def sync_annotations_to_local(self):
        """Copy tiny annotation JSON files from Drive to local SSD for fast reads."""
        self.local_json_folder.mkdir(parents=True, exist_ok=True)
        for json_name in ["llava_image_.json", "valley_.json", "chat.json"]:
            drive_src = self.drive_json_folder / json_name
            local_dst = self.local_json_folder / json_name
            if drive_src.exists():
                shutil.copy2(drive_src, local_dst)
                log_info(f"Synced {json_name} to local SSD ({drive_src.stat().st_size / (1024*1024):.1f} MB)")

    def download_and_prepare_all(self, parts_filter=None):
        """Full pipeline: download archives, extract on Drive, sync annotations."""
        log_header("Step 1: Downloading & Preparing Datasets on Google Drive")

        if self.verify_drive_dataset():
            log_success("All datasets already present and extracted on Google Drive!")
            self.sync_annotations_to_local()
            self.print_dataset_summary()
            return True

        # 0. Display storage diagnostics
        self.check_storage_space()

        # 1. Download annotations
        if not self.download_annotations():
            log_err("Annotation download failed.")
            return False

        # 2. Download & extract images
        if not self.download_image_archive():
            log_err("Image archive download failed.")
            return False
        if not self.extract_image_archive():
            log_err("Image archive extraction failed.")
            return False

        # Reclaim ~27 GB on Google Drive now that images are verified extracted
        self.cleanup_image_archive()

        # 3. Download & extract videos
        videos_downloaded = self.download_video_archives(parts_filter=parts_filter)
        if not videos_downloaded:
            log_warn("Video archives are not fully downloaded yet. Extraction postponed.")
            self.sync_annotations_to_local()
            self.print_dataset_summary()
            return False

        videos_extracted = self.extract_video_archives()
        if not videos_extracted:
            log_err("Video extraction failed.")
            self.sync_annotations_to_local()
            self.print_dataset_summary()
            return False

        # 4. Sync annotation JSONs to local SSD
        self.sync_annotations_to_local()

        log_header("Data Preparation Complete!")
        self.print_dataset_summary()
        return True

    def _fixup_nested_directory(self, target_dir: Path, folder_name: str):
        """Fixes nested directory extractions if archives contained root folders."""
        nested = target_dir / folder_name
        if nested.is_dir():
            log_info(f"Fixing nested directory in {target_dir}...")
            for item in nested.iterdir():
                dest = target_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(target_dir))
            shutil.rmtree(nested, ignore_errors=True)

    def print_dataset_summary(self):
        """Prints counts of images, videos, and annotation samples available."""
        self.check_storage_space()

        # Check Drive paths (where data lives)
        images_ok = self._has_min_files(self.drive_image_folder, min_count=10)
        videos_ok = self._has_min_files(self.drive_video_folder, min_count=10)

        log_info(f"Drive Image folder: {self.drive_image_folder} ({'Extracted (>558K files)' if images_ok else 'Missing/Incomplete'})")
        log_info(f"Drive Video folder: {self.drive_video_folder} ({'Extracted (>702K files)' if videos_ok else 'Missing/Incomplete'})")
        log_info(f"Image annotations: {self.drive_image_json} ({'Found' if self.drive_image_json.exists() else 'Missing'})")
        log_info(f"Video annotations: {self.drive_video_json} ({'Found' if self.drive_video_json.exists() else 'Missing'})")

        parts_status = self.get_valley_parts_status()
        log_info(f"Valley Video Archives: {parts_status['complete_count']}/12 complete ({parts_status['total_downloaded_gb']:.1f} GB / ~{parts_status['total_expected_gb']:.1f} GB)")
        if parts_status["incomplete_count"] > 0:
            log_warn(f"Incomplete parts ({parts_status['incomplete_count']}):")
            for p in parts_status["incomplete_parts"]:
                log_warn(f"  • {p['name']}: {p['actual_gb']:.2f} GB / ~{p['expected_gb']:.1f} GB (missing {p['missing_gb']:.2f} GB)")

    def create_demo_subset(self, num_samples: int = 100):
        """
        Creates a lightweight subset of annotations and sample media
        for rapid end-to-end dry runs or quick validation in Colab.
        Demo data goes on local SSD since it's tiny.
        """
        log_header(f"Creating Fast Demo/Verification Subset ({num_samples} samples)")
        self.local_image_folder.mkdir(parents=True, exist_ok=True)
        self.local_video_folder.mkdir(parents=True, exist_ok=True)
        self.local_json_folder.mkdir(parents=True, exist_ok=True)

        # Create varied dummy images & videos
        from PIL import Image
        colors = [
            (73, 109, 137), (180, 70, 70), (70, 160, 90),
            (200, 180, 60), (130, 80, 160)
        ]
        for c_idx, color in enumerate(colors):
            p = self.local_image_folder / f"sample_{c_idx}.jpg"
            if not p.exists():
                Image.new("RGB", (224, 224), color=color).save(p)

        # Create image annotations
        img_annots = []
        for i in range(num_samples):
            img_annots.append({
                "id": f"img_sample_{i}",
                "image": f"sample_{i % len(colors)}.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nProvide a brief description of the given image."},
                    {"from": "gpt", "value": f"This is a pretraining alignment sample number {i} for Video-LLaVA."}
                ]
            })
        with open(self.local_image_json, "w") as f:
            json.dump(img_annots, f, indent=2)

        # Create varied dummy videos
        vid_colors = ["blue", "red", "green", "yellow", "magenta"]
        for v_idx, v_color in enumerate(vid_colors):
            vp = self.local_video_folder / f"sample_{v_idx}.mp4"
            if not vp.exists():
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={v_color}:s=224x224:d=1", "-c:v", "libx264", str(vp)],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                except Exception:
                    vp.touch()

        # Create video annotations
        vid_annots = []
        for i in range(num_samples):
            vid_annots.append({
                "id": f"vid_sample_{i}",
                "video": f"sample_{i % len(vid_colors)}.mp4",
                "conversations": [
                    {"from": "human", "value": "<video>\nDescribe the key actions happening in this video."},
                    {"from": "gpt", "value": f"This is a video alignment sample number {i} for Video-LLaVA."}
                ]
            })
        with open(self.local_video_json, "w") as f:
            json.dump(vid_annots, f, indent=2)

        log_success(f"Generated verification subset with {num_samples} samples at {self.local_json_folder}")


# ==============================================================================
# Outbound Checkpoint Muler: Local SSD -> Google Drive Sync Daemon
# ==============================================================================

class CheckpointMuleDaemon:
    """
    Asynchronous background watcher that monitors the local SSD checkpoint folder
    and atomically mules saved checkpoints & projector adapters to Google Drive.
    """

    def __init__(self, local_ckpt_dir: str, drive_ckpt_dir: str, sync_interval_sec: int = 15, keep_local_ckpts: int = 1):
        self.local_ckpt_dir = Path(local_ckpt_dir)
        self.drive_ckpt_dir = Path(drive_ckpt_dir)
        self.sync_interval = sync_interval_sec
        self.keep_local_ckpts = keep_local_ckpts
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.synced_steps = set()

        self.local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.drive_ckpt_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start the background checkpoint synchronization thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        log_info(f"Checkpoint Mule Daemon started. Monitoring '{self.local_ckpt_dir}' -> '{self.drive_ckpt_dir}' every {self.sync_interval}s")

    def stop(self):
        """Stop the daemon and perform a final synchronous flush to Google Drive."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        log_info("Performing final checkpoint flush to Google Drive...")
        self.sync_now()
        log_success("Final checkpoint sync complete.")

    def _run_loop(self):
        while self.running:
            try:
                self.sync_now()
            except Exception as e:
                log_warn(f"Error during checkpoint sync: {e}")
            time.sleep(self.sync_interval)

    def sync_now(self):
        """Scans local checkpoints, syncs new ones to Google Drive, and manages local SSD space."""
        # 1. Sync standalone projector weights / state files if saved directly in output_dir
        for key_file in ["mm_projector.bin", "config.json", "trainer_state.json", "training_args.bin", "non_lora_trainables.bin"]:
            local_f = self.local_ckpt_dir / key_file
            if local_f.exists():
                drive_f = self.drive_ckpt_dir / key_file
                if not drive_f.exists() or local_f.stat().st_mtime > drive_f.stat().st_mtime:
                    shutil.copy2(local_f, drive_f)
                    log_success(f"[Mule -> Drive] Synced {key_file} to Google Drive.")

        # 2. Sync checkpoint subdirectories (checkpoint-XXX)
        ckpt_dirs = sorted(
            [d for d in self.local_ckpt_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
        )

        for ckpt in ckpt_dirs:
            ckpt_name = ckpt.name
            drive_dest = self.drive_ckpt_dir / ckpt_name

            # Check if this checkpoint is written (has trainer_state.json or pytorch_model/optimizer)
            if not (ckpt / "trainer_state.json").exists() and not (ckpt / "mm_projector.bin").exists() and not (ckpt / "optimizer.pt").exists():
                continue

            if ckpt_name not in self.synced_steps:
                log_info(f"[Mule -> Drive] Syncing {ckpt_name} to Google Drive...")
                temp_drive_dest = self.drive_ckpt_dir / f".{ckpt_name}.tmp"
                if temp_drive_dest.exists():
                    shutil.rmtree(temp_drive_dest, ignore_errors=True)

                shutil.copytree(ckpt, temp_drive_dest, dirs_exist_ok=True)
                if drive_dest.exists():
                    shutil.rmtree(drive_dest, ignore_errors=True)
                temp_drive_dest.rename(drive_dest)

                self.synced_steps.add(ckpt_name)
                log_success(f"[Mule -> Drive] Successfully secured {ckpt_name} in Google Drive!")

        # 3. Clean up older local checkpoints if local SSD is constrained
        if self.keep_local_ckpts > 0 and len(ckpt_dirs) > self.keep_local_ckpts:
            to_remove = ckpt_dirs[:-self.keep_local_ckpts]
            for old_ckpt in to_remove:
                if old_ckpt.name in self.synced_steps and (self.drive_ckpt_dir / old_ckpt.name).exists():
                    log_info(f"Pruning older local checkpoint {old_ckpt.name} from local SSD (safely preserved in Drive).")
                    shutil.rmtree(old_ckpt, ignore_errors=True)

    def restore_latest_checkpoint_from_drive(self) -> Optional[str]:
        """
        Finds the latest checkpoint on Google Drive and mules it down to local SSD
        so training can seamlessly resume with --resume_from_checkpoint.
        """
        drive_ckpts = sorted(
            [d for d in self.drive_ckpt_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
        )

        if not drive_ckpts:
            log_info("No existing checkpoints found on Google Drive. Starting fresh training.")
            return None

        latest_drive_ckpt = drive_ckpts[-1]
        local_dest = self.local_ckpt_dir / latest_drive_ckpt.name

        log_header(f"Found Existing Checkpoint on Google Drive: {latest_drive_ckpt.name}")
        log_info(f"Muling {latest_drive_ckpt.name} from Google Drive to local SSD for auto-resume...")

        if not local_dest.exists():
            shutil.copytree(latest_drive_ckpt, local_dest, dirs_exist_ok=True)
            log_success(f"Restored {latest_drive_ckpt.name} to {local_dest}")
        else:
            log_info(f"Local copy of {latest_drive_ckpt.name} is already present.")

        return str(local_dest)


# ==============================================================================
# Hardware Auto-Tuning Engine for Google Colab
# ==============================================================================

def detect_colab_hardware_and_tune(args=None) -> Dict:
    """
    Detects available GPU (T4, L4, V100, A100), VRAM size, and computes optimal
    batch size, gradient accumulation, fp16/bf16, and DeepSpeed settings.
    """
    import torch

    device_count = torch.cuda.device_count()
    if device_count == 0:
        log_err("=" * 60)
        log_err("CRITICAL ERROR: No CUDA GPU detected in this Colab session!")
        log_err("Video-LLaVA cannot run on CPU. Please enable a GPU in Google Colab:")
        log_err("  1. In the top menu, click 'Runtime' -> 'Change runtime type'.")
        log_err("  2. Under 'Hardware accelerator', select 'T4 GPU'.")
        log_err("  3. Click 'Save' and re-run the notebook.")
        log_err("=" * 60)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)
    major_cc = torch.cuda.get_device_capability(0)[0]

    log_header(f"GPU Hardware Detected: {gpu_name} ({vram_gb:.1f} GB VRAM, Compute {major_cc}.x)")

    supports_bf16 = major_cc >= 8
    target_effective_batch_size = 32

    if vram_gb >= 38.0:  # A100 (40GB / 80GB)
        micro_batch_size = 8
        grad_accum = max(1, target_effective_batch_size // (micro_batch_size * max(1, device_count)))
        num_workers = 4
        bits = 16
    elif vram_gb >= 22.0:  # L4 (24GB) or V100 (32GB)
        micro_batch_size = 4
        grad_accum = max(1, target_effective_batch_size // (micro_batch_size * max(1, device_count)))
        num_workers = 4
        bits = 16
    elif vram_gb >= 18.0:  # 20GB+ cards
        micro_batch_size = 2
        grad_accum = max(1, target_effective_batch_size // (micro_batch_size * max(1, device_count)))
        num_workers = 2
        bits = 8
    else:  # Tesla T4 (14.6GB), V100 (16GB), or smaller GPUs - micro_batch_size=1 fits safely in 14.6GB VRAM
        micro_batch_size = 1
        grad_accum = max(1, target_effective_batch_size // (micro_batch_size * max(1, device_count)))
        num_workers = 2
        bits = 8

    # Allow CLI overrides if explicitly passed
    if args is not None and getattr(args, "per_device_train_batch_size", None) is not None:
        micro_batch_size = args.per_device_train_batch_size
        grad_accum = max(1, target_effective_batch_size // (micro_batch_size * max(1, device_count)))
    if args is not None and getattr(args, "gradient_accumulation_steps", None) is not None:
        grad_accum = args.gradient_accumulation_steps

    config = {
        "fp16": not supports_bf16,
        "bf16": supports_bf16,
        "bits": bits,
        "per_device_train_batch_size": micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "dataloader_num_workers": num_workers,
        "device_name": gpu_name,
        "vram_gb": vram_gb,
        "effective_batch_size": micro_batch_size * grad_accum * max(1, device_count)
    }

    log_info(f"Hardware Auto-Tuning Configuration:")
    log_info(f"  • Backbone Quantization: {config['bits']}-bit")
    log_info(f"  • Precision: {'bfloat16 (bf16)' if config['bf16'] else 'float16 (fp16)'}")
    log_info(f"  • Learning Rate: {getattr(args, 'learning_rate', '1e-3')}")
    log_info(f"  • Micro Batch Size per GPU: {config['per_device_train_batch_size']}")
    log_info(f"  • Gradient Accumulation Steps: {config['gradient_accumulation_steps']}")
    log_info(f"  • Total Effective Batch Size: {config['effective_batch_size']}")
    log_info(f"  • Dataloader Workers: {config['dataloader_num_workers']}")

    return config


# ==============================================================================
# Pretraining Launcher
# ==============================================================================

class VideoLLaVAPretrainingEngine:
    """
    Configures arguments and launches Video-LLaVA Stage 1 Pretraining.
    """

    def __init__(self, args, hw_config: Dict, resume_checkpoint: Optional[str] = None):
        self.args = args
        self.hw_config = hw_config
        self.resume_checkpoint = resume_checkpoint

    def build_command(self) -> List[str]:
        cmd = [
            sys.executable,
            "-m", "videollava.train.train_mem",
            "--model_name_or_path", self.args.model_name_or_path,
            "--version", "v1",
            "--data_path", str(self.args.image_json), str(self.args.video_json),
            "--image_folder", str(self.args.image_folder),
            "--video_folder", str(self.args.video_folder),
            "--image_tower", self.args.image_tower,
            "--video_tower", self.args.video_tower,
            "--mm_projector_type", "mlp2x_gelu",
            "--tune_mm_mlp_adapter", "True",
            "--mm_vision_select_layer", "-2",
            "--mm_use_im_start_end", "False",
            "--mm_use_im_patch_token", "False",
            "--output_dir", str(self.args.local_output_dir),
            "--num_train_epochs", str(self.args.num_train_epochs),
            "--per_device_train_batch_size", str(self.hw_config["per_device_train_batch_size"]),
            "--per_device_eval_batch_size", "4",
            "--gradient_accumulation_steps", str(self.hw_config["gradient_accumulation_steps"]),
            "--eval_strategy", "no",
            "--save_strategy", "steps",
            "--save_steps", str(self.args.save_steps),
            "--save_total_limit", str(self.args.save_total_limit),
            "--learning_rate", str(self.args.learning_rate),
            "--weight_decay", "0.",
            "--warmup_ratio", "0.2" if "pt_json" in str(self.args.image_json) else "0.03",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "1",
            "--model_max_length", "2048",
            "--tokenizer_model_max_length", "3072",
            "--gradient_checkpointing", "True",
            "--max_grad_norm", "1.0",
            "--dataloader_num_workers", str(self.hw_config["dataloader_num_workers"]),
            "--lazy_preprocess", "True",
            "--report_to", "tensorboard",
            "--cache_dir", str(self.args.cache_dir)
        ]

        if self.hw_config.get("bits", 16) in [4, 8]:
            cmd.extend(["--bits", str(self.hw_config["bits"])])
            cmd.extend(["--optim", "paged_adamw_8bit"])

        if self.hw_config["bf16"]:
            cmd.extend(["--bf16", "True", "--tf32", "True"])
        else:
            cmd.extend(["--fp16", "True"])

        # DeepSpeed integration
        # Note: DeepSpeed does not support 4-bit / 8-bit quantized models (.to() calls crash bitsandbytes).
        # When bits in [4, 8], native PyTorch Trainer handles training with minimal VRAM overhead
        # because only the small mm_projector adapter is being optimized.
        if self.hw_config.get("bits", 16) not in [4, 8]:
            if self.args.deepspeed_config and os.path.exists(self.args.deepspeed_config):
                cmd.extend(["--deepspeed", self.args.deepspeed_config])
        else:
            log_info("Quantized backbone active: Using native PyTorch Trainer with FP16 and gradient accumulation (DeepSpeed bypassed to prevent .to() quantization conflicts).")

        return cmd

    def run(self):
        cmd = self.build_command()
        log_header("Step 3: Launching Video-LLaVA Pretraining")
        log_info(f"Execution Command:\n{' '.join(cmd)}\n")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
        env["WANDB_DISABLED"] = "true"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        process = subprocess.Popen(cmd, env=env)

        def signal_handler(sig, frame):
            log_warn("Interrupt received! Terminating pretraining process cleanly...")
            process.terminate()
            process.wait()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        ret_code = process.wait()
        if ret_code != 0:
            log_err(f"Pretraining exited with error code {ret_code}")
            sys.exit(ret_code)
        else:
            log_success("Pretraining process completed successfully!")


# ==============================================================================
# Main Orchestration CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video-LLaVA Stage 1 Pretraining & Data Muling Pipeline for Google Colab & Google Drive"
    )

    # Action mode
    parser.add_argument(
        "--action", type=str, default="all",
        choices=["all", "download", "extract", "train", "mule_out", "demo_setup", "status"],
        help="Action to perform: 'all' (download + extract + train), 'download' (download archives to Drive), "
             "'extract' (extract on Drive), 'train' (only training), 'mule_out' (manual checkpoint sync), "
             "'demo_setup' (create fast validation subset), or 'status'."
    )

    # Google Drive & Local Storage Paths
    parser.add_argument("--drive_root", type=str, default="/content/drive/MyDrive/Video-LLaVA",
                        help="Root folder in Google Drive for dataset archives and checkpoints.")
    parser.add_argument("--local_scratch_dir", type=str, default="/content/data",
                        help="Local SSD path for annotation JSONs and demo data.")
    parser.add_argument("--local_output_dir", type=str, default="/content/checkpoints/videollava-7b-pretrain",
                        help="Local SSD directory where checkpoints are written during training.")
    parser.add_argument("--cache_dir", type=str, default="/content/cache_dir",
                        help="Local HuggingFace model cache directory.")

    # Model & Vision Towers
    parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-7b-v1.5",
                        help="Base LLM model path or HF identifier.")
    parser.add_argument("--image_tower", type=str, default="LanguageBind/LanguageBind_Image",
                        help="LanguageBind Image Tower identifier.")
    parser.add_argument("--video_tower", type=str, default="LanguageBind/LanguageBind_Video_merge",
                        help="LanguageBind Video Tower identifier.")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for MLP projector.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Max checkpoints to keep.")
    parser.add_argument("--deepspeed_config", type=str, default="./scripts/zero2.json",
                        help="Path to DeepSpeed configuration json.")
    parser.add_argument("--demo_samples", type=int, default=0,
                        help="If > 0, generates a lightweight demo subset with X samples for fast testing.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None,
                        help="Override per-device train micro-batch size (defaults to auto-tuned).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Override gradient accumulation steps (defaults to auto-tuned).")
    parser.add_argument("--auto_resume", action="store_true", default=True,
                        help="Automatically check Google Drive for existing checkpoints to resume from.")
    parser.add_argument("--parts", nargs="+", type=str, default=None,
                        help="Filter specific Valley archive parts to download/resume (e.g. --parts 4 5 or --parts 004 005). Defaults to all incomplete parts.")
    parser.add_argument("--cleanup_image_archive", action="store_true", default=False,
                        help="Delete llava_image.zip on Drive to reclaim ~27 GB if images are already extracted.")

    return parser.parse_args()


def main():
    args = parse_args()
    log_header("Video-LLaVA Colab Pretraining & Data Muling Pipeline")

    mount_google_drive()

    drive_data_dir = os.path.join(args.drive_root, "datasets")
    drive_ckpt_dir = os.path.join(args.drive_root, "checkpoints", os.path.basename(args.local_output_dir))

    inbound_muler = InboundDataMuler(
        drive_data_dir=drive_data_dir,
        local_scratch_dir=args.local_scratch_dir
    )

    outbound_muler = CheckpointMuleDaemon(
        local_ckpt_dir=args.local_output_dir,
        drive_ckpt_dir=drive_ckpt_dir,
        sync_interval_sec=20,
        keep_local_ckpts=1
    )

    if args.cleanup_image_archive:
        inbound_muler.cleanup_image_archive()

    if args.action == "status":
        log_header("System & Storage Status")
        detect_colab_hardware_and_tune(args)
        inbound_muler.print_dataset_summary()
        log_info(f"Google Drive Checkpoint Directory: {drive_ckpt_dir}")
        existing_ckpts = list(Path(drive_ckpt_dir).glob("checkpoint-*"))
        log_info(f"Found {len(existing_ckpts)} checkpoints on Google Drive.")
        return

    # Demo mode: uses local SSD for tiny demo data
    if args.action == "demo_setup" or args.demo_samples > 0:
        inbound_muler.create_demo_subset(num_samples=args.demo_samples if args.demo_samples > 0 else 100)
        # For demo, point to local SSD paths
        args.image_folder = inbound_muler.local_image_folder
        args.video_folder = inbound_muler.local_video_folder
        args.image_json = inbound_muler.local_image_json
        args.video_json = inbound_muler.local_video_json
        if args.action == "demo_setup":
            return
    else:
        # Full mode: data lives on Google Drive, annotations synced to SSD
        args.image_folder = inbound_muler.drive_image_folder
        args.video_folder = inbound_muler.drive_video_folder
        args.image_json = inbound_muler.local_image_json  # Small JSON on SSD for fast reads
        args.video_json = inbound_muler.local_video_json

    # Download & extract datasets on Google Drive
    if args.action in ["all", "download", "extract"]:
        if args.demo_samples == 0:
            if not inbound_muler.verify_drive_dataset():
                if args.action == "extract":
                    prep_ok = inbound_muler.extract_video_archives()
                elif args.action == "download" and args.parts:
                    log_info(f"Direct download path: processing specified Valley archive parts {args.parts}...")
                    prep_ok = inbound_muler.download_video_archives(parts_filter=args.parts)
                else:
                    prep_ok = inbound_muler.download_and_prepare_all(parts_filter=args.parts)

                if not prep_ok and args.action == "all":
                    log_err("Dataset preparation was not completed (video archives are still downloading or need extraction).")
                    log_info("Pretraining halted safely. Run with '--action download' to finish downloading video archives.")
                    return
            else:
                log_success("Full datasets already present on Google Drive. Skipping download.")
                inbound_muler.sync_annotations_to_local()
        if args.action in ["download", "extract"]:
            return

    if args.action in ["all", "train"]:
        hw_config = detect_colab_hardware_and_tune(args)

        resume_ckpt = None
        if args.auto_resume and args.demo_samples == 0:
            resume_ckpt = outbound_muler.restore_latest_checkpoint_from_drive()
        elif args.demo_samples > 0:
            log_info("Demo mode active: Google Drive auto-resume disabled to guarantee fresh dry-run verification.")
            for old_ckpt in Path(args.local_output_dir).glob("checkpoint-*"):
                if old_ckpt.is_dir():
                    shutil.rmtree(old_ckpt, ignore_errors=True)

        outbound_muler.start()

        try:
            engine = VideoLLaVAPretrainingEngine(args, hw_config, resume_checkpoint=resume_ckpt)
            engine.run()
        finally:
            outbound_muler.stop()

    if args.action == "mule_out":
        log_info("Performing manual checkpoint synchronization to Google Drive...")
        outbound_muler.sync_now()
        log_success("Manual sync complete.")


if __name__ == "__main__":
    main()
