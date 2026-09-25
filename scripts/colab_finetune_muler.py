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
import struct
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
                parts = line.split()
                if len(parts) >= 2 and (parts[1] == mount_point or parts[1].startswith(mount_point + "/")):
                    is_mounted = True
                    break
    except Exception:
        is_mounted = os.path.ismount(mount_point)

    has_drive_folder = os.path.exists(os.path.join(mount_point, "MyDrive")) or os.path.exists(os.path.join(mount_point, "My Drive"))

    if is_mounted and has_drive_folder:
        _ensure_mydrive_symlink(mount_point)
        log_success(f"Google Drive verified and active at {mount_point}")
        return True

    # In Colab non-interactive subshells, drive.mount fails ('NoneType' object has no attribute 'kernel').
    # We attempt drive.mount only once if not yet mounted, without flushing or unmounting.
    try:
        log_info(f"Connecting to Google Drive at {mount_point}...")
        from google.colab import drive
        drive.mount(mount_point)
        _ensure_mydrive_symlink(mount_point)
        log_success(f"Google Drive mounted at {mount_point}")
        return True
    except Exception as e:
        log_err(f"Google Drive is not mounted ({e}).")
        log_err("Please mount Google Drive directly in a Colab notebook cell:")
        log_err("  from google.colab import drive; drive.mount('/content/drive')")
        return False


def _ensure_mydrive_symlink(mount_point: str = "/content/drive"):
    """Ensure both /content/drive/MyDrive and /content/drive/My Drive point to the valid Google Drive root."""
    my_drive_spaced = os.path.join(mount_point, "My Drive")
    my_drive_nospace = os.path.join(mount_point, "MyDrive")
    try:
        if os.path.exists(my_drive_spaced) and not os.path.exists(my_drive_nospace):
            try:
                os.symlink(my_drive_spaced, my_drive_nospace)
                log_info(f"Created symlink: {my_drive_nospace} -> {my_drive_spaced}")
            except Exception:
                pass
        elif os.path.exists(my_drive_nospace) and not os.path.exists(my_drive_spaced):
            try:
                os.symlink(my_drive_nospace, my_drive_spaced)
                log_info(f"Created symlink: {my_drive_spaced} -> {my_drive_nospace}")
            except Exception:
                pass
    except Exception:
        pass


def _test_writable_directory(folder: Path) -> bool:
    """Test if a directory exists or can be created and written to."""
    try:
        os.makedirs(str(folder), exist_ok=True)
        probe = folder / ".drive_write_probe"
        with open(probe, "wb") as f:
            f.write(b"probe")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def find_and_verify_drive_datasets(drive_root_input: str) -> tuple[Path, Path]:
    """
    Locates and verifies the true, writable Google Drive datasets directory.
    Discovers whether the Colab environment uses 'My Drive' (space) or 'MyDrive' (no space),
    verifying existence of known dataset files and confirming write access with a probe.
    """
    candidates: List[Path] = []

    # 1. From user input drive_root
    p_in = Path(drive_root_input)
    candidates.append(p_in / "datasets")
    p_str = str(p_in)
    if "MyDrive" in p_str:
        candidates.append(Path(p_str.replace("MyDrive", "My Drive")) / "datasets")
    elif "My Drive" in p_str:
        candidates.append(Path(p_str.replace("My Drive", "MyDrive")) / "datasets")

    # 2. Well-known standard Colab Google Drive locations
    for root_prefix in ["/content/drive/My Drive", "/content/drive/MyDrive"]:
        c = Path(root_prefix) / "Video-LLaVA" / "datasets"
        if c not in candidates:
            candidates.append(c)

    # 3. Dynamic glob discovery under /content/drive
    if os.path.exists("/content/drive"):
        for found in glob.glob("/content/drive/*/Video-LLaVA/datasets"):
            p_found = Path(found)
            if p_found not in candidates:
                candidates.append(p_found)
        for found in glob.glob("/content/drive/*/*/datasets"):
            p_found = Path(found)
            if p_found not in candidates and "Video-LLaVA" in str(p_found):
                candidates.append(p_found)

    known_markers = [
        "videochatgpt_tune_2.zip.001",
        "videochatgpt_tune_2.zip.002",
        "videochatgpt_tune_2.zip.003",
        "videochatgpt_tune_2.zip.005",
        "llava_image_tune_2.zip.001",
        "annotations.zip",
    ]

    # Priority 1: Candidate containing known dataset archives that is also writable
    for cand in candidates:
        try:
            has_marker = any((cand / marker).exists() for marker in known_markers)
            if has_marker and _test_writable_directory(cand):
                log_success(f"✓ Found active and writable Google Drive datasets directory: {cand}")
                return cand.parent, cand
        except Exception:
            pass

    # Priority 2: Candidate containing known dataset archives
    for cand in candidates:
        try:
            if any((cand / marker).exists() for marker in known_markers):
                log_info(f"Using Drive datasets directory with existing archives: {cand}")
                return cand.parent, cand
        except Exception:
            pass

    # Priority 3: Any candidate that exists and is writable
    for cand in candidates:
        try:
            if cand.exists() and _test_writable_directory(cand):
                log_success(f"Using writable Google Drive datasets directory: {cand}")
                return cand.parent, cand
        except Exception:
            pass

    # Priority 4: Any candidate whose parent exists and can be made writable
    for cand in candidates:
        try:
            if cand.parent.exists() and _test_writable_directory(cand):
                log_success(f"Created writable Google Drive datasets directory: {cand}")
                return cand.parent, cand
        except Exception:
            pass

    fallback_dir = Path(drive_root_input) / "datasets"
    log_warn(f"Drive probe fallback to input directory: {fallback_dir}")
    return Path(drive_root_input), fallback_dir


def get_actual_drive_path(path: Path) -> Path:
    """Resolve Google Drive paths robustly across 'MyDrive' vs 'My Drive' and symlinks."""
    p = Path(path)
    candidates = [p]
    p_str = str(p)
    if "MyDrive" in p_str:
        candidates.append(Path(p_str.replace("MyDrive", "My Drive")))
    elif "My Drive" in p_str:
        candidates.append(Path(p_str.replace("My Drive", "MyDrive")))

    # 1. First priority: any candidate where the path itself exists
    for c in candidates:
        try:
            if c.exists():
                return c.resolve() if c.is_symlink() else c
        except Exception:
            pass

    # 2. Second priority: any candidate where the parent directory exists
    for c in candidates:
        try:
            if c.parent.exists():
                resolved_parent = c.parent.resolve() if c.parent.is_symlink() else c.parent
                return resolved_parent / c.name
        except Exception:
            pass

    return p


# ==============================================================================
# Streaming Multi-Part Zip Extractor (Zero Intermediate File Architecture)
# ==============================================================================

class MultiPartStream:
    """Emulates a single contiguous, seekable binary stream across multiple split archive parts."""
    def __init__(self, paths: List[str]):
        self.paths = [str(p) for p in paths]
        self.sizes = []
        for p in self.paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing archive part: {p}")
            self.sizes.append(os.path.getsize(p))
        self.starts, t = [], 0
        for s in self.sizes:
            self.starts.append(t)
            t += s
        self.total = t
        self._pos = 0
        self._fhs = {}

    def _fh(self, i: int):
        if i not in self._fhs:
            self._fhs[i] = open(self.paths[i], 'rb')
        return self._fhs[i]

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self.total + pos
        self._pos = max(0, min(self._pos, self.total))
        return self._pos

    def read(self, n: int = -1) -> bytes:
        remain = (self.total - self._pos) if n < 0 else min(n, self.total - self._pos)
        buf = bytearray()
        while remain > 0:
            idx = 0
            for i in range(len(self.starts)):
                if self.starts[i] <= self._pos:
                    idx = i
                else:
                    break
            loc = self._pos - self.starts[idx]
            to_r = min(remain, self.sizes[idx] - loc)
            fh = self._fh(idx)
            fh.seek(loc)
            chunk = fh.read(to_r)
            if not chunk:
                break
            buf.extend(chunk)
            self._pos += len(chunk)
            remain -= len(chunk)
        return bytes(buf)

    def close(self):
        for fh in list(self._fhs.values()):
            try:
                fh.close()
            except Exception:
                pass
        self._fhs.clear()


def find_zip_data_offset_and_size(stream: MultiPartStream) -> tuple[int, int, str]:
    """Finds byte offset, uncompressed payload size, and filename of the inner archive."""
    stream.seek(0)
    sig = stream.read(4)
    if sig != b"PK\x03\x04":
        raise ValueError(f"Invalid ZIP signature: {sig}")
    stream.seek(18)
    comp_size_32, uncomp_size_32 = struct.unpack('<II', stream.read(8))
    stream.seek(26)
    fn_len, extra_len = struct.unpack('<HH', stream.read(4))
    stream.seek(30)
    fn = stream.read(fn_len).decode('utf-8', errors='ignore')
    extra = stream.read(extra_len)
    offset = 30 + fn_len + extra_len

    uncomp_size = uncomp_size_32
    idx = 0
    while idx + 4 <= len(extra):
        tag, sz = struct.unpack('<HH', extra[idx:idx+4])
        idx += 4
        if tag == 1 and sz >= 16:
            uncomp_size, _ = struct.unpack('<QQ', extra[idx:idx+16])
            break
        idx += sz

    log_info(f"Inner archive payload: '{fn}' (offset: {offset}, size: {uncomp_size / (1024**3):.2f} GB)")
    return offset, uncomp_size, fn


class StreamSlice:
    """Presents a bounded sub-slice of a stream as an independent seekable stream."""
    def __init__(self, base, offset: int, size: int):
        self.base = base
        self.offset = offset
        self.size = size
        self._pos = 0

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self.size + pos
        self._pos = max(0, min(self._pos, self.size))
        return self._pos

    def read(self, n: int = -1) -> bytes:
        remain = (self.size - self._pos) if n < 0 else min(n, self.size - self._pos)
        if remain <= 0:
            return b""
        self.base.seek(self.offset + self._pos)
        chunk = self.base.read(remain)
        self._pos += len(chunk)
        return chunk

    def close(self):
        pass


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
        self.drive_root, self.drive_data_dir = find_and_verify_drive_datasets(drive_root)
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
                check_path = path
                if not check_path.exists() and check_path.parent.exists():
                    check_path = check_path.parent
                if not check_path.exists() and "Google Drive" in label:
                    check_path = Path("/content/drive")

                resolved = check_path.resolve() if check_path.exists() else check_path
                if str(resolved) in seen:
                    continue
                seen.add(str(resolved))
                usage = shutil.disk_usage(str(resolved if resolved.exists() else check_path))
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

    def _flush_drive_fuse(self):
        try:
            log_info("Flushing filesystem write buffers to cloud...")
            os.sync()
            subprocess.run(["sync"], check=False)
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

        drive_path = self.drive_data_dir / drive_path.name
        os.makedirs(str(drive_path.parent), exist_ok=True)

        if not mount_google_drive():
            raise RuntimeError(f"Google Drive is not mounted! Refusing to write {drive_path.name} to local SSD.")

        if drive_path.exists():
            drive_path.unlink(missing_ok=True)
            time.sleep(0.5)

        start_t = time.time()
        fdst = None
        for attempt in range(1, 4):
            try:
                drive_path = self.drive_data_dir / drive_path.name
                os.makedirs(str(drive_path.parent), exist_ok=True)
                fdst = open(drive_path, "wb")
                break
            except (FileNotFoundError, OSError) as e:
                log_warn(f"Attempt {attempt}/3 to open {drive_path} failed: {e}")
                if attempt < 3:
                    if not mount_google_drive():
                        raise RuntimeError(f"Google Drive is not mounted! Refusing to write {drive_path.name} to local SSD.")
                    drive_path = self.drive_data_dir / drive_path.name
                    os.makedirs(str(drive_path.parent), exist_ok=True)
                    time.sleep(2)
                else:
                    raise

        try:
            open_mode = "r+b" if os.access(local_path, os.W_OK) else "rb"
            with open(local_path, open_mode) as fsrc:
                offset = 0
                last_log = time.time()
                total_bytes = local_path.stat().st_size
                while True:
                    buf = fsrc.read(chunk_size)
                    if not buf:
                        break
                    fdst.write(buf)
                    try:
                        if open_mode == "r+b" and hasattr(os, "fallocate"):
                            os.fallocate(fsrc.fileno(), 0x03, offset, len(buf))
                    except Exception:
                        pass
                    offset += len(buf)
                    if time.time() - last_log >= 15:
                        pct = (offset / total_bytes) * 100 if total_bytes > 0 else 0
                        speed = (offset / (1024 ** 2)) / max(time.time() - start_t, 1)
                        log_info(f"Drive transfer: {offset / (1024**3):.2f} / {size_gb:.1f} GB ({pct:.1f}%) [{speed:.1f} MB/s]")
                        last_log = time.time()

            fdst.flush()
            try:
                os.fsync(fdst.fileno())
            except Exception:
                pass
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

    def _direct_stream_from_url_to_drive(self, url: str, drive_target: Path, min_bytes: int, max_retries: int = 15) -> bool:
        drive_target = self.drive_data_dir / drive_target.name
        os.makedirs(str(drive_target.parent), exist_ok=True)

        if not mount_google_drive():
            raise RuntimeError(f"Google Drive is not mounted! Refusing to stream {drive_target.name} to local SSD.")

        import urllib.request

        for attempt in range(1, max_retries + 1):
            drive_target = self.drive_data_dir / drive_target.name
            curr_bytes = drive_target.stat().st_size if drive_target.exists() else 0
            if curr_bytes >= min_bytes:
                log_success(f"✓ {drive_target.name} is already complete on Google Drive ({curr_bytes / (1024**3):.2f} GB)!")
                return True

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            if curr_bytes > 0:
                headers["Range"] = f"bytes={curr_bytes}-"
                log_info(f"Resuming {drive_target.name} on Drive from byte {curr_bytes:,} ({curr_bytes / (1024**3):.2f} GB) [Attempt {attempt}/{max_retries}]...")
            else:
                log_info(f"Direct streaming {drive_target.name} to Google Drive (Zero local SSD staging) [Attempt {attempt}/{max_retries}]...")

            mode = "ab" if curr_bytes > 0 else "wb"
            fdst = None
            try:
                fdst = open(drive_target, mode)
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    content_len = resp.headers.get("Content-Length")
                    total_bytes = (int(content_len) + curr_bytes) if content_len else min_bytes
                    start_t = time.time()
                    last_log = start_t
                    chunk_size = 16 * 1024 * 1024
                    downloaded_this_run = 0
                    while True:
                        buf = resp.read(chunk_size)
                        if not buf:
                            break
                        fdst.write(buf)
                        downloaded_this_run += len(buf)
                        total_downloaded = curr_bytes + downloaded_this_run
                        if time.time() - last_log >= 15:
                            pct = (total_downloaded / total_bytes * 100) if total_bytes > 0 else 0
                            speed = (downloaded_this_run / (1024 ** 2)) / max(time.time() - start_t, 1)
                            log_info(f"Direct stream: {total_downloaded / (1024**3):.2f} / {total_bytes / (1024**3):.1f} GB ({pct:.1f}%) [{speed:.1f} MB/s]")
                            last_log = time.time()

                    fdst.flush()
                    try:
                        os.fsync(fdst.fileno())
                    except Exception:
                        pass
            except Exception as e:
                log_warn(f"Direct stream interrupted for {drive_target.name}: {e}. Retrying in 5s (attempt {attempt}/{max_retries})...")
                time.sleep(5)
            finally:
                if fdst and not fdst.closed:
                    try:
                        fdst.close()
                    except Exception:
                        pass

            self._prune_caches()
            self._flush_drive_fuse()

            if drive_target.exists() and drive_target.stat().st_size >= min_bytes:
                log_success(f"✓ Direct stream complete for {drive_target.name} ({drive_target.stat().st_size / (1024**3):.2f} GB)!")
                return True

        curr_size = drive_target.stat().st_size if drive_target.exists() else 0
        if curr_size >= min_bytes:
            return True
        log_err(f"Direct stream finished after {max_retries} attempts but size mismatch for {drive_target.name}: {curr_size} < {min_bytes}")
        return False

    def _download_part(self, url: str, part_name: str, drive_target: Path, min_bytes: int) -> bool:
        drive_target = self.drive_data_dir / part_name

        # Check both primary target in verified datasets folder and any alternative path variant
        target_candidates = [drive_target]
        alt = get_actual_drive_path(drive_target)
        if alt not in target_candidates:
            target_candidates.append(alt)

        # 1. PRIORITY: Check if already complete on Google Drive
        for tc in target_candidates:
            if tc.exists() and tc.stat().st_size >= min_bytes:
                log_success(f"✓ {part_name} is already complete on Google Drive ({tc.stat().st_size / (1024**3):.2f} GB)!")
                staged = Path("/content/_staging") / part_name
                staged.unlink(missing_ok=True)
                return True

        # 2. Check if already downloaded in local SSD staging
        staging_dir = Path("/content/_staging")
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_file = staging_dir / part_name

        if staged_file.exists() and staged_file.stat().st_size >= min_bytes:
            log_success(f"✓ {part_name} already in local SSD staging ({staged_file.stat().st_size / (1024**3):.2f} GB). Transferring to Drive...")
            return self._stream_copy_to_drive(staged_file, drive_target)

        # 3. Download via aria2c to local SSD with 16 parallel connections
        self._prune_caches()
        free_ssd = shutil.disk_usage(staging_dir).free / (1024 ** 3)
        has_aria2 = self._ensure_aria2()
        if has_aria2 and free_ssd >= 35.0:
            log_info(f"🚀 Downloading {part_name} via aria2c to local SSD (16 parallel streams, {free_ssd:.1f} GB free)...")
            cmd = [
                "aria2c",
                "-x", "16",
                "-s", "16",
                "-j", "16",
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
                return self._stream_copy_to_drive(staged_file, drive_target)

        # 4. Fallback: direct streaming
        log_info(f"Using direct stream fallback for {part_name} (Zero local SSD usage)...")
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
        # Clean up any zero-byte leftover archives from previous failed attempts
        for intermediate in ["llava_image_tune.zip", "videochatgpt_tune.zip"]:
            bad = self.drive_data_dir / intermediate
            if bad.exists() and bad.stat().st_size < 100_000_000:
                log_info(f"Cleaning up 0-byte/corrupted intermediate archive: {intermediate}")
                bad.unlink(missing_ok=True)

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

    def _streaming_extract(
        self,
        parts: List[str],
        stage_dir: Path,
        target_dataset_dir: Path,
        dataset_name: str,
        batch_size: int = 2500,
        ssd_low_gb: float = 20.0,
        workers: int = 8
    ) -> bool:
        """
        Extracts multi-part zip archives directly to Google Drive via Virtual Streaming Architecture.
        Emulates a continuous stream across split parts and mounts the inner archive with zipfile.
        Extracts files to local fast NVMe SSD in small batches, then moves them to Google Drive in parallel.
        Bypasses 7-Zip entirely and eliminates 72GB-172GB intermediate .zip files (Zero SSD exhaustion).
        """
        for p in parts:
            if not os.path.exists(p):
                log_err(f"Cannot extract {dataset_name}: missing archive part '{Path(p).name}'. Run with '--action download' first.")
                return False

        stage_dir.mkdir(parents=True, exist_ok=True)
        target_dataset_dir.mkdir(parents=True, exist_ok=True)

        def open_inner():
            for attempt in range(8):
                try:
                    if not os.path.exists(parts[0]):
                        try:
                            from google.colab import drive
                            drive.mount('/content/drive', force_remount=True)
                            time.sleep(10)
                            _ensure_mydrive_symlink('/content/drive')
                        except Exception:
                            pass
                    outer_stream = MultiPartStream(parts)
                    off, sz, _ = find_zip_data_offset_and_size(outer_stream)
                    inner_zip = zipfile.ZipFile(StreamSlice(outer_stream, off, sz))
                    return outer_stream, inner_zip
                except Exception as e:
                    log_warn(f"Drive not ready ({e}), retrying in 20s... (attempt {attempt+1}/8)")
                    time.sleep(20)
            raise RuntimeError(f"Failed to open {dataset_name} zip archive after 8 attempts.")

        def flush_staged():
            files = [f for f in stage_dir.rglob("*") if f.is_file()]
            if not files:
                return

            def _move(f: Path):
                rel = f.relative_to(stage_dir)
                rel_str = str(rel).replace("\\", "/")
                if rel_str.startswith(f"{dataset_name}/"):
                    dest = self.drive_data_dir / rel
                else:
                    dest = target_dataset_dir / rel

                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                else:
                    f.unlink(missing_ok=True)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_move, files))

        def do_drive_flush():
            log_info("⏳ Flushing Drive FUSE write cache to cloud servers (do NOT interrupt)...")
            try:
                os.sync()
                subprocess.run(["sync"], check=False)
            except Exception:
                pass
            try:
                from google.colab import drive
                drive.flush_and_unmount()
                time.sleep(10)
                drive.mount('/content/drive', force_remount=True)
                time.sleep(15)
                _ensure_mydrive_symlink('/content/drive')
            except Exception as e:
                log_warn(f"  Drive remount note: {e}")
            free = shutil.disk_usage('/content').free / (1024 ** 3) if os.path.exists('/content') else 100.0
            log_success(f"  ✅ Flush done. SSD: {free:.1f} GB free")

        # Flush any files remaining from previous interrupted run
        flush_staged()

        outer, inner = open_inner()
        all_files = [f for f in inner.namelist() if not f.endswith('/')]
        log_info(f"Total files in archive '{dataset_name}': {len(all_files):,}")

        # Check existing files on Drive to resume seamlessly if interrupted
        todo = []
        if not target_dataset_dir.exists() or not any(target_dataset_dir.iterdir()):
            todo = all_files
        else:
            log_info("Scanning existing files on Drive to resume...")
            existing_rel = set()
            for root, _, fs in os.walk(target_dataset_dir):
                r_p = Path(root)
                for f in fs:
                    existing_rel.add(str((r_p / f).relative_to(self.drive_data_dir)).replace("\\", "/"))
            todo = [
                f for f in all_files
                if (f if f.startswith(f"{dataset_name}/") else f"{dataset_name}/{f}") not in existing_rel
            ]

        done_count = len(all_files) - len(todo)
        log_info(f"Already done on Drive: {done_count:,} | Remaining: {len(todo):,}")

        if not todo:
            log_success(f"✓ All files for {dataset_name} are already extracted on Google Drive!")
            inner.close()
            outer.close()
            return True

        t0 = time.time()
        for i, fname in enumerate(todo):
            inner.extract(fname, stage_dir)

            need_flush_batch = (i > 0 and i % batch_size == 0)
            free_gb = shutil.disk_usage('/content').free / (1024 ** 3) if os.path.exists('/content') else 100.0
            need_drive_flush = free_gb < ssd_low_gb

            if need_flush_batch or need_drive_flush:
                flush_staged()
                elapsed = max(time.time() - t0, 1.0)
                rate = (i + 1) / elapsed
                eta_h = (len(todo) - i - 1) / rate / 3600
                pct = (done_count + i + 1) / len(all_files) * 100
                current_free = shutil.disk_usage('/content').free / (1024 ** 3) if os.path.exists('/content') else 100.0
                log_info(f"[{done_count+i+1:,}/{len(all_files):,}] {pct:.1f}% | {rate:.1f} f/s | ETA {eta_h:.1f}h | SSD {current_free:.1f}GB free")

            if need_drive_flush:
                inner.close()
                outer.close()
                flush_staged()
                if os.path.exists('/content/drive'):
                    do_drive_flush()
                outer, inner = open_inner()
                t0 = time.time()

        flush_staged()
        if os.path.exists('/content/drive'):
            do_drive_flush()
        inner.close()
        outer.close()
        shutil.rmtree(stage_dir, ignore_errors=True)
        log_success(f"✓ Streaming extraction complete for {dataset_name}!")
        return True

    def extract_datasets_on_drive(self, clean_zips: bool = False) -> bool:
        log_header("Extracting Fine-Tuning Datasets on Google Drive")

        # Proactively clean up any incomplete intermediate archives left from previous failed 7-Zip attempts
        for intermediate in ["llava_image_tune.zip", "videochatgpt_tune.zip"]:
            bad_zip = self.drive_data_dir / intermediate
            if bad_zip.exists():
                log_info(f"Removing intermediate 7-Zip archive {intermediate} ({bad_zip.stat().st_size / (1024**3):.1f} GB) to reclaim Drive storage...")
                bad_zip.unlink(missing_ok=True)

        # 1. Extract Image Tuning Dataset
        self.drive_image_folder.mkdir(parents=True, exist_ok=True)
        img_parts = [str(self.drive_data_dir / p) for p in self.IMAGE_TUNE_PARTS]
        img_stage = Path("/content/_stage_img") if Path("/content").exists() else self.local_scratch_dir / "_stage_img"
        log_info("Extracting Image Tuning dataset via Virtual Streaming Architecture (Zero intermediate .zip)...")
        ok = self._streaming_extract(
            parts=img_parts,
            stage_dir=img_stage,
            target_dataset_dir=self.drive_image_folder,
            dataset_name="llava_image_tune",
            batch_size=2500,
            ssd_low_gb=20.0,
            workers=8
        )
        if not ok:
            log_err("Failed to extract image tuning dataset.")
            return False

        if clean_zips:
            for p in self.IMAGE_TUNE_PARTS:
                (self.drive_data_dir / p).unlink(missing_ok=True)
            log_info("Cleaned up Image Tuning split archives.")

        # 2. Extract Video Tuning Dataset
        self.drive_video_folder.mkdir(parents=True, exist_ok=True)
        vid_parts = [str(self.drive_data_dir / p) for p in self.VIDEO_TUNE_PARTS]
        missing_vid = [p for p in vid_parts if not os.path.exists(p)]
        if missing_vid:
            log_warn(f"Video tuning parts not yet complete ({len(missing_vid)} missing). Run with '--action download' to finish downloading.")
            return False

        vid_stage = Path("/content/_stage_vid") if Path("/content").exists() else self.local_scratch_dir / "_stage_vid"
        log_info("Extracting Video Tuning dataset via Virtual Streaming Architecture (Zero intermediate .zip)...")
        ok = self._streaming_extract(
            parts=vid_parts,
            stage_dir=vid_stage,
            target_dataset_dir=self.drive_video_folder,
            dataset_name="videochatgpt_tune",
            batch_size=500,
            ssd_low_gb=20.0,
            workers=8
        )
        if not ok:
            log_err("Failed to extract video tuning dataset.")
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
    if not mount_google_drive():
        log_err("Aborting: Google Drive must be mounted before running this pipeline.")
        sys.exit(1)

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
        drive_ckpt_dir = muler.drive_root / "checkpoints/videollava-7b-finetune"
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
