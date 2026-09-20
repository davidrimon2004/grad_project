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

    def verify_drive_dataset(self) -> bool:
        """Check if REAL datasets are already extracted and ready on Google Drive."""
        image_json_ok = self.drive_image_json.is_file() and self.drive_image_json.stat().st_size > 0
        video_json_ok = self.drive_video_json.is_file() and self.drive_video_json.stat().st_size > 0

        num_images = sum(1 for _ in self.drive_image_folder.iterdir()) if self.drive_image_folder.is_dir() else 0
        num_videos = sum(1 for _ in self.drive_video_folder.iterdir()) if self.drive_video_folder.is_dir() else 0
        image_dir_ok = num_images >= self.MIN_REAL_IMAGE_FILES
        video_dir_ok = num_videos >= self.MIN_REAL_VIDEO_FILES

        return image_json_ok and video_json_ok and image_dir_ok and video_dir_ok

    def _wget_download(self, url: str, output_path: str, desc: str = ""):
        """Download a file using wget (no double-caching, direct write to destination)."""
        log_info(f"Downloading {desc or url}...")
        cmd = [
            "wget", "-c",  # -c enables resume of partial downloads
            "--progress=bar:force:noscroll",
            "-O", output_path,
            url
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            log_err(f"wget failed for {desc or url} (exit code {result.returncode})")
            return False
        log_success(f"Downloaded {desc}")
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
            # Extract to a temp dir first to find the JSONs
            extract_dir = self.drive_data_dir / "_annot_extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["unzip", "-q", "-o", str(annot_zip), "-d", str(extract_dir)], check=True)

            # Find and copy the JSON files we need
            found_any = False
            for json_name in ["llava_image_.json", "valley_.json", "chat.json"]:
                # Search recursively for the file
                matches = list(extract_dir.rglob(json_name))
                if matches:
                    shutil.copy2(matches[0], self.drive_json_folder / json_name)
                    log_success(f"Found and saved {json_name}")
                    found_any = True
                else:
                    log_warn(f"Could not find {json_name} in annotations zip")

            # If we didn't find specific names, look for any JSON files
            if not found_any:
                log_info("Searching for any annotation JSONs in the zip...")
                for jf in extract_dir.rglob("*.json"):
                    dest = self.drive_json_folder / jf.name
                    shutil.copy2(jf, dest)
                    log_info(f"  Extracted: {jf.name} ({jf.stat().st_size / 1024:.0f} KB)")

            # List what we have
            log_info("Contents of annotations folder:")
            for f in sorted(self.drive_json_folder.iterdir()):
                log_info(f"  {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")

            # Cleanup extraction temp dir
            shutil.rmtree(extract_dir, ignore_errors=True)
            return True
        else:
            log_err("Annotations zip download produced empty file")
            return False

    def download_image_archive(self):
        """Download llava_image.zip directly to Google Drive using wget."""
        image_archive = self.drive_data_dir / "llava_image.zip"

        if self.drive_image_folder.is_dir() and sum(1 for _ in self.drive_image_folder.iterdir()) >= self.MIN_REAL_IMAGE_FILES:
            log_success(f"Image dataset already extracted on Drive ({self.drive_image_folder})")
            return True

        if image_archive.exists() and image_archive.stat().st_size > 1_000_000:
            log_success(f"llava_image.zip already on Drive ({image_archive.stat().st_size / (1024**3):.2f} GB)")
        else:
            url = f"{self.HF_BASE_URL}/llava_image.zip"
            success = self._wget_download(url, str(image_archive), "llava_image.zip (~27 GB)")
            if not success:
                return False

        return True

    def download_video_archives(self):
        """Download valley_2.zip.* parts directly to Google Drive using wget."""
        # Check if already extracted
        if self.drive_video_folder.is_dir() and sum(1 for _ in self.drive_video_folder.iterdir()) >= self.MIN_REAL_VIDEO_FILES:
            log_success(f"Video dataset already extracted on Drive ({self.drive_video_folder})")
            return True

        for part_name in self.VALLEY_PARTS:
            part_file = self.drive_data_dir / part_name
            if part_file.exists() and part_file.stat().st_size > 1_000_000:
                log_info(f"{part_name} already on Drive ({part_file.stat().st_size / (1024**3):.2f} GB)")
                continue

            url = f"{self.HF_BASE_URL}/{part_name}"
            size_hint = "~42 GB" if part_name != "valley_2.zip.012" else "~2.3 GB"
            success = self._wget_download(url, str(part_file), f"{part_name} ({size_hint})")
            if not success:
                log_warn(f"Failed to download {part_name}, continuing with remaining parts...")

        return True

    def extract_image_archive(self):
        """Extract llava_image.zip directly on Google Drive."""
        if self.drive_image_folder.is_dir() and sum(1 for _ in self.drive_image_folder.iterdir()) >= self.MIN_REAL_IMAGE_FILES:
            log_success("Image dataset already extracted on Drive.")
            return True

        image_archive = self.drive_data_dir / "llava_image.zip"
        if not image_archive.exists():
            log_err("llava_image.zip not found on Drive. Download it first.")
            return False

        log_info(f"Extracting llava_image.zip on Google Drive ({image_archive.stat().st_size / (1024**3):.2f} GB)...")
        log_info("(This extracts directly on Drive — slower than SSD but data persists across sessions)")
        self.drive_image_folder.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["unzip", "-q", "-o", str(image_archive), "-d", str(self.drive_data_dir)],
            check=False
        )
        if result.returncode != 0:
            log_err(f"Image extraction failed (exit code {result.returncode})")
            return False

        self._fixup_nested_directory(self.drive_image_folder, "llava_image")
        num_files = sum(1 for _ in self.drive_image_folder.rglob("*") if _.is_file())
        log_success(f"Extracted {num_files} image files to {self.drive_image_folder}")
        return True

    def extract_video_archives(self):
        """Extract valley multi-part zip archives directly on Google Drive."""
        if self.drive_video_folder.is_dir() and sum(1 for _ in self.drive_video_folder.iterdir()) >= self.MIN_REAL_VIDEO_FILES:
            log_success("Video dataset already extracted on Drive.")
            return True

        first_part = self.drive_data_dir / "valley_2.zip.001"
        if not first_part.exists():
            log_err("valley_2.zip.001 not found on Drive. Download video archives first.")
            return False

        self.drive_video_folder.mkdir(parents=True, exist_ok=True)

        log_info("Extracting multi-part Valley video archives on Google Drive...")
        log_info("(This may take a while — extracting ~460 GB of videos directly on Drive)")

        # Use 7z for multi-part zip extraction
        has_7z = shutil.which("7z") or shutil.which("7za")
        if has_7z:
            seven_z = "7z" if shutil.which("7z") else "7za"
            cmd = [seven_z, "x", str(first_part), f"-o{self.drive_data_dir}", "-y"]
            result = subprocess.run(cmd, check=False)
        else:
            # Fallback: combine parts with cat, then unzip
            log_info("Combining multi-part archives with cat...")
            combined_zip = self.drive_data_dir / "valley_combined.zip"
            cat_cmd = f"cat {self.drive_data_dir}/valley_2.zip.* > {combined_zip}"
            result = subprocess.run(cat_cmd, shell=True, check=False)
            if result.returncode == 0:
                result = subprocess.run(
                    ["unzip", "-q", "-o", str(combined_zip), "-d", str(self.drive_data_dir)],
                    check=False
                )
                combined_zip.unlink(missing_ok=True)

        if result.returncode != 0:
            log_err(f"Video extraction failed (exit code {result.returncode})")
            return False

        self._fixup_nested_directory(self.drive_video_folder, "valley")
        num_files = sum(1 for _ in self.drive_video_folder.rglob("*") if _.is_file())
        log_success(f"Extracted {num_files} video files to {self.drive_video_folder}")
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

    def download_and_prepare_all(self):
        """Full pipeline: download archives, extract on Drive, sync annotations."""
        log_header("Step 1: Downloading & Preparing Datasets on Google Drive")

        if self.verify_drive_dataset():
            log_success("All datasets already present and extracted on Google Drive!")
            self.sync_annotations_to_local()
            self.print_dataset_summary()
            return

        # 1. Download annotations
        self.download_annotations()

        # 2. Download & extract images
        self.download_image_archive()
        self.extract_image_archive()

        # 3. Download & extract videos
        self.download_video_archives()
        self.extract_video_archives()

        # 4. Sync annotation JSONs to local SSD
        self.sync_annotations_to_local()

        log_header("Data Preparation Complete!")
        self.print_dataset_summary()

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
        # Check Drive paths (where data lives)
        num_images = sum(1 for _ in self.drive_image_folder.rglob("*") if _.is_file()) if self.drive_image_folder.exists() else 0
        num_videos = sum(1 for _ in self.drive_video_folder.rglob("*") if _.is_file()) if self.drive_video_folder.exists() else 0

        log_info(f"Drive Image folder: {self.drive_image_folder} ({num_images} files)")
        log_info(f"Drive Video folder: {self.drive_video_folder} ({num_videos} files)")
        log_info(f"Image annotations: {self.drive_image_json} ({'Found' if self.drive_image_json.exists() else 'Missing'})")
        log_info(f"Video annotations: {self.drive_video_json} ({'Found' if self.drive_video_json.exists() else 'Missing'}")

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

        # Create dummy image & video if none exist
        from PIL import Image
        dummy_img_path = self.local_image_folder / "sample_0.jpg"
        if not dummy_img_path.exists():
            img = Image.new("RGB", (224, 224), color=(73, 109, 137))
            img.save(dummy_img_path)

        # Create image annotations
        img_annots = []
        for i in range(num_samples):
            img_annots.append({
                "id": f"img_sample_{i}",
                "image": "sample_0.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nProvide a brief description of the given image."},
                    {"from": "gpt", "value": f"This is a pretraining alignment sample number {i} for Video-LLaVA."}
                ]
            })
        with open(self.local_image_json, "w") as f:
            json.dump(img_annots, f, indent=2)

        # Create dummy video if none exists
        dummy_vid_path = self.local_video_folder / "sample_0.mp4"
        if not dummy_vid_path.exists():
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=224x224:d=1", "-c:v", "libx264", str(dummy_vid_path)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception:
                dummy_vid_path.touch()

        # Create video annotations
        vid_annots = []
        for i in range(num_samples):
            vid_annots.append({
                "id": f"vid_sample_{i}",
                "video": "sample_0.mp4",
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
        log_warn("No CUDA GPU detected! Running on CPU (Pretraining will be extremely slow).")
        return {
            "fp16": False,
            "bf16": False,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "dataloader_num_workers": 2,
            "device_name": "CPU",
            "vram_gb": 0
        }

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
            "--dataloader_num_workers", str(self.hw_config["dataloader_num_workers"]),
            "--lazy_preprocess", "True",
            "--report_to", "tensorboard",
            "--cache_dir", str(self.args.cache_dir)
        ]

        if self.hw_config.get("bits", 16) in [4, 8]:
            cmd.extend(["--bits", str(self.hw_config["bits"])])

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
                inbound_muler.download_and_prepare_all()
            else:
                log_success("Full datasets already present on Google Drive. Skipping download.")
                inbound_muler.sync_annotations_to_local()
        if args.action in ["download", "extract"]:
            return

    if args.action in ["all", "train"]:
        hw_config = detect_colab_hardware_and_tune(args)

        resume_ckpt = None
        if args.auto_resume:
            resume_ckpt = outbound_muler.restore_latest_checkpoint_from_drive()

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
