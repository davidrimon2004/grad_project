#!/usr/bin/env python3
"""
Video-LLaVA Stage 1 Pretraining & Data Muling Pipeline for Google Colab & Google Drive.

Original Video-LLaVA Datasets:
  - Image Pretrain: LLaVA-558K (images in `llava_image/`, annotations `llava_image_.json`)
  - Video Pretrain: Valley Video 100K-702K (videos in `valley/`, annotations `valley_.json`)

Features:
  1. Automated Google Drive mounting and persistent storage management.
  2. Inbound Data Muler: Downloads/caches archives from HuggingFace to Drive, then mules
     and extracts them at high speed to Colab local ephemeral SSD (/content/data).
  3. Outbound Checkpoint Muler: Background daemon & Trainer synchronization that continuously
     syncs saved checkpoints and multimodal projector weights (mm_projector.bin) back to Google Drive.
  4. Auto-Resume: Automatically finds the latest checkpoint on Google Drive, mules it to local SSD,
     and resumes training seamlessly.
  5. Colab GPU Auto-Tuning: Auto-detects T4, L4, V100, A100 GPUs and configures batch size,
     gradient accumulation, fp16/bf16, and DeepSpeed/PyTorch settings.
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
# Inbound Data Muler: Google Drive <-> Local SSD
# ==============================================================================

class InboundDataMuler:
    """
    Manages downloading, storing in Google Drive, and muling (staging/extracting)
    the original Video-LLaVA pretraining datasets onto local high-speed SSD.
    """

    HF_DATASET_REPO = "LanguageBind/Video-LLaVA"
    
    # Official archives for Video-LLaVA Pretraining Stage 1
    # 1. LLaVA-558K Image archive: llava_image.zip
    # 2. Valley Video archive: valley_2.zip.001 to valley_2.zip.012
    # 3. Pretraining annotations: chat.json / llava_image_.json / valley_.json
    VALLEY_PARTS = [f"valley_2.zip.{i:03d}" for i in range(1, 13)]

    def __init__(self, drive_data_dir: str, local_scratch_dir: str):
        self.drive_data_dir = Path(drive_data_dir)
        self.local_scratch_dir = Path(local_scratch_dir)
        
        # Local paths for trainer
        self.local_image_folder = self.local_scratch_dir / "llava_image"
        self.local_video_folder = self.local_scratch_dir / "valley"
        self.local_json_folder = self.local_scratch_dir / "pt_json"
        self.local_image_json = self.local_json_folder / "llava_image_.json"
        self.local_video_json = self.local_json_folder / "valley_.json"

        # Create directories
        self.drive_data_dir.mkdir(parents=True, exist_ok=True)
        self.local_scratch_dir.mkdir(parents=True, exist_ok=True)
        self.local_json_folder.mkdir(parents=True, exist_ok=True)

    def verify_local_dataset(self) -> bool:
        """Check if datasets are already extracted and ready on local SSD."""
        image_json_ok = self.local_image_json.is_file() and self.local_image_json.stat().st_size > 0
        video_json_ok = self.local_video_json.is_file() and self.local_video_json.stat().st_size > 0
        image_dir_ok = self.local_image_folder.is_dir() and any(self.local_image_folder.iterdir())
        video_dir_ok = self.local_video_folder.is_dir() and any(self.local_video_folder.iterdir())

        return image_json_ok and video_json_ok and image_dir_ok and video_dir_ok

    def download_from_hf_to_drive(self, download_images: bool = True, download_videos: bool = True):
        """
        Download pretraining archives from HuggingFace directly to Google Drive
        so that datasets are saved permanently and never lost across Colab sessions.
        """
        log_header("Step 1: Downloading & Caching Datasets to Google Drive")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            log_info("Installing huggingface_hub for dataset download...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
            from huggingface_hub import hf_hub_download

        # 1. Download annotation JSON files
        log_info("Checking annotation files on Google Drive...")
        drive_json_dir = self.drive_data_dir / "pt_json"
        drive_json_dir.mkdir(parents=True, exist_ok=True)

        for json_name in ["llava_image_.json", "valley_.json", "chat.json"]:
            dest_file = drive_json_dir / json_name
            if not dest_file.exists():
                log_info(f"Downloading {json_name} from {self.HF_DATASET_REPO} to Google Drive...")
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=self.HF_DATASET_REPO,
                        filename=f"pt_json/{json_name}" if json_name != "chat.json" else json_name,
                        repo_type="dataset",
                        local_dir=str(self.drive_data_dir),
                        local_dir_use_symlinks=False
                    )
                    log_success(f"Saved {json_name} -> {downloaded_path}")
                except Exception as e:
                    log_warn(f"Could not download {json_name} directly: {e}")

        # 2. Download Image Archive (LLaVA 558K)
        if download_images:
            image_archive = self.drive_data_dir / "llava_image.zip"
            if not image_archive.exists():
                log_info(f"Downloading llava_image.zip to Google Drive ({self.drive_data_dir})...")
                try:
                    hf_hub_download(
                        repo_id=self.HF_DATASET_REPO,
                        filename="llava_image.zip",
                        repo_type="dataset",
                        local_dir=str(self.drive_data_dir),
                        local_dir_use_symlinks=False
                    )
                    log_success("llava_image.zip downloaded to Google Drive.")
                except Exception as e:
                    log_err(f"Error downloading llava_image.zip: {e}")
            else:
                log_success(f"llava_image.zip already exists on Google Drive ({image_archive.stat().st_size / (1024**3):.2f} GB)")

        # 3. Download Video Archive (Valley multi-part)
        if download_videos:
            for part in self.VALLEY_PARTS:
                part_file = self.drive_data_dir / part
                if not part_file.exists():
                    log_info(f"Downloading {part} to Google Drive...")
                    try:
                        hf_hub_download(
                            repo_id=self.HF_DATASET_REPO,
                            filename=part,
                            repo_type="dataset",
                            local_dir=str(self.drive_data_dir),
                            local_dir_use_symlinks=False
                        )
                        log_success(f"Downloaded {part}")
                    except Exception as e:
                        log_warn(f"Error downloading {part}: {e}")
                else:
                    log_info(f"Archive part {part} already exists on Drive.")

    def mule_archives_to_ssd(self, cleanup_zip_after_extract: bool = True):
        """
        Fast copy of dataset archives from Google Drive to local Colab NVMe/SSD,
        followed by high-speed local unzipping to avoid Drive FUSE I/O latency.
        """
        log_header("Step 2: Muling Data from Google Drive to Local Ephemeral SSD")

        if self.verify_local_dataset():
            log_success("All datasets are already muled and extracted on local SSD! Skipping extraction.")
            return

        local_archives_dir = self.local_scratch_dir / "archives"
        local_archives_dir.mkdir(parents=True, exist_ok=True)

        # 1. Mule Annotations
        drive_json_dir = self.drive_data_dir / "pt_json"
        if drive_json_dir.exists():
            log_info("Muling annotation JSONs to local SSD...")
            for f in drive_json_dir.glob("*.json"):
                shutil.copy2(f, self.local_json_folder / f.name)
        
        # If specific names exist at drive_data_dir root, copy them
        for jname in ["llava_image_.json", "valley_.json"]:
            root_j = self.drive_data_dir / jname
            if root_j.exists():
                shutil.copy2(root_j, self.local_json_folder / jname)

        # 2. Mule & Extract Image Dataset
        drive_img_zip = self.drive_data_dir / "llava_image.zip"
        if drive_img_zip.exists() and not (self.local_image_folder.exists() and any(self.local_image_folder.iterdir())):
            local_img_zip = local_archives_dir / "llava_image.zip"
            log_info(f"Muling {drive_img_zip.name} ({drive_img_zip.stat().st_size / (1024**3):.2f} GB) to local SSD...")
            shutil.copy2(drive_img_zip, local_img_zip)
            
            log_info("Extracting llava_image.zip on local SSD...")
            self.local_image_folder.mkdir(parents=True, exist_ok=True)
            subprocess.run(["unzip", "-q", "-o", str(local_img_zip), "-d", str(self.local_scratch_dir)], check=True)
            log_success("llava_image extracted successfully.")

            if cleanup_zip_after_extract and local_img_zip.exists():
                local_img_zip.unlink()
                log_info("Cleaned up local llava_image.zip archive to save SSD space.")
        elif (self.drive_data_dir / "llava_image").is_dir() and not (self.local_image_folder.exists() and any(self.local_image_folder.iterdir())):
            log_info("Copying uncompressed llava_image folder from Drive to local SSD...")
            shutil.copytree(self.drive_data_dir / "llava_image", self.local_image_folder, dirs_exist_ok=True)

        # 3. Mule & Extract Video Dataset (Valley)
        valley_parts_present = [self.drive_data_dir / p for p in self.VALLEY_PARTS if (self.drive_data_dir / p).exists()]
        single_valley_zip = self.drive_data_dir / "valley.zip"

        if not (self.local_video_folder.exists() and any(self.local_video_folder.iterdir())):
            self.local_video_folder.mkdir(parents=True, exist_ok=True)

            if len(valley_parts_present) > 0:
                log_info(f"Found {len(valley_parts_present)} Valley multi-part archives on Drive. Muling to local SSD...")
                for p in valley_parts_present:
                    shutil.copy2(p, local_archives_dir / p.name)

                # Combine or extract using 7z / cat
                log_info("Extracting multi-part valley archives on local SSD...")
                first_part = local_archives_dir / "valley_2.zip.001"
                
                # Check if 7z or p7zip is available
                has_7z = shutil.which("7z") or shutil.which("7za")
                if has_7z:
                    cmd = ["7z" if shutil.which("7z") else "7za", "x", str(first_part), f"-o{self.local_scratch_dir}", "-y"]
                    subprocess.run(cmd, check=True)
                else:
                    log_info("Combining multi-part archives with cat...")
                    combined_zip = local_archives_dir / "valley_combined.zip"
                    cat_cmd = f"cat {local_archives_dir}/valley_2.zip.* > {combined_zip}"
                    subprocess.run(cat_cmd, shell=True, check=True)
                    subprocess.run(["unzip", "-q", "-o", str(combined_zip), "-d", str(self.local_scratch_dir)], check=True)
                    if combined_zip.exists():
                        combined_zip.unlink()

                log_success("Valley videos extracted successfully.")

                if cleanup_zip_after_extract:
                    for p in local_archives_dir.glob("valley_2.zip.*"):
                        p.unlink()
            elif single_valley_zip.exists():
                local_vzip = local_archives_dir / "valley.zip"
                shutil.copy2(single_valley_zip, local_vzip)
                subprocess.run(["unzip", "-q", "-o", str(local_vzip), "-d", str(self.local_scratch_dir)], check=True)
                if cleanup_zip_after_extract:
                    local_vzip.unlink()
            elif (self.drive_data_dir / "valley").is_dir():
                log_info("Copying valley video folder from Drive to local SSD...")
                shutil.copytree(self.drive_data_dir / "valley", self.local_video_folder, dirs_exist_ok=True)

        # 4. Final Directory & Path Fixups
        self._fixup_nested_directories()

        log_success(f"Data Muling Complete! Datasets ready at {self.local_scratch_dir}")
        self.print_dataset_summary()

    def _fixup_nested_directories(self):
        """Fixes nested directory extractions if archives contained root folders."""
        for target_dir, folder_name in [(self.local_image_folder, "llava_image"), (self.local_video_folder, "valley")]:
            nested = target_dir / folder_name
            if nested.is_dir():
                log_info(f"Fixing nested directory in {target_dir}...")
                for item in nested.iterdir():
                    shutil.move(str(item), str(target_dir))
                shutil.rmtree(nested, ignore_errors=True)

    def print_dataset_summary(self):
        """Prints counts of images, videos, and annotation samples available."""
        num_images = len(list(self.local_image_folder.glob("*.*"))) if self.local_image_folder.exists() else 0
        num_videos = len(list(self.local_video_folder.glob("*.*"))) if self.local_video_folder.exists() else 0
        
        log_info(f"Local Image folder: {self.local_image_folder} ({num_images} files)")
        log_info(f"Local Video folder: {self.local_video_folder} ({num_videos} files)")
        log_info(f"Image annotations: {self.local_image_json} ({'Found' if self.local_image_json.exists() else 'Missing'})")
        log_info(f"Video annotations: {self.local_video_json} ({'Found' if self.local_video_json.exists() else 'Missing'})")

    def create_demo_subset(self, num_samples: int = 100):
        """
        Creates a lightweight subset of annotations and sample media
        for rapid end-to-end dry runs or quick validation in Colab.
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

def detect_colab_hardware_and_tune() -> Dict:
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
        micro_batch_size = 16
        grad_accum = max(1, target_effective_batch_size // (micro_batch_size * max(1, device_count)))
        num_workers = 8
    elif vram_gb >= 22.0:  # L4 (24GB) or V100 (32GB)
        micro_batch_size = 8
        grad_accum = max(1, target_effective_batch_size // (micro_batch_size * max(1, device_count)))
        num_workers = 4
    elif vram_gb >= 14.0:  # T4 (16GB) or V100 (16GB)
        micro_batch_size = 4
        grad_accum = max(1, target_effective_batch_size // (micro_batch_size * max(1, device_count)))
        num_workers = 4
    else:  # Small GPUs (<14GB)
        micro_batch_size = 2
        grad_accum = max(1, target_effective_batch_size // (micro_batch_size * max(1, device_count)))
        num_workers = 2

    config = {
        "fp16": not supports_bf16,
        "bf16": supports_bf16,
        "per_device_train_batch_size": micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "dataloader_num_workers": num_workers,
        "device_name": gpu_name,
        "vram_gb": vram_gb,
        "effective_batch_size": micro_batch_size * grad_accum * max(1, device_count)
    }

    log_info(f"Hardware Auto-Tuning Configuration:")
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
            "--evaluation_strategy", "no",
            "--save_strategy", "steps",
            "--save_steps", str(self.args.save_steps),
            "--save_total_limit", str(self.args.save_total_limit),
            "--learning_rate", str(self.args.learning_rate),
            "--weight_decay", "0.",
            "--warmup_ratio", "0.03",
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

        if self.hw_config["bf16"]:
            cmd.extend(["--bf16", "True", "--tf32", "True"])
        else:
            cmd.extend(["--fp16", "True"])

        # DeepSpeed integration
        if self.args.deepspeed_config and os.path.exists(self.args.deepspeed_config):
            cmd.extend(["--deepspeed", self.args.deepspeed_config])

        return cmd

    def run(self):
        cmd = self.build_command()
        log_header("Step 3: Launching Video-LLaVA Pretraining")
        log_info(f"Execution Command:\n{' '.join(cmd)}\n")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
        env["WANDB_DISABLED"] = "true"
        env["TOKENIZERS_PARALLELISM"] = "false"

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
        choices=["all", "mule_in", "train", "mule_out", "demo_setup", "status"],
        help="Action to perform: 'all' (mule + train + sync), 'mule_in' (download & extract), 'train' (only training), 'mule_out' (manual sync), 'demo_setup' (create fast validation subset), or 'status'."
    )

    # Google Drive & Local Storage Paths
    parser.add_argument("--drive_root", type=str, default="/content/drive/MyDrive/Video-LLaVA",
                        help="Root folder in Google Drive for dataset archives and checkpoints.")
    parser.add_argument("--local_scratch_dir", type=str, default="/content/data",
                        help="Fast local ephemeral SSD path for dataset extraction.")
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

    args.image_folder = inbound_muler.local_image_folder
    args.video_folder = inbound_muler.local_video_folder
    args.image_json = inbound_muler.local_image_json
    args.video_json = inbound_muler.local_video_json

    if args.action == "status":
        log_header("System & Storage Status")
        detect_colab_hardware_and_tune()
        inbound_muler.print_dataset_summary()
        log_info(f"Google Drive Checkpoint Directory: {drive_ckpt_dir}")
        existing_ckpts = list(Path(drive_ckpt_dir).glob("checkpoint-*"))
        log_info(f"Found {len(existing_ckpts)} checkpoints on Google Drive.")
        return

    if args.action == "demo_setup" or args.demo_samples > 0:
        inbound_muler.create_demo_subset(num_samples=args.demo_samples if args.demo_samples > 0 else 100)
        if args.action == "demo_setup":
            return

    if args.action in ["all", "mule_in"]:
        if not inbound_muler.verify_local_dataset() and args.demo_samples == 0:
            inbound_muler.download_from_hf_to_drive(download_images=True, download_videos=True)
            inbound_muler.mule_archives_to_ssd()
        if args.action == "mule_in":
            return

    if args.action in ["all", "train"]:
        hw_config = detect_colab_hardware_and_tune()

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
