"""
kaggle_embed.py  —  Video-LLaVA Embedding Pre-computation on Kaggle
====================================================================
Run this as a Kaggle Notebook (Python script mode).

PREREQUISITES (one-time setup):
1. On your LOCAL machine, install rclone and generate a Drive token:
       rclone config
       # Create a remote named "gdrive", type "drive", follow OAuth flow
       # Then find your config:  rclone config file
       # Copy the entire [gdrive] block from that file

2. In Kaggle → Your Account → Settings → "Add a new secret":
       Name:  RCLONE_CONF
       Value: (paste the [gdrive] block here, exactly as shown below)

       Example value:
       [gdrive]
       type = drive
       client_id =
       client_secret =
       scope = drive
       token = {"access_token":"...","token_type":"Bearer","refresh_token":"...","expiry":"..."}
       team_drive =

3. In this notebook, enable the "RCLONE_CONF" secret in the Secrets panel.

4. Set DRIVE_BASE below to match your Google Drive folder structure.

5. GPU: Enable T4 x2 or P100 in Settings → Accelerator.
   This script uses only one GPU (cuda:0).  T4 gives ~4-6 h for images,
   P100 gives ~3-4 h.  Videos take significantly longer due to video decoding.

KAGGLE SESSION LIMIT: 12 hours.  The script is fully resumable — just
re-run from scratch and it picks up where it left off.
"""

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← Edit these paths to match your Drive layout
# ══════════════════════════════════════════════════════════════════════════════

DRIVE_REMOTE   = "gdrive"                          # rclone remote name (from config above)
DRIVE_BASE     = "Video-LLaVA"                     # path inside your Drive root
DRIVE_MOUNT    = "/mnt/gdrive"                     # local mount point in Kaggle
EMBED_OUT_DIR  = f"{DRIVE_MOUNT}/{DRIVE_BASE}/embeddings"   # where HDF5 files are written

# What to embed:  "all" | "pretrain" | "finetune"
SPLIT  = "all"
# Which modality: "all" | "visual" | "text" | "images" | "videos"
ACTION = "all"

# Project visual tokens into LLM hidden dimension (4096) via mm_projector:
PROJECT_VISUAL = False

# LLM model for text token embeddings:
MODEL_NAME_OR_PATH = "lmsys/vicuna-7b-v1.5"

# Batch sizes (reduce if OOM):
IMAGE_BATCH = 32   # P100/T4 16 GB: 32 is safe for image embedding
VIDEO_BATCH = 2    # Videos are large; 2 is safe on 16 GB
TEXT_BATCH  = 256  # Fast on GPU for conversation tokenization and embedding lookup

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, sys, os

def run(cmd, **kw):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)

print("── Installing Python packages ──────────────────────────────────")
run("pip install -q h5py transformers accelerate")

print("\n── Installing system packages ──────────────────────────────────")
run("apt-get install -y -q rclone fuse3 > /dev/null 2>&1 || true")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Mount Google Drive via rclone using the Kaggle secret
# ══════════════════════════════════════════════════════════════════════════════

from kaggle_secrets import UserSecretsClient   # only available inside Kaggle
import pathlib

def mount_drive():
    # Write rclone config from Kaggle secret
    conf_dir = pathlib.Path.home() / ".config" / "rclone"
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_file = conf_dir / "rclone.conf"

    secrets = UserSecretsClient()
    rclone_conf = secrets.get_secret("RCLONE_CONF")
    conf_file.write_text(rclone_conf)
    print(f"rclone config written to {conf_file}")

    # Create mount point
    os.makedirs(DRIVE_MOUNT, exist_ok=True)

    # Mount (non-blocking daemon process)
    mount_cmd = (
        f"rclone mount {DRIVE_REMOTE}: {DRIVE_MOUNT} "
        f"--daemon "
        f"--vfs-cache-mode writes "
        f"--vfs-cache-max-size 8G "
        f"--buffer-size 256M "
        f"--transfers 8 "
        f"--log-level INFO "
        f"--log-file /tmp/rclone.log"
    )
    print(f"Mounting Drive at {DRIVE_MOUNT} ...")
    subprocess.Popen(mount_cmd, shell=True)

    # Wait for mount to be ready
    import time
    for i in range(30):
        time.sleep(2)
        if pathlib.Path(DRIVE_MOUNT).is_mount() or \
           len(list(pathlib.Path(DRIVE_MOUNT).iterdir())) > 0:
            print(f"Drive mounted successfully at {DRIVE_MOUNT}")
            return
        print(f"  waiting for mount... ({(i+1)*2}s)")

    raise RuntimeError("Drive did not mount within 60s. Check /tmp/rclone.log")

mount_drive()

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Clone / update the repo
# ══════════════════════════════════════════════════════════════════════════════

REPO_DIR = "/kaggle/working/grad_project"

if not os.path.exists(REPO_DIR):
    run(f"git clone https://github.com/davidrimon2004/grad_project.git {REPO_DIR}")
else:
    run(f"git -C {REPO_DIR} pull origin main")

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Install repo requirements
# ══════════════════════════════════════════════════════════════════════════════

if os.path.exists("requirements.txt"):
    run("pip install -q -r requirements.txt")

# LanguageBind towers need these:
run("pip install -q languagebind decord av")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Sanity checks
# ══════════════════════════════════════════════════════════════════════════════

import torch
from pathlib import Path

# GPU
if torch.cuda.is_available():
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu}  ({vram:.1f} GB VRAM)")
else:
    print("WARNING: No GPU detected. Embedding will be extremely slow.")

# Drive mount
drive_base_path = Path(DRIVE_MOUNT) / DRIVE_BASE
print(f"\nDrive base: {drive_base_path}")
print(f"  exists: {drive_base_path.exists()}")

datasets_dir = drive_base_path / "datasets"
for sub in ["llava_image", "valley", "llava_image_tune", "videochatgpt_tune",
            "pt_json", "ft_json"]:
    p = datasets_dir / sub
    print(f"  {sub}: {'OK' if p.exists() else 'MISSING'}")

# Storage
import shutil
_, _, free = shutil.disk_usage(DRIVE_MOUNT)
print(f"\nDrive free space: {free / 1e12:.2f} TB")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Run the embedding script
# ══════════════════════════════════════════════════════════════════════════════
#
# This is the main step.  It will take several hours.
# Re-running this cell after a disconnection resumes automatically.

project_flag = "--project_visual" if PROJECT_VISUAL else ""
embed_cmd = (
    f"python scripts/precompute_embeddings.py "
    f"  --action  {ACTION} "
    f"  --split   {SPLIT} "
    f"  --drive_base {drive_base_path} "
    f"  --output_dir {EMBED_OUT_DIR} "
    f"  --model_name_or_path {MODEL_NAME_OR_PATH} "
    f"  --image_batch_size {IMAGE_BATCH} "
    f"  --video_batch_size {VIDEO_BATCH} "
    f"  --text_batch_size {TEXT_BATCH} "
    f"  {project_flag} "
    f"  --dtype fp16"
)

print("Running:", embed_cmd)
subprocess.run(embed_cmd, shell=True, check=True)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Verify output
# ══════════════════════════════════════════════════════════════════════════════

import json as _json

out = Path(EMBED_OUT_DIR)
print(f"\n{'='*60}")
print(f"Output directory: {out}")
print(f"{'='*60}")

for fname in sorted(out.iterdir()):
    size_gb = fname.stat().st_size / 1e9
    print(f"  {fname.name:<40}  {size_gb:6.2f} GB")

if (out / "meta.json").exists():
    meta = _json.loads((out / "meta.json").read_text())
    print(f"\nmeta.json: {_json.dumps(meta, indent=2)}")

print("\nDone. Embeddings are on Google Drive and ready for training.")
print("Use --embed_cache_dir when launching training:")
print(f"  --embed_cache_dir {EMBED_OUT_DIR}")
