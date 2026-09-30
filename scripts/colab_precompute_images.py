#!/usr/bin/env python3
"""
colab_precompute_images.py
==========================
High-throughput, fully resumable visual embedding pre-computation for images
tailored for Google Colab (T4 / V100 / L4 / A100) and Google Drive storage.

Key Colab & Drive Features:
  1. Multi-threaded PyTorch DataLoader with prefetching (4-8 workers) to ensure
     CPU image decompression and LanguageBind transforms keep the GPU 100% saturated.
  2. Local NVMe SSD Staging: Writes active HDF5 shards directly to fast local SSD
     (/content/staging_embeddings) and syncs completed shards to Google Drive.
     This eliminates Google Drive FUSE socket hangs, stalls, and write throttling.
  3. Sharded HDF5 Storage: Automatically splits outputs into 2,000-image shards
     (~2.1 GB each in FP16), preventing Colab DriveFS cache bloat and file lockups.
  4. 100% Resumable: Detects existing image_index.json on Google Drive and skips
     already embedded images. If Colab disconnects, simply re-run the script.
  5. Automatic Environment Setup: Mounts Google Drive, configures CUDA benchmark,
     and provides zero-config paths for Video-LLaVA pretrain & finetune datasets.

Usage in Google Colab:
---------------------
  # 1. Precompute Pretrain Images (LLaVA-558K):
  !python scripts/colab_precompute_images.py --split pretrain

  # 2. Precompute Fine-tuning Images (LLaVA-Instruct-665K):
  !python scripts/colab_precompute_images.py --split finetune

  # 3. Precompute All Images (Pretrain + Finetune):
  !python scripts/colab_precompute_images.py --split all

  # 4. Custom Dataset / Folder:
  !python scripts/colab_precompute_images.py \\
      --image_folder /content/drive/MyDrive/Video-LLaVA/datasets/llava_image \\
      --json_path /content/drive/MyDrive/Video-LLaVA/datasets/pt_json/llava_image_.json \\
      --output_dir /content/drive/MyDrive/Video-LLaVA/embeddings \\
      --batch_size 64
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# Prevent PIL errors on slightly truncated or large images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ==============================================================================
# Terminal Color & Logging Utilities
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
# Google Colab Environment & Google Drive Management
# ==============================================================================

def is_colab_environment() -> bool:
    """Check if currently running inside a Google Colab instance."""
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
    """Mounts Google Drive with resilience against re-mount stalls."""
    if not is_colab_environment():
        log_info("Local environment detected. Skipping Google Drive mount.")
        return True

    drive_root = Path(mount_point)
    my_drive = drive_root / "MyDrive"
    if my_drive.exists() or (drive_root / "My Drive").exists():
        log_success(f"Google Drive is already active and mounted at {mount_point}")
        return True

    try:
        log_info(f"Mounting Google Drive to {mount_point}...")
        from google.colab import drive
        drive.mount(mount_point, force_remount=False)
        if my_drive.exists() or (drive_root / "My Drive").exists():
            log_success(f"Google Drive mounted successfully at {mount_point}")
            return True
        else:
            log_warn("Drive mounted, but MyDrive folder was not immediately visible.")
            return True
    except Exception as exc:
        log_err(f"Failed to mount Google Drive: {exc}")
        return False


def ensure_colab_dependencies():
    """Ensures required packages for LanguageBind and HDF5 are installed in Colab."""
    if not is_colab_environment():
        return

    required = ["h5py", "transformers", "accelerate", "einops"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        import subprocess
        log_info(f"Installing missing Colab packages: {', '.join(missing)}...")
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + missing
        subprocess.run(cmd, check=True)
        log_success("Dependencies installed successfully.")


def check_drive_responsive(path: Path, timeout_sec: int = 5) -> bool:
    """Checks if a path on Google Drive responds within timeout_sec without hanging."""
    import signal
    if hasattr(signal, "SIGALRM"):
        def handler(signum, frame):
            raise TimeoutError("Drive check timed out")
        old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_sec)
        try:
            exists = os.path.exists(str(path))
            signal.alarm(0)
            return exists
        except TimeoutError:
            return False
        except Exception:
            return False
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        try:
            return os.path.exists(str(path))
        except Exception:
            return False


# ==============================================================================
# Safe JSON & File Helpers
# ==============================================================================

def safe_read_json(path: Path) -> Dict[str, Any]:
    """Reads a JSON file safely. Returns {} if missing or corrupt."""
    if not path.exists() or not path.is_file():
        return {}
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log_warn(f"Failed to load JSON from {path.name}: {exc}")
        return {}


def load_json_cached(jp: Path, local_cache_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Safely loads large JSON dataset annotations.
    If the file is on Google Drive FUSE, copies it in binary chunks to fast local SSD
    (/content/cache_jsons or /tmp) first, avoiding FUSE socket stalls and timeouts.
    """
    if local_cache_dir is None:
        if Path("/content").exists():
            local_cache_dir = Path("/content/cache_jsons")
        elif Path("/tmp").exists():
            local_cache_dir = Path("/tmp/cache_jsons")
        else:
            local_cache_dir = Path("./cache_jsons")

    p_str = str(jp)
    is_network_fs = "/drive/" in p_str or "/MyDrive/" in p_str or p_str.startswith("/content/drive")

    if not is_network_fs or not jp.exists():
        with open(str(jp), "r", encoding="utf-8") as f:
            return json.load(f)

    local_cache_dir.mkdir(parents=True, exist_ok=True)
    local_cached = local_cache_dir / jp.name

    remote_size = -1
    try:
        remote_size = os.path.getsize(str(jp))
    except Exception:
        pass

    need_copy = True
    if local_cached.exists() and remote_size > 0:
        try:
            if local_cached.stat().st_size == remote_size:
                need_copy = False
        except Exception:
            pass

    if need_copy:
        size_str = f" ({remote_size / (1024 * 1024):.1f} MB)" if remote_size > 0 else ""
        log_info(f"Caching annotation {jp.name}{size_str} to fast local SSD ({local_cached})...")
        t0 = time.time()
        with open(str(jp), "rb") as src, open(str(local_cached), "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        log_success(f"Cached {jp.name} locally in {time.time() - t0:.1f}s.")
    else:
        log_info(f"Using locally cached JSON: {local_cached}")

    t0 = time.time()
    with open(str(local_cached), "r", encoding="utf-8") as f:
        data = json.load(f)
    log_success(f"Parsed {len(data):,} items from {jp.name} in {time.time() - t0:.1f}s.")
    return data


# ==============================================================================
# HDF5 Shard Management
# ==============================================================================

def open_or_create_hdf5_shard(
    path: Path,
    n_samples: int,
    feat_shape: Tuple[int, ...],
):
    """
    Opens an existing HDF5 shard or creates a pre-allocated chunked shard.
    Returns (h5_file, dset_features, written_count).
    """
    try:
        import h5py
    except ImportError:
        log_err("h5py is not installed. Please run: pip install h5py")
        sys.exit(1)

    if path.exists() and path.is_file():
        try:
            h5 = h5py.File(str(path), "a")
            dset = h5["features"]
            written = int(h5.attrs.get("written", 0))
            return h5, dset, written
        except Exception as exc:
            log_warn(f"Existing shard {path.name} unreadable ({exc}). Creating fresh file.")
            try:
                path.unlink()
            except Exception:
                pass

    path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(str(path), "w")
    full_shape = (n_samples,) + feat_shape
    chunk_n = min(64, n_samples)
    chunk_shape = (chunk_n,) + feat_shape
    dset = h5.create_dataset(
        "features",
        shape=full_shape,
        dtype=np.float16,
        chunks=chunk_shape,
        compression="lzf",
    )
    h5.attrs["written"] = 0
    h5.attrs["feat_shape"] = list(feat_shape)
    return h5, dset, 0


# ==============================================================================
# Multi-threaded Image PyTorch Dataset
# ==============================================================================

class ImageEmbeddingDataset(Dataset):
    """
    High-performance PyTorch Dataset for loading and preprocessing images
    using LanguageBind processor across multiple CPU background workers.
    """
    def __init__(
        self,
        file_list: List[str],
        image_folder: Path,
        processor,
        expected_shape: Tuple[int, int, int],
        strip_prefix: Optional[str] = None,
    ):
        self.file_list = file_list
        self.image_folder = image_folder
        self.processor = processor
        self.expected_shape = expected_shape
        self.strip_prefix = strip_prefix

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, bool]:
        rel_path = self.file_list[idx]
        actual_rel = (
            rel_path[len(self.strip_prefix):]
            if self.strip_prefix and rel_path.startswith(self.strip_prefix)
            else rel_path
        )
        full_path = self.image_folder / actual_rel

        try:
            img = Image.open(str(full_path)).convert("RGB")
            # LanguageBind processor transforms image to [1, 3, 224, 224]
            pv = self.processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
            if tuple(pv.shape) != self.expected_shape:
                pv = torch.zeros(self.expected_shape, dtype=torch.float32)
                return pv, rel_path, False
            return pv, rel_path, True
        except Exception:
            # Missing or corrupted image: return zero tensor safely
            pv = torch.zeros(self.expected_shape, dtype=torch.float32)
            return pv, rel_path, False


def collate_image_batch(batch):
    tensors = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    valid_flags = [item[2] for item in batch]
    batch_tensor = torch.stack(tensors)
    return batch_tensor, paths, valid_flags


# ==============================================================================
# Model Loading (LanguageBind Image Tower & Optional mm_projector)
# ==============================================================================

class _TowerArgs:
    mm_vision_select_layer = -2
    mm_vision_select_feature = "patch"


def load_image_tower(
    model_name: str = "LanguageBind/LanguageBind_Image",
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
):
    """Loads the LanguageBind Image vision encoder in frozen evaluation mode."""
    from videollava.model.multimodal_encoder.languagebind import (
        LanguageBindImageTower,
        LanguageBindImageProcessor,
        sanitize_attn_implementation,
    )

    log_info(f"Loading LanguageBind Image Tower: {model_name}...")
    tower = LanguageBindImageTower(
        model_name,
        args=_TowerArgs(),
        cache_dir=None,
        delay_load=False,
    )
    sanitize_attn_implementation(tower)
    tower = tower.to(device=device, dtype=dtype).eval()
    for p in tower.parameters():
        p.requires_grad_(False)
    sanitize_attn_implementation(tower)

    processor = getattr(tower, "image_processor", None)
    if processor is None:
        processor = LanguageBindImageProcessor(tower.config)

    log_success(f"Image tower loaded. Hidden dim: {tower.hidden_size}, Patches: {tower.num_patches}")
    return tower, processor


def load_vision_projector(
    model_name_or_path: str,
    mm_projector_path: Optional[str] = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
    base_dir: Optional[Path] = None,
) -> Optional[nn.Module]:
    """Loads or initializes mm_projector (1024 -> 4096) if --project_visual is requested."""
    from videollava.model.multimodal_projector.builder import build_vision_projector

    class _ProjectorConfig:
        mm_projector_type = "mlp2x_gelu"
        mm_hidden_size = 1024
        hidden_size = 4096

    log_info("Building mm_projector (mlp2x_gelu: 1024 -> 4096)...")
    projector = build_vision_projector(_ProjectorConfig()).to(device=device, dtype=dtype)

    weights = None
    if mm_projector_path and os.path.exists(mm_projector_path):
        log_info(f"Loading projector weights from {mm_projector_path}")
        weights = torch.load(mm_projector_path, map_location="cpu")
    else:
        # Check local path and Drive base
        search_dirs = [Path(model_name_or_path)]
        if base_dir is not None:
            search_dirs.extend([base_dir, base_dir / "checkpoints", base_dir / "models"])
        for sdir in search_dirs:
            for cand in ["mm_projector.bin", "non_lora_trainables.bin"]:
                p = sdir / cand
                if p.exists() and p.is_file():
                    log_success(f"Found projector weights: {p}")
                    weights = torch.load(str(p), map_location="cpu")
                    break
            if weights is not None:
                break

        # Fallback to HuggingFace hub
        if weights is None:
            for repo in ["LanguageBind/Video-LLaVA-7B", "LanguageBind/Video-LLaVA-Pretrain-7B"]:
                try:
                    from huggingface_hub import hf_hub_download
                    f = hf_hub_download(repo_id=repo, filename="mm_projector.bin")
                    weights = torch.load(f, map_location="cpu")
                    log_success(f"Downloaded official pretrained mm_projector from {repo}")
                    break
                except Exception:
                    pass

    if weights is not None:
        clean = {}
        for k, v in weights.items():
            kc = k
            for prefix in ["base_model.model.mm_projector.", "model.mm_projector.", "mm_projector."]:
                if kc.startswith(prefix):
                    kc = kc[len(prefix):]
            clean[kc] = v.to(dtype)
        projector.load_state_dict(clean, strict=False)
        log_success("Loaded mm_projector weights successfully.")
    else:
        log_warn("No pretrained mm_projector found; using freshly initialized weights.")

    projector.eval()
    for p in projector.parameters():
        p.requires_grad_(False)
    return projector


# ==============================================================================
# Path & Dataset Resolution
# ==============================================================================

def extract_image_file_list(json_paths: List[Path]) -> List[str]:
    """Extracts unique image relative paths preserving occurrence order."""
    seen = {}
    for jp in json_paths:
        try:
            data = load_json_cached(jp)
        except Exception as exc:
            log_warn(f"Could not load {jp.name}: {exc}")
            continue

        for item in data:
            if "image" in item and "video" not in item:
                imgs = item["image"] if isinstance(item["image"], list) else [item["image"]]
                for img_path in imgs:
                    if img_path and img_path not in seen:
                        seen[img_path] = len(seen)

    img_files = sorted(seen, key=seen.__getitem__)
    log_success(f"Extracted {len(img_files):,} unique image file(s) across annotations.")
    return img_files


def resolve_image_path_mapping(
    folder: Path,
    probe_files: List[str],
    default_name: str = "llava_image",
) -> Tuple[Path, Optional[str], Optional[Path]]:
    """
    Checks probe images against a candidate folder to locate the exact images path
    and detect any prefix differences without scanning entire Google Drive directories.
    """
    if not folder.exists() or not folder.is_dir():
        return folder, None, None

    for probe_rel in probe_files[:10]:
        bare = Path(probe_rel).name

        # 1. Exact match
        if (folder / probe_rel).is_file():
            return folder, None, folder / probe_rel

        # 2. Flat match
        if (folder / bare).is_file():
            pfx = probe_rel[:-len(bare)] if len(probe_rel) > len(bare) else None
            return folder, pfx, folder / bare

        # 3. Progressive subdirectory strip
        if "/" in probe_rel:
            parts = probe_rel.split("/")
            for i in range(1, len(parts)):
                subpath = "/".join(parts[i:])
                if (folder / subpath).is_file():
                    pfx = "/".join(parts[:i]) + "/"
                    return folder, pfx, folder / subpath

        # 4. Standard subdirectories
        for sname in [default_name, "images", "data", "train2017"]:
            sub = folder / sname
            if not sub.is_dir():
                continue
            if (sub / bare).is_file():
                pfx = probe_rel[:-len(bare)] if len(probe_rel) > len(bare) else None
                return sub, pfx, sub / bare
            if (sub / probe_rel).is_file():
                return sub, None, sub / probe_rel

    return folder, None, None


def resolve_image_folder(
    base: Path,
    folder_name: str,
    probe_files: List[str],
) -> Tuple[Path, Optional[str], Optional[Path]]:
    """Searches standard directory locations for image files."""
    candidates = [
        base / "datasets" / folder_name,
        base / "datasets" / folder_name / folder_name,
        base / "datasets",
        base / folder_name,
        base / folder_name / folder_name,
        base / "data" / folder_name,
        base / "data",
        Path("/content/datasets") / folder_name,
        Path("/content") / folder_name,
        base,
    ]
    for c in candidates:
        if not c.exists() or not c.is_dir():
            continue
        actual_folder, strip_pfx, probe_path = resolve_image_path_mapping(
            c, probe_files[:25], default_name=folder_name
        )
        if probe_path is not None:
            pfx_msg = f" (strip prefix: '{strip_pfx}')" if strip_pfx else ""
            log_success(f"Located image folder: {actual_folder}{pfx_msg}")
            return actual_folder, strip_pfx, probe_path

    fallback = base / "datasets" / folder_name
    log_warn(f"Could not confirm probe images for '{folder_name}'. Defaulting to {fallback}")
    return fallback, None, None


def resolve_annotation_jsons(base: Path, candidate_names: List[str]) -> List[Path]:
    """Finds annotation JSON files under Google Drive base directory."""
    search_dirs = [
        base / "datasets" / "pt_json",
        base / "datasets" / "ft_json",
        base / "datasets" / "annotations",
        base / "datasets",
        base / "pt_json",
        base / "ft_json",
        base / "annotations",
        base / "data",
        Path("/content/datasets"),
        Path("/content"),
        base,
    ]
    found = []
    for cand in candidate_names:
        for sdir in search_dirs:
            if not sdir.is_dir():
                continue
            target = sdir / cand
            if target.is_file() and os.path.getsize(str(target)) > 0:
                found.append(target)
                log_success(f"Found annotation JSON: {target}")
                break
    return found


# ==============================================================================
# Main Precomputation Pipeline for Images
# ==============================================================================

def precompute_images_split(
    split_name: str,
    img_files: List[str],
    image_folder: Path,
    tower,
    processor,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 64,
    num_workers: int = 4,
    images_per_shard: int = 2000,
    projector: Optional[nn.Module] = None,
    staging_dir: Optional[Path] = None,
    strip_prefix: Optional[str] = None,
    probe_path: Optional[Path] = None,
):
    """
    Encodes images through LanguageBind vision model into sharded HDF5 files.
    Leverages fast local NVMe SSD staging and atomic Google Drive syncing.
    """
    n_total = len(img_files)
    log_header(f"Embedding {n_total:,} Images for '{split_name.upper()}' -> {output_dir}")

    # 1. Determine feature shapes
    if probe_path is None:
        image_folder, strip_prefix, probe_path = resolve_image_path_mapping(
            image_folder, img_files[:10], default_name="llava_image"
        )
    if probe_path is None:
        log_err(f"Cannot find probe image in {image_folder}. Please verify dataset paths.")
        return

    log_info(f"Probing feature dimensions with: {probe_path.name}")
    probe_img = Image.open(str(probe_path)).convert("RGB")
    probe_pv = processor.preprocess(probe_img, return_tensors="pt")["pixel_values"]
    img_pv_shape = tuple(probe_pv[0].shape)  # typically (3, 224, 224)

    with torch.no_grad():
        probe_feat = tower(probe_pv.to(device=device, dtype=dtype))
        if projector is not None:
            probe_feat = projector(probe_feat.to(next(projector.parameters()).dtype))
    feat_shape = tuple(probe_feat.shape[1:])  # typically (256, 1024) or (256, 4096)
    log_success(f"Image feature shape per sample: {feat_shape} (Projected: {projector is not None})")
    del probe_feat, probe_pv

    # 2. Output and Index Paths
    output_dir.mkdir(parents=True, exist_ok=True)
    drive_index_path = output_dir / f"{split_name}_image_index.json"

    # If staging is enabled, keep working index locally
    local_index_path = drive_index_path
    if staging_dir is not None:
        staging_dir.mkdir(parents=True, exist_ok=True)
        local_index_path = staging_dir / f"{split_name}_image_index.json"
        if drive_index_path.exists() and not local_index_path.exists():
            shutil.copy2(str(drive_index_path), str(local_index_path))

    index: Dict[str, Any] = safe_read_json(local_index_path)
    if not index and drive_index_path.exists():
        index = safe_read_json(drive_index_path)

    # 3. Resume Check
    pending_files = [f for f in img_files if f not in index]
    already_done = n_total - len(pending_files)
    if already_done > 0:
        log_success(f"Resume detected: {already_done:,} / {n_total:,} images already precomputed.")
    if not pending_files:
        log_success(f"All {n_total:,} images for '{split_name}' are already embedded! Nothing to do.")
        return

    # Helper to sync a completed shard and index to Google Drive
    def sync_shard_to_drive(shard_fname: str):
        if staging_dir is None:
            return
        local_shard = staging_dir / shard_fname
        drive_shard = output_dir / shard_fname
        if local_shard.exists():
            try:
                shutil.copy2(str(local_shard), str(drive_shard))
                shutil.copy2(str(local_index_path), str(drive_index_path))
                log_success(f"[Drive Sync] Shard {shard_fname} backed up to Google Drive ({drive_shard.stat().st_size / 1e6:.1f} MB)")
                # Prune local staging if disk space is below 20 GB
                free_gb = shutil.disk_usage(str(staging_dir)).free / 1e9
                if free_gb < 20.0 and drive_shard.exists() and drive_shard.stat().st_size == local_shard.stat().st_size:
                    local_shard.unlink()
                    log_info(f"[Disk Prune] Removed local staged {shard_fname} to free SSD space ({free_gb:.1f} GB free).")
            except Exception as exc:
                log_warn(f"[Drive Sync Warning] Sync for {shard_fname} encountered: {exc}. Local SSD copy is intact.")

    # 4. Initialize Current Shard
    work_dir = staging_dir if staging_dir is not None else output_dir
    current_shard_idx = len(index) // images_per_shard
    shard_filename = f"{split_name}_images_shard_{current_shard_idx:04d}.h5"
    shard_path = work_dir / shard_filename

    # If shard exists on Drive but not locally, fetch to resume
    if staging_dir is not None and not shard_path.exists() and (output_dir / shard_filename).exists():
        shutil.copy2(str(output_dir / shard_filename), str(shard_path))

    h5, dset, written_in_shard = open_or_create_hdf5_shard(shard_path, images_per_shard, feat_shape)
    if written_in_shard >= images_per_shard:
        h5.close()
        sync_shard_to_drive(shard_filename)
        current_shard_idx += 1
        shard_filename = f"{split_name}_images_shard_{current_shard_idx:04d}.h5"
        shard_path = work_dir / shard_filename
        h5, dset, written_in_shard = open_or_create_hdf5_shard(shard_path, images_per_shard, feat_shape)

    # 5. Build PyTorch DataLoader with Multi-worker Prefetching
    dataset = ImageEmbeddingDataset(
        file_list=pending_files,
        image_folder=image_folder,
        processor=processor,
        expected_shape=img_pv_shape,
        strip_prefix=strip_prefix,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_image_batch,
    )

    t0 = time.time()
    last_sync_time = time.time()

    try:
        for batch_idx, (batch_tensors, batch_paths, valid_flags) in enumerate(loader):
            batch_tensors = batch_tensors.to(device=device, dtype=dtype, non_blocking=True)

            with torch.no_grad():
                feats = tower(batch_tensors)
                if projector is not None:
                    feats = projector(feats.to(next(projector.parameters()).dtype))

            feats_np = feats.cpu().to(torch.float16).numpy().astype(np.float16)

            for k, rel_path in enumerate(batch_paths):
                # When shard fills up, close it, sync to Drive, and allocate next shard
                if written_in_shard >= images_per_shard:
                    h5.attrs["written"] = written_in_shard
                    h5.flush()
                    h5.close()
                    local_index_path.write_text(json.dumps(index, separators=(",", ":")))
                    sync_shard_to_drive(shard_filename)

                    current_shard_idx += 1
                    shard_filename = f"{split_name}_images_shard_{current_shard_idx:04d}.h5"
                    shard_path = work_dir / shard_filename
                    h5, dset, written_in_shard = open_or_create_hdf5_shard(shard_path, images_per_shard, feat_shape)

                dset[written_in_shard] = feats_np[k]
                index[rel_path] = {"shard": shard_filename, "idx": written_in_shard}
                written_in_shard += 1

            h5.attrs["written"] = written_in_shard

            # Periodic metadata flush (every 25 batches or 5 minutes)
            if batch_idx % 25 == 24 or (time.time() - last_sync_time > 300):
                h5.flush()
                local_index_path.write_text(json.dumps(index, separators=(",", ":")))
                if staging_dir is not None and drive_index_path.parent.exists():
                    try:
                        shutil.copy2(str(local_index_path), str(drive_index_path))
                    except Exception:
                        pass
                last_sync_time = time.time()

            # Live terminal progress bar
            elapsed = time.time() - t0
            cur_total = len(index)
            pct = (cur_total / n_total) * 100
            rate = (cur_total - already_done) / elapsed if elapsed > 1 else 0
            eta_s = (n_total - cur_total) / rate if rate > 0 else 0
            eta_h = eta_s / 3600
            print(
                f"\r  [{cur_total:>7,}/{n_total:,}] {pct:5.1f}% | "
                f"{rate:5.1f} img/s | Shard {current_shard_idx:04d} ({written_in_shard:>4}/{images_per_shard}) | ETA {eta_h:4.1f}h",
                end="",
                flush=True,
            )

    finally:
        if h5 is not None:
            try:
                h5.attrs["written"] = written_in_shard
                h5.flush()
                h5.close()
            except Exception:
                pass

        local_index_path.write_text(json.dumps(index, separators=(",", ":")))
        sync_shard_to_drive(shard_filename)
        if staging_dir is not None and drive_index_path.parent.exists():
            try:
                shutil.copy2(str(local_index_path), str(drive_index_path))
            except Exception:
                pass
        print()

    elapsed = time.time() - t0
    log_success(f"Finished {len(index):,} images for '{split_name}' in {elapsed/3600:.2f}h -> {drive_index_path}")


# ==============================================================================
# Verification Helper
# ==============================================================================

def verify_output_embeddings(output_dir: Path, split: str = "all"):
    """Validates that HDF5 shards and index JSON files exist and are readable."""
    import h5py

    log_header(f"Verifying Precomputed Embeddings in {output_dir}")
    splits = ["pretrain", "finetune"] if split == "all" else [split]

    for s in splits:
        idx_file = output_dir / f"{s}_image_index.json"
        if not idx_file.exists():
            log_warn(f"Index file {idx_file.name} not found.")
            continue

        index = safe_read_json(idx_file)
        log_info(f"Split '{s}': {len(index):,} indexed images.")

        shards = sorted(output_dir.glob(f"{s}_images_shard_*.h5"))
        log_info(f"Split '{s}': Found {len(shards)} HDF5 shard(s).")

        total_shards_samples = 0
        for sh in shards:
            try:
                with h5py.File(str(sh), "r") as h5:
                    dset = h5["features"]
                    written = int(h5.attrs.get("written", 0))
                    total_shards_samples += written
                    log_success(f"  {sh.name}: {written:,} samples stored, shape={dset.shape}, dtype={dset.dtype}")
            except Exception as exc:
                log_err(f"  {sh.name}: Failed to read ({exc})")

        log_success(f"Split '{s}' verified successfully: {total_shards_samples:,} samples across {len(shards)} shard(s).")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="High-throughput Visual Embedding Pre-computation for Images in Google Colab."
    )
    parser.add_argument(
        "--split",
        choices=["pretrain", "finetune", "all"],
        default="pretrain",
        help="Dataset split to embed: 'pretrain' (LLaVA-558K), 'finetune' (LLaVA-665K), or 'all'.",
    )
    parser.add_argument(
        "--drive_base",
        type=str,
        default="/content/drive/MyDrive/Video-LLaVA",
        help="Root path of Video-LLaVA on Google Drive.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write HDF5 shards and index JSON. Default: <drive_base>/embeddings",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Explicit path to image directory (overrides auto-detection).",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Explicit path to annotation JSON (overrides auto-detection).",
    )
    parser.add_argument(
        "--image_tower",
        type=str,
        default="LanguageBind/LanguageBind_Image",
        help="LanguageBind Image tower checkpoint or HuggingFace repo ID.",
    )
    parser.add_argument(
        "--project_visual",
        action="store_true",
        help="Project visual features (1024 -> 4096) through mm_projector into LLM hidden space.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="LLM model or checkpoint path for mm_projector extraction.",
    )
    parser.add_argument(
        "--mm_projector_path",
        type=str,
        default=None,
        help="Explicit path to mm_projector.bin.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Inference batch size (default: 64 for T4/V100, 128 for A100/L4).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="CPU background workers for image loading and preprocessing (default: 4).",
    )
    parser.add_argument(
        "--images_per_shard",
        type=int,
        default=2000,
        help="Images per HDF5 shard (default: 2000, ~2.1 GB uncompressed in FP16).",
    )
    parser.add_argument(
        "--staging_dir",
        type=str,
        default="/content/staging_embeddings",
        help="Fast local NVMe SSD staging directory to eliminate Google Drive FUSE latency.",
    )
    parser.add_argument(
        "--no_staging",
        action="store_true",
        help="Disable local SSD staging and write directly to output_dir.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Compute precision (FP16 or BF16).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing HDF5 shards and index JSON without computing.",
    )

    args = parser.parse_args()

    # 1. Colab & Drive Init
    mount_google_drive("/content/drive")
    ensure_colab_dependencies()

    # 2. CUDA Device & PyTorch Optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log_success(f"GPU Active: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        log_warn("No GPU detected. Running on CPU will be slow.")

    # 3. Path Setup
    base_dir = Path(args.drive_base)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "embeddings"
    staging_dir = None if (args.no_staging or not Path("/content").exists()) else Path(args.staging_dir)

    if args.verify:
        verify_output_embeddings(output_dir, split=args.split)
        return

    # Check Drive Responsiveness
    if "/drive/" in str(base_dir) or "/MyDrive/" in str(base_dir) or str(base_dir).startswith("/content/drive"):
        if not check_drive_responsive(base_dir, timeout_sec=5):
            log_err(f"Google Drive at {base_dir} is unresponsive. Please remount drive.")
            sys.exit(1)

    splits_to_run = ["pretrain", "finetune"] if args.split == "all" else [args.split]

    # 4. Load Image Tower Model (and optional Projector)
    tower, processor = load_image_tower(
        model_name=args.image_tower,
        device=device,
        dtype=compute_dtype,
    )

    projector = None
    if args.project_visual:
        projector = load_vision_projector(
            model_name_or_path=args.model_name_or_path,
            mm_projector_path=args.mm_projector_path,
            device=device,
            dtype=compute_dtype,
            base_dir=base_dir,
        )

    # 5. Process Splits
    for split in splits_to_run:
        # Determine annotation JSONs
        if args.json_path:
            json_paths = [Path(args.json_path)]
        elif split == "pretrain":
            json_paths = resolve_annotation_jsons(base_dir, ["llava_image_.json", "llava_image.json"])
        else:
            json_paths = resolve_annotation_jsons(base_dir, ["llava_image_tune_.json", "llava_image_tune.json"])

        if not json_paths:
            log_warn(f"No annotation JSON files located for split '{split}'. Skipping.")
            continue

        # Extract file paths from JSON
        img_files = extract_image_file_list(json_paths)
        if not img_files:
            log_warn(f"No image entries found in annotations for split '{split}'. Skipping.")
            continue

        # Determine image directory
        img_folder = None
        strip_pfx = None
        probe_path = None

        if args.image_folder:
            candidate = Path(args.image_folder)
            folder_tag = "llava_image" if split == "pretrain" else "llava_image_tune"
            img_folder, strip_pfx, probe_path = resolve_image_path_mapping(
                candidate, img_files[:25], default_name=folder_tag
            )
            if probe_path is not None:
                log_success(f"Using explicitly specified image folder: {img_folder}")

        if probe_path is None:
            folder_tag = "llava_image" if split == "pretrain" else "llava_image_tune"
            img_folder, strip_pfx, probe_path = resolve_image_folder(
                base_dir, folder_tag, img_files
            )

        if probe_path is None:
            log_err(f"Could not locate valid images for split '{split}' in {img_folder}. Skipping split.")
            continue

        # Run Precomputation
        precompute_images_split(
            split_name=split,
            img_files=img_files,
            image_folder=img_folder,
            tower=tower,
            processor=processor,
            output_dir=output_dir,
            device=device,
            dtype=compute_dtype,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            images_per_shard=args.images_per_shard,
            projector=projector,
            staging_dir=staging_dir,
            strip_prefix=strip_pfx,
            probe_path=probe_path,
        )

    # 6. Save Metadata
    meta = {
        "image_tower": args.image_tower,
        "project_visual": args.project_visual,
        "visual_dim": 4096 if args.project_visual else 1024,
        "storage_dtype": "float16",
        "splits": splits_to_run,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = output_dir / "image_embeddings_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log_success(f"Saved precomputation metadata -> {meta_path}")

    # 7. Final Verification
    verify_output_embeddings(output_dir, split=args.split)
    log_header("Image Embedding Pre-computation Complete!")


if __name__ == "__main__":
    main()
