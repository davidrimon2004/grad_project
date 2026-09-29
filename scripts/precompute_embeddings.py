"""
precompute_embeddings.py
========================
Pre-computes and caches:
  1. Visual token embeddings (LanguageBind encoder features, optionally projected
     to LLM embedding dimension via mm_projector) for images and videos.
  2. Textual token embeddings (Vicuna embed_tokens) along with input_ids, labels,
     and offsets for conversation sequences.

Supports both pretraining and fine-tuning datasets for Video-LLaVA.

Output layout on Google Drive / local disk:
    <drive_base>/embeddings/
        # Visual features:
        pretrain_images.h5          -- [N_img, N_patches, dim] fp16
        pretrain_videos.h5          -- [N_vid, T, N_patches, dim] fp16
        finetune_images.h5          -- [N_img, N_patches, dim] fp16
        finetune_videos.h5          -- [N_vid, T, N_patches, dim] fp16
        pretrain_image_index.json   -- {"relative/path.jpg": row_idx, ...}
        pretrain_video_index.json   -- {"relative/path.mp4": row_idx, ...}
        finetune_image_index.json
        finetune_video_index.json

        # Textual features:
        pretrain_text.h5            -- Extensible: text_embeddings [Total_tokens, 4096],
                                       token_ids, labels, offsets, vocab_embeddings
        pretrain_text_index.json    -- {sample_id_or_idx: {"offset": o, "length": l, ...}}
        finetune_text.h5
        finetune_text_index.json

        meta.json                   -- Dims, token counts, config, and timestamp

Usage:
    # 1. Embed everything (visual + text for pretrain + finetune):
    python scripts/precompute_embeddings.py --action all --split all

    # 2. Embed only textual tokens:
    python scripts/precompute_embeddings.py --action text --split all

    # 3. Embed only visual tokens (images + videos):
    python scripts/precompute_embeddings.py --action visual --split pretrain

    # 4. Project visual tokens to LLM dimension (4096) via mm_projector:
    python scripts/precompute_embeddings.py --action all --split all --project_visual

    # Point training at the cache:
    python scripts/colab_pretrain_muler.py --action train \\
        --embed_cache_dir /content/drive/MyDrive/Video-LLaVA/embeddings
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers

# Add repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from videollava import conversation as conversation_lib
from videollava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
)
from videollava.mm_utils import tokenizer_image_token


# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)

def log_ok(msg: str):
    print(f"[{_ts()}] OK  {msg}", flush=True)

def log_warn(msg: str):
    print(f"[{_ts()}] WARN  {msg}", flush=True)

def log_err(msg: str):
    print(f"[{_ts()}] ERR  {msg}", flush=True)

def log_header(msg: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {msg}\n{bar}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Safe I/O and Google Drive helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_exists(p: Path) -> bool:
    try:
        return os.path.exists(str(p))
    except (OSError, Exception):
        return False

def _safe_is_dir(p: Path) -> bool:
    try:
        return os.path.isdir(str(p))
    except (OSError, Exception):
        return False

def _safe_is_file(p: Path) -> bool:
    try:
        return os.path.isfile(str(p))
    except (OSError, Exception):
        return False

def check_drive_responsive(path: Path, timeout_sec: int = 5) -> bool:
    """Checks if a path on Google Drive responds within timeout_sec."""
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


def _safe_read_json(path: Path) -> Dict[str, Any]:
    """Safely loads a JSON index or dictionary file."""
    if not _safe_is_file(path):
        return {}
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log_warn(f"Failed to load JSON from {path}: {exc}")
        return {}


def load_json_cached(jp: Path, cache_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Safely loads a JSON dataset file. If the file resides on Google Drive FUSE,
    it copies it in 8MB binary chunks to a local fast cache directory
    (e.g. /tmp/cache_jsons) first, preventing FUSE socket timeouts and Python
    text decoding stalls over network filesystem sockets.
    """
    if cache_dir is None:
        if Path("/content").exists():
            cache_dir = Path("/content/cache_jsons")
        elif Path("/tmp").exists():
            cache_dir = Path("/tmp/cache_jsons")
        else:
            cache_dir = Path("./cache_jsons")

    p_str = str(jp)
    is_network_fs = "/drive/" in p_str or "/MyDrive/" in p_str or p_str.startswith("/content/drive")

    if not is_network_fs:
        with open(str(jp), "r", encoding="utf-8") as f:
            return json.load(f)

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_cached = cache_dir / jp.name

    try:
        remote_size = os.path.getsize(str(jp))
    except Exception:
        remote_size = -1

    need_copy = True
    if local_cached.exists() and remote_size > 0:
        try:
            if local_cached.stat().st_size == remote_size:
                need_copy = False
        except Exception:
            pass

    if need_copy:
        size_str = f" ({remote_size / (1024 * 1024):.1f} MB)" if remote_size > 0 else ""
        log(f"Caching {jp.name}{size_str} to local fast storage ({local_cached})...")
        t0 = time.time()
        with open(str(jp), "rb") as src, open(str(local_cached), "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        elapsed = time.time() - t0
        log_ok(f"Cached {jp.name} locally in {elapsed:.1f}s.")
    else:
        log_ok(f"Using locally cached JSON: {local_cached}")

    t0 = time.time()
    with open(str(local_cached), "r", encoding="utf-8") as f:
        data = json.load(f)
    log_ok(f"Parsed {len(data):,} items from {jp.name} in {time.time() - t0:.1f}s.")
    return data



# ──────────────────────────────────────────────────────────────────────────────
# HDF5 helpers: Visual
# ──────────────────────────────────────────────────────────────────────────────

def _open_or_create_hdf5(path: Path, n_samples: int, feat_shape: Tuple[int, ...]):
    """
    Opens an existing HDF5 file or creates one pre-allocated for n_samples.
    Returns (h5_file, features_dataset, written_count).
    """
    try:
        import h5py
    except ImportError:
        log_err("h5py not installed. Run: pip install h5py")
        sys.exit(1)

    if _safe_exists(path):
        try:
            h5 = h5py.File(str(path), "a")
            dset = h5["features"]
            written = int(h5.attrs.get("written", 0))
            log(f"  Resuming {path.name}: {written:,} / {dset.shape[0]:,} already done")
            return h5, dset, written
        except Exception as exc:
            log_warn(f"Existing file {path.name} is corrupt or unreadable ({exc}). Overwriting with fresh file.")
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
    uncompressed_gb = float(np.prod(full_shape)) * 2 / 1e9
    log(f"  Created {path.name}: shape={full_shape}  ({uncompressed_gb:.2f} GB uncompressed)")
    return h5, dset, 0


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 helpers: Textual
# ──────────────────────────────────────────────────────────────────────────────

def _open_or_create_text_hdf5(
    path: Path,
    vocab_weight: np.ndarray,
    hidden_size: int,
    save_raw_embeds: bool = False,
):
    """
    Opens an existing text HDF5 file or creates a compact/extensible one.
    When save_raw_embeds=False (default), stores pre-tokenized token_ids and labels
    (~500 MB total for 1.26M samples) without the redundant 1 TB float16 expansion.
    Returns (h5_file, dset_embeds, dset_tokens, dset_labels, dset_offsets, written_samples, total_tokens).
    """
    try:
        import h5py
    except ImportError:
        log_err("h5py not installed. Run: pip install h5py")
        sys.exit(1)

    if _safe_exists(path):
        try:
            h5 = h5py.File(str(path), "a")
            dset_embeds = h5["text_embeddings"] if "text_embeddings" in h5 else None
            dset_tokens = h5["token_ids"]
            dset_labels = h5["labels"]
            dset_offsets = h5["offsets"]
            written = int(h5.attrs.get("written_samples", 0))
            total_tokens = int(h5.attrs.get("total_tokens", 0))
            log(f"  Resuming {path.name}: {written:,} samples, {total_tokens:,} tokens already done")
            return h5, dset_embeds, dset_tokens, dset_labels, dset_offsets, written, total_tokens
        except Exception as exc:
            log_warn(f"Existing file {path.name} is corrupt or unreadable ({exc}). Overwriting with fresh file.")
            try:
                path.unlink()
            except Exception:
                pass

    path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(str(path), "w")

    # Store full vocabulary embedding table once for reference (260 MB)
    h5.create_dataset(
        "vocab_embeddings",
        data=vocab_weight.astype(np.float16),
        dtype=np.float16,
        compression="lzf",
    )

    dset_embeds = None
    if save_raw_embeds:
        dset_embeds = h5.create_dataset(
            "text_embeddings",
            shape=(0, hidden_size),
            maxshape=(None, hidden_size),
            dtype=np.float16,
            chunks=(2048, hidden_size),
            compression="lzf",
        )

    dset_tokens = h5.create_dataset(
        "token_ids",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int32,
        chunks=(16384,),
        compression="lzf",
    )
    dset_labels = h5.create_dataset(
        "labels",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int32,
        chunks=(16384,),
        compression="lzf",
    )
    dset_offsets = h5.create_dataset(
        "offsets",
        shape=(1,),
        maxshape=(None,),
        dtype=np.int64,
        chunks=(8192,),
    )
    dset_offsets[0] = 0

    h5.attrs["written_samples"] = 0
    h5.attrs["total_tokens"] = 0
    h5.attrs["hidden_size"] = hidden_size
    mode_str = "with 4096-dim float16 embeddings" if save_raw_embeds else "compact mode (pre-tokenized token_ids + labels ~500 MB)"
    log(f"  Created {path.name}: {mode_str}")
    return h5, dset_embeds, dset_tokens, dset_labels, dset_offsets, 0, 0


# ──────────────────────────────────────────────────────────────────────────────
# Staging & Syncing helpers (Fast local NVMe SSD <-> Google Drive)
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_staged_paths(
    h5_path: Path,
    index_path: Path,
    staging_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Optional[Path], Optional[Path]]:
    """
    If staging_dir is provided, ensures heavy random I/O occurs on fast local SSD,
    and returns (work_h5, work_index, target_h5, target_index).
    Resumes seamlessly by copying existing data from target storage to staging_dir.
    """
    if staging_dir is None:
        return h5_path, index_path, None, None

    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    work_h5 = staging_dir / h5_path.name
    work_index = staging_dir / index_path.name

    try:
        free_gb = shutil.disk_usage(str(staging_dir)).free / 1e9
        log(f"  [Staging] Local NVMe staging active: {staging_dir} ({free_gb:.1f} GB free)")
    except Exception:
        pass

    # If local working file does not exist, but storage target exists, copy down to resume
    if not work_h5.exists() and h5_path.exists():
        log(f"  [Staging] Fetching existing {h5_path.name} from storage to local SSD ({h5_path.stat().st_size / 1e6:.1f} MB)...")
        shutil.copy2(str(h5_path), str(work_h5))
        if index_path.exists():
            shutil.copy2(str(index_path), str(work_index))
        log_ok(f"  [Staging] Staged existing checkpoint to {work_h5}")
    elif work_h5.exists() and h5_path.exists():
        if h5_path.stat().st_size > work_h5.stat().st_size:
            log(f"  [Staging] Storage file is larger ({h5_path.stat().st_size / 1e6:.1f} MB vs {work_h5.stat().st_size / 1e6:.1f} MB). Fetching from storage...")
            shutil.copy2(str(h5_path), str(work_h5))
            if index_path.exists():
                shutil.copy2(str(index_path), str(work_index))
        else:
            log(f"  [Staging] Using existing local working file: {work_h5} ({work_h5.stat().st_size / 1e6:.1f} MB)")
    elif work_h5.exists():
        log(f"  [Staging] Found existing local working file: {work_h5} ({work_h5.stat().st_size / 1e6:.1f} MB)")

    return work_h5, work_index, h5_path, index_path


def _sync_staged_files(
    work_h5: Path,
    work_index: Path,
    target_h5: Optional[Path],
    target_index: Optional[Path],
):
    """
    Syncs working files from local SSD staging to long-term storage (e.g. Google Drive).
    Catches network/FUSE exceptions so a transient Drive error doesn't crash the script.
    """
    if target_h5 is None:
        return
    try:
        target_h5.parent.mkdir(parents=True, exist_ok=True)
        # Sync index first (fast metadata)
        if work_index.exists() and target_index is not None:
            shutil.copy2(str(work_index), str(target_index))
        # Sync HDF5
        if work_h5.exists():
            shutil.copy2(str(work_h5), str(target_h5))
            log(f"\n  [Sync] Successfully backed up {work_h5.name} to {target_h5.parent} ({work_h5.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        log_warn(f"\n  [Sync Notice] Sync to Drive encountered: {exc}. Local progress on SSD is safe; will retry next sync.")


# ──────────────────────────────────────────────────────────────────────────────
# Tower & Model Loading
# ──────────────────────────────────────────────────────────────────────────────

class _TowerArgs:
    mm_vision_select_layer = -2
    mm_vision_select_feature = "patch"


def load_image_tower(model_name: str, device: torch.device, dtype: torch.dtype):
    """Loads the LanguageBind Image tower in eval/frozen mode."""
    from videollava.model.multimodal_encoder.languagebind import (
        LanguageBindImageTower,
        LanguageBindImageProcessor,
        sanitize_attn_implementation,
    )
    log(f"Loading image tower: {model_name}")
    tower = LanguageBindImageTower(model_name, args=_TowerArgs(), cache_dir=None, delay_load=False)
    sanitize_attn_implementation(tower)
    tower = tower.to(device=device, dtype=dtype).eval()
    for p in tower.parameters():
        p.requires_grad_(False)
    sanitize_attn_implementation(tower)
    processor = getattr(tower, 'image_processor', None)
    if processor is None:
        processor = LanguageBindImageProcessor(tower.config)
    log_ok(f"Image tower ready  hidden_size={tower.hidden_size}")
    return tower, processor


def load_video_tower(model_name: str, device: torch.device, dtype: torch.dtype):
    """Loads the LanguageBind Video tower in eval/frozen mode."""
    from videollava.model.multimodal_encoder.languagebind import (
        LanguageBindVideoTower,
        LanguageBindVideoProcessor,
        sanitize_attn_implementation,
    )
    log(f"Loading video tower: {model_name}")
    tower = LanguageBindVideoTower(model_name, args=_TowerArgs(), cache_dir=None, delay_load=False)
    sanitize_attn_implementation(tower)
    tower = tower.to(device=device, dtype=dtype).eval()
    for p in tower.parameters():
        p.requires_grad_(False)
    sanitize_attn_implementation(tower)
    processor = getattr(tower, 'video_processor', None)
    if processor is None:
        processor = LanguageBindVideoProcessor(tower.config)
    log_ok(f"Video tower ready  hidden_size={tower.hidden_size}")
    return tower, processor


def load_projector(
    model_name_or_path: str,
    mm_projector_path: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    base_dir: Optional[Path] = None,
):
    """
    Loads or initializes mm_projector (MLP 1024 -> 4096) to project visual features.
    """
    from videollava.model.multimodal_projector.builder import build_vision_projector

    class _ProjectorConfig:
        mm_projector_type = "mlp2x_gelu"
        mm_hidden_size = 1024
        hidden_size = 4096

    log("Building mm_projector (mlp2x_gelu: 1024 -> 4096)...")
    projector = build_vision_projector(_ProjectorConfig()).to(device=device, dtype=dtype)

    weights = None
    if mm_projector_path and os.path.exists(mm_projector_path):
        log(f"Loading mm_projector from {mm_projector_path}")
        weights = torch.load(mm_projector_path, map_location="cpu")
    else:
        # 1. Check inside local path or Drive base
        search_dirs = [Path(model_name_or_path)]
        if base_dir is not None:
            search_dirs.extend([base_dir, base_dir / "checkpoints", base_dir / "models"])
        for sdir in search_dirs:
            for candidate in ["mm_projector.bin", "non_lora_trainables.bin"]:
                p = sdir / candidate
                if p.exists() and p.is_file():
                    log_ok(f"Found mm_projector weights: {p}")
                    weights = torch.load(str(p), map_location="cpu")
                    break
            if weights is not None:
                break

        # 2. Check HF hub repo from model_name_or_path
        if weights is None and "/" in model_name_or_path:
            try:
                from huggingface_hub import hf_hub_download
                for candidate in ["mm_projector.bin", "non_lora_trainables.bin"]:
                    try:
                        f = hf_hub_download(repo_id=model_name_or_path, filename=candidate)
                        weights = torch.load(f, map_location="cpu")
                        log_ok(f"Downloaded mm_projector weights from {model_name_or_path}/{candidate}")
                        break
                    except Exception:
                        pass
            except Exception:
                pass

        # 3. Automatic fallback to official Video-LLaVA pretrained projector weights
        if weights is None:
            for repo in ["LanguageBind/Video-LLaVA-7B", "LanguageBind/Video-LLaVA-Pretrain-7B"]:
                try:
                    from huggingface_hub import hf_hub_download
                    f = hf_hub_download(repo_id=repo, filename="mm_projector.bin")
                    weights = torch.load(f, map_location="cpu")
                    log_ok(f"Downloaded official pretrained mm_projector weights from {repo}")
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
        log_ok("Loaded mm_projector weights successfully.")
    else:
        log_warn("No pretrained mm_projector weights found; using initialized projector weights.")

    projector.eval()
    for p in projector.parameters():
        p.requires_grad_(False)
    return projector


def load_text_embedder(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Loads tokenizer and Vicuna embed_tokens layer with low-memory CPU extraction.
    """
    log(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    log(f"Extracting embed_tokens from: {model_name_or_path}")
    from transformers import AutoModelForCausalLM

    try:
        # Load only embedding layer into CPU memory, then transfer to GPU
        m = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        embed_tokens = m.get_input_embeddings().to(device=device, dtype=dtype)
        vocab_weight = embed_tokens.weight.detach().cpu().to(torch.float16).numpy()
        hidden_size = embed_tokens.embedding_dim
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        log_warn(f"Standard loading embed_tokens: {exc}")
        m = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        embed_tokens = m.get_input_embeddings()
        vocab_weight = embed_tokens.weight.detach().cpu().to(torch.float16).numpy()
        hidden_size = embed_tokens.embedding_dim

    embed_tokens.eval()
    for p in embed_tokens.parameters():
        p.requires_grad_(False)

    log_ok(f"Text embedder ready  vocab_size={vocab_weight.shape[0]:,}  hidden_size={hidden_size}")
    return tokenizer, embed_tokens, vocab_weight, hidden_size


# ──────────────────────────────────────────────────────────────────────────────
# Image embedding
# ──────────────────────────────────────────────────────────────────────────────

def embed_images(
    file_list: List[str],
    image_folder: Path,
    tower,
    processor,
    h5_path: Path,
    index_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 64,
    projector: Optional[torch.nn.Module] = None,
    staging_dir: Optional[Path] = None,
    sync_interval_mins: float = 15.0,
    images_per_shard: int = 2000,
    enable_sharding: bool = True,
    strip_prefix: Optional[str] = None,
    probe_path: Optional[Path] = None,
):
    """
    Encodes images through LanguageBind (and optional mm_projector) to HDF5.
    When enable_sharding=True (default), writes into 5 GB shards (default 2,000 images/shard)
    so Google Drive uploads and clears each shard from Colab SSD cache without overflow.
    """
    from PIL import Image as PILImage

    n_total = len(file_list)
    log_header(f"Embedding {n_total:,} images  ->  {h5_path.parent}")

    # Probe feature shape
    if probe_path is None:
        image_folder, strip_prefix, probe_path = resolve_media_path_mapping(
            image_folder, file_list[:5], default_name="llava_image"
        )
    if probe_path is None:
        log_err(f"Could not find any probe image in image_folder: {image_folder}. Check paths.")
        return

    log_ok(f"Probing image feature shape using: {probe_path}")
    probe_img = PILImage.open(str(probe_path)).convert("RGB")
    probe_tensor = processor.preprocess(probe_img, return_tensors="pt")["pixel_values"]
    img_pv_shape = tuple(probe_tensor[0].shape)
    with torch.no_grad():
        probe_feat = tower(probe_tensor.to(device=device, dtype=dtype))
        if projector is not None:
            probe_feat = projector(probe_feat.to(next(projector.parameters()).dtype))
    feat_shape = tuple(probe_feat.shape[1:])
    log(f"  Feature shape per image: {feat_shape} (projected={projector is not None})")
    del probe_feat, probe_tensor

    index: Dict[str, Any] = _safe_read_json(index_path)

    # Resume check: filter files already processed
    pending_files = [f for f in file_list if f not in index]
    already_done = n_total - len(pending_files)
    if already_done > 0:
        log_ok(f"Resuming: {already_done:,} / {n_total:,} images already cached in index.")
    if not pending_files:
        log_ok(f"All {n_total:,} images are already embedded!")
        return

    t0 = time.time()
    h5 = None

    if enable_sharding:
        current_shard_idx = len(index) // images_per_shard
        shard_filename = f"{h5_path.stem}_shard_{current_shard_idx:04d}.h5"
        shard_path = h5_path.parent / shard_filename
        h5, dset, written_in_shard = _open_or_create_hdf5(shard_path, images_per_shard, feat_shape)
        if written_in_shard >= images_per_shard:
            h5.close()
            current_shard_idx += 1
            shard_filename = f"{h5_path.stem}_shard_{current_shard_idx:04d}.h5"
            shard_path = h5_path.parent / shard_filename
            h5, dset, written_in_shard = _open_or_create_hdf5(shard_path, images_per_shard, feat_shape)

        try:
            for batch_start in range(0, len(pending_files), batch_size):
                batch_files = pending_files[batch_start : batch_start + batch_size]
                tensors: List[torch.Tensor] = []
                valid_files: List[str] = []

                for rel_path in batch_files:
                    actual_rel = rel_path[len(strip_prefix):] if strip_prefix and rel_path.startswith(strip_prefix) else rel_path
                    full = image_folder / actual_rel
                    try:
                        img = PILImage.open(str(full)).convert("RGB")
                        pv = processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
                        if tuple(pv.shape) != img_pv_shape:
                            pv = torch.zeros(img_pv_shape, dtype=dtype)
                        tensors.append(pv)
                    except Exception as exc:
                        log_warn(f"  Skipping {rel_path}: {exc}")
                        tensors.append(torch.zeros(img_pv_shape, dtype=dtype))
                    valid_files.append(rel_path)

                batch_tensor = torch.stack(tensors).to(device=device, dtype=dtype)
                with torch.no_grad():
                    feats = tower(batch_tensor)
                    if projector is not None:
                        feats = projector(feats.to(next(projector.parameters()).dtype))

                feats_np = feats.cpu().to(torch.float16).numpy().astype(np.float16)

                for k, rel_path in enumerate(valid_files):
                    if written_in_shard >= images_per_shard:
                        h5.attrs["written"] = written_in_shard
                        h5.flush()
                        h5.close()
                        log_ok(f"  [Shard {current_shard_idx:04d}] Full ({written_in_shard:,} images) -> {shard_filename}")
                        index_path.write_text(json.dumps(index, separators=(",", ":")))

                        current_shard_idx += 1
                        shard_filename = f"{h5_path.stem}_shard_{current_shard_idx:04d}.h5"
                        shard_path = h5_path.parent / shard_filename
                        h5, dset, written_in_shard = _open_or_create_hdf5(shard_path, images_per_shard, feat_shape)

                    dset[written_in_shard] = feats_np[k]
                    index[rel_path] = {"shard": shard_filename, "idx": written_in_shard}
                    written_in_shard += 1

                h5.attrs["written"] = written_in_shard

                if (batch_start // batch_size) % 50 == 49:
                    h5.flush()
                    index_path.write_text(json.dumps(index, separators=(",", ":")))

                elapsed = time.time() - t0
                cur_total = len(index)
                pct = cur_total / n_total * 100
                rate = (cur_total - already_done) / elapsed if elapsed > 1 else 0
                eta_s = (n_total - cur_total) / rate if rate > 0 else float("inf")
                print(
                    f"\r  [{cur_total:>7,}/{n_total:,}] {pct:5.1f}% | "
                    f"{rate:6.0f} img/s | shard {current_shard_idx:04d} ({written_in_shard}/{images_per_shard}) | ETA {eta_s/3600:.1f}h",
                    end="", flush=True,
                )
        finally:
            if h5 is not None:
                try:
                    h5.attrs["written"] = written_in_shard
                    h5.flush()
                    h5.close()
                except Exception:
                    pass
            index_path.write_text(json.dumps(index, separators=(",", ":")))
            print()
    else:
        # Legacy single-file mode
        work_h5, work_index, target_h5, target_index = _prepare_staged_paths(
            h5_path, index_path, staging_dir
        )
        h5, dset, written = _open_or_create_hdf5(work_h5, n_total, feat_shape)
        try:
            for batch_start in range(written, n_total, batch_size):
                batch_files = file_list[batch_start : batch_start + batch_size]
                tensors = []
                valid_files = []
                for rel_path in batch_files:
                    actual_rel = rel_path[len(strip_prefix):] if strip_prefix and rel_path.startswith(strip_prefix) else rel_path
                    full = image_folder / actual_rel
                    try:
                        img = PILImage.open(str(full)).convert("RGB")
                        pv = processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
                        if tuple(pv.shape) != img_pv_shape:
                            pv = torch.zeros(img_pv_shape, dtype=dtype)
                        tensors.append(pv)
                    except Exception:
                        tensors.append(torch.zeros(img_pv_shape, dtype=dtype))
                    valid_files.append(rel_path)
                batch_tensor = torch.stack(tensors).to(device=device, dtype=dtype)
                with torch.no_grad():
                    feats = tower(batch_tensor)
                    if projector is not None:
                        feats = projector(feats.to(next(projector.parameters()).dtype))
                feats_np = feats.cpu().to(torch.float16).numpy().astype(np.float16)
                end_row = batch_start + len(valid_files)
                dset[batch_start:end_row] = feats_np
                for k, rel_path in enumerate(valid_files):
                    index[rel_path] = batch_start + k
                written = end_row
                h5.attrs["written"] = written
        finally:
            h5.flush()
            h5.close()
            work_index.write_text(json.dumps(index, separators=(",", ":")))
            print()

    elapsed = time.time() - t0
    log_ok(f"Done {len(index):,} images in {elapsed/3600:.2f}h  ->  {index_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Video embedding
# ──────────────────────────────────────────────────────────────────────────────

def embed_videos(
    file_list: List[str],
    video_folder: Path,
    tower,
    processor,
    h5_path: Path,
    index_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 4,
    projector: Optional[torch.nn.Module] = None,
    staging_dir: Optional[Path] = None,
    sync_interval_mins: float = 15.0,
    videos_per_shard: int = 250,
    enable_sharding: bool = True,
    strip_prefix: Optional[str] = None,
    probe_path: Optional[Path] = None,
):
    """
    Encodes videos through LanguageBind (and optional mm_projector) to HDF5.
    When enable_sharding=True (default), writes into 5 GB shards (default 250 videos/shard)
    so Google Drive uploads and clears each shard from Colab SSD cache without overflow.
    """
    n_total = len(file_list)
    log_header(f"Embedding {n_total:,} videos  ->  {h5_path.parent}")

    if probe_path is None:
        video_folder, strip_prefix, probe_path = resolve_media_path_mapping(
            video_folder, file_list[:5], default_name="valley"
        )
    if probe_path is None:
        log_err(f"Could not find any probe video in video_folder: {video_folder}. Check paths.")
        return

    log_ok(f"Probing video feature shape using: {probe_path}")
    probe_pv = processor(str(probe_path), return_tensors="pt")["pixel_values"].to(
        device=device, dtype=dtype
    )
    pv_shape = tuple(probe_pv[0].shape)
    with torch.no_grad():
        probe_feat = tower(probe_pv)
        if projector is not None:
            probe_feat = projector(probe_feat.to(next(projector.parameters()).dtype))
    feat_shape = tuple(probe_feat.shape[1:])
    log(f"  Pixel values shape per video: {pv_shape}")
    log(f"  Feature shape per video: {feat_shape} (projected={projector is not None})")
    del probe_feat, probe_pv

    index: Dict[str, Any] = _safe_read_json(index_path)

    pending_files = [f for f in file_list if f not in index]
    already_done = n_total - len(pending_files)
    if already_done > 0:
        log_ok(f"Resuming: {already_done:,} / {n_total:,} videos already cached in index.")
    if not pending_files:
        log_ok(f"All {n_total:,} videos are already embedded!")
        return

    t0 = time.time()
    h5 = None

    if enable_sharding:
        current_shard_idx = len(index) // videos_per_shard
        shard_filename = f"{h5_path.stem}_shard_{current_shard_idx:04d}.h5"
        shard_path = h5_path.parent / shard_filename
        h5, dset, written_in_shard = _open_or_create_hdf5(shard_path, videos_per_shard, feat_shape)
        if written_in_shard >= videos_per_shard:
            h5.close()
            current_shard_idx += 1
            shard_filename = f"{h5_path.stem}_shard_{current_shard_idx:04d}.h5"
            shard_path = h5_path.parent / shard_filename
            h5, dset, written_in_shard = _open_or_create_hdf5(shard_path, videos_per_shard, feat_shape)

        try:
            for batch_start in range(0, len(pending_files), batch_size):
                batch_files = pending_files[batch_start : batch_start + batch_size]
                pvs: List[torch.Tensor] = []
                valid_files: List[str] = []

                for rel_path in batch_files:
                    actual_rel = rel_path[len(strip_prefix):] if strip_prefix and rel_path.startswith(strip_prefix) else rel_path
                    full = str(video_folder / actual_rel)
                    try:
                        pv = processor(full, return_tensors="pt")["pixel_values"][0]
                        if tuple(pv.shape) != pv_shape:
                            log_warn(f"  Unexpected shape for {rel_path}: {pv.shape} != {pv_shape}")
                            pv = torch.zeros(pv_shape, dtype=dtype)
                        pvs.append(pv)
                    except Exception as exc:
                        log_warn(f"  Skipping {rel_path}: {exc}")
                        pvs.append(torch.zeros(pv_shape, dtype=dtype))
                    valid_files.append(rel_path)

                batch_tensor = torch.stack(pvs).to(device=device, dtype=dtype)
                with torch.no_grad():
                    feats = tower(batch_tensor)
                    if projector is not None:
                        feats = projector(feats.to(next(projector.parameters()).dtype))

                feats_np = feats.cpu().to(torch.float16).numpy().astype(np.float16)

                for k, rel_path in enumerate(valid_files):
                    if written_in_shard >= videos_per_shard:
                        h5.attrs["written"] = written_in_shard
                        h5.flush()
                        h5.close()
                        log_ok(f"  [Shard {current_shard_idx:04d}] Full ({written_in_shard:,} videos) -> {shard_filename}")
                        index_path.write_text(json.dumps(index, separators=(",", ":")))

                        current_shard_idx += 1
                        shard_filename = f"{h5_path.stem}_shard_{current_shard_idx:04d}.h5"
                        shard_path = h5_path.parent / shard_filename
                        h5, dset, written_in_shard = _open_or_create_hdf5(shard_path, videos_per_shard, feat_shape)

                    dset[written_in_shard] = feats_np[k]
                    index[rel_path] = {"shard": shard_filename, "idx": written_in_shard}
                    written_in_shard += 1

                h5.attrs["written"] = written_in_shard

                if (batch_start // batch_size) % 50 == 49:
                    h5.flush()
                    index_path.write_text(json.dumps(index, separators=(",", ":")))

                elapsed = time.time() - t0
                cur_total = len(index)
                pct = cur_total / n_total * 100
                rate = (cur_total - already_done) / elapsed if elapsed > 1 else 0
                eta_s = (n_total - cur_total) / rate if rate > 0 else float("inf")
                print(
                    f"\r  [{cur_total:>6,}/{n_total:,}] {pct:5.1f}% | "
                    f"{rate:.2f} vid/s | shard {current_shard_idx:04d} ({written_in_shard}/{videos_per_shard}) | ETA {eta_s/3600:.1f}h",
                    end="", flush=True,
                )
        finally:
            if h5 is not None:
                try:
                    h5.attrs["written"] = written_in_shard
                    h5.flush()
                    h5.close()
                except Exception:
                    pass
            index_path.write_text(json.dumps(index, separators=(",", ":")))
            print()
    else:
        work_h5, work_index, target_h5, target_index = _prepare_staged_paths(
            h5_path, index_path, staging_dir
        )
        h5, dset, written = _open_or_create_hdf5(work_h5, n_total, feat_shape)
        try:
            for batch_start in range(written, n_total, batch_size):
                batch_files = file_list[batch_start : batch_start + batch_size]
                pvs = []
                valid_files = []
                for rel_path in batch_files:
                    actual_rel = rel_path[len(strip_prefix):] if strip_prefix and rel_path.startswith(strip_prefix) else rel_path
                    full = str(video_folder / actual_rel)
                    try:
                        pv = processor(full, return_tensors="pt")["pixel_values"][0]
                        if tuple(pv.shape) != pv_shape:
                            pv = torch.zeros(pv_shape, dtype=dtype)
                        pvs.append(pv)
                    except Exception:
                        pvs.append(torch.zeros(pv_shape, dtype=dtype))
                    valid_files.append(rel_path)
                batch_tensor = torch.stack(pvs).to(device=device, dtype=dtype)
                with torch.no_grad():
                    feats = tower(batch_tensor)
                    if projector is not None:
                        feats = projector(feats.to(next(projector.parameters()).dtype))
                feats_np = feats.cpu().to(torch.float16).numpy().astype(np.float16)
                end_row = batch_start + len(valid_files)
                dset[batch_start:end_row] = feats_np
                for k, rel_path in enumerate(valid_files):
                    index[rel_path] = batch_start + k
                written = end_row
                h5.attrs["written"] = written
        finally:
            h5.flush()
            h5.close()
            work_index.write_text(json.dumps(index, separators=(",", ":")))
            print()

    elapsed = time.time() - t0
    log_ok(f"Done {len(index):,} videos in {elapsed/3600:.2f}h  ->  {index_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Textual embedding
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_sample_text(
    sample: Dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
    conv_template_name: str = "vicuna_v1",
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Tokenizes a single conversation sample with roles and labels masked.
    Returns (input_ids, labels) as 1D LongTensors, where <image> is IMAGE_TOKEN_INDEX.
    """
    conversations = sample.get("conversations", [])
    if not conversations:
        return None

    sources = copy.deepcopy(conversations)

    # Replace <video> with 8 <image> tokens
    has_image = ("image" in sample) or ("video" in sample)
    for sentence in sources:
        val = sentence.get("value", "")
        if DEFAULT_VIDEO_TOKEN in val:
            sentence["value"] = val.replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN * 8)

    # Format conversations with template
    conv = conversation_lib.conv_templates[conv_template_name].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    if roles.get(sources[0].get("from")) != conv.roles[0]:
        sources = sources[1:]
    if not sources:
        return None

    conv.messages = []
    for j, s in enumerate(sources):
        role = roles.get(s.get("from"), conv.roles[j % 2])
        conv.append_message(role, s.get("value", ""))
    prompt = conv.get_prompt()

    # Tokenize
    if has_image:
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]

    # Create target labels with human turns masked to IGNORE_INDEX (-100)
    target = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    rounds = prompt.split(conv.sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX
    for rou in rounds:
        if rou == "":
            break
        parts = rou.split(sep)
        if len(parts) != 2:
            break
        parts[0] += sep

        if has_image:
            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
        else:
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len
    target[cur_len:] = IGNORE_INDEX

    return input_ids, target


def embed_text(
    json_paths: List[Path],
    tokenizer: transformers.PreTrainedTokenizer,
    embed_tokens_layer: Optional[torch.nn.Module],
    vocab_weight: np.ndarray,
    hidden_size: int,
    h5_path: Path,
    index_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 256,
    staging_dir: Optional[Path] = None,
    sync_interval_mins: float = 15.0,
    save_raw_embeds: bool = False,
):
    """
    Precomputes pre-tokenized token_ids, labels, and offsets into an HDF5 file.
    By default, saves in compact mode (~500 MB for 1.26M samples) without the
    redundant 1 TB float16 expansion, enabling fast tokenization without filling SSD cache.
    """
    # Load all items across JSONs
    all_samples: List[Dict[str, Any]] = []
    for jp in json_paths:
        if not _safe_is_file(jp):
            log_warn(f"JSON not found, skipping: {jp}")
            continue
        try:
            data = load_json_cached(jp, staging_dir)
            all_samples.extend(data)
        except Exception as exc:
            log_warn(f"Failed to parse {jp}: {exc}")

    n_samples = len(all_samples)
    log_header(f"Embedding text for {n_samples:,} conversation samples  ->  {h5_path.name}")

    work_h5, work_index, target_h5, target_index = _prepare_staged_paths(
        h5_path, index_path, staging_dir
    )
    h5, dset_embeds, dset_tokens, dset_labels, dset_offsets, written, total_tokens = (
        _open_or_create_text_hdf5(work_h5, vocab_weight, hidden_size, save_raw_embeds=save_raw_embeds)
    )

    index: Dict[str, Any] = _safe_read_json(work_index)

    t0 = time.time()
    last_sync = time.time()
    sync_interval_sec = max(60.0, sync_interval_mins * 60.0)

    try:
        for batch_start in range(written, n_samples, batch_size):
            batch_samples = all_samples[batch_start : batch_start + batch_size]

            batch_token_ids_list: List[np.ndarray] = []
            batch_labels_list: List[np.ndarray] = []
            batch_sample_meta: List[Dict[str, Any]] = []

            for k, s in enumerate(batch_samples):
                sample_idx = batch_start + k
                sid = str(s.get("id", sample_idx))
                parsed = _preprocess_sample_text(s, tokenizer)
                if parsed is None:
                    # dummy empty token
                    toks = np.array([tokenizer.eos_token_id or 2], dtype=np.int32)
                    labs = np.array([IGNORE_INDEX], dtype=np.int32)
                else:
                    inp, lab = parsed
                    # Replace IMAGE_TOKEN_INDEX (-200) with pad token for embed_tokens lookup
                    toks = inp.numpy().astype(np.int32)
                    labs = lab.numpy().astype(np.int32)

                batch_token_ids_list.append(toks)
                batch_labels_list.append(labs)

                # Media info
                mtype = "none"
                mpath = None
                if "image" in s:
                    mtype = "image"
                    mpath = s["image"]
                elif "video" in s:
                    mtype = "video"
                    mpath = s["video"]

                batch_sample_meta.append({
                    "id": sid,
                    "idx": sample_idx,
                    "media_type": mtype,
                    "media_path": mpath,
                })

            # Concatenate token IDs and labels
            flat_tokens = np.concatenate(batch_token_ids_list)
            flat_labels = np.concatenate(batch_labels_list)
            num_tokens_in_batch = len(flat_tokens)

            cur_tokens = dset_tokens.shape[0]
            new_tokens = cur_tokens + num_tokens_in_batch

            if save_raw_embeds and dset_embeds is not None and embed_tokens_layer is not None:
                # Clamp negative token ids (like -200) to 0 for embedding lookup
                lookup_tokens = np.where(flat_tokens >= 0, flat_tokens, 0)
                tokens_tensor = torch.from_numpy(lookup_tokens).to(device=device, dtype=torch.long)
                with torch.no_grad():
                    embeds = embed_tokens_layer(tokens_tensor)
                    mask_non_text = torch.from_numpy(flat_tokens < 0).to(device=device)
                    if mask_non_text.any():
                        embeds[mask_non_text] = 0
                embeds_np = embeds.cpu().to(torch.float16).numpy().astype(np.float16)
                dset_embeds.resize((new_tokens, hidden_size))
                dset_embeds[cur_tokens:new_tokens] = embeds_np

            # Append to compact token_ids and labels
            dset_tokens.resize((new_tokens,))
            dset_tokens[cur_tokens:new_tokens] = flat_tokens

            dset_labels.resize((new_tokens,))
            dset_labels[cur_tokens:new_tokens] = flat_labels

            # Update offsets and JSON index
            cur_offset = cur_tokens
            offsets_to_append = []
            for meta, toks in zip(batch_sample_meta, batch_token_ids_list):
                length = len(toks)
                offsets_to_append.append(cur_offset + length)
                index_entry = {
                    "offset": int(cur_offset),
                    "length": int(length),
                    "media_type": meta["media_type"],
                    "media_path": meta["media_path"],
                }
                index[str(meta["id"])] = index_entry
                index[str(meta["idx"])] = index_entry
                cur_offset += length

            cur_offsets_len = dset_offsets.shape[0]
            new_offsets_len = cur_offsets_len + len(offsets_to_append)
            dset_offsets.resize((new_offsets_len,))
            dset_offsets[cur_offsets_len:new_offsets_len] = np.array(offsets_to_append, dtype=np.int64)

            written = batch_start + len(batch_samples)
            total_tokens = new_tokens
            h5.attrs["written_samples"] = written
            h5.attrs["total_tokens"] = total_tokens

            # Flush periodically to local disk
            if (batch_start // batch_size) % 50 == 49:
                h5.flush()
                work_index.write_text(json.dumps(index, separators=(",", ":")))
                if target_h5 is not None and (time.time() - last_sync >= sync_interval_sec):
                    _sync_staged_files(work_h5, work_index, target_h5, target_index)
                    last_sync = time.time()

            elapsed = time.time() - t0
            pct = written / n_samples * 100
            rate_samp = written / elapsed if elapsed > 1 else 0
            rate_tok = total_tokens / elapsed if elapsed > 1 else 0
            eta_s = (n_samples - written) / rate_samp if rate_samp > 0 else float("inf")
            print(
                f"\r  [{written:>7,}/{n_samples:,}] {pct:5.1f}% | "
                f"{rate_samp:5.0f} samp/s | {rate_tok:7.0f} tok/s | ETA {eta_s/3600:.1f}h",
                end="", flush=True,
            )
    finally:
        h5.flush()
        h5.close()
        work_index.write_text(json.dumps(index, separators=(",", ":")))
        if target_h5 is not None:
            _sync_staged_files(work_h5, work_index, target_h5, target_index)
        print()

    elapsed = time.time() - t0
    final_dest = target_h5 if target_h5 is not None else work_h5
    log_ok(f"Done {written:,} samples ({total_tokens:,} tokens) in {elapsed/3600:.2f}h  ->  {final_dest}")
    log_ok(f"Index saved  ->  {target_index if target_index is not None else work_index}")


# ──────────────────────────────────────────────────────────────────────────────
# File extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_file_lists(
    json_paths: List[Path],
    modality: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Parses data JSON files and returns (image_files, video_files) as
    deduplicated lists of relative paths.
    
    If modality == 'videos', skips pure image or pure NLP annotation files.
    If modality == 'images', skips pure video or pure NLP annotation files.
    """
    seen_images: Dict[str, int] = {}
    seen_videos: Dict[str, int] = {}

    for jp in json_paths:
        if not _safe_is_file(jp):
            log_warn(f"JSON not found, skipping: {jp}")
            continue

        jp_name = jp.name.lower()
        if modality == "videos":
            if "image" in jp_name or "nlp" in jp_name:
                log(f"Skipping non-video annotation file: {jp.name}")
                continue
        elif modality == "images":
            if "video" in jp_name or "valley" in jp_name or "nlp" in jp_name:
                log(f"Skipping non-image annotation file: {jp.name}")
                continue

        try:
            data = load_json_cached(jp)
        except Exception as exc:
            log_warn(f"Failed to load {jp}: {exc}")
            continue

        for item in data:
            if modality != "videos" and "image" in item and "video" not in item:
                files = item["image"] if isinstance(item["image"], list) else [item["image"]]
                for f in files:
                    if f not in seen_images:
                        seen_images[f] = len(seen_images)
            elif modality != "images" and "video" in item:
                files = item["video"] if isinstance(item["video"], list) else [item["video"]]
                for f in files:
                    if f not in seen_videos:
                        seen_videos[f] = len(seen_videos)

    image_files = sorted(seen_images, key=seen_images.__getitem__)
    video_files = sorted(seen_videos, key=seen_videos.__getitem__)
    return image_files, video_files


def resolve_json_paths(base: Path, candidate_groups: List[Any]) -> List[Path]:
    """
    Finds existing annotation JSON files across common dataset/annotation directory structures.
    candidate_groups can be a list of filenames or a list of alias groups (e.g. [['a.json', 'b.json']]).
    Stops searching an alias group once a matching file is found. Never performs recursive rglob
    over network filesystems like Google Drive.
    """
    found: List[Path] = []

    if not _safe_exists(base):
        log_err(f"Base path does not exist: {base}")
        log_err("Google Drive is NOT mounted or disconnected! Run in Colab: drive.mount('/content/drive', force_remount=True)")
        return []

    if not check_drive_responsive(base, timeout_sec=5):
        log_err(f"Google Drive at {base} is UNRESPONSIVE (timed out after 5s).")
        log_err("Drive FUSE socket has hung. Run in Colab: drive.mount('/content/drive', force_remount=True)")
        return []

    common_subdirs = [
        base / "datasets" / "annotations",
        base / "datasets" / "pt_json",
        base / "datasets" / "ft_json",
        base / "datasets",
        base / "data" / "annotations",
        base / "data" / "pt_json",
        base / "data" / "ft_json",
        base / "data",
        base / "annotations",
        base / "pt_json",
        base / "ft_json",
        base / "download" / "annotations",
        base / "download",
        base,
    ]
    for item in candidate_groups:
        group = item if isinstance(item, list) else [item]
        matched = None
        for name in group:
            for sdir in common_subdirs:
                if not _safe_is_dir(sdir):
                    continue
                p = sdir / name
                if _safe_is_file(p):
                    try:
                        if os.path.getsize(str(p)) > 0:
                            matched = p
                            break
                    except Exception:
                        continue
            if matched is not None:
                break
        if matched and matched not in found:
            found.append(matched)
            try:
                rel = matched.relative_to(base)
            except Exception:
                rel = matched
            log_ok(f"Found annotation: {rel}")
        elif not matched:
            log_warn(f"Could not find any of {group} in standard subdirectories under {base}")
    return found


def resolve_media_path_mapping(
    folder: Path,
    probe_files: List[str],
    default_name: str = "",
) -> Tuple[Path, Optional[str], Optional[Path]]:
    """
    Given a candidate folder and probe relative paths from annotation JSON,
    finds the actual location on disk and determines if a relative prefix needs
    to be stripped or adjusted.
    Tests at most 5 probe files with direct os.path.isfile checks.
    NEVER scans directories or calls iterdir() over Google Drive FUSE.
    Returns: (actual_folder, strip_prefix, valid_probe_path)
    """
    # 1. Direct targeted path checks across first 25 probe files
    for probe_rel in probe_files[:25]:
        direct = folder / probe_rel
        if _safe_is_file(direct):
            return folder, None, direct

        for name in [folder.name, default_name]:
            if name and probe_rel.startswith(f"{name}/"):
                stripped = probe_rel[len(name) + 1:]
                candidate = folder / stripped
                if _safe_is_file(candidate):
                    return folder, f"{name}/", candidate

        if _safe_is_file(folder.parent / probe_rel):
            return folder.parent, None, folder.parent / probe_rel

        if _safe_is_file(folder / folder.name / probe_rel):
            return folder / folder.name, None, folder / folder.name / probe_rel

        if default_name and probe_rel.startswith(f"{default_name}/"):
            stripped = probe_rel[len(default_name) + 1:]
            candidate = folder / default_name / stripped
            if _safe_is_file(candidate):
                return folder / default_name, f"{default_name}/", candidate

    # 2. If direct checks failed, peek at first few items physically in folder
    if _safe_is_dir(folder):
        sample_names = []
        try:
            with os.scandir(str(folder)) as it:
                for _, entry in zip(range(10), it):
                    sample_names.append(entry.name)
        except Exception:
            pass

        if not sample_names:
            log_warn(f"Directory {folder} exists but is EMPTY!")
            return folder, None, None

        # Check if sample files match any probe files by basename (fast O(1) in-memory lookup)
        probe_basenames = {Path(p).name: p for p in probe_files[:1000]}
        for entry_name in sample_names:
            if entry_name in probe_basenames:
                matched_probe = probe_basenames[entry_name]
                actual_file = folder / entry_name
                prefix = matched_probe.rsplit("/", 1)[0] + "/" if "/" in matched_probe else None
                return folder, prefix, actual_file

        # Check subdirectories inside folder (e.g. folder / 'videos' or folder / 'valley')
        for entry_name in sample_names:
            sub = folder / entry_name
            if _safe_is_dir(sub):
                for probe_rel in probe_files[:10]:
                    bare_name = Path(probe_rel).name
                    if _safe_is_file(sub / bare_name):
                        prefix = probe_rel.rsplit("/", 1)[0] + "/" if "/" in probe_rel else None
                        return sub, prefix, sub / bare_name
                    if _safe_is_file(sub / probe_rel):
                        return sub, None, sub / probe_rel

        log_warn(f"Folder {folder} contains {len(sample_names)} items (e.g. {sample_names[:3]}), but none match probe paths (e.g. {probe_files[:2]})")

    return folder, None, None


def resolve_media_folder(
    base: Path,
    default_name: str,
    probe_files: List[str],
) -> Tuple[Path, Optional[str], Optional[Path]]:
    """
    Locates the directory where media probe files exist under base.
    Uses targeted single-file existence checks without any directory listing.
    """
    candidates = [
        base / "datasets" / default_name,
        base / "datasets" / default_name / default_name,
        base / "datasets",
        base / default_name,
        base / default_name / default_name,
        base / "data" / default_name,
        base / "data",
        base,
    ]
    for c in candidates:
        if not _safe_is_dir(c):
            continue
        actual_folder, strip_prefix, probe_path = resolve_media_path_mapping(
            c, probe_files[:25], default_name=default_name
        )
        if probe_path is not None:
            prefix_msg = f" (strip prefix: '{strip_prefix}')" if strip_prefix else ""
            log_ok(f"Found media folder for '{default_name}': {actual_folder}{prefix_msg}")
            return actual_folder, strip_prefix, probe_path

    # Fallback if no probe matched
    fallback = base / "datasets" / default_name
    log_warn(f"Could not confirm media probe files for '{default_name}'. Defaulting to {fallback}")
    return fallback, None, None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute visual and textual token embeddings for Video-LLaVA."
    )
    parser.add_argument(
        "--action",
        choices=["images", "videos", "visual", "text", "all"],
        default="all",
        help="Which modality to embed.",
    )
    parser.add_argument(
        "--split",
        choices=["pretrain", "finetune", "all"],
        default="all",
        help="Which dataset split to embed.",
    )
    parser.add_argument(
        "--drive_base",
        type=str,
        default="/content/drive/MyDrive/Video-LLaVA",
        help="Root of Video-LLaVA data on Google Drive or local filesystem.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write HDF5 files. Default: <drive_base>/embeddings",
    )
    parser.add_argument(
        "--project_visual",
        action="store_true",
        help="If set, projects visual features through mm_projector into LLM hidden dimension (4096).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="LLM model or checkpoint for embed_tokens and tokenizer.",
    )
    parser.add_argument(
        "--mm_projector_path",
        type=str,
        default=None,
        help="Explicit path to mm_projector.bin (used when --project_visual is enabled).",
    )
    parser.add_argument(
        "--image_tower",
        type=str,
        default="LanguageBind/LanguageBind_Image",
    )
    parser.add_argument(
        "--video_tower",
        type=str,
        default="LanguageBind/LanguageBind_Video_merge",
    )
    parser.add_argument("--image_batch_size", type=int, default=64)
    parser.add_argument("--video_batch_size", type=int, default=4)
    parser.add_argument("--text_batch_size", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Encoder compute precision (storage is always fp16).",
    )
    parser.add_argument(
        "--staging_dir",
        type=str,
        default=None,
        help="Local staging directory (e.g. /content/staging_embeddings) on fast local NVMe to prevent Google Drive FUSE timeouts. Periodically synced to Drive.",
    )
    parser.add_argument(
        "--no_staging",
        action="store_true",
        help="Disable local SSD staging and write directly to Google Drive.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Explicit directory path for images (overrides auto-detection).",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help="Explicit directory path for videos (overrides auto-detection).",
    )
    parser.add_argument(
        "--sync_interval_mins",
        type=float,
        default=15.0,
        help="Minutes between periodic syncs from local staging to Drive (default: 15.0).",
    )
    parser.add_argument(
        "--images_per_shard",
        type=int,
        default=2000,
        help="Images per HDF5 shard (default: 2000, ~4.2 GB uncompressed). Prevents Colab SSD cache overflow.",
    )
    parser.add_argument(
        "--videos_per_shard",
        type=int,
        default=250,
        help="Videos per HDF5 shard (default: 250, ~4.2 GB uncompressed). Prevents Colab SSD cache overflow.",
    )
    parser.add_argument(
        "--no_sharding",
        action="store_true",
        help="Disable HDF5 sharding and write a single monolithic file (not recommended on Google Drive FUSE).",
    )
    parser.add_argument(
        "--save_raw_text_embeddings",
        action="store_true",
        help="Save raw 4096-dim float16 embeddings (~1 TB). Default is False (compact pre-tokenized token_ids & labels ~500 MB).",
    )
    args = parser.parse_args()

    # -- device / dtype ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"GPU: {gpu}  ({vram:.1f} GB VRAM)")
    else:
        log_warn("No GPU detected. Embedding on CPU will be slow.")

    # -- paths ----------------------------------------------------------------
    base = Path(args.drive_base)
    if "/drive/" in str(base) or "/MyDrive/" in str(base) or str(base).startswith("/content/drive"):
        log("Checking Google Drive connectivity...")
        if not _safe_exists(base) or not check_drive_responsive(base, timeout_sec=5):
            log_err(f"Google Drive base path is inaccessible or unresponsive: {base}")
            log_err("The Google Drive connection timed out or is unmounted. Run in Colab:")
            log_err("  from google.colab import drive")
            log_err("  drive.flush_and_unmount()")
            log_err("  drive.mount('/content/drive', force_remount=True)")
            sys.exit(1)
        log_ok(f"Google Drive is responsive: {base}")

    out = Path(args.output_dir) if args.output_dir else base / "embeddings"
    out.mkdir(parents=True, exist_ok=True)

    # -- staging setup --------------------------------------------------------
    staging_dir = None
    if args.no_staging or (args.staging_dir and args.staging_dir.lower() in ["none", "false", "no"]):
        staging_dir = None
        log_ok("Writing directly to Google Drive (local NVMe SSD staging disabled).")
    elif args.staging_dir:
        staging_dir = Path(args.staging_dir)
    elif "/drive/" in str(out) or "/MyDrive/" in str(out) or str(out).startswith("/content/drive"):
        # Check local SSD capacity
        if Path("/content").exists():
            free_gb = shutil.disk_usage("/content").free / 1e9
            # If dataset is huge (e.g. pretrain images: ~1.1 TB) and SSD only has ~40 GB free, write directly to Drive!
            if free_gb < 80.0 and args.action in ["images", "visual", "all"] and (args.split in ["pretrain", "all"]):
                log_warn(f"Local NVMe SSD has only {free_gb:.1f} GB free, but pretrain visual dataset requires much more space.")
                log_ok("Writing directly to Google Drive to prevent local disk space exhaustion.")
                staging_dir = None
            else:
                staging_dir = Path("/content/staging_embeddings")
                log_ok(f"Detected Google Drive destination. Using local NVMe staging: {staging_dir}")

    splits_to_run = ["pretrain", "finetune"] if args.split == "all" else [args.split]

    run_images = args.action in ["images", "visual", "all"]
    run_videos = args.action in ["videos", "visual", "all"]
    run_text = args.action in ["text", "all"]

    log_header(f"Resolving Dataset Paths under {base}")
    cfg = {}
    if "pretrain" in splits_to_run:
        pretrain_jsons = resolve_json_paths(
            base,
            [
                ["llava_image_.json", "llava_image.json"],
                ["valley_.json", "valley.json"],
            ],
        )
        cfg["pretrain"] = {
            "jsons": pretrain_jsons,
            "image_folder_name": "llava_image",
            "video_folder_name": "valley",
        }

    if "finetune" in splits_to_run:
        finetune_jsons = resolve_json_paths(
            base,
            [
                ["llava_image_tune_.json", "llava_image_tune.json"],
                ["videochatgpt_tune_.json", "videochatgpt_.json", "videochatgpt_tune.json"],
                ["nlp_tune.json"],
            ],
        )
        cfg["finetune"] = {
            "jsons": finetune_jsons,
            "image_folder_name": "llava_image_tune",
            "video_folder_name": "videochatgpt_tune",
        }

    # -- lazy model loading --------------------------------------------------
    image_tower = image_proc = None
    video_tower = video_proc = None
    projector = None
    tokenizer = embed_tokens_layer = vocab_weight = None
    hidden_size = 4096

    if args.project_visual and (run_images or run_videos):
        projector = load_projector(
            args.model_name_or_path,
            args.mm_projector_path,
            device=device,
            dtype=dtype,
            base_dir=base,
        )

    if run_images:
        image_tower, image_proc = load_image_tower(args.image_tower, device, dtype)

    if run_videos:
        video_tower, video_proc = load_video_tower(args.video_tower, device, dtype)

    if run_text:
        tokenizer, embed_tokens_layer, vocab_weight, hidden_size = load_text_embedder(
            args.model_name_or_path,
            device=device,
            dtype=dtype,
        )

    # -- per-split embedding --------------------------------------------------
    for split in splits_to_run:
        log_header(f"Split: {split.upper()}")
        scfg = cfg[split]

        if not scfg["jsons"]:
            log_warn(f"No annotation JSONs found for split '{split}'. Skipping.")
            continue

        # 1. Images
        if run_images:
            img_files, _ = extract_file_lists(scfg["jsons"], modality="images")
            if img_files:
                resolved_img_folder = None
                img_strip_prefix = None
                img_probe = None
                if args.image_folder:
                    candidate = Path(args.image_folder)
                    resolved_img_folder, img_strip_prefix, img_probe = resolve_media_path_mapping(
                        candidate, img_files[:25], default_name=scfg["image_folder_name"]
                    )
                    if img_probe is not None:
                        log_ok(f"Using explicitly specified image folder: {resolved_img_folder}")

                if img_probe is None:
                    resolved_img_folder, img_strip_prefix, img_probe = resolve_media_folder(
                        base, scfg["image_folder_name"], img_files
                    )

                if img_probe is None:
                    log_warn(f"No valid probe image found for split '{split}'. Skipping image embedding for this split.")
                else:
                    embed_images(
                        file_list=img_files,
                        image_folder=resolved_img_folder,
                        tower=image_tower,
                        processor=image_proc,
                        h5_path=out / f"{split}_images.h5",
                        index_path=out / f"{split}_image_index.json",
                        device=device,
                        dtype=dtype,
                        batch_size=args.image_batch_size,
                        projector=projector,
                        staging_dir=staging_dir,
                        sync_interval_mins=args.sync_interval_mins,
                        images_per_shard=args.images_per_shard,
                        enable_sharding=not args.no_sharding,
                        strip_prefix=img_strip_prefix,
                        probe_path=img_probe,
                    )

        # 2. Videos
        if run_videos:
            _, vid_files = extract_file_lists(scfg["jsons"], modality="videos")
            if vid_files:
                resolved_vid_folder = None
                vid_strip_prefix = None
                vid_probe = None
                if args.video_folder:
                    candidate = Path(args.video_folder)
                    resolved_vid_folder, vid_strip_prefix, vid_probe = resolve_media_path_mapping(
                        candidate, vid_files[:25], default_name=scfg["video_folder_name"]
                    )
                    if vid_probe is not None:
                        log_ok(f"Using explicitly specified video folder: {resolved_vid_folder}")

                if vid_probe is None:
                    resolved_vid_folder, vid_strip_prefix, vid_probe = resolve_media_folder(
                        base, scfg["video_folder_name"], vid_files
                    )

                if vid_probe is None:
                    log_warn(f"No valid probe video found for split '{split}'. Skipping video embedding for this split.")
                else:
                    embed_videos(
                        file_list=vid_files,
                        video_folder=resolved_vid_folder,
                        tower=video_tower,
                        processor=video_proc,
                        h5_path=out / f"{split}_videos.h5",
                        index_path=out / f"{split}_video_index.json",
                        device=device,
                        dtype=dtype,
                        batch_size=args.video_batch_size,
                        projector=projector,
                        staging_dir=staging_dir,
                        sync_interval_mins=args.sync_interval_mins,
                        videos_per_shard=args.videos_per_shard,
                        enable_sharding=not args.no_sharding,
                        strip_prefix=vid_strip_prefix,
                        probe_path=vid_probe,
                    )

        # 3. Text
        if run_text:
            embed_text(
                json_paths=scfg["jsons"],
                tokenizer=tokenizer,
                embed_tokens_layer=embed_tokens_layer,
                vocab_weight=vocab_weight,
                hidden_size=hidden_size,
                h5_path=out / f"{split}_text.h5",
                index_path=out / f"{split}_text_index.json",
                device=device,
                dtype=dtype,
                batch_size=args.text_batch_size,
                staging_dir=staging_dir,
                sync_interval_mins=args.sync_interval_mins,
                save_raw_embeds=args.save_raw_text_embeddings,
            )

    # -- write meta -----------------------------------------------------------
    meta = {
        "image_tower": args.image_tower,
        "video_tower": args.video_tower,
        "model_name_or_path": args.model_name_or_path,
        "project_visual": args.project_visual,
        "visual_dim": 4096 if args.project_visual else 1024,
        "text_hidden_size": hidden_size,
        "storage_dtype": "float16",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "Visual features are in {split}_images.h5 and {split}_videos.h5. "
            "Textual features and embeddings are in {split}_text.h5. "
            "Pass --embed_cache_dir <dir> during training to bypass frozen encoders "
            "and embedding lookups."
        ),
    }
    meta_json_str = json.dumps(meta, indent=2)
    (out / "meta.json").write_text(meta_json_str)
    if staging_dir is not None and staging_dir.exists():
        (staging_dir / "meta.json").write_text(meta_json_str)
    log_ok(f"All done! Embeddings written to {out}")


if __name__ == "__main__":
    main()
