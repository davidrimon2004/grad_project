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

    if path.exists():
        h5 = h5py.File(str(path), "a")
        dset = h5["features"]
        written = int(h5.attrs.get("written", 0))
        log(f"  Resuming {path.name}: {written:,} / {dset.shape[0]:,} already done")
        return h5, dset, written

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

def _open_or_create_text_hdf5(path: Path, vocab_weight: np.ndarray, hidden_size: int):
    """
    Opens an existing text HDF5 file or creates an extensible one.
    Returns (h5_file, dset_embeds, dset_tokens, dset_labels, dset_offsets, written_samples, total_tokens).
    """
    try:
        import h5py
    except ImportError:
        log_err("h5py not installed. Run: pip install h5py")
        sys.exit(1)

    if path.exists():
        h5 = h5py.File(str(path), "a")
        dset_embeds = h5["text_embeddings"]
        dset_tokens = h5["token_ids"]
        dset_labels = h5["labels"]
        dset_offsets = h5["offsets"]
        written = int(h5.attrs.get("written_samples", 0))
        total_tokens = int(h5.attrs.get("total_tokens", 0))
        log(f"  Resuming {path.name}: {written:,} samples, {total_tokens:,} tokens already done")
        return h5, dset_embeds, dset_tokens, dset_labels, dset_offsets, written, total_tokens

    path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(str(path), "w")

    # Store full vocabulary embedding table once for easy offline lookup
    h5.create_dataset(
        "vocab_embeddings",
        data=vocab_weight.astype(np.float16),
        dtype=np.float16,
        compression="lzf",
    )

    # Extensible datasets
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
        chunks=(8192,),
        compression="lzf",
    )
    dset_labels = h5.create_dataset(
        "labels",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int32,
        chunks=(8192,),
        compression="lzf",
    )
    dset_offsets = h5.create_dataset(
        "offsets",
        shape=(1,),
        maxshape=(None,),
        dtype=np.int64,
        chunks=(4096,),
    )
    dset_offsets[0] = 0

    h5.attrs["written_samples"] = 0
    h5.attrs["total_tokens"] = 0
    h5.attrs["hidden_size"] = hidden_size
    log(f"  Created extensible {path.name}: hidden_size={hidden_size}")
    return h5, dset_embeds, dset_tokens, dset_labels, dset_offsets, 0, 0


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
    )
    log(f"Loading image tower: {model_name}")
    tower = LanguageBindImageTower(model_name, args=_TowerArgs(), cache_dir=None, delay_load=False)
    tower = tower.to(device=device, dtype=dtype).eval()
    for p in tower.parameters():
        p.requires_grad_(False)
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
    )
    log(f"Loading video tower: {model_name}")
    tower = LanguageBindVideoTower(model_name, args=_TowerArgs(), cache_dir=None, delay_load=False)
    tower = tower.to(device=device, dtype=dtype).eval()
    for p in tower.parameters():
        p.requires_grad_(False)
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
        # Check inside model_name_or_path
        for candidate in ["mm_projector.bin", "non_lora_trainables.bin"]:
            p = Path(model_name_or_path) / candidate
            if p.exists():
                log(f"Loading mm_projector from {p}")
                weights = torch.load(str(p), map_location="cpu")
                break

        # Check HF hub if model_name_or_path is a repo id
        if weights is None and "/" in model_name_or_path:
            try:
                from huggingface_hub import hf_hub_download
                for candidate in ["mm_projector.bin", "non_lora_trainables.bin"]:
                    try:
                        f = hf_hub_download(repo_id=model_name_or_path, filename=candidate)
                        weights = torch.load(f, map_location="cpu")
                        log(f"Downloaded mm_projector weights from {model_name_or_path}/{candidate}")
                        break
                    except Exception:
                        pass
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
):
    """
    Encodes images through LanguageBind (and optional mm_projector) to HDF5.
    """
    from PIL import Image as PILImage

    n_total = len(file_list)
    log_header(f"Embedding {n_total:,} images  ->  {h5_path.name}")

    # Probe feature shape
    for probe_rel in file_list[:10]:
        probe_path = image_folder / probe_rel
        if probe_path.exists():
            break
    else:
        log_err("Could not find any probe image in image_folder. Check paths.")
        return

    probe_img = PILImage.open(str(probe_path)).convert("RGB")
    probe_tensor = processor.preprocess(probe_img, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        probe_feat = tower(probe_tensor.to(device=device, dtype=dtype))
        if projector is not None:
            probe_feat = projector(probe_feat.to(next(projector.parameters()).dtype))
    feat_shape = tuple(probe_feat.shape[1:])
    log(f"  Feature shape per image: {feat_shape} (projected={projector is not None})")
    del probe_feat, probe_tensor

    h5, dset, written = _open_or_create_hdf5(h5_path, n_total, feat_shape)

    index: Dict[str, int] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())

    t0 = time.time()
    try:
        for batch_start in range(written, n_total, batch_size):
            batch_files = file_list[batch_start: batch_start + batch_size]
            tensors: List[torch.Tensor] = []
            valid_files: List[str] = []

            for rel_path in batch_files:
                full = image_folder / rel_path
                try:
                    img = PILImage.open(str(full)).convert("RGB")
                    pv = processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
                    tensors.append(pv)
                except Exception as exc:
                    log_warn(f"  Skipping {rel_path}: {exc}")
                    tensors.append(torch.zeros((3, 224, 224), dtype=dtype))
                valid_files.append(rel_path)

            batch_tensor = torch.stack(tensors).to(device=device, dtype=dtype)
            with torch.no_grad():
                feats = tower(batch_tensor)  # [B, N, D]
                if projector is not None:
                    feats = projector(feats.to(next(projector.parameters()).dtype))

            feats_np = feats.cpu().to(torch.float16).numpy().astype(np.float16)
            end_row = batch_start + len(valid_files)
            dset[batch_start:end_row] = feats_np

            for k, rel_path in enumerate(valid_files):
                index[rel_path] = batch_start + k

            written = end_row
            h5.attrs["written"] = written

            # Flush periodically
            if (batch_start // batch_size) % 500 == 499:
                h5.flush()
                index_path.write_text(json.dumps(index, separators=(",", ":")))

            elapsed = time.time() - t0
            pct = written / n_total * 100
            rate = written / elapsed if elapsed > 1 else 0
            eta_s = (n_total - written) / rate if rate > 0 else float("inf")
            print(
                f"\r  [{written:>7,}/{n_total:,}] {pct:5.1f}% | "
                f"{rate:6.0f} img/s | ETA {eta_s/3600:.1f}h",
                end="", flush=True,
            )
    finally:
        h5.flush()
        h5.close()
        index_path.write_text(json.dumps(index, separators=(",", ":")))
        print()

    elapsed = time.time() - t0
    log_ok(f"Done {written:,} images in {elapsed/3600:.2f}h  ->  {h5_path}")
    log_ok(f"Index saved  ->  {index_path}  ({len(index):,} entries)")


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
):
    """
    Encodes videos through LanguageBind (and optional mm_projector) to HDF5.
    """
    n_total = len(file_list)
    log_header(f"Embedding {n_total:,} videos  ->  {h5_path.name}")

    # Probe feature shape
    for probe_rel in file_list[:10]:
        probe_path = video_folder / probe_rel
        if probe_path.exists():
            break
    else:
        log_err("Could not find any probe video in video_folder. Check paths.")
        return

    probe_pv = processor(str(probe_path), return_tensors="pt")["pixel_values"].to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        probe_feat = tower(probe_pv)
        if projector is not None:
            probe_feat = projector(probe_feat.to(next(projector.parameters()).dtype))
    feat_shape = tuple(probe_feat.shape[1:])
    log(f"  Feature shape per video: {feat_shape} (projected={projector is not None})")
    del probe_feat, probe_pv

    h5, dset, written = _open_or_create_hdf5(h5_path, n_total, feat_shape)

    index: Dict[str, int] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())

    t0 = time.time()
    try:
        for batch_start in range(written, n_total, batch_size):
            batch_files = file_list[batch_start: batch_start + batch_size]
            pvs: List[torch.Tensor] = []
            valid_files: List[str] = []

            for rel_path in batch_files:
                full = str(video_folder / rel_path)
                try:
                    pv = processor(full, return_tensors="pt")["pixel_values"][0]
                    pvs.append(pv)
                except Exception as exc:
                    log_warn(f"  Skipping {rel_path}: {exc}")
                    pvs.append(torch.zeros((8, 3, 224, 224), dtype=dtype))
                valid_files.append(rel_path)

            batch_tensor = torch.stack(pvs).to(device=device, dtype=dtype)
            with torch.no_grad():
                feats = tower(batch_tensor)  # [B, T, N, D]
                if projector is not None:
                    feats = projector(feats.to(next(projector.parameters()).dtype))

            feats_np = feats.cpu().to(torch.float16).numpy().astype(np.float16)
            end_row = batch_start + len(valid_files)
            dset[batch_start:end_row] = feats_np

            for k, rel_path in enumerate(valid_files):
                index[rel_path] = batch_start + k

            written = end_row
            h5.attrs["written"] = written

            if (batch_start // batch_size) % 100 == 99:
                h5.flush()
                index_path.write_text(json.dumps(index, separators=(",", ":")))

            elapsed = time.time() - t0
            pct = written / n_total * 100
            rate = written / elapsed if elapsed > 1 else 0
            eta_s = (n_total - written) / rate if rate > 0 else float("inf")
            print(
                f"\r  [{written:>6,}/{n_total:,}] {pct:5.1f}% | "
                f"{rate:.2f} vid/s | ETA {eta_s/3600:.1f}h",
                end="", flush=True,
            )
    finally:
        h5.flush()
        h5.close()
        index_path.write_text(json.dumps(index, separators=(",", ":")))
        print()

    elapsed = time.time() - t0
    log_ok(f"Done {written:,} videos in {elapsed/3600:.2f}h  ->  {h5_path}")
    log_ok(f"Index saved  ->  {index_path}")


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
    embed_tokens_layer: torch.nn.Module,
    vocab_weight: np.ndarray,
    hidden_size: int,
    h5_path: Path,
    index_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 256,
):
    """
    Precomputes textual token embeddings, token_ids, labels, and offsets into an
    extensible HDF5 file. Fully resumable.
    """
    # Load all items across JSONs
    all_samples: List[Dict[str, Any]] = []
    for jp in json_paths:
        if not jp.exists():
            log_warn(f"JSON not found, skipping: {jp}")
            continue
        try:
            data = json.loads(jp.read_text())
            all_samples.extend(data)
        except Exception as exc:
            log_warn(f"Failed to parse {jp}: {exc}")

    n_samples = len(all_samples)
    log_header(f"Embedding text for {n_samples:,} conversation samples  ->  {h5_path.name}")

    h5, dset_embeds, dset_tokens, dset_labels, dset_offsets, written, total_tokens = (
        _open_or_create_text_hdf5(h5_path, vocab_weight, hidden_size)
    )

    index: Dict[str, Any] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except Exception:
            index = {}

    t0 = time.time()
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

            # Concatenate token IDs and look up embeddings on GPU
            flat_tokens = np.concatenate(batch_token_ids_list)
            flat_labels = np.concatenate(batch_labels_list)
            num_tokens_in_batch = len(flat_tokens)

            # Clamp negative token ids (like -200) to 0 for embedding lookup
            lookup_tokens = np.where(flat_tokens >= 0, flat_tokens, 0)
            tokens_tensor = torch.from_numpy(lookup_tokens).to(device=device, dtype=torch.long)

            with torch.no_grad():
                embeds = embed_tokens_layer(tokens_tensor)  # [num_tokens, hidden_size]
                # Zero out embeddings for image tokens (-200)
                mask_non_text = torch.from_numpy(flat_tokens < 0).to(device=device)
                if mask_non_text.any():
                    embeds[mask_non_text] = 0

            embeds_np = embeds.cpu().to(torch.float16).numpy().astype(np.float16)

            # Append to HDF5
            cur_tokens = dset_embeds.shape[0]
            new_tokens = cur_tokens + num_tokens_in_batch
            dset_embeds.resize((new_tokens, hidden_size))
            dset_embeds[cur_tokens:new_tokens] = embeds_np

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

            # Flush periodically
            if (batch_start // batch_size) % 100 == 99:
                h5.flush()
                index_path.write_text(json.dumps(index, separators=(",", ":")))

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
        index_path.write_text(json.dumps(index, separators=(",", ":")))
        print()

    elapsed = time.time() - t0
    log_ok(f"Done {written:,} samples ({total_tokens:,} tokens) in {elapsed/3600:.2f}h  ->  {h5_path}")
    log_ok(f"Index saved  ->  {index_path}")


# ──────────────────────────────────────────────────────────────────────────────
# File extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_file_lists(json_paths: List[Path]) -> Tuple[List[str], List[str]]:
    """
    Parses data JSON files and returns (image_files, video_files) as
    deduplicated lists of relative paths.
    """
    seen_images: Dict[str, int] = {}
    seen_videos: Dict[str, int] = {}

    for jp in json_paths:
        if not jp.exists():
            log_warn(f"JSON not found, skipping: {jp}")
            continue
        try:
            data = json.loads(jp.read_text())
        except Exception as exc:
            log_warn(f"Failed to parse {jp}: {exc}")
            continue

        for item in data:
            if "image" in item and "video" not in item:
                files = item["image"] if isinstance(item["image"], list) else [item["image"]]
                for f in files:
                    if f not in seen_images:
                        seen_images[f] = len(seen_images)
            elif "video" in item:
                files = item["video"] if isinstance(item["video"], list) else [item["video"]]
                for f in files:
                    if f not in seen_videos:
                        seen_videos[f] = len(seen_videos)

    image_files = sorted(seen_images, key=seen_images.__getitem__)
    video_files = sorted(seen_videos, key=seen_videos.__getitem__)
    return image_files, video_files


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
    out = Path(args.output_dir) if args.output_dir else base / "embeddings"
    out.mkdir(parents=True, exist_ok=True)

    datasets_dir = base / "datasets"

    cfg = {
        "pretrain": {
            "jsons": [
                datasets_dir / "pt_json" / "llava_image_.json",
                datasets_dir / "pt_json" / "valley_.json",
            ],
            "image_folder": datasets_dir / "llava_image",
            "video_folder": datasets_dir / "valley",
        },
        "finetune": {
            "jsons": [
                datasets_dir / "ft_json" / "llava_image_tune_.json",
                datasets_dir / "ft_json" / "videochatgpt_.json",
                datasets_dir / "ft_json" / "nlp_tune.json",
            ],
            "image_folder": datasets_dir / "llava_image_tune",
            "video_folder": datasets_dir / "videochatgpt_tune",
        },
    }

    splits_to_run = ["pretrain", "finetune"] if args.split == "all" else [args.split]

    run_images = args.action in ["images", "visual", "all"]
    run_videos = args.action in ["videos", "visual", "all"]
    run_text = args.action in ["text", "all"]

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

        # 1. Images
        if run_images:
            img_files, _ = extract_file_lists(scfg["jsons"])
            if img_files:
                embed_images(
                    file_list=img_files,
                    image_folder=scfg["image_folder"],
                    tower=image_tower,
                    processor=image_proc,
                    h5_path=out / f"{split}_images.h5",
                    index_path=out / f"{split}_image_index.json",
                    device=device,
                    dtype=dtype,
                    batch_size=args.image_batch_size,
                    projector=projector,
                )

        # 2. Videos
        if run_videos:
            _, vid_files = extract_file_lists(scfg["jsons"])
            if vid_files:
                embed_videos(
                    file_list=vid_files,
                    video_folder=scfg["video_folder"],
                    tower=video_tower,
                    processor=video_proc,
                    h5_path=out / f"{split}_videos.h5",
                    index_path=out / f"{split}_video_index.json",
                    device=device,
                    dtype=dtype,
                    batch_size=args.video_batch_size,
                    projector=projector,
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
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log_ok(f"All done! Embeddings written to {out}")


if __name__ == "__main__":
    main()
