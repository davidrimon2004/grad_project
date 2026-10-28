#!/usr/bin/env python3
"""
smoke_test.py — Video-LLaVA Training Pipeline Smoke Test
=========================================================
Tests the full forward + backward pass WITHOUT DeepSpeed or real data.
Catches NaN losses, broken gradients, and config/import errors
BEFORE you commit to a multi-hour training run on the workstation.

Usage (single GPU, from project root):
    python scripts/smoke_test.py

Options:
    --model_path   HF model id or local path  (default: lmsys/vicuna-7b-v1.5)
    --image_tower  LanguageBind image tower    (default: LanguageBind/LanguageBind_Image)
    --video_tower  LanguageBind video tower    (default: LanguageBind/LanguageBind_Video_merge)
    --steps        Number of training steps    (default: 10)
    --batch_size   Samples per step            (default: 2)
    --device       cuda device index           (default: 0)
    --pretrain     Test pretrain mode (default: True)
    --finetune     Also test finetune mode
    --lora         Wrap model in LoRA for finetune test
    --cache_dir    HF cache directory          (default: ./cache_dir)
    --no_video     Skip video modality tests (faster)
    --fp16         Use fp16 instead of bf16
"""

import os
import sys
import math
import time
import argparse
import json
import traceback
from pathlib import Path

# ── Ensure project root is on path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

# ───────────────────────────────────────────────────────────────────────────
# Console helpers
# ───────────────────────────────────────────────────────────────────────────
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[96m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
SEP  = "=" * 70

_results: list = []  # list of (name, ok, msg)


def check(name: str, ok: bool, msg: str = "") -> bool:
    tag  = PASS if ok else FAIL
    line = f"  {tag}  {name}"
    if msg:
        line += f"  -- {msg}"
    print(line)
    _results.append((name, ok, msg))
    return ok


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def assert_finite(tensor: torch.Tensor, name: str = "tensor") -> bool:
    if tensor is None:
        return check(f"{name} is not None", False, "Got None")
    nan = torch.isnan(tensor).any().item()
    inf = torch.isinf(tensor).any().item()
    ok  = (not nan) and (not inf)
    msg = ("NaN! " if nan else "") + ("Inf!" if inf else "")
    return check(f"{name} is finite", ok, msg.strip() if not ok else "")


# ───────────────────────────────────────────────────────────────────────────
# Fake tensor helpers (no real images/videos needed)
# ───────────────────────────────────────────────────────────────────────────
def fake_image(dtype=torch.bfloat16) -> torch.Tensor:
    """[3, 224, 224] fake image pixel tensor."""
    return torch.randn(3, 224, 224, dtype=dtype)


def fake_video(T=8, dtype=torch.bfloat16) -> torch.Tensor:
    """[T, 3, 224, 224] fake video tensor (LanguageBind frame layout)."""
    return torch.randn(T, 3, 224, 224, dtype=dtype)


# ───────────────────────────────────────────────────────────────────────────
# Forward / backward loop
# ───────────────────────────────────────────────────────────────────────────
def run_training_loop(
    model,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    steps: int,
    mode_label: str,
    include_video: bool = True,
) -> bool:
    """
    Run `steps` synthetic training steps.
    Returns True only if EVERY step produced a finite, non-zero loss.
    """
    section(f"Forward/Backward — {mode_label}")

    from videollava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
    from videollava.train.train import DataCollatorForSupervisedDataset
    from videollava.mm_utils import tokenizer_image_token

    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        check("Has trainable parameters", False, "No params have requires_grad=True!")
        return False

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    T = 8   # LanguageBind default frames
    all_ok = True

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(steps):
        optimizer.zero_grad()

        # ── Build synthetic batch ────────────────────────────────────────────
        instances = []
        for b in range(batch_size):
            use_video = include_video and (b % 2 == 1)

            if use_video:
                prompt = DEFAULT_VIDEO_TOKEN + "\nWhat happens here?"
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\nDescribe this image."

            input_ids = tokenizer_image_token(
                prompt, tokenizer, return_tensors="pt"
            ).squeeze(0)

            answer     = "Smoke test answer for step " + str(step) + "."
            answer_ids = tokenizer(answer, return_tensors="pt").input_ids.squeeze(0)

            full_ids    = torch.cat([input_ids, answer_ids])
            full_labels = torch.cat([
                torch.full_like(input_ids, IGNORE_INDEX),
                answer_ids,
            ])

            img_tensor = fake_video(T=T, dtype=dtype) if use_video else fake_image(dtype=dtype)

            instances.append({
                "input_ids": full_ids,
                "labels":    full_labels,
                "image":     [img_tensor],
            })

        # ── Collate ──────────────────────────────────────────────────────────
        collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        batch = collator(instances)

        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images         = [img.to(device) for img in batch["images"]]

        # ── Forward ──────────────────────────────────────────────────────────
        try:
            with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=images,
                )
            loss = outputs.loss
        except Exception as exc:
            check(f"Step {step+1:02d} forward", False, repr(exc))
            traceback.print_exc()
            all_ok = False
            continue

        # ── Loss checks ──────────────────────────────────────────────────────
        if not assert_finite(loss, f"Step {step+1:02d} loss"):
            all_ok = False
            continue

        loss_val = loss.item()

        if math.isnan(loss_val) or math.isinf(loss_val):
            check(f"Step {step+1:02d} loss numeric", False, f"loss={loss_val}")
            all_ok = False
            continue

        if loss_val == 0.0:
            check(f"Step {step+1:02d} loss > 0", False,
                  "Loss=0.0 — all labels may be IGNORE_INDEX masked")
            all_ok = False

        # ── Backward ─────────────────────────────────────────────────────────
        try:
            loss.backward()
        except Exception as exc:
            check(f"Step {step+1:02d} backward", False, repr(exc))
            traceback.print_exc()
            all_ok = False
            continue

        # Spot-check gradients on first 5 trainable params
        for n, p in [(n, p) for n, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None][:5]:
            if not assert_finite(p.grad, f"  grad/{n}"):
                all_ok = False

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mem_info = ""
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            mem_info = f" | Peak VRAM: {peak_gb:.2f} GB"
        print(f"         loss = {loss_val:.6f}{mem_info}")
        check(f"Step {step+1:02d}", True, f"loss={loss_val:.4f}{mem_info}")

    return all_ok


# ───────────────────────────────────────────────────────────────────────────
# Fast Architectural Unit Test (No weight downloads required)
# ───────────────────────────────────────────────────────────────────────────
def run_fast_unit_test(dtype=torch.bfloat16):
    """
    Validates Gate v2, cross-attention fusion, multimodal input preparation,
    loss computation, and backward gradient flow WITHOUT needing HuggingFace model weights.
    """
    section("FAST ARCHITECTURAL UNIT TEST — Gate v2 & Multimodal Architecture")
    from unittest.mock import MagicMock
    if "decord" not in sys.modules:
        sys.modules['decord'] = MagicMock()
        sys.modules['decord.bridge'] = MagicMock()

    from videollava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from videollava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

    class _DummyConfig:
        hidden_size = 64
        mha_fusion_heads = 4
        bf16 = (dtype == torch.bfloat16)
        fp16 = (dtype == torch.float16)
        tokenizer_model_max_length = 512
        tokenizer_padding_side = "right"

    class _DummySubModel(nn.Module, LlavaMetaModel):
        def __init__(self, config):
            nn.Module.__init__(self)
            self.config = config
            self.embed_tokens = nn.Embedding(1000, config.hidden_size)
            self.early_fusion_attn = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.mha_fusion_heads,
                batch_first=True
            )
            self.early_fusion_ln = nn.LayerNorm(config.hidden_size)
            self.early_fusion_kv_ln = nn.LayerNorm(config.hidden_size)
            self.early_fusion_gate_linear = nn.Linear(2 * config.hidden_size, 1)
            nn.init.zeros_(self.early_fusion_gate_linear.weight)
            nn.init.zeros_(self.early_fusion_gate_linear.bias)

    class _DummyTower(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
        def forward(self, x):
            return x

    class _DummyLlavaModel(nn.Module, LlavaMetaForCausalLM):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = _DummySubModel(config)
            self.image_tower = _DummyTower(config.hidden_size)

        def get_model(self):
            return self.model

        def get_image_tower(self):
            return self.image_tower

        def get_video_tower(self):
            return None

        def encode_images(self, images):
            return torch.randn(len(images), 8, self.config.hidden_size, dtype=dtype, requires_grad=True)

    cfg = _DummyConfig()
    model = _DummyLlavaModel(cfg).to(dtype=dtype)

    # 1. Gate v2 Step 0 Identity Check
    Nt, Nv, d = 12, 36, cfg.hidden_size
    text_embeds = torch.randn(Nt, d, dtype=dtype, requires_grad=True)
    visual_embeds = torch.randn(Nv, d, dtype=dtype, requires_grad=True)

    fused = model.cross_attention_fuse(text_embeds, visual_embeds)
    diff = (fused - text_embeds).abs().max().item()
    check("Gate v2 init identity (output == text at step 0)", diff == 0.0, f"diff={diff}")

    # 2. Gate v2 Gradients
    (fused.sum()).backward()
    gl = model.get_model().early_fusion_gate_linear
    w_ok = gl.weight.grad is not None
    b_ok = gl.bias.grad is not None
    check("Gate v2 weight gradient exists", w_ok)
    check("Gate v2 bias gradient exists", b_ok)
    if w_ok:
        assert_finite(gl.weight.grad, "Gate v2 weight grad")
    if b_ok:
        assert_finite(gl.bias.grad, "Gate v2 bias grad")

    # 3. Multimodal Input Packing & Loss
    input_ids = torch.tensor([
        [10, 11, IMAGE_TOKEN_INDEX, 12, 13, 14],
        [IMAGE_TOKEN_INDEX, 20, 21, 22, 0, 0]
    ], dtype=torch.long)
    labels = torch.tensor([
        [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 12, 13, 14],
        [IGNORE_INDEX, 20, 21, 22, IGNORE_INDEX, IGNORE_INDEX]
    ], dtype=torch.long)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0]
    ], dtype=torch.long)
    images = [
        torch.randn(3, 224, 224, dtype=dtype),
        torch.randn(3, 224, 224, dtype=dtype)
    ]

    res = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=labels,
        images=images
    )
    (input_ids_out, position_ids, attention_mask_out, past_key_values, inputs_embeds, labels_out) = res
    check("Multimodal input packing succeeds", inputs_embeds is not None and inputs_embeds.shape[0] == 2)

    head = nn.Linear(cfg.hidden_size, 1000).to(dtype=dtype)
    logits = head(inputs_embeds)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loss = loss_fn(logits.view(-1, 1000), labels_out.view(-1))
    check("Synthetic loss finite & positive", not torch.isnan(loss) and not torch.isinf(loss) and loss.item() > 0,
          f"loss={loss.item():.4f}")
    loss.backward()
    check("End-to-end multimodal backward pass succeeds", True)
    _print_summary()


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Video-LLaVA smoke test")
    parser.add_argument("--model_path",  default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--image_tower", default="LanguageBind/LanguageBind_Image")
    parser.add_argument("--video_tower", default="LanguageBind/LanguageBind_Video_merge")
    parser.add_argument("--steps",       type=int,  default=10)
    parser.add_argument("--batch_size",  type=int,  default=2)
    parser.add_argument("--device",      type=int,  default=0)
    parser.add_argument("--pretrain",    action="store_true", default=True)
    parser.add_argument("--finetune",    action="store_true", default=False)
    parser.add_argument("--lora",        action="store_true", default=False)
    parser.add_argument("--cache_dir",   default="./cache_dir")
    parser.add_argument("--no_video",    action="store_true", default=False)
    parser.add_argument("--fp16",        action="store_true", default=False)
    parser.add_argument("--fast_unit_test", action="store_true", default=False,
                        help="Run lightning-fast architectural & Gate v2 unit tests without downloading weights")
    args = parser.parse_args()

    dtype  = torch.float16 if args.fp16 else torch.bfloat16
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    if args.fast_unit_test:
        run_fast_unit_test(dtype=dtype)
        return

    # ── 1. Environment ────────────────────────────────────────────────────────
    section("1 — Environment")
    print(f"  {INFO}  Python  : {sys.version.split()[0]}")
    print(f"  {INFO}  PyTorch : {torch.__version__}")
    print(f"  {INFO}  Device  : {device}  |  dtype={dtype}")

    cuda_ok = torch.cuda.is_available()
    check("CUDA available", cuda_ok,
          f"{torch.cuda.device_count()} GPU(s)" if cuda_ok else "No GPU — CPU only (very slow)")
    if cuda_ok:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  {INFO}  GPU {i}: {props.name}  "
                  f"({props.total_memory / 1e9:.1f} GB)")

    for pkg in ["transformers", "deepspeed", "peft", "languagebind", "decord", "flash_attn"]:
        try:
            mod = __import__(pkg)
            check(f"import {pkg}", True, getattr(mod, "__version__", "ok"))
        except ImportError as e:
            ok = (pkg == "flash_attn")  # flash_attn is optional
            check(f"import {pkg}", ok,
                  ("optional — training will still work" if ok else str(e)))

    # ── 2. Project imports ────────────────────────────────────────────────────
    section("2 — Project imports")
    try:
        from videollava.model import LlavaLlamaForCausalLM  # noqa: F401
        check("import LlavaLlamaForCausalLM", True)
    except Exception as e:
        check("import LlavaLlamaForCausalLM", False, str(e))
        traceback.print_exc()
        _print_summary()
        return

    try:
        from videollava.train.train import DataCollatorForSupervisedDataset  # noqa: F401
        from videollava.mm_utils import tokenizer_image_token               # noqa: F401
        check("import training utilities", True)
    except Exception as e:
        check("import training utilities", False, str(e))
        traceback.print_exc()

    # ── 3. Tokenizer ──────────────────────────────────────────────────────────
    section("3 — Tokenizer")
    try:
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        check("Tokenizer loaded", True, f"vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        check("Tokenizer loaded", False, str(e))
        traceback.print_exc()
        _print_summary()
        return

    # ── 4. Model load ─────────────────────────────────────────────────────────
    section("4 — Model load  (this downloads weights on first run)")
    try:
        from videollava.model import LlavaLlamaForCausalLM
        model = LlavaLlamaForCausalLM.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            torch_dtype=dtype,
        )
        model.config.use_cache = False
        total_params = sum(p.numel() for p in model.parameters()) / 1e9
        check("Base model loaded", True, f"{total_params:.2f}B params")
    except Exception as e:
        check("Base model loaded", False, str(e))
        traceback.print_exc()
        _print_summary()
        return

    # ── 5. Vision towers ──────────────────────────────────────────────────────
    section("5 — Vision towers")

    class _ModelArgs:
        image_tower             = args.image_tower
        video_tower             = None if args.no_video else args.video_tower
        mm_vision_select_layer  = -2
        mm_vision_select_feature = "patch"
        pretrain_mm_mlp_adapter = None
        mm_projector_type       = "mlp2x_gelu"
        mm_use_im_start_end     = False
        mm_use_im_patch_token   = False

    try:
        model.get_model().initialize_vision_modules(
            model_args=_ModelArgs(), fsdp=None
        )
        check("initialize_vision_modules", True)
    except Exception as e:
        check("initialize_vision_modules", False, str(e))
        traceback.print_exc()
        _print_summary()
        return

    # Move image tower and fix HF 4.45+ position_ids bug
    try:
        image_tower = model.get_image_tower()
        image_tower.to(dtype=dtype, device=device)
        sub = getattr(image_tower, "image_tower", image_tower)
        emb = getattr(sub, "embeddings", None)
        if emb is not None and hasattr(emb, "position_embedding"):
            n = emb.position_embedding.weight.shape[0]
            emb.position_ids = torch.arange(n, device=device).unsqueeze(0)
        check("Image tower on device", True)
    except Exception as e:
        check("Image tower on device", False, str(e))

    if not args.no_video:
        try:
            video_tower = model.get_video_tower()
            video_tower.to(dtype=dtype, device=device)
            sub = getattr(video_tower, "video_tower", video_tower)
            emb = getattr(sub, "embeddings", None)
            if emb is not None and hasattr(emb, "position_embedding"):
                n = emb.position_embedding.weight.shape[0]
                emb.position_ids = torch.arange(n, device=device).unsqueeze(0)
            check("Video tower on device", True)
        except Exception as e:
            check("Video tower on device", False, str(e))

    # Set the config flags that the real trainer sets
    model.config.bf16 = (dtype == torch.bfloat16)
    model.config.fp16 = (dtype == torch.float16)
    model.config.image_aspect_ratio       = "square"
    model.config.tokenizer_padding_side   = "right"
    model.config.tokenizer_model_max_length = 512
    model.config.mm_use_im_start_end      = False
    model.config.mm_use_im_patch_token    = False

    # Verify custom early_fusion components
    for attr in ["early_fusion_attn", "early_fusion_ln",
                 "early_fusion_kv_ln", "early_fusion_gate_linear"]:
        check(f"model.{attr} exists", hasattr(model.get_model(), attr))

    # ── 6. Pretrain mode ──────────────────────────────────────────────────────
    if args.pretrain:
        section("6 — PRETRAIN MODE  (mm_projector + early_fusion trainable)")

        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        for name, param in model.get_model().named_parameters():
            if "early_fusion" in name:
                param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        check("Trainable params > 0", trainable > 0,
              f"{trainable/1e6:.1f}M / {total/1e9:.2f}B")

        model.to(device)
        if cuda_ok:
            torch.cuda.empty_cache()

        ok = run_training_loop(
            model, tokenizer, device, dtype,
            batch_size=args.batch_size,
            steps=args.steps,
            mode_label="PRETRAIN",
            include_video=(not args.no_video),
        )
        check("Pretrain: all steps finite", ok)

        # Verify gate_linear gradients (Gate v2)
        gate_linear = getattr(model.get_model(), "early_fusion_gate_linear", None)
        if gate_linear is not None:
            w_grad = gate_linear.weight.grad
            b_grad = gate_linear.bias.grad
            w_ok = w_grad is not None
            b_ok = b_grad is not None
            check("early_fusion_gate_linear.weight has gradient", w_ok)
            check("early_fusion_gate_linear.bias has gradient",   b_ok)
            if w_ok:
                assert_finite(w_grad, "gate_linear.weight.grad")
            if b_ok:
                assert_finite(b_grad, "gate_linear.bias.grad")

    # ── 7. Finetune mode ──────────────────────────────────────────────────────
    if args.finetune:
        section("7 — FINETUNE MODE" + (" + LoRA" if args.lora else " (full)"))

        model.requires_grad_(True)

        if args.lora:
            try:
                from peft import LoraConfig, get_peft_model
                from videollava.train.train import find_all_linear_names
                lora_cfg = LoraConfig(
                    r=16, lora_alpha=32,
                    target_modules=find_all_linear_names(model),
                    lora_dropout=0.05, bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_cfg)
                model.print_trainable_parameters()
                check("LoRA applied", True)
            except Exception as e:
                check("LoRA applied", False, str(e))
                traceback.print_exc()

        model.to(device)
        if cuda_ok:
            torch.cuda.empty_cache()

        ok = run_training_loop(
            model, tokenizer, device, dtype,
            batch_size=args.batch_size,
            steps=args.steps,
            mode_label="FINETUNE" + (" + LoRA" if args.lora else " (full)"),
            include_video=(not args.no_video),
        )
        check("Finetune: all steps finite", ok)

    # ── 8. GPU memory & OOM Headroom Analysis ─────────────────────────────────
    section("8 — GPU Memory & OOM Safety Analysis")
    if cuda_ok:
        for i in range(torch.cuda.device_count()):
            alloc    = torch.cuda.memory_allocated(i) / (1024 ** 3)
            peak     = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
            reserv   = torch.cuda.memory_reserved(i)  / (1024 ** 3)
            total    = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            headroom = total - peak
            pct      = 100 * peak / total
            print(f"  {INFO}  GPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"         Peak VRAM : {peak:.2f} GB / {total:.2f} GB ({pct:.1f}%)")
            print(f"         Headroom  : {headroom:.2f} GB available")
            print(f"         Reserved  : {reserv:.2f} GB | Current alloc: {alloc:.2f} GB")

            if headroom < 1.0:
                check(f"GPU {i} VRAM OOM safety (<90%)", False,
                      f"CRITICAL: Only {headroom:.2f} GB headroom left! High risk of CUDA OOM on variable sequence lengths.")
            elif headroom < 2.5:
                check(f"GPU {i} VRAM OOM safety", True,
                      f"WARNING: {headroom:.2f} GB headroom. Safe for tested batch size ({args.batch_size}), but reduce batch size if OOM occurs on long videos.")
            else:
                check(f"GPU {i} VRAM OOM safety (>2.5 GB headroom)", True,
                      f"SAFE: {headroom:.2f} GB headroom remaining.")

    _print_summary()


def _print_summary():
    section("SUMMARY")
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    for name, ok, msg in _results:
        tag    = PASS if ok else FAIL
        suffix = f"  -- {msg}" if (not ok and msg) else ""
        print(f"  {tag}  {name}{suffix}")
    print()
    if failed == 0:
        print(f"\033[92m  [OK]  ALL {passed} CHECKS PASSED -- Safe to start training!\033[0m\n")
    else:
        print(f"\033[91m  [FAIL]  {failed}/{passed+failed} CHECKS FAILED -- "
              f"Fix issues before training!\033[0m\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
