# Video-LLaVA Project Handover & Run Guide

**Project Directory:** `/home/aitech/hdd/grad_project_david/grad_project_david`  
**Date:** November 4, 2026  
**Hardware Environment:** 3× NVIDIA GeForce RTX 4060 Ti (16 GB VRAM each), Intel Core i9-13900K (`no_turbo=1`), 94 GB RAM  
**Conda Environment:** `videollava` (`/home/aitech/miniconda3/envs/videollava`)

---

## 1. Executive Summary

In this session, we:
1. **Audited and revised code changes** from the *"Video-LLaVA Pretraining: Issues and Solutions"* report, fixing outstanding technical debt, broken guards, and dead code.
2. **Transitioned the entire pipeline from mixed video+image to strictly image-only**:
   - Filtered out all video samples (eliminating missing video read errors and video decoding bottlenecks).
   - Disabled the LanguageBind video tower, freeing **~1.8 GB VRAM per GPU**.
   - Connected the official pretrained projector weights (`/home/aitech/hdd/videollava_models/mm_projector.bin`).
3. **Configured robust hardware safety**:
   - Lowered `tokenizer_model_max_length` from 3072 to **2048** (preventing VRAM allocation spikes).
   - Set `per_device_train_batch_size = 1` and `gradient_accumulation_steps = 16` (effective batch size = 48 across 3 GPUs).
   - VRAM usage is capped at **~9.5–9.7 GB** per GPU, leaving **>6 GB of headroom** on all 3 cards (zero OOM risk).
4. **Stopped the stalled mixed-modality process** (which was taking ~210s/step) and **launched the clean image-only pretraining job** (`PID 147185`).

---

## 2. Detailed Summary of Changes Made

### A. Code Corrections & Bug Fixes
* **[`videollava/model/multimodal_encoder/builder.py`](file:///home/aitech/hdd/grad_project_david/grad_project_david/videollava/model/multimodal_encoder/builder.py)**:
  - Added `if image_tower is None: return None` and `if video_tower is None: return None` checks to prevent `AttributeError: 'NoneType' object has no attribute 'endswith'` when omitting `video_tower`.
* **[`videollava/model/language_model/llava_llama.py`](file:///home/aitech/hdd/grad_project_david/grad_project_david/videollava/model/language_model/llava_llama.py)**:
  - Removed persistent debug hooks (`DBG_DTYPE` forward hooks on `q_proj`), eliminating unnecessary forward-pass overhead.
* **[`videollava/model/llava_arch.py`](file:///home/aitech/hdd/grad_project_david/grad_project_david/videollava/model/llava_arch.py)**:
  - Added `self.early_fusion_kv_ln = nn.LayerNorm(config.hidden_size)` directly inside `LlavaMetaModel.__init__` so that checkpoint loading and model initialization always contain all early-fusion submodules.
  - Added `hasattr(config, "mm_hidden_size")` guard to avoid `AttributeError` when instantiating from raw backbone configs prior to vision module initialization.
* **[`videollava/train/train.py`](file:///home/aitech/hdd/grad_project_david/grad_project_david/videollava/train/train.py)**:
  - Fixed dead code in `preprocess_multimodal` where `raise ValueError` preceded a replace statement.
  - In `LazySupervisedDataset.__init__`, added automatic filtering of any entry containing `'video'`, logging the filtered count and ensuring only clean image/text data is used.
  - Updated `modality_lengths` property to only check for `'image'`.
* **Repository Cleanup**:
  - Deleted 9 leftover `.bak` and `.orig` files.
  - Created [`.gitattributes`](file:///home/aitech/hdd/grad_project_david/grad_project_david/.gitattributes) to enforce Unix line endings (`eol=lf`) across all `.py`, `.sh`, `.json`, and `.md` files.

### B. Training Scripts Updated
* **[`scripts/v1_5/pretrain.sh`](file:///home/aitech/hdd/grad_project_david/grad_project_david/scripts/v1_5/pretrain.sh)**:
  - Removed `valley_.json` (700k video dataset) $\to$ now only trains on `llava_image_.json` (558,128 samples).
  - Removed `--video_folder` and `--video_tower`.
  - Added `--pretrain_mm_mlp_adapter /home/aitech/hdd/videollava_models/mm_projector.bin` to initialize directly from the official pretrained projector weights.
  - Configured `--save_steps 250` (saves checkpoint every ~5–8 minutes).
  - Set `--model_max_length 2048` and `--tokenizer_model_max_length 2048`.
  - Set default `PER_DEVICE_BATCH=1` and `GRAD_ACCUM=16` with DeepSpeed ZeRO-3 (`scripts/zero3.json`).
* **[`scripts/v1_5/finetune.sh`](file:///home/aitech/hdd/grad_project_david/grad_project_david/scripts/v1_5/finetune.sh) & [`finetune_lora.sh`](file:///home/aitech/hdd/grad_project_david/grad_project_david/scripts/v1_5/finetune_lora.sh)**:
  - Removed `videochatgpt_tune_.json`. Kept `llava_image_tune_.json` and `nlp_tune.json`.
  - Removed `--video_folder` and `--video_tower`.
  - Applied the same memory-safe batch and sequence length parameters (`max_length=2048`, `batch=1`, `accum=16`).
* **[`/home/aitech/run_pretrain.sh`](file:///home/aitech/run_pretrain.sh)**:
  - Fully self-contained launcher updated to invoke `scripts/v1_5/pretrain.sh` directly with all NCCL exports, logging to `/home/aitech/pretrain_log_images.txt`.

---

## 3. Current Live Status

* **Process ID**: `PID 147185` (running `bash /home/aitech/run_pretrain.sh`)
* **Total Dataset**: 558,128 image samples
* **Total Steps**: **11,627 steps** (1 epoch with effective batch size 48)
* **GPU Utilization**: 100% on GPU 0, 1, 2
* **VRAM Usage**: ~9.5–9.7 GB / 16.3 GB per GPU (~6.6 GB safety headroom)
* **GPU Temperatures**: Stable at 48°C – 56°C

---

## 4. Important Commands Reference

### Monitoring Commands
```bash
# 1. Watch live training steps, loss, and learning rate
tail -f /home/aitech/pretrain_log_images.txt

# 2. Monitor GPU utilization, VRAM, and temperatures
watch -n 2 nvidia-smi

# 3. Check saved checkpoints
ls -lh /home/aitech/hdd/grad_project_david/grad_project_david/checkpoints/videollava-7b-pretrain/

# 4. View pretrain status logs
cat /home/aitech/pretrain_status.txt
```

### Process Management Commands
```bash
# Stop training at any time
pkill -f train_mem.py

# Relaunch pretraining (runs in background with nohup)
nohup bash /home/aitech/run_pretrain.sh > /dev/null 2>&1 &

# Relaunch pretraining (interactive foreground in terminal)
bash /home/aitech/run_pretrain.sh
```

*(Note: If training is ever stopped, re-running `run_pretrain.sh` automatically detects the latest `checkpoint-<step>` and resumes training seamlessly).*

---

## 5. Next Steps

### Phase 1: Monitor Pretraining Completion
* Let the current run proceed through the 11,627 steps.
* Checkpoints are written every **250 steps** to:
  `/home/aitech/hdd/grad_project_david/grad_project_david/checkpoints/videollava-7b-pretrain/checkpoint-<step>/`
* Upon completion (epoch 1.0), the final weights will be stored at:
  `/home/aitech/hdd/grad_project_david/grad_project_david/checkpoints/videollava-7b-pretrain/mm_projector.bin`

### Phase 2: Stage 2 Fine-Tuning
Once pretraining finishes, launch Stage 2 fine-tuning on the image instruction dataset:
```bash
cd /home/aitech/hdd/grad_project_david/grad_project_david
source /home/aitech/miniconda3/etc/profile.d/conda.sh
conda activate videollava

# Full fine-tuning (with DeepSpeed ZeRO-3 offload)
bash scripts/v1_5/finetune.sh

# OR LoRA fine-tuning (faster, parameter-efficient)
bash scripts/v1_5/finetune_lora.sh
```

### Phase 3: Validation & Benchmarking
Run the evaluation scripts on standard multimodal benchmarks:
```bash
# E.g. ScienceQA
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/eval_image_sqa.sh

# E.g. POPE
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/eval_image_pope.sh

# E.g. TextVQA
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/eval_image_textvqa.sh
```

