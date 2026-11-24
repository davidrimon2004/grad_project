#!/usr/bin/env python3
"""
Turnkey POPE Benchmark Evaluation & Multi-Checkpoint Comparison Pipeline for Google Colab.

Supports comparing any combination of:
  - Untrained (Step 0): vicuna-7b-v1.5 + LanguageBind_Image + random mm_projector + tanh(g)=0
  - final (Stage 1, step 867)
  - checkpoint-4750 (Stage 1, step 4750)
  - Official LanguageBind/Video-LLaVA-7B (Stage 2 Baseline)

Usage:
  python scripts/eval_pope_colab.py --action prepare
  python scripts/eval_pope_colab.py --action download_ckpt --ckpt_name all
  python scripts/eval_pope_colab.py --action inspect_ckpt --ckpt_name all
  python scripts/eval_pope_colab.py --action eval --model_path ... --model_base ... [--untrained]
  python scripts/eval_pope_colab.py --action compare --metrics_files m1.json m2.json m3.json
  python scripts/eval_pope_colab.py --action all
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


CHECKPOINT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "final": {
        "folder_id": "1H4L3d9FT2l5HrejUZ2pYbLupNeh4fEr2",
        "drive_subpath": "final",
        "label": "final (Stage 1, step 867)",
        "files": {
            "config.json": "13eK91eIFA-nZFx7_AZbB_o4pKtbMOjZL",
            "mm_projector.bin": "1cl7itjrNiUJS8mnbnm3kxhRBnw5VftGc",
            "trainer_state.json": "1DbEH8mNrTD7gbcIA1Ti0QAer_WrQ-Vez",
        },
    },
    "checkpoint-4750": {
        "folder_id": "1z8qaEQiOmmV8A68I42yYI17UBN54UqMY",
        "drive_subpath": "checkpoint-4750",
        "label": "checkpoint-4750 (Stage 1, step 4750)",
        "files": {
            "config.json": "1zj3Elfl7nbk7sLURy6hA49C8A7hTZour",
            "mm_projector.bin": "1LuPa2Y9V1e3zAiqDWTgCEJCpi5K8959H",
            "trainer_state.json": "182EpKwBVNpdrn3uaFOw7YazF2cAzfAqw",
        },
    },
}

# Backward-compatible aliases
FINAL_DRIVE_FOLDER_ID = CHECKPOINT_REGISTRY["final"]["folder_id"]
FINAL_DRIVE_FILES = CHECKPOINT_REGISTRY["final"]["files"]

POPE_SPLITS = ("adversarial", "popular", "random")
POPE_BASE_URL = (
    "https://raw.githubusercontent.com/AoiDragon/POPE/"
    "e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco"
)
COCO_VAL2014_BASE_URL = "http://images.cocodataset.org/val2014"
POPE_PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."

DEFAULT_MODEL_BASE = "lmsys/vicuna-7b-v1.5"
DEFAULT_BASELINE_MODEL = "LanguageBind/Video-LLaVA-7B"


def _download_file(url: str, dest: Path, retries: int = 3, timeout: int = 30) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            if tmp_dest.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file from {url}")
            tmp_dest.replace(dest)
            return dest
        except Exception as exc:
            last_err = exc
            if tmp_dest.exists():
                tmp_dest.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 6))
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_err}")


def _verify_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def build_llava_pope_questions(coco_dir: Path, out_jsonl: Path) -> List[Dict[str, Any]]:
    """Combine adversarial/popular/random POPE JSONs into a unified LLaVA-format JSONL."""
    combined: List[Dict[str, Any]] = []
    qid = 1
    for split in POPE_SPLITS:
        split_file = coco_dir / f"coco_pope_{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing POPE split file: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                item = json.loads(raw_line)
                prompt_text = item["text"].rstrip()
                if "Answer the question using a single word or phrase." not in prompt_text:
                    prompt_text = prompt_text + POPE_PROMPT_SUFFIX
                combined.append(
                    {
                        "question_id": qid,
                        "split_question_id": item.get("question_id", qid),
                        "image": item["image"],
                        "text": prompt_text,
                        "category": split,
                    }
                )
                qid += 1

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")
    return combined


def prepare_pope_data(eval_dir: Path, max_workers: int = 16) -> Dict[str, Any]:
    """Download POPE split files, generate llava_pope_test.jsonl, and fetch the 500 unique COCO images."""
    coco_dir = eval_dir / "coco"
    val_dir = eval_dir / "val2014"
    coco_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] Downloading POPE split annotations into {coco_dir}...")
    for split in POPE_SPLITS:
        fname = f"coco_pope_{split}.json"
        dest = coco_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  - Using cached {fname}")
        else:
            url = f"{POPE_BASE_URL}/{fname}"
            _download_file(url, dest)
            print(f"  - Downloaded {fname}")

    question_jsonl = eval_dir / "llava_pope_test.jsonl"
    questions = build_llava_pope_questions(coco_dir, question_jsonl)
    unique_images = sorted({q["image"] for q in questions})
    print(f"[prepare] Built {question_jsonl} ({len(questions)} questions across {len(unique_images)} unique images).")

    to_download = [img for img in unique_images if not _verify_image(val_dir / img)]
    print(f"[prepare] COCO val2014 images: {len(unique_images) - len(to_download)} cached, {len(to_download)} to download.")

    if to_download:
        failed: List[Tuple[str, str]] = []

        def _fetch_img(img_name: str) -> str:
            dest = val_dir / img_name
            url = f"{COCO_VAL2014_BASE_URL}/{img_name}"
            _download_file(url, dest, retries=3, timeout=30)
            if not _verify_image(dest):
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"Corrupted image downloaded: {img_name}")
            return img_name

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_img, img): img for img in to_download}
            for fut in as_completed(futures):
                img = futures[fut]
                try:
                    fut.result()
                    completed += 1
                    if completed % 100 == 0 or completed == len(to_download):
                        print(f"  - Downloaded {completed}/{len(to_download)} images")
                except Exception as exc:
                    failed.append((img, str(exc)))

        if failed:
            sample_err = "; ".join(f"{name}: {err}" for name, err in failed[:5])
            raise RuntimeError(f"Failed to download {len(failed)} COCO images ({sample_err})")

    return {
        "eval_dir": str(eval_dir),
        "question_file": str(question_jsonl),
        "num_questions": len(questions),
        "num_images": len(unique_images),
    }


def download_one_checkpoint(ckpt_name: str, drive_ckpt_path: Path, local_ckpt_dir: Path) -> Path:
    """Copy a registered checkpoint from mounted Drive if available, else download via gdown."""
    if ckpt_name not in CHECKPOINT_REGISTRY:
        raise ValueError(f"Unknown checkpoint '{ckpt_name}'. Available: {list(CHECKPOINT_REGISTRY.keys())}")
    file_ids = CHECKPOINT_REGISTRY[ckpt_name]["files"]

    local_ckpt_dir.mkdir(parents=True, exist_ok=True)
    required = ("config.json", "mm_projector.bin")
    optional = ("trainer_state.json",)

    if all((local_ckpt_dir / f).exists() and (local_ckpt_dir / f).stat().st_size > 0 for f in required):
        print(f"[download_ckpt] '{ckpt_name}' already present at {local_ckpt_dir}")
        return local_ckpt_dir

    if drive_ckpt_path.exists() and all((drive_ckpt_path / f).exists() for f in required):
        print(f"[download_ckpt] Copying '{ckpt_name}' from mounted Drive: {drive_ckpt_path} -> {local_ckpt_dir}")
        for fname in required + optional:
            src = drive_ckpt_path / fname
            if src.exists():
                shutil.copy2(src, local_ckpt_dir / fname)
        return local_ckpt_dir

    print(f"[download_ckpt] Drive path '{drive_ckpt_path}' not found; downloading '{ckpt_name}' via gdown...")
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required to download from Google Drive. Run `pip install gdown`.") from exc

    for fname, file_id in file_ids.items():
        dest = local_ckpt_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        out = gdown.download(url, str(dest), quiet=False)
        if out is None or not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError(f"gdown failed to download {ckpt_name}/{fname} (id={file_id}).")
    return local_ckpt_dir


def ensure_untrained_config(local_ckpt_root: Path) -> Path:
    """Create a config-only folder (no mm_projector.bin) for untrained step-0 evaluation."""
    untrained_dir = local_ckpt_root / "untrained_step0"
    untrained_dir.mkdir(parents=True, exist_ok=True)
    target_cfg = untrained_dir / "config.json"
    if not target_cfg.exists():
        for src_name in ("final", "checkpoint-4750"):
            candidate_cfg = local_ckpt_root / src_name / "config.json"
            if candidate_cfg.exists():
                shutil.copy2(candidate_cfg, target_cfg)
                break
        else:
            # Download just config.json from final if neither folder exists yet
            try:
                import gdown

                gdown.download(
                    f"https://drive.google.com/uc?id={CHECKPOINT_REGISTRY['final']['files']['config.json']}",
                    str(target_cfg),
                    quiet=False,
                )
            except Exception as exc:
                raise RuntimeError(f"Unable to create {target_cfg}: {exc}") from exc

    # Ensure no stale weights exist in untrained_step0
    stale_bin = untrained_dir / "mm_projector.bin"
    if stale_bin.exists():
        stale_bin.unlink()
    return untrained_dir


def download_checkpoint(
    ckpt_name: str,
    drive_ckpt_root: Path,
    local_ckpt_root: Path,
    drive_ckpt_path: Optional[Path] = None,
    local_ckpt_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Download one or all registered checkpoints and prepare the untrained_step0 config directory."""
    names = list(CHECKPOINT_REGISTRY.keys()) if ckpt_name == "all" else [ckpt_name]
    paths: Dict[str, Path] = {}
    for name in names:
        d_path = drive_ckpt_path if (drive_ckpt_path is not None and ckpt_name != "all") else (drive_ckpt_root / name)
        l_dir = local_ckpt_dir if (local_ckpt_dir is not None and ckpt_name != "all") else (local_ckpt_root / name)
        paths[name] = download_one_checkpoint(name, d_path, l_dir)

    paths["untrained_step0"] = ensure_untrained_config(local_ckpt_root)
    return paths


def inspect_checkpoint(ckpt_dir: Path) -> Dict[str, Any]:
    """Inspect mm_projector.bin weights, learned gate value, and trainer_state.json."""
    import torch

    bin_path = ckpt_dir / "mm_projector.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"Missing {bin_path}")

    state = torch.load(bin_path, map_location="cpu")
    groups: Dict[str, Dict[str, Any]] = {}
    gate_raw: Optional[float] = None
    tanh_g: Optional[float] = None
    gate_v2_weight_norm: Optional[float] = None
    gate_v2_weight_max: Optional[float] = None
    gate_v2_bias: Optional[float] = None
    gate_v2_tanh_bias: Optional[float] = None

    for k, v in state.items():
        if "mm_projector" in k:
            grp = "mm_projector"
        elif "early_fusion" in k:
            grp = "early_fusion"
        else:
            grp = "other"
        entry = groups.setdefault(grp, {"tensors": 0, "params": 0})
        entry["tensors"] += 1
        entry["params"] += int(v.numel())
        if "early_fusion_gate_linear.weight" in k:
            vf = v.float()
            gate_v2_weight_norm = float(vf.norm().item())
            gate_v2_weight_max = float(vf.abs().max().item())
        elif "early_fusion_gate_linear.bias" in k and v.numel() == 1:
            gate_v2_bias = float(v.float().item())
            gate_v2_tanh_bias = float(torch.tanh(v.float()).item())
        elif "early_fusion_gate" in k and v.numel() == 1:
            gate_raw = float(v.float().item())
            tanh_g = float(torch.tanh(v.float()).item())

    summary: Dict[str, Any] = {
        "checkpoint_dir": str(ckpt_dir),
        "groups": groups,
    }
    if gate_v2_weight_norm is not None:
        summary["gate_v2"] = {
            "weight_l2_norm": gate_v2_weight_norm,
            "weight_max_abs": gate_v2_weight_max,
            "bias": gate_v2_bias,
            "tanh_bias": gate_v2_tanh_bias,
        }
    if gate_raw is not None:
        summary["gate_scalar"] = {"gate_raw": gate_raw, "tanh_g": tanh_g}

    trainer_path = ckpt_dir / "trainer_state.json"
    if trainer_path.exists():
        with open(trainer_path, "r", encoding="utf-8") as f:
            ts = json.load(f)
        losses = [x for x in ts.get("log_history", []) if "loss" in x]
        summary["trainer_state"] = {
            "global_step": ts.get("global_step"),
            "max_steps": ts.get("max_steps"),
            "epoch": ts.get("epoch"),
            "min_loss": min((x["loss"] for x in losses), default=None),
            "last_loss": losses[-1]["loss"] if losses else None,
        }

    print(f"=== Checkpoint Inspection: {ckpt_dir.name} ===")
    print(json.dumps(summary, indent=2))
    return summary


def resolve_precision(precision: str) -> Tuple[bool, Any, str]:
    """Select (load_4bit, torch_dtype, label) based on user choice and GPU VRAM."""
    import torch

    req = precision.lower()
    if req == "auto":
        if not torch.cuda.is_available():
            return False, torch.float32, "fp32-cpu"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb < 20.0:
            return True, torch.float16, f"4bit-fp16 ({vram_gb:.1f}GB VRAM)"
        if torch.cuda.is_bf16_supported():
            return False, torch.bfloat16, f"bf16 ({vram_gb:.1f}GB VRAM)"
        return False, torch.float16, f"fp16 ({vram_gb:.1f}GB VRAM)"

    if req == "4bit":
        return True, torch.float16, "4bit-fp16"
    if req == "bf16":
        return False, torch.bfloat16, "bf16"
    if req == "fp16":
        return False, torch.float16, "fp16"
    raise ValueError(f"Unsupported precision '{precision}'. Choose from auto, fp16, bf16, 4bit.")


def normalize_pope_answer(text: str) -> str:
    """Apply the exact POPE answer normalization used in videollava/eval/eval_pope.py."""
    if "." in text:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def compute_split_metrics(pred_texts: List[str], label_texts: List[str]) -> Dict[str, Any]:
    """Compute TP/FP/TN/FN, accuracy, precision, recall, F1, and yes_ratio safely."""
    if len(pred_texts) != len(label_texts):
        raise ValueError(f"Length mismatch: {len(pred_texts)} predictions vs {len(label_texts)} labels.")
    if len(pred_texts) == 0:
        raise ValueError("Cannot compute POPE metrics on 0 samples.")

    preds = [0 if normalize_pope_answer(t) == "no" else 1 for t in pred_texts]
    labels = [0 if lbl == "no" else 1 for lbl in label_texts]

    tp = fp = tn = fn = 0
    for pred, lbl in zip(preds, labels):
        if pred == 1 and lbl == 1:
            tp += 1
        elif pred == 1 and lbl == 0:
            fp += 1
        elif pred == 0 and lbl == 0:
            tn += 1
        else:
            fn += 1

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float(tp + tn) / float(tp + tn + fp + fn)
    yes_ratio = float(preds.count(1)) / float(len(preds))

    return {
        "samples": len(preds),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
    }


def compute_pope_metrics(
    answers_by_qid: Dict[int, Dict[str, Any]],
    questions: List[Dict[str, Any]],
    coco_dir: Path,
    max_samples_per_split: int = 0,
) -> Dict[str, Any]:
    """Score answers split-by-split against coco_pope_{split}.json in exact question order."""
    split_metrics: Dict[str, Dict[str, Any]] = {}

    for split in POPE_SPLITS:
        split_questions = [q for q in questions if q["category"] == split]
        if max_samples_per_split > 0:
            split_questions = split_questions[:max_samples_per_split]

        label_path = coco_dir / f"coco_pope_{split}.json"
        with open(label_path, "r", encoding="utf-8") as f:
            all_labels = [json.loads(line)["label"] for line in f if line.strip()]
        labels = all_labels[: len(split_questions)]

        preds: List[str] = []
        for q in split_questions:
            qid = q["question_id"]
            if qid not in answers_by_qid:
                raise KeyError(f"Missing answer for question_id={qid} ({split})")
            preds.append(answers_by_qid[qid]["text"])

        split_metrics[split] = compute_split_metrics(preds, labels)

    macro = {}
    for metric_key in ("accuracy", "precision", "recall", "f1", "yes_ratio"):
        macro[metric_key] = sum(split_metrics[s][metric_key] for s in POPE_SPLITS) / len(POPE_SPLITS)

    return {"splits": split_metrics, "macro_avg": macro}


def select_eval_questions(questions: List[Dict[str, Any]], max_samples_per_split: int) -> List[Dict[str, Any]]:
    if max_samples_per_split <= 0:
        return questions
    selected: List[Dict[str, Any]] = []
    counts = {s: 0 for s in POPE_SPLITS}
    for q in questions:
        cat = q["category"]
        if cat in counts and counts[cat] < max_samples_per_split:
            selected.append(q)
            counts[cat] += 1
    return selected


def run_pope_eval(
    model_path: str,
    model_base: Optional[str],
    eval_dir: Path,
    answers_file: Path,
    metrics_file: Path,
    conv_mode: str = "vicuna_v1",
    precision: str = "auto",
    max_samples_per_split: int = 0,
    max_new_tokens: int = 16,
    save_every: int = 100,
    label: Optional[str] = None,
    untrained: bool = False,
) -> Dict[str, Any]:
    """Run POPE inference (with resume support) and compute exact metrics."""
    import torch
    from PIL import Image
    import shortuuid
    from tqdm import tqdm

    from videollava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from videollava.conversation import SeparatorStyle, conv_templates
    from videollava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from videollava.model.builder import load_pretrained_model
    from videollava.utils import disable_torch_init

    question_file = eval_dir / "llava_pope_test.jsonl"
    image_folder = eval_dir / "val2014"
    coco_dir = eval_dir / "coco"
    if not question_file.exists():
        raise FileNotFoundError(f"Missing {question_file}. Run --action prepare first.")

    with open(question_file, "r", encoding="utf-8") as f:
        all_questions = [json.loads(line) for line in f if line.strip()]
    questions = select_eval_questions(all_questions, max_samples_per_split)

    answers_by_qid: Dict[int, Dict[str, Any]] = {}
    if answers_file.exists():
        with open(answers_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                answers_by_qid[rec["question_id"]] = rec

    pending = [q for q in questions if q["question_id"] not in answers_by_qid]
    model_name = get_model_name_from_path(model_path)
    run_label = label or ("Untrained (Step 0)" if untrained else model_name)
    print(f"[eval] {run_label}: {len(questions)} target questions ({len(answers_by_qid)} cached, {len(pending)} pending).")

    load_4bit, torch_dtype, prec_desc = resolve_precision(precision)
    if pending:
        disable_torch_init()
        if untrained:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
        print(f"[eval] Loading model='{model_path}' (base={model_base}, untrained={untrained}) in {prec_desc}...")
        tokenizer, model, processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_4bit=load_4bit,
            load_video_tower=False,
            torch_dtype=torch_dtype,
            load_mm_projector=not untrained,
        )
        model.config.use_cache = True
        model.eval()
        image_processor = processor["image"]
        embed_dtype = model.get_model().embed_tokens.weight.dtype
        input_dtype = embed_dtype if embed_dtype.is_floating_point else torch_dtype

        answers_file.parent.mkdir(parents=True, exist_ok=True)
        conv_template = conv_templates[conv_mode]
        stop_str = conv_template.sep if conv_template.sep_style != SeparatorStyle.TWO else conv_template.sep2

        t0 = time.time()
        with open(answers_file, "a", encoding="utf-8") as ans_f:
            for idx, line in enumerate(tqdm(pending, desc=f"POPE ({run_label})"), start=1):
                qid = line["question_id"]
                qs = line["text"]
                if getattr(model.config, "mm_use_im_start_end", False):
                    qs_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs_prompt = DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv = conv_template.copy()
                conv.append_message(conv.roles[0], qs_prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(model.device)

                image_path = image_folder / line["image"]
                image = Image.open(image_path).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)
                image_tensor = image_tensor.to(device=model.device, dtype=input_dtype)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                    )

                in_len = input_ids.shape[1]
                if output_ids.shape[1] > in_len and torch.equal(output_ids[:, :in_len], input_ids):
                    gen_ids = output_ids[:, in_len:]
                else:
                    gen_ids = output_ids

                output_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                if stop_str and output_text.endswith(stop_str):
                    output_text = output_text[: -len(stop_str)].strip()

                record = {
                    "question_id": qid,
                    "prompt": qs,
                    "text": output_text,
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_name,
                    "category": line["category"],
                    "metadata": {},
                }
                answers_by_qid[qid] = record
                ans_f.write(json.dumps(record) + "\n")
                if idx % save_every == 0:
                    ans_f.flush()
            ans_f.flush()

        elapsed = time.time() - t0
        print(f"[eval] Finished {len(pending)} questions in {elapsed:.1f}s ({len(pending)/max(elapsed, 1e-6):.2f} q/s).")

        del model, tokenizer, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = compute_pope_metrics(
        answers_by_qid=answers_by_qid,
        questions=questions,
        coco_dir=coco_dir,
        max_samples_per_split=max_samples_per_split,
    )
    payload = {
        "label": run_label,
        "model_path": str(model_path),
        "model_base": model_base,
        "untrained": untrained,
        "precision": prec_desc,
        "conv_mode": conv_mode,
        "max_samples_per_split": max_samples_per_split,
        "answers_file": str(answers_file),
        **metrics,
    }
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n=== POPE Results: {run_label} ===")
    for split in POPE_SPLITS:
        s = metrics["splits"][split]
        print(
            f"  {split:12s} | Acc: {s['accuracy']*100:6.2f}% | Prec: {s['precision']*100:6.2f}% | "
            f"Rec: {s['recall']*100:6.2f}% | F1: {s['f1']*100:6.2f}% | Yes%: {s['yes_ratio']*100:6.2f}% (n={s['samples']})"
        )
    m = metrics["macro_avg"]
    print(
        f"  {'MACRO AVG':12s} | Acc: {m['accuracy']*100:6.2f}% | Prec: {m['precision']*100:6.2f}% | "
        f"Rec: {m['recall']*100:6.2f}% | F1: {m['f1']*100:6.2f}% | Yes%: {m['yes_ratio']*100:6.2f}%"
    )
    return payload


def generate_comparison_report(
    models: List[Dict[str, Any]],
    output_dir: Path,
    names: Optional[List[str]] = None,
    drive_output_dir: Optional[Path] = None,
    extra_files_to_sync: Optional[List[Path]] = None,
) -> Path:
    """Build an N-model comparison Markdown + JSON report and optionally sync to Drive."""
    if not models:
        raise ValueError("At least one model metrics dictionary is required for comparison.")
    if names is not None and len(names) != len(models):
        raise ValueError(f"Expected {len(models)} names, got {len(names)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_names = [
        (names[i] if names is not None else models[i].get("label", f"Model {i+1}"))
        for i in range(len(models))
    ]
    ref_name = resolved_names[0]

    md_lines = [
        "# POPE Benchmark Comparison Report",
        "",
        "## Evaluated Models",
    ]
    for name, mdl in zip(resolved_names, models):
        md_lines.append(
            f"- **{name}**: `{mdl.get('model_path')}` (base=`{mdl.get('model_base')}`, precision=`{mdl.get('precision')}`)"
        )

    md_lines.extend(
        [
            "",
            "> **Context Note:** `Untrained (Step 0)`, `final` (step 867 / 11,627), and `checkpoint-4750` (step 4,750 / 11,627) all use a frozen Vicuna-7B LLM backbone (Stage 1 alignment). If included, `LanguageBind/Video-LLaVA-7B` is a complete Stage 2 instruction-tuned reference.",
            "",
            f"## Macro Average Summary (Δ F1 vs `{ref_name}`)",
            "",
            "| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | Yes Ratio (%) | Δ F1 (pp) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    ref_macro_f1 = models[0]["macro_avg"]["f1"]
    for idx, (name, mdl) in enumerate(zip(resolved_names, models)):
        m = mdl["macro_avg"]
        delta_str = "—" if idx == 0 else f"{(m['f1'] - ref_macro_f1) * 100.0:+.2f}"
        md_lines.append(
            f"| {name} | {m['accuracy']*100:.2f} | {m['precision']*100:.2f} | "
            f"{m['recall']*100:.2f} | {m['f1']*100:.2f} | {m['yes_ratio']*100:.2f} | {delta_str} |"
        )

    md_lines.extend(
        [
            "",
            f"## Per-Split Breakdown (Δ F1 vs `{ref_name}`)",
            "",
            "| Split | Model | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | Yes Ratio (%) | Δ F1 (pp) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for split in POPE_SPLITS:
        ref_f1 = models[0]["splits"][split]["f1"]
        for idx, (name, mdl) in enumerate(zip(resolved_names, models)):
            s = mdl["splits"][split]
            delta_str = "—" if idx == 0 else f"{(s['f1'] - ref_f1) * 100.0:+.2f}"
            md_lines.append(
                f"| {split.capitalize()} | {name} | {s['accuracy']*100:.2f} | {s['precision']*100:.2f} | "
                f"{s['recall']*100:.2f} | {s['f1']*100:.2f} | {s['yes_ratio']*100:.2f} | {delta_str} |"
            )

    md_text = "\n".join(md_lines) + "\n"
    report_path = output_dir / "pope_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    combined_json: Dict[str, Any] = {
        "names": resolved_names,
        "models": [{"name": n, **m} for n, m in zip(resolved_names, models)],
    }
    if len(models) == 2:
        combined_json["candidate_name"] = resolved_names[0]
        combined_json["baseline_name"] = resolved_names[1]
        combined_json["candidate"] = models[0]
        combined_json["baseline"] = models[1]

    json_path = output_dir / "pope_comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_json, f, indent=2)

    print("\n" + md_text)

    if drive_output_dir is not None:
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.exists() or drive_output_dir.parent.exists():
            drive_output_dir.mkdir(parents=True, exist_ok=True)
            files_to_copy = [report_path, json_path]
            for path_item in extra_files_to_sync or []:
                if path_item and path_item.exists():
                    files_to_copy.append(path_item)
            for src in files_to_copy:
                shutil.copy2(src, drive_output_dir / src.name)
            print(f"[compare] Synced {len(files_to_copy)} files to Google Drive: {drive_output_dir}")

    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POPE Benchmark Evaluation & Comparison for Video-LLaVA / MuLER")
    parser.add_argument(
        "--action",
        required=True,
        choices=["prepare", "download_ckpt", "inspect_ckpt", "eval", "compare", "all"],
        help="Pipeline stage to execute.",
    )
    parser.add_argument("--eval_dir", type=Path, default=Path("/content/eval/pope"))
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="all",
        choices=["final", "checkpoint-4750", "all"],
        help="Which registered checkpoint(s) to download/inspect.",
    )
    parser.add_argument(
        "--drive_ckpt_root",
        type=Path,
        default=Path("/content/drive/MyDrive/Video-LLaVA/checkpoints/videollava-7b-pretrain"),
    )
    parser.add_argument(
        "--local_ckpt_root",
        type=Path,
        default=Path("/content/checkpoints/videollava-7b-pretrain"),
    )
    # Legacy single-checkpoint overrides
    parser.add_argument("--drive_ckpt_path", type=Path, default=None)
    parser.add_argument("--local_ckpt_dir", type=Path, default=None)

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--untrained", action="store_true", help="Skip loading mm_projector.bin to evaluate step-0 model.")
    parser.add_argument("--baseline_model_path", type=str, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--skip_official_baseline", action="store_true")
    parser.add_argument("--answers_file", type=Path, default=None)
    parser.add_argument("--metrics_file", type=Path, default=None)

    # N-model comparison args + legacy 2-model fallback
    parser.add_argument("--metrics_files", type=Path, nargs="+", default=None)
    parser.add_argument("--names", type=str, nargs="+", default=None)
    parser.add_argument("--candidate_metrics", type=Path, default=Path("/content/eval/pope/results/final_metrics.json"))
    parser.add_argument("--baseline_metrics", type=Path, default=Path("/content/eval/pope/results/baseline_metrics.json"))
    parser.add_argument("--candidate_name", type=str, default="final (Stage 1, step 867)")
    parser.add_argument("--baseline_name", type=str, default="Official Video-LLaVA-7B (Stage 2 Baseline)")

    parser.add_argument("--output_dir", type=Path, default=Path("/content/eval/pope/results"))
    parser.add_argument(
        "--drive_output_dir",
        type=Path,
        default=Path("/content/drive/MyDrive/Video-LLaVA/eval_results/pope"),
    )
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp16", "bf16", "4bit"])
    parser.add_argument("--max_samples_per_split", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.action == "prepare":
        prepare_pope_data(args.eval_dir)
    elif args.action == "download_ckpt":
        download_checkpoint(
            ckpt_name=args.ckpt_name,
            drive_ckpt_root=args.drive_ckpt_root,
            local_ckpt_root=args.local_ckpt_root,
            drive_ckpt_path=args.drive_ckpt_path,
            local_ckpt_dir=args.local_ckpt_dir,
        )
    elif args.action == "inspect_ckpt":
        if args.local_ckpt_dir is not None:
            inspect_checkpoint(args.local_ckpt_dir)
        else:
            names = list(CHECKPOINT_REGISTRY.keys()) if args.ckpt_name == "all" else [args.ckpt_name]
            for name in names:
                inspect_checkpoint(args.local_ckpt_root / name)
    elif args.action == "eval":
        model_path = args.model_path or str(args.local_ckpt_dir or (args.local_ckpt_root / ("untrained_step0" if args.untrained else "final")))
        safe_tag = (args.label or Path(model_path).name).replace("/", "_").replace(" ", "_")
        answers_file = args.answers_file or (args.eval_dir / "answers" / f"{safe_tag}.jsonl")
        metrics_file = args.metrics_file or (args.output_dir / f"{safe_tag}_metrics.json")
        run_pope_eval(
            model_path=model_path,
            model_base=args.model_base,
            eval_dir=args.eval_dir,
            answers_file=answers_file,
            metrics_file=metrics_file,
            conv_mode=args.conv_mode,
            precision=args.precision,
            max_samples_per_split=args.max_samples_per_split,
            max_new_tokens=args.max_new_tokens,
            save_every=args.save_every,
            label=args.label,
            untrained=args.untrained,
        )
    elif args.action == "compare":
        if args.metrics_files:
            loaded_models = []
            extra_sync: List[Path] = []
            for mf in args.metrics_files:
                with open(mf, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                loaded_models.append(m_data)
                extra_sync.append(mf)
                if m_data.get("answers_file"):
                    extra_sync.append(Path(m_data["answers_file"]))
            generate_comparison_report(
                models=loaded_models,
                output_dir=args.output_dir,
                names=args.names,
                drive_output_dir=args.drive_output_dir,
                extra_files_to_sync=extra_sync,
            )
        else:
            with open(args.candidate_metrics, "r", encoding="utf-8") as f:
                cand = json.load(f)
            with open(args.baseline_metrics, "r", encoding="utf-8") as f:
                base = json.load(f)
            extra = [
                args.candidate_metrics,
                args.baseline_metrics,
                Path(cand["answers_file"]) if cand.get("answers_file") else None,
                Path(base["answers_file"]) if base.get("answers_file") else None,
            ]
            generate_comparison_report(
                models=[cand, base],
                output_dir=args.output_dir,
                names=[args.candidate_name, args.baseline_name],
                drive_output_dir=args.drive_output_dir,
                extra_files_to_sync=[p for p in extra if p is not None],
            )
    elif args.action == "all":
        prepare_pope_data(args.eval_dir)
        ckpt_paths = download_checkpoint(
            ckpt_name="all",
            drive_ckpt_root=args.drive_ckpt_root,
            local_ckpt_root=args.local_ckpt_root,
        )
        for name in CHECKPOINT_REGISTRY:
            inspect_checkpoint(ckpt_paths[name])

        base_llm = args.model_base or DEFAULT_MODEL_BASE
        eval_specs = [
            {
                "name": "Untrained (Step 0)",
                "model_path": str(ckpt_paths["untrained_step0"]),
                "model_base": base_llm,
                "untrained": True,
                "answers": args.eval_dir / "answers" / "untrained_step0.jsonl",
                "metrics": args.output_dir / "untrained_step0_metrics.json",
            },
            {
                "name": "final (Stage 1, step 867)",
                "model_path": str(ckpt_paths["final"]),
                "model_base": base_llm,
                "untrained": False,
                "answers": args.eval_dir / "answers" / "final_stage1_step867.jsonl",
                "metrics": args.output_dir / "final_metrics.json",
            },
            {
                "name": "checkpoint-4750 (Stage 1, step 4750)",
                "model_path": str(ckpt_paths["checkpoint-4750"]),
                "model_base": base_llm,
                "untrained": False,
                "answers": args.eval_dir / "answers" / "checkpoint_4750_stage1.jsonl",
                "metrics": args.output_dir / "checkpoint_4750_metrics.json",
            },
        ]
        if not args.skip_official_baseline:
            eval_specs.append(
                {
                    "name": "Official Video-LLaVA-7B (Stage 2 Baseline)",
                    "model_path": args.baseline_model_path,
                    "model_base": None,
                    "untrained": False,
                    "answers": args.eval_dir / "answers" / "videollava_7b_baseline.jsonl",
                    "metrics": args.output_dir / "baseline_metrics.json",
                }
            )

        evaluated_models = []
        extra_sync: List[Path] = []
        for spec in eval_specs:
            res = run_pope_eval(
                model_path=spec["model_path"],
                model_base=spec["model_base"],
                eval_dir=args.eval_dir,
                answers_file=spec["answers"],
                metrics_file=spec["metrics"],
                conv_mode=args.conv_mode,
                precision=args.precision,
                max_samples_per_split=args.max_samples_per_split,
                max_new_tokens=args.max_new_tokens,
                save_every=args.save_every,
                label=spec["name"],
                untrained=spec["untrained"],
            )
            evaluated_models.append(res)
            extra_sync.extend([spec["metrics"], spec["answers"]])

        generate_comparison_report(
            models=evaluated_models,
            output_dir=args.output_dir,
            names=[s["name"] for s in eval_specs],
            drive_output_dir=args.drive_output_dir,
            extra_files_to_sync=extra_sync,
        )


if __name__ == "__main__":
    main()
