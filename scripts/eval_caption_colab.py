#!/usr/bin/env python3
"""
Turnkey COCO Image Captioning (Karpathy Test Split) Evaluation & Comparison Pipeline for Google Colab.

Designed for Stage 1 (pretraining) checkpoints whose primary task is visual-to-text alignment
and image captioning rather than instruction-tuned VQA. Evaluates and compares:
  - Untrained (Step 0): random mm_projector + zeroed Gate v2
  - Stage 1 Projector-Only (No Fusion): checkpoint-800 mm_projector + zeroed Gate v2
  - checkpoint-800 (Stage 1, step 800, pre-spike)
  - final (Stage 1, step 867)
  - checkpoint-4750 (Stage 1, step 4750)
  - Optional Official LanguageBind/Video-LLaVA-7B (Stage 2 reference)

Metrics computed in pure Python (no Java / CoreNLP dependency):
  - CIDEr (TF-IDF n-gram cosine similarity x 10, Vedantam et al.)
  - BLEU-1 & BLEU-4 (corpus BLEU with closest-reference brevity penalty)
  - ROUGE-L (LCS F-measure, beta=1.2)
  - Distinct-1, Avg Token Length, Empty % (hallucination / mode-collapse diagnostics)

Usage:
  python scripts/eval_caption_colab.py --action prepare --num_images 500
  python scripts/eval_caption_colab.py --action download_ckpt --ckpt_name all
  python scripts/eval_caption_colab.py --action eval --model_path ... --model_base ...
  python scripts/eval_caption_colab.py --action compare --metrics_files m1.json m2.json ...
  python scripts/eval_caption_colab.py --action all
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple


CHECKPOINT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "checkpoint-800": {
        "folder_id": "18vvNxd9wqi9kqYb5BXAzk5lnJ2mFNZQS",
        "label": "checkpoint-800 (Stage 1, step 800)",
        "files": {
            "config.json": "1r9WQ6FbhH4437eDXl_zJWLcFQef7g5J4",
            "mm_projector.bin": "1VK854XM1kB2P0KPUGr1WnBCCftwul1iB",
            "trainer_state.json": "1lhoHWj-g-sJpdyOA6l112ds2TJmznkjN",
        },
    },
    "final": {
        "folder_id": "1H4L3d9FT2l5HrejUZ2pYbLupNeh4fEr2",
        "label": "final (Stage 1, step 867)",
        "files": {
            "config.json": "13eK91eIFA-nZFx7_AZbB_o4pKtbMOjZL",
            "mm_projector.bin": "1cl7itjrNiUJS8mnbnm3kxhRBnw5VftGc",
            "trainer_state.json": "1DbEH8mNrTD7gbcIA1Ti0QAer_WrQ-Vez",
        },
    },
    "checkpoint-4750": {
        "folder_id": "1z8qaEQiOmmV8A68I42yYI17UBN54UqMY",
        "label": "checkpoint-4750 (Stage 1, step 4750)",
        "files": {
            "config.json": "1zj3Elfl7nbk7sLURy6hA49C8A7hTZour",
            "mm_projector.bin": "1LuPa2Y9V1e3zAiqDWTgCEJCpi5K8959H",
            "trainer_state.json": "182EpKwBVNpdrn3uaFOw7YazF2cAzfAqw",
        },
    },
}

KARPATHY_COCO_URL = "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json"
COCO_VAL2014_BASE_URL = "http://images.cocodataset.org/val2014"
DEFAULT_MODEL_BASE = "lmsys/vicuna-7b-v1.5"
DEFAULT_BASELINE_MODEL = "LanguageBind/Video-LLaVA-7B"
DEFAULT_PROMPT = "Describe the image concisely in one sentence."


def _download_file(url: str, dest: Path, retries: int = 3, timeout: int = 45) -> Path:
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


def build_karpathy_test_references(
    dataset_coco_json: Path, out_jsonl: Path, num_images: int = 500
) -> List[Dict[str, Any]]:
    """Extract deterministic Karpathy test-split records from dataset_coco.json."""
    with open(dataset_coco_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    test_items = [
        img for img in raw.get("images", []) if img.get("split") == "test" and img.get("filename")
    ]
    test_items.sort(key=lambda x: int(x.get("cocoid", x.get("imgid", 0))))
    if num_images > 0:
        test_items = test_items[:num_images]

    records: List[Dict[str, Any]] = []
    for idx, img in enumerate(test_items, start=1):
        refs = [s["raw"].strip() for s in img.get("sentences", []) if s.get("raw", "").strip()]
        if not refs:
            continue
        records.append(
            {
                "image_id": idx,
                "cocoid": int(img.get("cocoid", idx)),
                "filename": img["filename"],
                "filepath": img.get("filepath", "val2014"),
                "references": refs,
            }
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    return records


def prepare_caption_data(
    eval_dir: Path,
    num_images: int = 500,
    reuse_val2014_dir: Optional[Path] = Path("/content/eval/pope/val2014"),
    max_workers: int = 16,
) -> Dict[str, Any]:
    """Download Karpathy COCO annotations and fetch the selected COCO val2014 images."""
    dataset_json = eval_dir / "dataset_coco.json"
    val_dir = eval_dir / "val2014"
    val_dir.mkdir(parents=True, exist_ok=True)

    if dataset_json.exists() and dataset_json.stat().st_size > 0:
        print(f"[prepare] Using cached {dataset_json}")
    else:
        print(f"[prepare] Downloading Karpathy COCO splits to {dataset_json}...")
        _download_file(KARPATHY_COCO_URL, dataset_json, timeout=90)

    ref_jsonl = eval_dir / "karpathy_test_references.jsonl"
    records = build_karpathy_test_references(dataset_json, ref_jsonl, num_images=num_images)
    print(f"[prepare] Wrote {len(records)} Karpathy test references to {ref_jsonl}.")

    reused = 0
    to_download: List[Dict[str, Any]] = []
    for rec in records:
        fname = rec["filename"]
        dest = val_dir / fname
        if _verify_image(dest):
            continue
        if reuse_val2014_dir is not None:
            cached = reuse_val2014_dir / fname
            if _verify_image(cached):
                shutil.copy2(cached, dest)
                reused += 1
                continue
        to_download.append(rec)

    print(
        f"[prepare] Images: {len(records) - len(to_download) - reused} cached, "
        f"{reused} reused from POPE cache, {len(to_download)} to download."
    )

    if to_download:
        failed: List[Tuple[str, str]] = []

        def _fetch(rec: Dict[str, Any]) -> str:
            fname = rec["filename"]
            subdir = rec.get("filepath", "val2014")
            dest = val_dir / fname
            url = f"http://images.cocodataset.org/{subdir}/{fname}"
            _download_file(url, dest, retries=3, timeout=30)
            if not _verify_image(dest):
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"Corrupted image downloaded: {fname}")
            return fname

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_fetch, r): r["filename"] for r in to_download}
            for fut in as_completed(futs):
                fname = futs[fut]
                try:
                    fut.result()
                    done += 1
                    if done % 100 == 0 or done == len(to_download):
                        print(f"  - Downloaded {done}/{len(to_download)} images")
                except Exception as exc:
                    failed.append((fname, str(exc)))

        if failed:
            sample_err = "; ".join(f"{n}: {e}" for n, e in failed[:5])
            raise RuntimeError(f"Failed to download {len(failed)} COCO images ({sample_err})")

    return {
        "eval_dir": str(eval_dir),
        "reference_file": str(ref_jsonl),
        "num_images": len(records),
    }


def download_one_checkpoint(ckpt_name: str, drive_ckpt_path: Path, local_ckpt_dir: Path) -> Path:
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
        out = gdown.download(f"https://drive.google.com/uc?id={file_id}", str(dest), quiet=False)
        if out is None or not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError(f"gdown failed to download {ckpt_name}/{fname} (id={file_id}).")
    return local_ckpt_dir


def ensure_untrained_config(local_ckpt_root: Path) -> Path:
    untrained_dir = local_ckpt_root / "untrained_step0"
    untrained_dir.mkdir(parents=True, exist_ok=True)
    target_cfg = untrained_dir / "config.json"
    if not target_cfg.exists():
        for src_name in ("checkpoint-800", "final", "checkpoint-4750"):
            candidate_cfg = local_ckpt_root / src_name / "config.json"
            if candidate_cfg.exists():
                shutil.copy2(candidate_cfg, target_cfg)
                break
        else:
            import gdown

            gdown.download(
                f"https://drive.google.com/uc?id={CHECKPOINT_REGISTRY['checkpoint-800']['files']['config.json']}",
                str(target_cfg),
                quiet=False,
            )
    stale_bin = untrained_dir / "mm_projector.bin"
    if stale_bin.exists():
        stale_bin.unlink()
    return untrained_dir


def download_checkpoint(
    ckpt_name: str, drive_ckpt_root: Path, local_ckpt_root: Path
) -> Dict[str, Path]:
    names = list(CHECKPOINT_REGISTRY.keys()) if ckpt_name == "all" else [ckpt_name]
    paths: Dict[str, Path] = {}
    for name in names:
        paths[name] = download_one_checkpoint(name, drive_ckpt_root / name, local_ckpt_root / name)
    paths["untrained_step0"] = ensure_untrained_config(local_ckpt_root)
    return paths


# ==============================================================================
# Pure-Python Standard Captioning Metrics (BLEU-1/4, ROUGE-L, CIDEr, Distinct-1)
# ==============================================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize_caption(text: str) -> List[str]:
    """Lowercase alphanumeric/apostrophe word tokenization matching standard caption eval."""
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_corpus_bleu(
    hyps_tok: Sequence[List[str]], refs_tok_list: Sequence[List[List[str]]], max_n: int = 4
) -> float:
    """Compute corpus-level BLEU-N (0-100) with closest reference length brevity penalty."""
    if not hyps_tok:
        return 0.0

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, refs in zip(hyps_tok, refs_tok_list):
        c_len = len(hyp)
        hyp_len += c_len
        closest_ref = min(refs, key=lambda r: (abs(len(r) - c_len), len(r)))
        ref_len += len(closest_ref)

        for n in range(1, max_n + 1):
            hyp_ng = _ngrams(hyp, n)
            if not hyp_ng:
                continue
            total_counts[n - 1] += sum(hyp_ng.values())
            max_ref_ng: Dict[Tuple[str, ...], int] = {}
            for ref in refs:
                for gram, cnt in _ngrams(ref, n).items():
                    if cnt > max_ref_ng.get(gram, 0):
                        max_ref_ng[gram] = cnt
            clipped_counts[n - 1] += sum(min(cnt, max_ref_ng.get(gram, 0)) for gram, cnt in hyp_ng.items())

    if hyp_len == 0:
        return 0.0

    precisions: List[float] = []
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        precisions.append(clipped_counts[n] / float(total_counts[n]))

    log_avg = sum(math.log(p) for p in precisions) / float(max_n)
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / float(hyp_len)))
    return 100.0 * bp * math.exp(log_avg)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[n]


def compute_rouge_l(
    hyps_tok: Sequence[List[str]], refs_tok_list: Sequence[List[List[str]]], beta: float = 1.2
) -> float:
    """Compute corpus average ROUGE-L F-measure (0-100), matching pycocoevalcap beta=1.2."""
    if not hyps_tok:
        return 0.0
    beta_sq = beta * beta
    scores: List[float] = []
    for hyp, refs in zip(hyps_tok, refs_tok_list):
        if not hyp:
            scores.append(0.0)
            continue
        best = 0.0
        for ref in refs:
            if not ref:
                continue
            lcs = _lcs_length(hyp, ref)
            if lcs == 0:
                continue
            prec = lcs / float(len(hyp))
            rec = lcs / float(len(ref))
            denom = rec + beta_sq * prec
            if denom > 0:
                f = ((1.0 + beta_sq) * prec * rec) / denom
                if f > best:
                    best = f
        scores.append(best)
    return 100.0 * sum(scores) / float(len(scores))


def compute_cider(
    hyps_tok: Sequence[List[str]],
    refs_tok_list: Sequence[List[List[str]]],
    n: int = 4,
    sigma: float = 6.0,
) -> float:
    """Compute CIDEr score (Vedantam et al., scaled x100 / standard 0-10+ -> 0-1000? Standard CIDEr is x10 = 0-150+)."""
    num_docs = len(refs_tok_list)
    if num_docs == 0:
        return 0.0

    # Document frequency of each n-gram across images (each image can contribute at most once per n-gram)
    doc_freq: List[Counter] = [Counter() for _ in range(n)]
    ref_ngrams_per_img: List[List[List[Counter]]] = []
    for refs in refs_tok_list:
        img_refs: List[List[Counter]] = []
        seen_in_img: List[set] = [set() for _ in range(n)]
        for ref in refs:
            ref_ns: List[Counter] = []
            for k in range(1, n + 1):
                c = _ngrams(ref, k)
                ref_ns.append(c)
                seen_in_img[k - 1].update(c.keys())
            img_refs.append(ref_ns)
        for k in range(n):
            for gram in seen_in_img[k]:
                doc_freq[k][gram] += 1
        ref_ngrams_per_img.append(img_refs)

    log_num_docs = math.log(max(float(num_docs), 1.0))

    def _tfidf_vec(counts: Counter, df_map: Counter) -> Tuple[Dict[Tuple[str, ...], float], float]:
        vec: Dict[Tuple[str, ...], float] = {}
        norm_sq = 0.0
        total = float(sum(counts.values()))
        if total <= 0.0:
            return vec, 0.0
        for gram, cnt in counts.items():
            df = float(df_map.get(gram, 0))
            idf = max(0.0, log_num_docs - math.log(max(df, 1.0))) if num_docs > 1 else 1.0
            val = (cnt / total) * idf
            if val != 0.0:
                vec[gram] = val
                norm_sq += val * val
        return vec, math.sqrt(norm_sq)

    img_scores: List[float] = []
    for hyp, refs, ref_ns_list in zip(hyps_tok, refs_tok_list, ref_ngrams_per_img):
        if not hyp:
            img_scores.append(0.0)
            continue
        hyp_ns = [_ngrams(hyp, k) for k in range(1, n + 1)]
        order_scores: List[float] = []
        for k in range(n):
            h_vec, h_norm = _tfidf_vec(hyp_ns[k], doc_freq[k])
            if h_norm == 0.0:
                order_scores.append(0.0)
                continue
            ref_sims = 0.0
            for ref, ref_ns in zip(refs, ref_ns_list):
                r_vec, r_norm = _tfidf_vec(ref_ns[k], doc_freq[k])
                if r_norm == 0.0:
                    continue
                dot = sum(val * r_vec[g] for g, val in h_vec.items() if g in r_vec)
                cos = dot / (h_norm * r_norm)
                delta = float(len(hyp) - len(ref))
                length_penalty = math.exp(-(delta * delta) / (2.0 * sigma * sigma))
                ref_sims += cos * length_penalty
            order_scores.append(10.0 * (ref_sims / float(max(len(refs), 1))))
        img_scores.append(sum(order_scores) / float(n))

    return sum(img_scores) / float(len(img_scores))


def compute_caption_metrics(
    predictions_by_id: Dict[int, Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute CIDEr, BLEU-1, BLEU-4, ROUGE-L, and generation health diagnostics."""
    if not references:
        raise ValueError("Cannot compute caption metrics on 0 reference samples.")

    hyps_raw: List[str] = []
    hyps_tok: List[List[str]] = []
    refs_tok_list: List[List[List[str]]] = []

    for rec in references:
        img_id = rec["image_id"]
        if img_id not in predictions_by_id:
            raise KeyError(f"Missing caption prediction for image_id={img_id}")
        pred_text = predictions_by_id[img_id]["text"].strip()
        hyps_raw.append(pred_text)
        hyps_tok.append(tokenize_caption(pred_text))
        refs_tok_list.append([tokenize_caption(r) for r in rec["references"]])

    bleu_1 = compute_corpus_bleu(hyps_tok, refs_tok_list, max_n=1)
    bleu_4 = compute_corpus_bleu(hyps_tok, refs_tok_list, max_n=4)
    rouge_l = compute_rouge_l(hyps_tok, refs_tok_list)
    cider = compute_cider(hyps_tok, refs_tok_list)

    total_tokens = sum(len(t) for t in hyps_tok)
    unique_tokens = len({tok for seq in hyps_tok for tok in seq})
    empty_count = sum(1 for s in hyps_raw if not s)
    # Detect per-caption repetition collapse (ratio of unique tokens within a caption < 0.5 and len >= 4)
    repetitive_count = sum(
        1 for seq in hyps_tok if len(seq) >= 4 and (len(set(seq)) / float(len(seq))) < 0.6
    )

    return {
        "samples": len(references),
        "cider": cider,
        "bleu_1": bleu_1,
        "bleu_4": bleu_4,
        "rouge_l": rouge_l,
        "avg_len": total_tokens / float(len(hyps_tok)),
        "distinct_1": (unique_tokens / float(total_tokens)) if total_tokens > 0 else 0.0,
        "empty_rate": empty_count / float(len(hyps_raw)),
        "repetition_rate": repetitive_count / float(len(hyps_tok)),
    }


def resolve_precision(precision: str) -> Tuple[bool, Any, str]:
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
    raise ValueError(f"Unsupported precision '{precision}'.")


def run_caption_eval(
    model_path: str,
    model_base: Optional[str],
    eval_dir: Path,
    answers_file: Path,
    metrics_file: Path,
    conv_mode: str = "vicuna_v1",
    prompt_text: str = DEFAULT_PROMPT,
    precision: str = "auto",
    max_samples: int = 0,
    max_new_tokens: int = 32,
    save_every: int = 50,
    label: Optional[str] = None,
    untrained: bool = False,
    zero_gate: bool = False,
) -> Dict[str, Any]:
    """Run COCO Karpathy caption generation (resumable) and compute CIDEr/BLEU/ROUGE-L."""
    import torch
    from PIL import Image
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

    ref_file = eval_dir / "karpathy_test_references.jsonl"
    image_folder = eval_dir / "val2014"
    if not ref_file.exists():
        raise FileNotFoundError(f"Missing {ref_file}. Run --action prepare first.")

    with open(ref_file, "r", encoding="utf-8") as f:
        references = [json.loads(line) for line in f if line.strip()]
    if max_samples > 0:
        references = references[:max_samples]

    preds_by_id: Dict[int, Dict[str, Any]] = {}
    if answers_file.exists():
        with open(answers_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                preds_by_id[rec["image_id"]] = rec

    pending = [r for r in references if r["image_id"] not in preds_by_id]
    model_name = get_model_name_from_path(model_path)
    run_label = label or ("Untrained (Step 0)" if untrained else model_name)
    print(f"[eval] {run_label}: {len(references)} images ({len(preds_by_id)} cached, {len(pending)} pending).")

    load_4bit, torch_dtype, prec_desc = resolve_precision(precision)
    if pending:
        disable_torch_init()
        if untrained:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
        print(
            f"[eval] Loading model='{model_path}' (base={model_base}, untrained={untrained}, zero_gate={zero_gate}) in {prec_desc}..."
        )
        tokenizer, model, processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_4bit=load_4bit,
            load_video_tower=False,
            torch_dtype=torch_dtype,
            load_mm_projector=not untrained,
            zero_early_fusion_gate=zero_gate,
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
            for idx, rec in enumerate(tqdm(pending, desc=f"Caption ({run_label})"), start=1):
                img_id = rec["image_id"]
                if getattr(model.config, "mm_use_im_start_end", False):
                    qs_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
                else:
                    qs_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

                conv = conv_template.copy()
                conv.append_message(conv.roles[0], qs_prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(model.device)
                attention_mask = torch.ones_like(input_ids)

                image = Image.open(image_folder / rec["filename"]).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)
                image_tensor = image_tensor.to(device=model.device, dtype=input_dtype)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
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

                out_rec = {
                    "image_id": img_id,
                    "cocoid": rec["cocoid"],
                    "filename": rec["filename"],
                    "prompt": prompt_text,
                    "text": output_text,
                    "model_id": model_name,
                }
                preds_by_id[img_id] = out_rec
                ans_f.write(json.dumps(out_rec) + "\n")
                if idx % save_every == 0:
                    ans_f.flush()
            ans_f.flush()

        elapsed = time.time() - t0
        print(f"[eval] Finished {len(pending)} captions in {elapsed:.1f}s ({len(pending)/max(elapsed, 1e-6):.2f} img/s).")

        del model, tokenizer, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = compute_caption_metrics(preds_by_id, references)
    sample_previews = [
        {
            "image_id": r["image_id"],
            "filename": r["filename"],
            "prediction": preds_by_id[r["image_id"]]["text"],
            "reference_0": r["references"][0],
        }
        for r in references[:5]
    ]
    payload = {
        "label": run_label,
        "model_path": str(model_path),
        "model_base": model_base,
        "untrained": untrained,
        "zero_gate": zero_gate,
        "precision": prec_desc,
        "conv_mode": conv_mode,
        "prompt": prompt_text,
        "max_samples": max_samples,
        "answers_file": str(answers_file),
        "metrics": metrics,
        "sample_previews": sample_previews,
    }
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n=== COCO Caption Results: {run_label} (n={metrics['samples']}) ===")
    print(
        f"  CIDEr: {metrics['cider']:6.2f} | BLEU-1: {metrics['bleu_1']:5.2f} | "
        f"BLEU-4: {metrics['bleu_4']:5.2f} | ROUGE-L: {metrics['rouge_l']:5.2f} | "
        f"AvgLen: {metrics['avg_len']:4.1f} | Distinct-1: {metrics['distinct_1']:.3f} | "
        f"Rep%: {metrics['repetition_rate']*100:4.1f}% | Empty%: {metrics['empty_rate']*100:4.1f}%"
    )
    return payload


def generate_caption_comparison_report(
    models: List[Dict[str, Any]],
    output_dir: Path,
    names: Optional[List[str]] = None,
    drive_output_dir: Optional[Path] = None,
    extra_files_to_sync: Optional[List[Path]] = None,
) -> Path:
    """Build an N-model COCO Captioning comparison Markdown + JSON report and sync to Drive."""
    if not models:
        raise ValueError("At least one model metrics payload is required.")
    if names is not None and len(names) != len(models):
        raise ValueError(f"Expected {len(models)} names, got {len(names)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_names = [
        (names[i] if names is not None else models[i].get("label", f"Model {i+1}"))
        for i in range(len(models))
    ]
    ref_name = resolved_names[0]
    ref_cider = models[0]["metrics"]["cider"]

    md_lines = [
        "# COCO Image Captioning Benchmark (Karpathy Test Split)",
        "",
        f"## Metrics Summary (Δ CIDEr vs `{ref_name}`)",
        "",
        "| Model | Samples | CIDEr | Δ CIDEr | BLEU-1 | BLEU-4 | ROUGE-L | Avg Len | Distinct-1 | Rep (%) | Empty (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for idx, (name, mdl) in enumerate(zip(resolved_names, models)):
        m = mdl["metrics"]
        delta_str = "—" if idx == 0 else f"{m['cider'] - ref_cider:+.2f}"
        md_lines.append(
            f"| {name} | {m['samples']} | {m['cider']:.2f} | {delta_str} | "
            f"{m['bleu_1']:.2f} | {m['bleu_4']:.2f} | {m['rouge_l']:.2f} | "
            f"{m['avg_len']:.1f} | {m['distinct_1']:.3f} | "
            f"{m['repetition_rate']*100:.1f} | {m['empty_rate']*100:.1f} |"
        )

    previews_0 = models[0].get("sample_previews", [])
    if previews_0:
        md_lines.extend(["", "## Sample Captions (First 5 Test Images)", ""])
        for i, prev in enumerate(previews_0):
            md_lines.append(f"### Image {prev['image_id']} (`{prev['filename']}`)")
            md_lines.append(f"- **Ground Truth (COCO):** {prev['reference_0']}")
            for name, mdl in zip(resolved_names, models):
                p_list = mdl.get("sample_previews", [])
                pred_txt = p_list[i]["prediction"] if i < len(p_list) else ""
                md_lines.append(f"- **{name}:** {pred_txt!r}")
            md_lines.append("")

    md_text = "\n".join(md_lines) + "\n"
    report_path = output_dir / "caption_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    combined_json = {
        "names": resolved_names,
        "models": [{"name": n, **m} for n, m in zip(resolved_names, models)],
    }
    json_path = output_dir / "caption_comparison_results.json"
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
    parser = argparse.ArgumentParser(description="COCO Image Captioning Evaluation for Stage 1 / Stage 2 Video-LLaVA")
    parser.add_argument(
        "--action",
        required=True,
        choices=["prepare", "download_ckpt", "eval", "compare", "all"],
    )
    parser.add_argument("--eval_dir", type=Path, default=Path("/content/eval/coco_caption"))
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--reuse_val2014_dir", type=Path, default=Path("/content/eval/pope/val2014"))
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="all",
        choices=["checkpoint-800", "final", "checkpoint-4750", "all"],
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
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--zero_gate", action="store_true", help="Load mm_projector but zero early-fusion gate.")
    parser.add_argument("--baseline_model_path", type=str, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--skip_official_baseline", action="store_true")
    parser.add_argument("--answers_file", type=Path, default=None)
    parser.add_argument("--metrics_file", type=Path, default=None)
    parser.add_argument("--metrics_files", type=Path, nargs="+", default=None)
    parser.add_argument("--names", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("/content/eval/coco_caption/results"))
    parser.add_argument(
        "--drive_output_dir",
        type=Path,
        default=Path("/content/drive/MyDrive/Video-LLaVA/eval_results/coco_caption"),
    )
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp16", "bf16", "4bit"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.action == "prepare":
        prepare_caption_data(
            eval_dir=args.eval_dir,
            num_images=args.num_images,
            reuse_val2014_dir=args.reuse_val2014_dir,
        )
    elif args.action == "download_ckpt":
        download_checkpoint(args.ckpt_name, args.drive_ckpt_root, args.local_ckpt_root)
    elif args.action == "eval":
        model_path = args.model_path or str(
            args.local_ckpt_root / ("untrained_step0" if args.untrained else "checkpoint-800")
        )
        safe_tag = (args.label or Path(model_path).name).replace("/", "_").replace(" ", "_")
        answers_file = args.answers_file or (args.eval_dir / "answers" / f"{safe_tag}.jsonl")
        metrics_file = args.metrics_file or (args.output_dir / f"{safe_tag}_metrics.json")
        run_caption_eval(
            model_path=model_path,
            model_base=args.model_base,
            eval_dir=args.eval_dir,
            answers_file=answers_file,
            metrics_file=metrics_file,
            conv_mode=args.conv_mode,
            prompt_text=args.prompt,
            precision=args.precision,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            save_every=args.save_every,
            label=args.label,
            untrained=args.untrained,
            zero_gate=args.zero_gate,
        )
    elif args.action == "compare":
        if not args.metrics_files:
            raise ValueError("--metrics_files is required for --action compare.")
        loaded = []
        extra_sync: List[Path] = []
        for mf in args.metrics_files:
            with open(mf, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            loaded.append(m_data)
            extra_sync.append(mf)
            if m_data.get("answers_file"):
                extra_sync.append(Path(m_data["answers_file"]))
        generate_caption_comparison_report(
            models=loaded,
            output_dir=args.output_dir,
            names=args.names,
            drive_output_dir=args.drive_output_dir,
            extra_files_to_sync=extra_sync,
        )
    elif args.action == "all":
        prepare_caption_data(
            eval_dir=args.eval_dir,
            num_images=args.num_images,
            reuse_val2014_dir=args.reuse_val2014_dir,
        )
        ckpt_paths = download_checkpoint("all", args.drive_ckpt_root, args.local_ckpt_root)
        base_llm = args.model_base or DEFAULT_MODEL_BASE

        eval_specs = [
            {
                "name": "Untrained (Step 0)",
                "model_path": str(ckpt_paths["untrained_step0"]),
                "model_base": base_llm,
                "untrained": True,
                "zero_gate": False,
                "answers": args.eval_dir / "answers" / "cap_untrained.jsonl",
                "metrics": args.output_dir / "cap_untrained_metrics.json",
            },
            {
                "name": "Stage 1 Projector-Only (No Fusion)",
                "model_path": str(ckpt_paths["checkpoint-800"]),
                "model_base": base_llm,
                "untrained": False,
                "zero_gate": True,
                "answers": args.eval_dir / "answers" / "cap_proj_only.jsonl",
                "metrics": args.output_dir / "cap_proj_only_metrics.json",
            },
            {
                "name": "checkpoint-800 (Stage 1, step 800)",
                "model_path": str(ckpt_paths["checkpoint-800"]),
                "model_base": base_llm,
                "untrained": False,
                "zero_gate": False,
                "answers": args.eval_dir / "answers" / "cap_ckpt800.jsonl",
                "metrics": args.output_dir / "cap_ckpt800_metrics.json",
            },
            {
                "name": "final (Stage 1, step 867)",
                "model_path": str(ckpt_paths["final"]),
                "model_base": base_llm,
                "untrained": False,
                "zero_gate": False,
                "answers": args.eval_dir / "answers" / "cap_final.jsonl",
                "metrics": args.output_dir / "cap_final_metrics.json",
            },
            {
                "name": "checkpoint-4750 (Stage 1, step 4750)",
                "model_path": str(ckpt_paths["checkpoint-4750"]),
                "model_base": base_llm,
                "untrained": False,
                "zero_gate": False,
                "answers": args.eval_dir / "answers" / "cap_ckpt4750.jsonl",
                "metrics": args.output_dir / "cap_ckpt4750_metrics.json",
            },
        ]
        if not args.skip_official_baseline:
            eval_specs.append(
                {
                    "name": "Official Video-LLaVA-7B (Stage 2 Ref)",
                    "model_path": args.baseline_model_path,
                    "model_base": None,
                    "untrained": False,
                    "zero_gate": False,
                    "answers": args.eval_dir / "answers" / "cap_official_ref.jsonl",
                    "metrics": args.output_dir / "cap_official_ref_metrics.json",
                }
            )

        evaluated = []
        extra_sync: List[Path] = []
        for spec in eval_specs:
            res = run_caption_eval(
                model_path=spec["model_path"],
                model_base=spec["model_base"],
                eval_dir=args.eval_dir,
                answers_file=spec["answers"],
                metrics_file=spec["metrics"],
                conv_mode=args.conv_mode,
                prompt_text=args.prompt,
                precision=args.precision,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                save_every=args.save_every,
                label=spec["name"],
                untrained=spec["untrained"],
                zero_gate=spec["zero_gate"],
            )
            evaluated.append(res)
            extra_sync.extend([spec["metrics"], spec["answers"]])

        generate_caption_comparison_report(
            models=evaluated,
            output_dir=args.output_dir,
            names=[s["name"] for s in eval_specs],
            drive_output_dir=args.drive_output_dir,
            extra_files_to_sync=extra_sync,
        )


if __name__ == "__main__":
    main()
