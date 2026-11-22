#!/usr/bin/env bash
# Precompute reusable LanguageBind [256, 1024] image embeddings + pre-tokenized text
# directly onto the HDD (/home/aitech/hdd/videollava_embeddings).
#
# Usage:
#   bash scripts/precompute_hdd_cache.sh text      # CPU-only (~300 MB, safe while GPUs are training!)
#   bash scripts/precompute_hdd_cache.sh subsets   # Create 10k/50k/100k ablation JSON subsets on HDD
#   bash scripts/precompute_hdd_cache.sh images    # Multi-GPU [256, 1024] fp16 visual cache (~230 GB on HDD; run when GPUs are free)
#   bash scripts/precompute_hdd_cache.sh all       # Run text + subsets + images
#   bash scripts/precompute_hdd_cache.sh eval <json_path> <image_folder> <split_name>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate videollava

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATA_BASE="${DATA_BASE:-/home/aitech/hdd/videollava_data}"
CACHE_OUT="${CACHE_OUT:-/home/aitech/hdd/videollava_embeddings}"
MODEL_PATH="${MODEL_PATH:-/home/aitech/hdd/videollava_models/vicuna-7b-v1.5}"
IMAGE_TOWER="${IMAGE_TOWER:-/home/aitech/hdd/videollava_models/LanguageBind_Image}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-192}"
IMAGES_PER_SHARD="${IMAGES_PER_SHARD:-20000}"

mkdir -p "${CACHE_OUT}"

MODE="${1:-help}"

run_text() {
    echo ">>> Pre-tokenizing pretrain text (CPU-only, compact ~300 MB) -> ${CACHE_OUT}"
    CUDA_VISIBLE_DEVICES="" python3 scripts/precompute_embeddings.py \
        --action text \
        --split pretrain \
        --image_only \
        --drive_base "${DATA_BASE}" \
        --output_dir "${CACHE_OUT}" \
        --model_name_or_path "${MODEL_PATH}" \
        --no_staging
}

run_subsets() {
    echo ">>> Creating reproducible 10k / 50k / 100k ablation subsets -> ${CACHE_OUT}/subsets"
    python3 scripts/create_ablation_subsets.py \
        --input "${DATA_BASE}/pt_json/llava_image_.json" \
        --output_dir "${CACHE_OUT}/subsets" \
        --sizes 10000 50000 100000 \
        --seed 42
}

run_images() {
    echo ">>> Precomputing LanguageBind [256, 1024] fp16 image embeddings -> ${CACHE_OUT}"
    echo "    (Pre-projector features: reusable across all early-fusion & projector ablations)"
    python3 scripts/precompute_embeddings.py \
        --action images \
        --split pretrain \
        --image_only \
        --image_aspect_ratio square \
        --drive_base "${DATA_BASE}" \
        --image_folder "${DATA_BASE}" \
        --output_dir "${CACHE_OUT}" \
        --image_tower "${IMAGE_TOWER}" \
        --dtype bf16 \
        --image_batch_size "${IMAGE_BATCH_SIZE}" \
        --images_per_shard "${IMAGES_PER_SHARD}" \
        --num_workers "${NUM_WORKERS}" \
        --multi_gpu \
        --no_staging
}

case "${MODE}" in
    text)
        run_text
        ;;
    subsets)
        run_subsets
        ;;
    images)
        run_images
        ;;
    all)
        run_subsets
        run_text
        run_images
        ;;
    eval)
        JSON_P="${2:?Usage: $0 eval <json_path> <image_folder> <split_name>}"
        IMG_DIR="${3:?Usage: $0 eval <json_path> <image_folder> <split_name>}"
        SPLIT_N="${4:?Usage: $0 eval <json_path> <image_folder> <split_name>}"
        python3 scripts/precompute_embeddings.py \
            --action images \
            --image_only \
            --image_aspect_ratio pad \
            --json_path "${JSON_P}" \
            --custom_split_name "${SPLIT_N}" \
            --drive_base "${DATA_BASE}" \
            --image_folder "${IMG_DIR}" \
            --output_dir "${CACHE_OUT}/eval" \
            --image_tower "${IMAGE_TOWER}" \
            --dtype bf16 \
            --image_batch_size "${IMAGE_BATCH_SIZE}" \
            --images_per_shard 20000 \
            --num_workers "${NUM_WORKERS}" \
            --multi_gpu \
            --no_staging
        ;;
    *)
        echo "Usage: $0 {text|subsets|images|all|eval <json> <img_dir> <split_name>}"
        exit 1
        ;;
esac
