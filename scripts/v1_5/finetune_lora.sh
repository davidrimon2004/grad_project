#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

JSON_FOLDER="${PROJECT_ROOT}/llava_all_image_video/ft_json"
IMAGE_FOLDER="${PROJECT_ROOT}/llava_all_image_video"

NUM_GPUS=${NUM_GPUS:-3}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
# zero3.json keeps per-GPU memory well under 16GB (3x RTX 4060 Ti).
# Override with DEEPSPEED_CONFIG=./scripts/zero2_offload.json on 24GB+ GPUs if desired.
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"./scripts/zero3.json"}

MODEL_PATH=${MODEL_PATH:-"/home/aitech/hdd/videollava_models/vicuna-7b-v1.5"}
if [ ! -d "${MODEL_PATH}" ]; then
    MODEL_PATH="lmsys/vicuna-7b-v1.5"
fi

IMAGE_TOWER_PATH=${IMAGE_TOWER_PATH:-"/home/aitech/hdd/videollava_models/LanguageBind_Image"}
if [ ! -d "${IMAGE_TOWER_PATH}" ]; then
    IMAGE_TOWER_PATH="LanguageBind/LanguageBind_Image"
fi

# Best Stage 1 checkpoint (pre-spike step 800: CIDEr 1.01, ROUGE-L 50.01)
PRETRAIN_ADAPTER=${PRETRAIN_ADAPTER:-"./checkpoints/videollava-7b-pretrain/checkpoint-800/mm_projector.bin"}
if [ ! -f "${PRETRAIN_ADAPTER}" ]; then
    echo "[ERROR] PRETRAIN_ADAPTER not found: ${PRETRAIN_ADAPTER}" >&2
    echo "        Pass PRETRAIN_ADAPTER=/path/to/checkpoint-800/mm_projector.bin or place it at the default path." >&2
    exit 1
fi

DATA_PATH=${DATA_PATH:-"${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/nlp_tune.json"}
for json_file in ${DATA_PATH}; do
    if [ ! -f "${json_file}" ]; then
        echo "[ERROR] Training JSON not found: ${json_file}" >&2
        exit 1
    fi
done

FREEZE_MM_PROJECTOR=${FREEZE_MM_PROJECTOR:-False}
FREEZE_PROJECTOR_ARG=""
if [ "${FREEZE_MM_PROJECTOR}" = "True" ]; then
    FREEZE_PROJECTOR_ARG="--freeze_mm_projector True"
fi

EMBED_CACHE_DIR=${EMBED_CACHE_DIR:-""}
EMBED_CACHE_ARG=""
if [ -n "${EMBED_CACHE_DIR}" ] && [ -d "${EMBED_CACHE_DIR}" ]; then
    EMBED_CACHE_ARG="--embed_cache_dir ${EMBED_CACHE_DIR}"
fi

OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/videollava-7b-lora"}
LORA_R=${LORA_R:-128}
LORA_ALPHA=${LORA_ALPHA:-256}
LEARNING_RATE=${LEARNING_RATE:-"2e-4"}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-"2e-5"}
MHA_FUSION_HEADS=${MHA_FUSION_HEADS:-"8"}
NUM_EPOCHS=${NUM_EPOCHS:-"1"}
SAVE_STEPS=${SAVE_STEPS:-250}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
WARMUP_RATIO=${WARMUP_RATIO:-"0.05"}

export GDRIVE_REMOTE=${GDRIVE_REMOTE:-"gdrive:Video-LLaVA/checkpoints"}
export GDRIVE_AUTO_UPLOAD=${GDRIVE_AUTO_UPLOAD:-"1"}
export STATUS_FILE=${STATUS_FILE:-"${OUTPUT_DIR}/finetune_lora_status.txt"}
export DIVERGE_GUARD=${DIVERGE_GUARD:-"1"}
export DIVERGE_RATIO=${DIVERGE_RATIO:-"2.0"}
export PATH="$HOME/.local/bin:$PATH"

echo "============================================================"
echo " Video-LLaVA Stage 2 LoRA + Early-Fusion Gate v2"
echo "============================================================"
echo "  MODEL_PATH:        ${MODEL_PATH}"
echo "  IMAGE_TOWER_PATH:  ${IMAGE_TOWER_PATH}"
echo "  PRETRAIN_ADAPTER:  ${PRETRAIN_ADAPTER}"
echo "  DATA_PATH:         ${DATA_PATH}"
echo "  OUTPUT_DIR:        ${OUTPUT_DIR}"
echo "  DEEPSPEED_CONFIG:  ${DEEPSPEED_CONFIG}"
echo "  LEARNING_RATE:     ${LEARNING_RATE} (LoRA)"
echo "  MM_PROJECTOR_LR:   ${MM_PROJECTOR_LR} (mm_projector + early_fusion)"
echo "  SAVE_STEPS:        ${SAVE_STEPS} (keep ${SAVE_TOTAL_LIMIT})"
echo "  GDRIVE_REMOTE:     ${GDRIVE_REMOTE} (auto=${GDRIVE_AUTO_UPLOAD})"
echo "============================================================"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --num_gpus ${NUM_GPUS} videollava/train/train_mem.py \
    --lora_enable True --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA} --mm_projector_lr ${MM_PROJECTOR_LR} \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${IMAGE_TOWER_PATH} \
    ${FREEZE_PROJECTOR_ARG} \
    ${EMBED_CACHE_ARG} \
    --mha_fusion_heads ${MHA_FUSION_HEADS} \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ${PRETRAIN_ADAPTER} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --tokenizer_model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
