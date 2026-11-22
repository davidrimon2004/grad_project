#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

JSON_FOLDER="${PROJECT_ROOT}/llava_all_image_video/pt_json"
IMAGE_FOLDER="${PROJECT_ROOT}/llava_all_image_video"

NUM_GPUS=${NUM_GPUS:-3}
# Conservative hardware settings for 16GB VRAM GPUs:
# micro-batch=1 and gradient_accumulation=16 ensures no OOM while maintaining effective batch size of 48 (3 GPUs * 1 * 16)
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"./scripts/zero3.json"}

MODEL_PATH=${MODEL_PATH:-"/home/aitech/hdd/videollava_models/vicuna-7b-v1.5"}
if [ ! -d "${MODEL_PATH}" ]; then
    MODEL_PATH="lmsys/vicuna-7b-v1.5"
fi

IMAGE_TOWER_PATH=${IMAGE_TOWER_PATH:-"/home/aitech/hdd/videollava_models/LanguageBind_Image"}
if [ ! -d "${IMAGE_TOWER_PATH}" ]; then
    IMAGE_TOWER_PATH="LanguageBind/LanguageBind_Image"
fi

PRETRAIN_PROJECTOR_PATH=${PRETRAIN_PROJECTOR_PATH:-"/home/aitech/hdd/videollava_models/mm_projector.bin"}
PROJECTOR_ARG=""
if [ -f "${PRETRAIN_PROJECTOR_PATH}" ]; then
    PROJECTOR_ARG="--pretrain_mm_mlp_adapter ${PRETRAIN_PROJECTOR_PATH}"
fi

FREEZE_MM_PROJECTOR=${FREEZE_MM_PROJECTOR:-True}
FREEZE_PROJECTOR_ARG=""
if [ "${FREEZE_MM_PROJECTOR}" = "True" ]; then
    FREEZE_PROJECTOR_ARG="--freeze_mm_projector True"
fi

# Optional reusable embedding cache (pre-computed LanguageBind [256, 1024] + pre-tokenized text)
EMBED_CACHE_DIR=${EMBED_CACHE_DIR:-""}
EMBED_CACHE_ARG=""
if [ -n "${EMBED_CACHE_DIR}" ] && [ -d "${EMBED_CACHE_DIR}" ]; then
    EMBED_CACHE_ARG="--embed_cache_dir ${EMBED_CACHE_DIR}"
fi

DATA_PATH=${DATA_PATH:-"${JSON_FOLDER}/llava_image_.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/videollava-7b-pretrain"}
LEARNING_RATE=${LEARNING_RATE:-"5e-5"}
NUM_EPOCHS=${NUM_EPOCHS:-"1"}
MHA_FUSION_HEADS=${MHA_FUSION_HEADS:-"8"}
SAVE_STEPS=${SAVE_STEPS:-50}

export GDRIVE_REMOTE=${GDRIVE_REMOTE:-"gdrive:Video-LLaVA/checkpoints"}
export GDRIVE_AUTO_UPLOAD=${GDRIVE_AUTO_UPLOAD:-"1"}
export PATH="$HOME/.local/bin:$PATH"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --num_gpus ${NUM_GPUS} videollava/train/train_mem.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${IMAGE_TOWER_PATH} \
    ${PROJECTOR_ARG} \
    ${FREEZE_PROJECTOR_ARG} \
    ${EMBED_CACHE_ARG} \
    --mha_fusion_heads ${MHA_FUSION_HEADS} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 3 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
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
