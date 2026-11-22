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

SAVE_STEPS=${SAVE_STEPS:-250}

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --num_gpus ${NUM_GPUS} videollava/train/train_mem.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${JSON_FOLDER}/llava_image_.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${IMAGE_TOWER_PATH} \
    ${PROJECTOR_ARG} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/videollava-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
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
