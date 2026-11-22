#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

JSON_FOLDER="${PROJECT_ROOT}/llava_all_image_video/ft_json"
IMAGE_FOLDER="${PROJECT_ROOT}/llava_all_image_video"

NUM_GPUS=${NUM_GPUS:-3}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"./scripts/zero3_offload.json"}

MODEL_PATH=${MODEL_PATH:-"/home/aitech/hdd/videollava_models/vicuna-7b-v1.5"}
if [ ! -d "${MODEL_PATH}" ]; then
    MODEL_PATH="lmsys/vicuna-7b-v1.5"
fi

IMAGE_TOWER_PATH=${IMAGE_TOWER_PATH:-"/home/aitech/hdd/videollava_models/LanguageBind_Image"}
if [ ! -d "${IMAGE_TOWER_PATH}" ]; then
    IMAGE_TOWER_PATH="LanguageBind/LanguageBind_Image"
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --num_gpus ${NUM_GPUS} videollava/train/train_mem.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${IMAGE_TOWER_PATH} \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/videollava-7b-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/videollava-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
