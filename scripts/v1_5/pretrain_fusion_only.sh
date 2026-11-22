#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

JSON_FOLDER="${PROJECT_ROOT}/llava_all_image_video/pt_json"
IMAGE_FOLDER="${PROJECT_ROOT}/llava_all_image_video"

NUM_GPUS=${NUM_GPUS:-3}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"./scripts/zero3.json"}

MODEL_PATH=${MODEL_PATH:-"/home/aitech/hdd/videollava_models/vicuna-7b-v1.5"}
IMAGE_TOWER_PATH=${IMAGE_TOWER_PATH:-"/home/aitech/hdd/videollava_models/LanguageBind_Image"}
PRETRAIN_PROJECTOR_PATH=${PRETRAIN_PROJECTOR_PATH:-"/home/aitech/hdd/videollava_models/mm_projector.bin"}

DATA_PATH=${DATA_PATH:-"${JSON_FOLDER}/llava_image_.json"}
OUT_DIR=${OUT_DIR:-"./checkpoints/videollava-7b-fusion-lr5e-5"}
LR=${LR:-"5e-5"}
NUM_EPOCHS=${NUM_EPOCHS:-"1"}
MHA_FUSION_HEADS=${MHA_FUSION_HEADS:-"8"}
SAVE_STEPS=${SAVE_STEPS:-250}

export GDRIVE_REMOTE=${GDRIVE_REMOTE:-"gdrive:Video-LLaVA/checkpoints"}
export GDRIVE_AUTO_UPLOAD=${GDRIVE_AUTO_UPLOAD:-"0"}
export PATH="$HOME/.local/bin:$PATH"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --num_gpus ${NUM_GPUS} videollava/train/train_mem.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${IMAGE_TOWER_PATH} \
    --pretrain_mm_mlp_adapter ${PRETRAIN_PROJECTOR_PATH} \
    --freeze_mm_projector True \
    --mha_fusion_heads ${MHA_FUSION_HEADS} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 2 \
    --learning_rate ${LR} \
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
