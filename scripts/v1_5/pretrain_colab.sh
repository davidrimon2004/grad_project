#!/bin/bash
# ==============================================================================
# Video-LLaVA Stage 1 Pretraining on Google Colab with Google Drive Data Muling
# ==============================================================================
# Usage:
#   bash scripts/v1_5/pretrain_colab.sh [--demo]
# ==============================================================================

set -e

DRIVE_ROOT="/content/drive/MyDrive/Video-LLaVA"
LOCAL_DATA="/content/data"
LOCAL_CKPT="/content/checkpoints/videollava-7b-pretrain"

echo "========================================================"
echo " Starting Video-LLaVA Pretraining with Data Muling"
echo " Google Drive Root: ${DRIVE_ROOT}"
echo " Local Ephemeral SSD: ${LOCAL_DATA}"
echo "========================================================"

if [ "$1" == "--demo" ]; then
    echo "Running with Demo verification subset..."
    python scripts/colab_pretrain_muler.py \
        --action all \
        --demo_samples 100 \
        --drive_root "${DRIVE_ROOT}" \
        --local_scratch_dir "${LOCAL_DATA}" \
        --local_output_dir "${LOCAL_CKPT}" \
        --num_train_epochs 1.0 \
        --save_steps 50
else
    echo "Running full Video-LLaVA Stage 1 Pretraining..."
    python scripts/colab_pretrain_muler.py \
        --action all \
        --drive_root "${DRIVE_ROOT}" \
        --local_scratch_dir "${LOCAL_DATA}" \
        --local_output_dir "${LOCAL_CKPT}" \
        --num_train_epochs 1.0 \
        --save_steps 500
fi
