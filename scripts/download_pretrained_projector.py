#!/usr/bin/env python3
"""
download_pretrained_projector.py
=================================
Downloads the official pretrained Video-LLaVA mm_projector weights from HuggingFace.

Usage:
    python scripts/download_pretrained_projector.py [--output_dir ./checkpoints/downloaded_projector]
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser(description="Download official pretrained Video-LLaVA mm_projector")
    parser.add_argument(
        "--repo_id",
        default="LanguageBind/Video-LLaVA-Pretrain-7B",
        help="HuggingFace repository ID (default: LanguageBind/Video-LLaVA-Pretrain-7B)"
    )
    parser.add_argument(
        "--output_dir",
        default="./checkpoints/downloaded_projector",
        help="Directory to save mm_projector.bin"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Downloading mm_projector.bin from {args.repo_id}...")

    downloaded_path = hf_hub_download(
        repo_id=args.repo_id,
        filename="mm_projector.bin",
        local_dir=args.output_dir
    )
    print(f"Successfully downloaded to: {downloaded_path}")
    print("\nTo use this with pretraining (training ONLY early-fusion MHA + Gate v2):")
    print(f"    --pretrain_mm_mlp_adapter {downloaded_path}")
    print("    --freeze_mm_projector True")

if __name__ == "__main__":
    main()
