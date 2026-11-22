#!/usr/bin/env python3
"""
create_ablation_subsets.py
==========================
Creates deterministic, reproducible random subsets (e.g. 10k, 50k, 100k samples)
of `llava_image_.json` so multiple early-fusion ablations can run 5-50x faster
while reusing the EXACT same precomputed HDF5 embedding + text cache on HDD.

Usage:
  python scripts/create_ablation_subsets.py \
      --input /home/aitech/hdd/videollava_data/pt_json/llava_image_.json \
      --output_dir /home/aitech/hdd/videollava_embeddings/subsets \
      --sizes 10000 50000 100000 --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible ablation subset JSONs.")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/aitech/hdd/videollava_data/pt_json/llava_image_.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/aitech/hdd/videollava_embeddings/subsets",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10000, 50000, 100000],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {inp} ...", flush=True)
    data = json.loads(inp.read_text(encoding="utf-8"))
    image_only = [s for s in data if "image" in s and "video" not in s]
    print(f"Loaded {len(image_only):,} image-only samples.", flush=True)

    rng = random.Random(args.seed)
    indices = list(range(len(image_only)))
    rng.shuffle(indices)

    for size in sorted(args.sizes):
        k = min(size, len(image_only))
        # Preserve original ordering inside each subset for better HDD HDF5 locality
        chosen = sorted(indices[:k])
        subset = [image_only[i] for i in chosen]
        label = f"{k // 1000}k" if k % 1000 == 0 else str(k)
        out_path = out_dir / f"llava_image_{label}_seed{args.seed}.json"
        out_path.write_text(json.dumps(subset), encoding="utf-8")
        print(f"Wrote {len(subset):,} samples -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
