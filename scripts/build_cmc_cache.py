"""
Build the CMC (Camera Motion Compensation) affine cache.

For each selected val sequence, reads the raw infrared video frame-by-frame
at the configured stride and estimates the partial affine between consecutive
SAMPLED frames via sparse Lucas-Kanade optical flow + RANSAC.

Saves one JSON per sequence to cache/affines/<seq_name>.json.
Each file records frame_stride so stale caches are detected at load time.

Usage
-----
    python scripts/build_cmc_cache.py --config configs/default.yaml
    python scripts/build_cmc_cache.py --config configs/default.yaml --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import dataset_from_config
from src.tracker.cmc import CMC, save_affine_cache, load_affine_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if cache exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir   = Path(cfg["paths"]["cache_affines"])
    frame_stride = int(cfg["data"]["frame_stride"])

    print(f"[build_cmc_cache]  frame_stride={frame_stride}")
    print(f"[build_cmc_cache]  cache_dir={cache_dir}\n")

    dataset = dataset_from_config(cfg)
    cmc = CMC(cfg)

    t0 = time.time()
    for i, seq in enumerate(dataset):
        # Check existing cache
        if not args.force:
            try:
                existing = load_affine_cache(seq.name, cache_dir, frame_stride)
                print(f"[{i+1:>2}/{len(dataset)}] {seq.name}: cached, skipping")
                continue
            except FileNotFoundError:
                pass
            except ValueError as e:
                print(f"[{i+1:>2}/{len(dataset)}] {seq.name}: {e}  → rebuilding")

        print(f"[{i+1:>2}/{len(dataset)}] {seq.name} ({len(seq)} sampled frames)…",
              end=" ", flush=True)
        t1 = time.time()

        cmc.reset()
        affines = []

        for record in seq:
            frame = seq.get_frame(record.frame_idx)
            if frame is None:
                # Store identity if frame cannot be decoded
                M = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            else:
                M_np = cmc.apply(frame)
                M = M_np.tolist()

            affines.append({"frame_idx": record.frame_idx, "M": M})

        save_affine_cache(seq.name, frame_stride, affines, cache_dir)
        n_identity = sum(
            1 for a in affines
            if abs(a["M"][0][0]-1)<1e-6 and abs(a["M"][0][2])<1e-3 and abs(a["M"][1][2])<1e-3
        )
        print(f"  [{time.time()-t1:.1f}s]  identity_affines={n_identity}/{len(affines)}")

    total = time.time() - t0
    n_files = len(list(cache_dir.glob("*.json")))
    print(f"\nDone. {n_files} affine cache files in {cache_dir}  ({total:.1f}s total)")


if __name__ == "__main__":
    main()
