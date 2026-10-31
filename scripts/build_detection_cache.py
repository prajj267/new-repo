"""
Build the detection cache for all selected val sequences.

Current mode: GT simulation (no pretrained weights available).
See src/detector/yolov5_wrapper.py and design.md §4.1 for the disclaimer.

Saves one JSON file per sequence to cache/detections/<seq_name>.json.
Each file records frame_stride, detector_weights, torch_version.

Amendment B recall gate
───────────────────────
After the first 3 sequences, recall@0.50 and recall@0.25 are printed.
  • Simulation mode: recall is always ~0.95 (matches GT minus dropout).
    Gate is informational only and never blocks in simulation mode.
  • Real detector: gate blocks if recall@0.25 < 5%.

Usage
─────
    python scripts/build_detection_cache.py --config configs/default.yaml
    python scripts/build_detection_cache.py --config configs/default.yaml --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import AntiUAVDataset, dataset_from_config
from src.detector.yolov5_wrapper import YOLOv5Detector, compute_recall

RECALL_GATE_SEQS = 3
RECALL_MIN = 0.05


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_path(seq_name: str, cache_dir: Path) -> Path:
    return cache_dir / f"{seq_name}.json"


def load_cache_validate(seq_name: str, cache_dir: Path, expected_stride: int) -> dict | None:
    """Load cache, validating frame_stride. Returns None if not found.
    Raises ValueError on stride mismatch (Amendment C).
    """
    p = cache_path(seq_name, cache_dir)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    cached = data.get("frame_stride")
    if cached != expected_stride:
        raise ValueError(
            f"Cache stride mismatch for {seq_name}: "
            f"cached={cached}, config={expected_stride}. "
            "Re-run with --force to rebuild."
        )
    return data


def save_cache(seq_name, frame_stride, detector_weights, torch_version, frames, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_path(seq_name, cache_dir)
    with open(p, "w") as f:
        json.dump({
            "seq_name":        seq_name,
            "frame_stride":    frame_stride,
            "detector_weights": detector_weights,
            "torch_version":   torch_version,
            "frames":          frames,
        }, f)
    return p


# ---------------------------------------------------------------------------
# Process one sequence
# ---------------------------------------------------------------------------

def process_sequence(seq, detector: YOLOv5Detector) -> tuple:
    """Run the detector over all sampled frames.

    Returns (frames_data, all_dets, all_records).
    """
    frames_data = []
    all_dets    = []
    all_records = []

    for record in seq:
        if detector.is_simulation:
            dets = detector.detect_from_gt(record.box_xyxy, record.visible)
        else:
            frame_bgr = seq.get_frame(record.frame_idx)
            dets = detector.detect(frame_bgr) if frame_bgr is not None else []

        frames_data.append({
            "frame_idx": record.frame_idx,
            "boxes":     [d["box_xyxy"] for d in dets],
            "scores":    [d["score"]    for d in dets],
        })
        all_dets.append(dets)
        all_records.append(record)

    n_with_dets = sum(1 for f in frames_data if f["boxes"])
    print(f"  frames={len(frames_data)}  with_dets={n_with_dets}", end="")
    return frames_data, all_dets, all_records


# ---------------------------------------------------------------------------
# Recall gate
# ---------------------------------------------------------------------------

def recall_gate(all_dets_flat, all_records_flat, cfg, is_simulation: bool) -> None:
    img_w = int(cfg["data"]["img_w"])
    img_h = int(cfg["data"]["img_h"])
    r50 = compute_recall(all_dets_flat, all_records_flat, 0.50, img_w, img_h)
    r25 = compute_recall(all_dets_flat, all_records_flat, 0.25, img_w, img_h)
    n_gt   = sum(1 for r in all_records_flat if r.visible)
    n_dets = sum(len(d) for d in all_dets_flat)

    print(f"\n{'='*55}")
    print(f"  RECALL GATE (first {RECALL_GATE_SEQS} sequences)")
    print(f"  GT visible frames : {n_gt}")
    print(f"  Total detections  : {n_dets}")
    print(f"  Recall@0.50       : {r50:.4f}")
    print(f"  Recall@0.25       : {r25:.4f}")
    if is_simulation:
        print(f"  Mode: GT simulation — gate is informational only")
    print(f"{'='*55}\n")

    if not is_simulation and r25 < RECALL_MIN:
        print("STOPPING: recall@0.25 < 5% gate threshold.")
        print("Options:")
        print("  1. detector.conf_thresh: 0.01")
        print("  2. detector.imgsz: 1280")
        print("  3. detector.class_agnostic: true")
        print("  4. Provide Anti-UAV baseline weights in reference/")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import torch
    torch_version    = torch.__version__
    cache_dir        = Path(cfg["paths"]["cache_detections"])
    frame_stride     = int(cfg["data"]["frame_stride"])
    detector_weights = cfg["detector"]["weights"]

    print(f"[build_detection_cache]  torch={torch_version}")
    print(f"[build_detection_cache]  weights={detector_weights}")
    print(f"[build_detection_cache]  frame_stride={frame_stride}")
    print(f"[build_detection_cache]  cache_dir={cache_dir}\n")

    dataset = dataset_from_config(cfg)
    print(f"Dataset: {dataset}")
    print(f"Sequences: {dataset.sequence_names}\n")

    detector = YOLOv5Detector(cfg)

    all_dets_gate: list    = []
    all_records_gate: list = []
    gate_checked = False
    t0 = time.time()

    for i, seq in enumerate(dataset):
        # Check / validate existing cache
        if not args.force:
            try:
                existing = load_cache_validate(seq.name, cache_dir, frame_stride)
                if existing is not None:
                    print(f"[{i+1:>2}/{len(dataset)}] {seq.name}: cached, skipping")
                    continue
            except ValueError as e:
                print(f"[{i+1:>2}/{len(dataset)}] {seq.name}: {e}  → rebuilding")

        print(f"[{i+1:>2}/{len(dataset)}] {seq.name} ({len(seq)} frames)…", end=" ")
        t1 = time.time()

        frames_data, dets, records = process_sequence(seq, detector)

        save_cache(seq.name, frame_stride, detector_weights,
                   torch_version, frames_data, cache_dir)
        print(f"  [{time.time()-t1:.1f}s]")

        # Accumulate for recall gate
        if not gate_checked:
            all_dets_gate.extend(dets)
            all_records_gate.extend(records)
            if i + 1 >= RECALL_GATE_SEQS:
                recall_gate(all_dets_gate, all_records_gate, cfg, detector.is_simulation)
                gate_checked = True

    total = time.time() - t0
    n_files = len(list(cache_dir.glob("*.json")))
    print(f"\nDone. {n_files} cache files in {cache_dir}  ({total:.1f}s total)")


if __name__ == "__main__":
    main()
