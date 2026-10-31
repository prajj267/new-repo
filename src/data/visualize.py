"""
Draw ground-truth bounding boxes on sample frames and save as JPEG images.

Stage 1 sanity check: inspect results/viz/*.jpg to verify that green boxes
sit on the UAV and that frames where exist=0 show no box.

Usage:
    python src/data/visualize.py --config configs/default.yaml
    python src/data/visualize.py --config configs/default.yaml --n 20 --seed 7
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import AntiUAVDataset, AntiUAVSequence, FrameRecord, dataset_from_config

# Colours (BGR)
BOX_COLOUR = (0, 220, 0)       # green  — visible UAV
ABSENT_COLOUR = (0, 0, 200)    # red    — absent label
TEXT_COLOUR = (255, 255, 255)  # white


def draw_frame(
    frame: np.ndarray,
    record: FrameRecord,
    seq_name: str,
) -> np.ndarray:
    """Draw one annotation onto a copy of frame.

    Args:
        frame:    BGR image (H, W, 3).
        record:   FrameRecord for this frame.
        seq_name: Sequence name shown as overlay text.
    Returns:
        Annotated copy of frame.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    header = f"{seq_name}  f={record.frame_idx}"
    cv2.putText(out, header, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, TEXT_COLOUR, 2, cv2.LINE_AA)

    if record.visible and record.box_xyxy is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in record.box_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOUR, 2)
        bw, bh = x2 - x1, y2 - y1
        cv2.putText(
            out, f"{bw}x{bh}px",
            (x1, max(y1 - 5, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLOUR, 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            out, "exist=0 (no box)",
            (6, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, ABSENT_COLOUR, 1, cv2.LINE_AA,
        )

    return out


def visualize(
    dataset: AntiUAVDataset,
    out_dir: Path,
    n: int = 20,
    seed: int = 0,
) -> None:
    """Save n annotated frames to out_dir.

    Tries to include ~20% frames with exist=0 so both code paths are checked.

    Args:
        dataset: Loaded AntiUAVDataset.
        out_dir: Output directory (created if absent).
        n:       Number of frames to save.
        seed:    Random seed for frame selection.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect (seq, record) pairs from all sequences
    visible_pairs: list = []
    absent_pairs: list = []

    for seq in dataset:
        for record in seq:
            pair = (seq, record)
            if record.visible:
                visible_pairs.append(pair)
            else:
                absent_pairs.append(pair)

    rng = random.Random(seed)
    n_absent = min(max(n // 5, 1), len(absent_pairs))
    n_visible = n - n_absent

    chosen = (
        rng.sample(visible_pairs, min(n_visible, len(visible_pairs))) +
        rng.sample(absent_pairs, n_absent)
    )
    rng.shuffle(chosen)

    saved = 0
    skipped = 0
    for seq, record in chosen:
        frame = seq.get_frame(record.frame_idx)
        if frame is None:
            print(f"  WARNING: could not decode frame {record.frame_idx} from {seq.name}")
            skipped += 1
            continue

        annotated = draw_frame(frame, record, seq.name)
        fname = f"{seq.name}_{record.frame_idx:05d}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1

    print(f"Saved {saved} images to {out_dir}  (skipped {skipped} missing frames)")
    if saved > 0:
        print("INSPECT MANUALLY:")
        print("  Green box must sit on the UAV.")
        print("  'exist=0' frames must have no box.")
        print("  If boxes are shifted, fix xywh_tl_to_xyxy in loader.py.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise Anti-UAV GT annotations.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n", type=int, default=20, help="Number of frames to save")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Override output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = dataset_from_config(cfg)
    print(f"Dataset: {dataset}")
    print(f"Sequences: {dataset.sequence_names}")
    print(f"Total sampled frames: {dataset.total_sampled_frames()}")

    out_dir = Path(args.out) if args.out else Path("results/viz")
    visualize(dataset, out_dir, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
