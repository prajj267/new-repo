"""
Evaluation utilities — MOTA, IDF1, FP, FN, IDS via motmetrics.

Frames where exist=0 have NO GT box.  Any tracker hypothesis on those
frames is counted as a false positive.  This is intentional and matches
the paper's evaluation protocol for single-target tracking.

Usage
-----
    from src.evaluate import evaluate_sequence, evaluate_dataset
"""

from __future__ import annotations

from typing import Dict, List, Optional

import motmetrics as mm
import numpy as np


# ---------------------------------------------------------------------------
# Per-sequence accumulator
# ---------------------------------------------------------------------------

def evaluate_sequence(
    gt_records,        # iterable of FrameRecord (from loader)
    tracker_outputs,   # iterable of (frame_idx, tracks_xyxy (M,4), track_ids (M,))
    iou_thresh: float = 0.5,
) -> mm.MOTAccumulator:
    """Build a motmetrics accumulator for one sequence.

    IMPORTANT: Frames with exist=0 have no GT box.  Any tracker hypothesis
    on such a frame is a false positive (counted in the accumulator).

    Args:
        gt_records:      Iterable of FrameRecord objects (loader output).
                         Only records where visible=True contribute GT.
        tracker_outputs: Iterable of (frame_idx, tracks_xyxy, track_ids).
                         tracks_xyxy : (M, 4) [x1,y1,x2,y2]
                         track_ids   : (M,)   integer IDs
                         Pass empty arrays for frames with no hypotheses.
        iou_thresh:      IoU threshold for a TP match (default 0.5).
    Returns:
        Filled motmetrics MOTAccumulator.
    """
    acc = mm.MOTAccumulator(auto_id=True)

    gt_map:  Dict[int, Optional[List[float]]] = {}
    for rec in gt_records:
        gt_map[rec.frame_idx] = rec.box_xyxy if rec.visible else None

    for frame_idx, tracks_xyxy, track_ids in tracker_outputs:
        # Ground truth for this frame
        gt_box = gt_map.get(frame_idx, None)
        if gt_box is not None:
            gt_ids   = [0]                    # single target → id always 0
            gt_boxes = np.array([gt_box])     # (1, 4)
        else:
            gt_ids   = []
            gt_boxes = np.empty((0, 4))

        # Hypotheses
        hyp_ids = list(track_ids.astype(int)) if len(track_ids) > 0 else []
        hyp_boxes = np.array(tracks_xyxy) if len(tracks_xyxy) > 0 else np.empty((0, 4))

        # Distance matrix (IoU-based; mm uses 1-IoU as distance)
        if len(gt_ids) > 0 and len(hyp_ids) > 0:
            dists = mm.distances.iou_matrix(
                gt_boxes, hyp_boxes, max_iou=1.0 - iou_thresh
            )
        else:
            dists = np.empty((len(gt_ids), len(hyp_ids)))

        acc.update(gt_ids, hyp_ids, dists)

    return acc


# ---------------------------------------------------------------------------
# Aggregate over multiple sequences
# ---------------------------------------------------------------------------

def evaluate_dataset(
    accs: List[mm.MOTAccumulator],
    seq_names: List[str],
) -> Dict[str, float]:
    """Compute aggregate MOTA, IDF1, FP, FN, IDS from a list of accumulators.

    Args:
        accs:      List of per-sequence accumulators.
        seq_names: Corresponding sequence names (for per-seq breakdown).
    Returns:
        Dict with keys: mota, idf1, fp, fn, ids, num_frames,
        and nested 'per_seq' dict.
    """
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=["num_frames", "mota", "idf1",
                 "num_false_positives", "num_misses", "num_switches"],
        names=seq_names,
        generate_overall=True,
    )

    overall = summary.loc["OVERALL"]

    per_seq = {}
    for name in seq_names:
        row = summary.loc[name]
        per_seq[name] = {
            "mota": _safe(row["mota"]),
            "idf1": _safe(row["idf1"]),
            "fp":   int(row["num_false_positives"]),
            "fn":   int(row["num_misses"]),
            "ids":  int(row["num_switches"]),
        }

    return {
        "mota":       _safe(overall["mota"]),
        "idf1":       _safe(overall["idf1"]),
        "fp":         int(overall["num_false_positives"]),
        "fn":         int(overall["num_misses"]),
        "ids":        int(overall["num_switches"]),
        "num_frames": int(overall["num_frames"]),
        "per_seq":    per_seq,
    }


def _safe(v) -> float:
    """Return float, mapping NaN to 0.0."""
    try:
        f = float(v)
        return 0.0 if (f != f) else f   # NaN check
    except (TypeError, ValueError):
        return 0.0
