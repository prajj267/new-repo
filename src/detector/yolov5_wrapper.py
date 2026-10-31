"""
Detector module.

Current mode: GT-based simulated detector  (source="gt_simulation")
─────────────────────────────────────────────────────────────────────
No pretrained weights were available on this machine — internet access
to download.pytorch.org and github.com is blocked (DNS failure).

The simulated detector returns the GT box + Gaussian noise and random
dropout so the tracker ablations can be run in isolation from detector
recall.  Every cache file records "detector_weights": "gt_simulation"
so the fallback is always visible in results.

REPORT DISCLAIMER (must appear in docs/report.md):
  "No pretrained detector weights were available on the test machine
  (internet access blocked).  A GT-based simulated detector is used:
  GT boxes with Gaussian noise (centre σ=5 px, size σ=2 px) and 5%
  random frame dropout.  This isolates the tracker ablation from
  detector recall and means results are NOT comparable with Table 3
  of the paper."

To use real weights when available
────────────────────────────────────
1. Place a .pt file at e.g. reference/yolov5s.pt
2. Set in configs/default.yaml:
       detector.weights: reference/yolov5s.pt
       detector.source: local_pt
3. Re-run:  python scripts/build_detection_cache.py --force

YOLOv5 compatibility note (kept for when weights become available):
  torch >= 2.6 defaults weights_only=True in torch.load which breaks
  YOLOv5 checkpoint loading.  The _WeightsOnlyPatch context manager
  applies weights_only=False for the trusted local file.
  Documented in design.md §4.3.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# torch.load patch (for future use with real weights)
# ---------------------------------------------------------------------------

class _WeightsOnlyPatch:
    """Temporarily sets weights_only=False in torch.load.

    Safe only for trusted local weight files.
    Applied for the duration of model loading, then restored.
    """
    def __enter__(self):
        self._orig = torch.load
        _orig = self._orig
        def patched(f, *a, **kw):
            kw["weights_only"] = False
            return _orig(f, *a, **kw)
        torch.load = patched
        return self

    def __exit__(self, *_):
        torch.load = self._orig


# ---------------------------------------------------------------------------
# GT simulation detector
# ---------------------------------------------------------------------------

class GTSimulationDetector:
    """Simulates a detector by returning GT boxes with Gaussian noise.

    This is used when no pretrained weights are available.
    Output format matches the real detector so the cache schema is identical.

    Args:
        centre_sigma: std of Gaussian noise on cx, cy (pixels)
        size_sigma:   std of Gaussian noise on w, h (pixels)
        dropout:      probability of dropping a visible frame's detection
        seed:         RNG seed (incremented per-frame to stay reproducible)
    """

    def __init__(
        self,
        centre_sigma: float = 5.0,
        size_sigma: float = 2.0,
        dropout: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.centre_sigma = centre_sigma
        self.size_sigma = size_sigma
        self.dropout = dropout
        self._rng = np.random.default_rng(seed)
        self._call_count = 0

    def detect_from_gt(
        self,
        box_xyxy: Optional[List[float]],
        visible: bool,
    ) -> List[dict]:
        """Generate a simulated detection from the GT box.

        Args:
            box_xyxy: GT box [x1, y1, x2, y2] or None
            visible:  whether the UAV is annotated as visible
        Returns:
            List with one detection dict, or empty list if dropped/invisible.
            Dict: {"box_xyxy": [x1,y1,x2,y2], "score": float, "class_id": 0}
        """
        self._call_count += 1
        if not visible or box_xyxy is None:
            return []

        # Random dropout
        if self._rng.random() < self.dropout:
            return []

        x1, y1, x2, y2 = box_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = x2 - x1
        h  = y2 - y1

        # Add Gaussian noise
        cx += float(self._rng.normal(0, self.centre_sigma))
        cy += float(self._rng.normal(0, self.centre_sigma))
        w  += float(self._rng.normal(0, self.size_sigma))
        h  += float(self._rng.normal(0, self.size_sigma))
        w   = max(1.0, w)
        h   = max(1.0, h)

        nx1 = cx - w / 2.0
        ny1 = cy - h / 2.0
        nx2 = cx + w / 2.0
        ny2 = cy + h / 2.0

        # Simulated confidence: high but not perfect
        score = float(np.clip(self._rng.normal(0.85, 0.08), 0.3, 0.99))

        return [{
            "box_xyxy": [nx1, ny1, nx2, ny2],
            "score": score,
            "class_id": 0,
        }]


# ---------------------------------------------------------------------------
# YOLOv5 real detector (for use when weights are available)
# ---------------------------------------------------------------------------

class YOLOv5RealDetector:
    """Load a local YOLOv5 .pt weights file and run inference.

    Used when detector.source = "local_pt".

    Args:
        weights_path: absolute or relative path to .pt file
        cfg:          full config dict
    """

    def __init__(self, weights_path: str, cfg: dict) -> None:
        det = cfg.get("detector", {})
        self.conf_thresh = float(det.get("conf_thresh", 0.25))
        self.iou_thresh  = float(det.get("iou_thresh_nms", 0.45))
        self.imgsz       = int(det.get("imgsz", 640))

        vendor = _REPO_ROOT / "reference" / "yolov5"
        if not (vendor / "hubconf.py").exists():
            raise FileNotFoundError(
                f"YOLOv5 repo not found at {vendor}.\n"
                "Run: python scripts/build_detection_cache.py with source='hub' first."
            )

        with _WeightsOnlyPatch():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = torch.hub.load(
                    str(vendor), "custom",
                    path=str(weights_path),
                    source="local",
                    force_reload=False,
                    verbose=False,
                )
        self._model.conf = self.conf_thresh
        self._model.iou  = self.iou_thresh
        self._model.eval()

    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        frame_rgb = frame_bgr[..., ::-1].copy()
        with torch.no_grad():
            results = self._model(frame_rgb, size=self.imgsz)
        preds = results.xyxy[0].cpu().numpy()
        out = []
        for row in preds:
            x1, y1, x2, y2, score, cls_id = row
            if score >= self.conf_thresh:
                out.append({
                    "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(score),
                    "class_id": int(cls_id),
                })
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class YOLOv5Detector:
    """Unified detector interface.

    Dispatches to GTSimulationDetector or YOLOv5RealDetector based on
    detector.source in the config.

    Args:
        cfg: full config dict
    """

    def __init__(self, cfg: dict) -> None:
        det = cfg.get("detector", {})
        source = str(det.get("source", "gt_simulation"))
        self._source = source
        self._cfg = cfg

        if source == "gt_simulation":
            self._sim = GTSimulationDetector(
                centre_sigma=float(det.get("sim_centre_sigma", 5.0)),
                size_sigma=float(det.get("sim_size_sigma", 2.0)),
                dropout=float(det.get("sim_dropout", 0.05)),
                seed=int(det.get("sim_seed", 42)),
            )
            self._real: Optional[YOLOv5RealDetector] = None
            print("[YOLOv5Detector] Using GT simulation (no weights available)")

        elif source == "local_pt":
            weights = det.get("weights", "")
            if not Path(weights).exists():
                raise FileNotFoundError(
                    f"Weights file not found: {weights}\n"
                    "Set detector.weights to the correct path in configs/default.yaml."
                )
            self._real = YOLOv5RealDetector(weights, cfg)
            self._sim = None
            print(f"[YOLOv5Detector] Loaded real weights: {weights}")

        else:
            raise ValueError(
                f"detector.source={source!r} not supported without internet.\n"
                "Set source: 'gt_simulation' or source: 'local_pt' with a local .pt file."
            )

    @property
    def is_simulation(self) -> bool:
        return self._source == "gt_simulation"

    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        """Run detection on a BGR frame (real weights path)."""
        if self._real is not None:
            return self._real.detect(frame_bgr)
        raise RuntimeError("detect() called on simulation detector; use detect_from_gt()")

    def detect_from_gt(
        self,
        box_xyxy: Optional[List[float]],
        visible: bool,
    ) -> List[dict]:
        """Generate simulated detections from GT (simulation path)."""
        if self._sim is not None:
            return self._sim.detect_from_gt(box_xyxy, visible)
        # Real detector: ignore GT, return empty (caller should use detect())
        raise RuntimeError("detect_from_gt() called on real detector; use detect()")


# ---------------------------------------------------------------------------
# Recall (real detector path — not used in simulation mode)
# ---------------------------------------------------------------------------

def _iou_xyxy(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    iw  = max(0.0, ix2 - ix1);  ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(aa + ab - inter, 1e-7)


def compute_recall(
    detections_per_frame: List[List[dict]],
    gt_records,
    iou_thresh: float = 0.5,
    img_w: int = 640,
    img_h: int = 512,
) -> float:
    """Compute detection recall at iou_thresh.

    Args:
        detections_per_frame: list of detection lists (one per frame)
        gt_records:           list of FrameRecord objects (same frames)
        iou_thresh:           IoU threshold for true positive
    Returns:
        Recall in [0, 1]
    """
    tp = 0;  total_gt = 0
    for dets, rec in zip(detections_per_frame, gt_records):
        if not rec.visible or rec.box_xyxy is None:
            continue
        total_gt += 1
        for det in dets:
            if _iou_xyxy(det["box_xyxy"], rec.box_xyxy) >= iou_thresh:
                tp += 1
                break
    return float(tp) / max(total_gt, 1)
