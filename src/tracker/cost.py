"""
Cost matrix functions for FLIT tracker — Eq. 10-11.

All functions are intentionally separated from the assignment step so
each can be unit-tested independently.

Inputs
------
tracks_xyxy : (T, 4) ndarray  [x1, y1, x2, y2] pixels
dets_xyxy   : (D, 4) ndarray  [x1, y1, x2, y2] pixels
scores      : (D,)   ndarray  detection confidence in [0, 1]

Eq. 10:  C_IoU-L2 = gamma * C_L2 + (1-gamma) * C_IoU
         C_IoU = 1 - IoU,   values in [0, 1]
         C_L2  = ||centre_t - centre_d|| / norm,  values in [0, ∞)

Eq. 11:  C_final = 1 - (1 - C_IoU-L2) * S_conf
         S_conf  = scores broadcast to shape (T, D)

Design notes
------------
- norm options: "diagonal" (÷√(W²+H²) ≈ 820 for 640×512) or "fixed" (÷512).
  Config key: tracker.l2_norm  [OQ-6]
- Frames where exist=0 contribute no GT box.  Any hypothesis on those frames
  is a FP in evaluation (stated in evaluate.py docstring).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# IoU cost  (1 - IoU)
# ---------------------------------------------------------------------------

def iou_cost(
    tracks_xyxy: np.ndarray,
    dets_xyxy: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute 1-IoU cost matrix.

    Args:
        tracks_xyxy: (T, 4) float64  [x1, y1, x2, y2]
        dets_xyxy:   (D, 4) float64  [x1, y1, x2, y2]
        eps:         division guard
    Returns:
        (T, D) float64 ndarray, values in [0, 1]
    """
    T = tracks_xyxy.shape[0]
    D = dets_xyxy.shape[0]
    if T == 0 or D == 0:
        return np.empty((T, D), dtype=np.float64)

    # Broadcast intersection
    tx1 = tracks_xyxy[:, 0:1];  ty1 = tracks_xyxy[:, 1:2]
    tx2 = tracks_xyxy[:, 2:3];  ty2 = tracks_xyxy[:, 3:4]
    dx1 = dets_xyxy[:, 0];      dy1 = dets_xyxy[:, 1]
    dx2 = dets_xyxy[:, 2];      dy2 = dets_xyxy[:, 3]

    ix1 = np.maximum(tx1, dx1)   # (T, D)
    iy1 = np.maximum(ty1, dy1)
    ix2 = np.minimum(tx2, dx2)
    iy2 = np.minimum(ty2, dy2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_t = (tx2 - tx1) * (ty2 - ty1)          # (T, 1)
    area_d = (dx2 - dx1) * (dy2 - dy1)           # (D,)
    union  = area_t + area_d - inter + eps

    iou = inter / union
    return 1.0 - np.clip(iou, 0.0, 1.0)


# ---------------------------------------------------------------------------
# L2 cost (normalised Euclidean centre distance)
# ---------------------------------------------------------------------------

def l2_cost(
    tracks_xyxy: np.ndarray,
    dets_xyxy: np.ndarray,
    norm: str = "diagonal",
    img_w: float = 640.0,
    img_h: float = 512.0,
    fixed_norm: float = 512.0,
) -> np.ndarray:
    """Compute normalised L2 centre-distance cost matrix.

    Args:
        tracks_xyxy: (T, 4) [x1, y1, x2, y2]
        dets_xyxy:   (D, 4) [x1, y1, x2, y2]
        norm:        "diagonal" → divide by √(W²+H²); "fixed" → divide by fixed_norm
        img_w, img_h: image dimensions (for diagonal normalisation)
        fixed_norm:  denominator when norm="fixed"
    Returns:
        (T, D) float64 ndarray, values in [0, ∞)
    """
    T = tracks_xyxy.shape[0]
    D = dets_xyxy.shape[0]
    if T == 0 or D == 0:
        return np.empty((T, D), dtype=np.float64)

    # Centres
    t_cx = (tracks_xyxy[:, 0] + tracks_xyxy[:, 2]) / 2.0   # (T,)
    t_cy = (tracks_xyxy[:, 1] + tracks_xyxy[:, 3]) / 2.0
    d_cx = (dets_xyxy[:, 0] + dets_xyxy[:, 2]) / 2.0       # (D,)
    d_cy = (dets_xyxy[:, 1] + dets_xyxy[:, 3]) / 2.0

    dx = t_cx[:, None] - d_cx[None, :]   # (T, D)
    dy = t_cy[:, None] - d_cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)

    if norm == "diagonal":
        denom = np.sqrt(img_w**2 + img_h**2)
    elif norm == "fixed":
        denom = fixed_norm
    else:
        raise ValueError(f"Unknown l2 norm option: {norm!r}. Use 'diagonal' or 'fixed'.")

    return dist / max(denom, 1e-7)


# ---------------------------------------------------------------------------
# Fused cost  Eq. 10
# ---------------------------------------------------------------------------

def fused_cost(
    tracks_xyxy: np.ndarray,
    dets_xyxy: np.ndarray,
    gamma: float = 0.25,
    norm: str = "diagonal",
    img_w: float = 640.0,
    img_h: float = 512.0,
    fixed_norm: float = 512.0,
) -> np.ndarray:
    """Fused IoU-L2 cost — Eq. 10.

    C = gamma * C_L2 + (1 - gamma) * C_IoU

    Args:
        tracks_xyxy, dets_xyxy: box arrays
        gamma:      fusion weight; 0 = IoU only, 1 = L2 only, 0.25 = paper default
        norm:       L2 normalisation strategy
        img_w/img_h: image dimensions
        fixed_norm: denominator for norm="fixed"
    Returns:
        (T, D) float64 ndarray
    """
    C_iou = iou_cost(tracks_xyxy, dets_xyxy)
    C_l2  = l2_cost(tracks_xyxy, dets_xyxy, norm, img_w, img_h, fixed_norm)
    return gamma * C_l2 + (1.0 - gamma) * C_iou


# ---------------------------------------------------------------------------
# Final cost  Eq. 11
# ---------------------------------------------------------------------------

def final_cost(
    C_fused: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Confidence-weighted final cost — Eq. 11.

    C_final = 1 - (1 - C_fused) * S_conf

    S_conf is the detection score broadcast to shape (T, D).
    Higher-confidence detections get lower cost (preferred in assignment).

    Args:
        C_fused: (T, D) fused cost matrix
        scores:  (D,)   detection confidence scores in [0, 1]
    Returns:
        (T, D) float64 ndarray
    """
    if C_fused.shape[0] == 0 or C_fused.shape[1] == 0:
        return C_fused.copy()
    S = scores[np.newaxis, :]   # (1, D) broadcast to (T, D)
    return 1.0 - (1.0 - C_fused) * S


# ---------------------------------------------------------------------------
# Config-driven wrapper
# ---------------------------------------------------------------------------

def build_cost_matrix(
    tracks_xyxy: np.ndarray,
    dets_xyxy: np.ndarray,
    scores: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Build the full (T, D) final cost matrix from config.

    Args:
        tracks_xyxy: (T, 4)
        dets_xyxy:   (D, 4)
        scores:      (D,)
        cfg:         full config dict
    Returns:
        (T, D) final cost matrix
    """
    tc = cfg.get("tracker", {})
    dc = cfg.get("data", {})
    gamma      = float(tc.get("gamma", 0.25))
    norm       = str(tc.get("l2_norm", "diagonal"))
    img_w      = float(dc.get("img_w", 640.0))
    img_h      = float(dc.get("img_h", 512.0))
    fixed_norm = float(tc.get("l2_norm_fixed", 512.0))

    C = fused_cost(tracks_xyxy, dets_xyxy, gamma, norm, img_w, img_h, fixed_norm)
    return final_cost(C, scores)
