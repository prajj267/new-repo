"""
SSQ loss functions — Eq. 2-8 from:
  Wang et al., "Dist-Tracker: A Small Object-aware Detector and Tracker
  for UAV Tracking", CVPR Workshops 2025.

All box tensors are in [cx, cy, w, h] format, pixel units by default.
Use xywh_tl_to_cxcywh() from src/data/loader.py to convert loader output.

Functions
---------
ciou_loss        Eq. 2  — CIoU baseline
scale_loss       Eq. 3  — Wasserstein-style centre + size term
shape_loss       Eq. 4  — aspect-ratio-weighted relative size mismatch
compute_beta     Eq. 6  — sliding-window quality normaliser (Python float, no grad)
quality_loss     Eq. 5  — beta-weighted IoU gap
ssq_loss         Eq. 7-8 — combined loss with step-dependent schedule

Open questions documented in docs/open_questions.md:
  OQ-1  w_s value
  OQ-1b size_cov_scale (1.0 vs NWD 0.25)
  OQ-2  pixel vs normalised coords
  OQ-3  shape_scale exponent
  OQ-5  sigma value
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EPS = 1e-7  # matches config default loss.eps


def _box_iou_elementwise(p: Tensor, t: Tensor, eps: float = _EPS) -> Tensor:
    """Per-element IoU between two sets of [cx,cy,w,h] boxes.

    Args:
        p: predicted boxes  (..., 4)  [cx, cy, w, h]
        t: target boxes     (..., 4)  [cx, cy, w, h]
        eps: division guard
    Returns:
        IoU values  (...,)  in [0, 1]
    """
    # Convert to xyxy
    p_x1 = p[..., 0] - p[..., 2] / 2
    p_y1 = p[..., 1] - p[..., 3] / 2
    p_x2 = p[..., 0] + p[..., 2] / 2
    p_y2 = p[..., 1] + p[..., 3] / 2

    t_x1 = t[..., 0] - t[..., 2] / 2
    t_y1 = t[..., 1] - t[..., 3] / 2
    t_x2 = t[..., 0] + t[..., 2] / 2
    t_y2 = t[..., 1] + t[..., 3] / 2

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area_p = p[..., 2] * p[..., 3]
    area_t = t[..., 2] * t[..., 3]
    union_area = area_p + area_t - inter_area + eps

    return inter_area / union_area


def _maybe_normalise(box: Tensor, img_w: float, img_h: float) -> Tensor:
    """Optionally normalise [cx,cy,w,h] by image dimensions.

    Args:
        box:   (..., 4) tensor [cx, cy, w, h] in pixels
        img_w: image width
        img_h: image height
    Returns:
        Normalised box (..., 4) in [0,1]
    """
    scale = box.new_tensor([img_w, img_h, img_w, img_h])
    return box / scale


# ---------------------------------------------------------------------------
# Eq. 2 — CIoU loss (baseline)
# ---------------------------------------------------------------------------

def ciou_loss(p: Tensor, t: Tensor, eps: float = _EPS) -> Tensor:
    """CIoU loss (Zheng et al. 2020).  Eq. 2 baseline.

    Args:
        p:   predicted boxes (..., 4) [cx, cy, w, h], pixels
        t:   target boxes    (..., 4) [cx, cy, w, h], pixels
        eps: division guard
    Returns:
        CIoU loss per box (...,), differentiable w.r.t. p
    """
    iou = _box_iou_elementwise(p, t, eps)

    # Enclosing box
    p_x1 = p[..., 0] - p[..., 2] / 2;  p_x2 = p[..., 0] + p[..., 2] / 2
    p_y1 = p[..., 1] - p[..., 3] / 2;  p_y2 = p[..., 1] + p[..., 3] / 2
    t_x1 = t[..., 0] - t[..., 2] / 2;  t_x2 = t[..., 0] + t[..., 2] / 2
    t_y1 = t[..., 1] - t[..., 3] / 2;  t_y2 = t[..., 1] + t[..., 3] / 2

    c_x1 = torch.min(p_x1, t_x1);  c_x2 = torch.max(p_x2, t_x2)
    c_y1 = torch.min(p_y1, t_y1);  c_y2 = torch.max(p_y2, t_y2)

    c_diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Centre distance
    d_sq = (p[..., 0] - t[..., 0]) ** 2 + (p[..., 1] - t[..., 1]) ** 2

    # Aspect ratio penalty
    atan_t = torch.atan(t[..., 2] / (t[..., 3] + eps))
    atan_p = torch.atan(p[..., 2] / (p[..., 3] + eps))
    v = (4 / (math.pi ** 2)) * (atan_p - atan_t) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return 1 - iou + d_sq / c_diag_sq + alpha * v


# ---------------------------------------------------------------------------
# Eq. 3 — Scale loss
# ---------------------------------------------------------------------------

def scale_loss(
    p: Tensor,
    t: Tensor,
    w_s: float = 0.6,
    size_cov_scale: float = 1.0,
    use_normalised_coords: bool = False,
    img_w: float = 640.0,
    img_h: float = 512.0,
    use_nwd_exp: bool = False,
    nwd_exp_C: float = 25.0,
    eps: float = _EPS,
) -> Tensor:
    """Scale loss — Eq. 3.

    L_scale = ||mu_p - mu_t||^2
            + w_s * size_cov_scale * ((w_p - w_t)^2 + (h_p - h_t)^2)

    Derivation: Eq. 3 uses the Frobenius norm of (Sigma_p^(1/2) - Sigma_t^(1/2))
    with Sigma = diag(w^2, h^2), giving Sigma^(1/2) = diag(w, h), so the
    Frobenius norm reduces to (w_p-w_t)^2 + (h_p-h_t)^2.
    size_cov_scale=1.0 matches literal Eq. 3; set to 0.25 for NWD convention.
    See docs/open_questions.md OQ-1b.

    Args:
        p:                    predicted boxes (..., 4) [cx, cy, w, h]
        t:                    target boxes    (..., 4) [cx, cy, w, h]
        w_s:                  weight on the size term (paper: > 0.5)  [OQ-1]
        size_cov_scale:       factor on size term (1.0=paper, 0.25=NWD)  [OQ-1b]
        use_normalised_coords: normalise by image dimensions before computing  [OQ-2]
        img_w, img_h:         image dimensions for normalisation
        use_nwd_exp:          wrap result in 1-exp(-L/C)  [OQ-2b]
        nwd_exp_C:            constant C for NWD exponential
        eps:                  division guard
    Returns:
        Scale loss per box (...,), differentiable w.r.t. p
    """
    if use_normalised_coords:
        p = _maybe_normalise(p, img_w, img_h)
        t = _maybe_normalise(t, img_w, img_h)

    # Centre distance term
    centre_sq = (p[..., 0] - t[..., 0]) ** 2 + (p[..., 1] - t[..., 1]) ** 2

    # Size Frobenius term
    size_sq = (p[..., 2] - t[..., 2]) ** 2 + (p[..., 3] - t[..., 3]) ** 2

    L = centre_sq + w_s * size_cov_scale * size_sq

    if use_nwd_exp:
        L = 1.0 - torch.exp(-L / (nwd_exp_C + eps))

    return L


# ---------------------------------------------------------------------------
# Eq. 4 — Shape loss
# ---------------------------------------------------------------------------

def shape_loss(
    p: Tensor,
    t: Tensor,
    shape_scale: float = 2.0,
    eps: float = _EPS,
) -> Tensor:
    """Shape loss — Eq. 4.

    Weights the relative width and height errors by the GT aspect ratio,
    so the loss is invariant to overall scale but sensitive to shape mismatch.

    L_shape = 2*(w_t^s / (w_t^s + h_t^s)) * (|w_p-w_t| / (max(w_p,w_t)+eps))^2
            + 2*(h_t^s / (w_t^s + h_t^s)) * (|h_p-h_t| / (max(h_p,h_t)+eps))^2

    Returns 0 when p == t.

    Args:
        p:           predicted boxes (..., 4) [cx, cy, w, h], pixels
        t:           target boxes    (..., 4) [cx, cy, w, h], pixels
        shape_scale: exponent s on w_t, h_t  [OQ-3, default 2.0]
        eps:         division guard
    Returns:
        Shape loss per box (...,), differentiable w.r.t. p
    """
    wt = t[..., 2]
    ht = t[..., 3]

    wt_s = wt.pow(shape_scale)
    ht_s = ht.pow(shape_scale)
    denom = wt_s + ht_s + eps

    w_weight = 2.0 * wt_s / denom   # in [0, 2]
    h_weight = 2.0 * ht_s / denom

    wp = p[..., 2]
    hp = p[..., 3]

    rel_w = (wp - wt).abs() / (torch.max(wp, wt) + eps)
    rel_h = (hp - ht).abs() / (torch.max(hp, ht) + eps)

    return w_weight * rel_w.pow(2) + h_weight * rel_h.pow(2)


# ---------------------------------------------------------------------------
# Eq. 6 — Beta (sliding window, no gradient)
# ---------------------------------------------------------------------------

def compute_beta(
    iou_val: float,
    window: deque,
    eps: float = _EPS,
) -> float:
    """Compute beta for the quality loss — Eq. 6.

    beta = (1 - iou_val) / (E[1 - iou] + eps)
    where E[.] is the sliding mean over the last 100 iterations.

    beta is a Python float — it carries NO gradient. The caller must pass
    iou_val as a Python float (use .item() on a tensor before calling).

    Args:
        iou_val: current IoU as a Python float (must be detached, e.g. tensor.item())
        window:  collections.deque(maxlen=100) maintained by the caller
        eps:     division guard
    Returns:
        beta as a Python float (no gradient)
    Raises:
        TypeError if iou_val is a Tensor (guards against accidental gradient path)
    """
    if isinstance(iou_val, Tensor):
        raise TypeError(
            "compute_beta: iou_val must be a Python float (call .item() first). "
            "beta must NOT carry a gradient."
        )

    window.append(float(iou_val))
    mean_iou = sum(window) / len(window)
    beta = (1.0 - float(iou_val)) / (mean_iou + eps)
    return beta  # Python float, no Tensor, no gradient


# ---------------------------------------------------------------------------
# Eq. 5 — Quality loss
# ---------------------------------------------------------------------------

def quality_loss(
    p: Tensor,
    t: Tensor,
    beta: float,
    sigma: float = 2.0,
    eps: float = _EPS,
) -> Tensor:
    """Quality loss — Eq. 5.

    L_quality = beta / sigma^(beta - 1 + eps) * (1 - IoU)^sigma

    beta is a Python float — no gradient flows through it.
    sigma controls how sharply the loss focuses on hard examples.

    Args:
        p:     predicted boxes (..., 4) [cx, cy, w, h], pixels
        t:     target boxes    (..., 4) [cx, cy, w, h], pixels
        beta:  quality normaliser, Python float (from compute_beta)  [OQ-5]
        sigma: exponent on (1 - IoU)  [OQ-5, default 2.0]
        eps:   division guard
    Returns:
        Quality loss per box (...,), differentiable w.r.t. p
    Raises:
        TypeError if beta is a Tensor
    """
    if isinstance(beta, Tensor):
        raise TypeError(
            "quality_loss: beta must be a Python float, not a Tensor. "
            "Use compute_beta() which returns a Python float."
        )

    iou = _box_iou_elementwise(p, t, eps)
    # sigma^(beta - 1 + eps): scalar, computed in Python — no gradient path
    denom = float(sigma) ** (float(beta) - 1.0 + float(eps))
    coeff = float(beta) / (denom + float(eps))

    return coeff * (1.0 - iou).pow(sigma)


# ---------------------------------------------------------------------------
# Eq. 7-8 — Combined SSQ loss
# ---------------------------------------------------------------------------

def ssq_loss(
    p: Tensor,
    t: Tensor,
    step: int,
    total_steps: int,
    window: deque,
    w_s: float = 0.6,
    size_cov_scale: float = 1.0,
    shape_scale: float = 2.0,
    sigma: float = 2.0,
    use_normalised_coords: bool = False,
    img_w: float = 640.0,
    img_h: float = 512.0,
    use_nwd_exp: bool = False,
    nwd_exp_C: float = 25.0,
    eps: float = _EPS,
) -> Tensor:
    """Combined SSQ loss — Eq. 7-8.

    r = step / total_steps
    L_SSQ = 0.7*(1-r)*(L_scale + L_shape) + 0.3*(1+r)*L_quality

    The 0.7/0.3 and 1±r coefficients are fixed by the paper.

    At step=0:           weights 0.7 on (scale+shape), 0.3 on quality.
    At step=total_steps: weights 0.0 on (scale+shape), 0.6 on quality.

    This function updates `window` internally via compute_beta.

    Args:
        p, t:              boxes (..., 4) [cx, cy, w, h], pixels
        step:              current optimisation step (0-indexed)
        total_steps:       total steps in the run
        window:            deque(maxlen=100) maintained by caller; updated here
        w_s:               scale loss weight  [OQ-1]
        size_cov_scale:    Frobenius factor  [OQ-1b]
        shape_scale:       shape exponent  [OQ-3]
        sigma:             quality exponent  [OQ-5]
        use_normalised_coords: normalise before scale_loss  [OQ-2]
        img_w, img_h:      image dims for normalisation
        use_nwd_exp:       NWD exponential wrapping  [OQ-2b]
        nwd_exp_C:         NWD constant
        eps:               division guard
    Returns:
        SSQ loss per box (...,), differentiable w.r.t. p
    """
    r = float(step) / max(float(total_steps), 1.0)

    Ls = scale_loss(
        p, t,
        w_s=w_s,
        size_cov_scale=size_cov_scale,
        use_normalised_coords=use_normalised_coords,
        img_w=img_w,
        img_h=img_h,
        use_nwd_exp=use_nwd_exp,
        nwd_exp_C=nwd_exp_C,
        eps=eps,
    )
    Lsh = shape_loss(p, t, shape_scale=shape_scale, eps=eps)

    # Compute beta from current IoU (detached — no gradient)
    with torch.no_grad():
        iou_vals = _box_iou_elementwise(p, t, eps)
        iou_mean = iou_vals.mean().item()  # scalar Python float

    beta = compute_beta(iou_mean, window, eps)

    Lq = quality_loss(p, t, beta=beta, sigma=sigma, eps=eps)

    return 0.7 * (1.0 - r) * (Ls + Lsh) + 0.3 * (1.0 + r) * Lq


# ---------------------------------------------------------------------------
# Config-driven wrappers
# ---------------------------------------------------------------------------

def ssq_loss_from_cfg(
    p: Tensor,
    t: Tensor,
    step: int,
    total_steps: int,
    window: deque,
    cfg: dict,
) -> Tensor:
    """Call ssq_loss using values from the loaded config dict.

    Args:
        p, t:          boxes (..., 4) [cx, cy, w, h]
        step:          current step
        total_steps:   total steps
        window:        sliding deque (maintained by caller)
        cfg:           full config dict (from configs/default.yaml)
    Returns:
        SSQ loss tensor (...,)
    """
    lc = cfg.get("loss", {})
    dc = cfg.get("data", {})
    return ssq_loss(
        p, t, step, total_steps, window,
        w_s=float(lc.get("w_s", 0.6)),
        size_cov_scale=float(lc.get("size_cov_scale", 1.0)),
        shape_scale=float(lc.get("shape_scale", 2.0)),
        sigma=float(lc.get("sigma", 2.0)),
        use_normalised_coords=bool(lc.get("use_normalised_coords", False)),
        img_w=float(dc.get("img_w", 640.0)),
        img_h=float(dc.get("img_h", 512.0)),
        use_nwd_exp=bool(lc.get("use_nwd_exp", False)),
        nwd_exp_C=float(lc.get("nwd_exp_C", 25.0)),
        eps=float(lc.get("eps", _EPS)),
    )


def scale_loss_from_cfg(p: Tensor, t: Tensor, cfg: dict) -> Tensor:
    lc = cfg.get("loss", {})
    dc = cfg.get("data", {})
    return scale_loss(
        p, t,
        w_s=float(lc.get("w_s", 0.6)),
        size_cov_scale=float(lc.get("size_cov_scale", 1.0)),
        use_normalised_coords=bool(lc.get("use_normalised_coords", False)),
        img_w=float(dc.get("img_w", 640.0)),
        img_h=float(dc.get("img_h", 512.0)),
        use_nwd_exp=bool(lc.get("use_nwd_exp", False)),
        nwd_exp_C=float(lc.get("nwd_exp_C", 25.0)),
        eps=float(lc.get("eps", _EPS)),
    )


def shape_loss_from_cfg(p: Tensor, t: Tensor, cfg: dict) -> Tensor:
    lc = cfg.get("loss", {})
    return shape_loss(
        p, t,
        shape_scale=float(lc.get("shape_scale", 2.0)),
        eps=float(lc.get("eps", _EPS)),
    )
