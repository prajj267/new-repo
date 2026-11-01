"""
Camera Motion Compensation (CMC).

Estimates the inter-frame affine warp due to camera motion using sparse
Lucas-Kanade optical flow between consecutive SAMPLED frames.

Stride semantics (design.md §1.3, Amendment C):
  Affines are computed between frame[i] and frame[i+stride] (the previous
  sampled frame), not between raw consecutive frames.
  The CMC cache records frame_stride; a mismatch raises ValueError.

Algorithm:
  1. Detect Shi-Tomasi corners in the previous sampled frame.
  2. Track them with Lucas-Kanade pyramidal optical flow.
  3. Estimate a partial affine (translation + rotation, no shear/scale)
     from matched pairs via RANSAC.
  4. Apply the forward transform to Kalman-predicted [cx, cy].
     (Forward = maps prev → curr; this moves the prediction into alignment
      with the current frame.  Applying the inverse is a common bug.)

Config keys (under cmc.):
  enabled, max_corners, quality_level, min_distance, block_size, ransac_thresh

The _IDENTITY matrix is returned whenever estimation fails (< 4 points
tracked, or RANSAC fails), so the tracker degrades gracefully.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_IDENTITY = np.eye(2, 3, dtype=np.float64)   # no-motion fallback

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class CMC:
    """Per-sequence camera motion estimator.

    Call reset() at the start of each new sequence.
    Call apply(frame) for each sampled frame in order.

    Args:
        cfg: Full config dict.
    """

    def __init__(self, cfg: dict) -> None:
        cc = cfg.get("cmc", {})
        self.enabled:       bool  = bool(cc.get("enabled", True))
        self._max_corners:  int   = int(cc.get("max_corners", 200))
        self._quality:      float = float(cc.get("quality_level", 0.01))
        self._min_dist:     int   = int(cc.get("min_distance", 30))
        self._block_size:   int   = int(cc.get("block_size", 3))
        self._ransac_thresh: float = float(cc.get("ransac_thresh", 3.0))

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts:  Optional[np.ndarray] = None   # shape (N,1,2) float32

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state.  Call at the start of each new sequence."""
        self._prev_gray = None
        self._prev_pts  = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Estimate affine warp from the previous sampled frame to `frame`.

        Args:
            frame: Current BGR or grayscale uint8 image (H, W[, 3]).
        Returns:
            (2, 3) float64 affine matrix M.
            Identity matrix on the first frame or when estimation fails.
        """
        gray = self._to_gray(frame)

        if not self.enabled or self._prev_gray is None:
            self._update_state(gray)
            return _IDENTITY.copy()

        if self._prev_pts is None or len(self._prev_pts) < 4:
            self._update_state(gray)
            return _IDENTITY.copy()

        M = self._estimate(self._prev_gray, gray, self._prev_pts)
        self._update_state(gray)
        return M

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _detect_corners(self, gray: np.ndarray) -> Optional[np.ndarray]:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._max_corners,
            qualityLevel=self._quality,
            minDistance=self._min_dist,
            blockSize=self._block_size,
        )
        return pts  # (N,1,2) float32 or None

    def _estimate(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> np.ndarray:
        """Run LK flow and RANSAC affine estimation.  Returns identity on failure."""
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **_LK_PARAMS
        )
        if curr_pts is None or status is None:
            return _IDENTITY.copy()

        mask = status.ravel() == 1
        if mask.sum() < 4:
            return _IDENTITY.copy()

        src = prev_pts[mask].reshape(-1, 2).astype(np.float32)
        dst = curr_pts[mask].reshape(-1, 2).astype(np.float32)

        M, inlier_mask = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._ransac_thresh,
        )
        if M is None:
            return _IDENTITY.copy()
        return M.astype(np.float64)

    def _update_state(self, gray: np.ndarray) -> None:
        self._prev_gray = gray
        self._prev_pts  = self._detect_corners(gray)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_affine_cache(
    seq_name: str,
    frame_stride: int,
    affines: list,      # list of {"frame_idx": int, "M": [[…],[…]]}
    cache_dir: Path,
) -> Path:
    """Save affines for one sequence to cache/affines/<seq_name>.json.

    Args:
        seq_name:     Sequence name.
        frame_stride: Must be stored so stale caches can be detected.
        affines:      List of {"frame_idx": int, "M": list[list[float]]}
        cache_dir:    Output directory.
    Returns:
        Path to the saved file.
    """
    import json
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{seq_name}.json"
    with open(p, "w") as f:
        json.dump({"seq_name": seq_name, "frame_stride": frame_stride, "affines": affines}, f)
    return p


def load_affine_cache(seq_name: str, cache_dir: Path, expected_stride: int) -> list:
    """Load affines for one sequence.

    Args:
        seq_name:        Sequence name.
        cache_dir:       Directory containing <seq_name>.json
        expected_stride: Must match stored frame_stride.
    Returns:
        List of {"frame_idx": int, "M": list[list[float]]}
    Raises:
        FileNotFoundError if cache file does not exist.
        ValueError if frame_stride in cache != expected_stride.
    """
    import json
    p = cache_dir / f"{seq_name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"CMC cache not found: {p}.\n"
            "Run: python scripts/build_cmc_cache.py --config configs/default.yaml"
        )
    with open(p) as f:
        data = json.load(f)
    cached_stride = data.get("frame_stride")
    if cached_stride != expected_stride:
        raise ValueError(
            f"CMC cache stride mismatch for {seq_name}: "
            f"cached={cached_stride}, expected={expected_stride}. "
            "Re-run build_cmc_cache.py."
        )
    return data["affines"]


def affines_to_dict(affines: list) -> dict:
    """Convert list of affine records to {frame_idx: M(2x3 ndarray)} dict."""
    result = {}
    for rec in affines:
        M = np.array(rec["M"], dtype=np.float64)
        result[rec["frame_idx"]] = M
    return result
