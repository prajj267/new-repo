"""
Kalman filter track — BoT-SORT 8-state formulation.

State:       [cx, cy, w, h, vx, vy, vw, vh]
Observation: [cx, cy, w, h]

Design decisions (documented in design.md §5.1, OQ-9):
  F  = constant-velocity transition (I_8 + offdiag 1s)
  H  = [I_4 | 0_{4x4}]
  Q  = diag([1,1,1,1,0.01,0.01,0.0001,0.0001]) * Q_scale
  R  = diag([1,1,10,10]) * R_scale
  P0 = diag([2w,2h,2w,2h,25w²,25h²,25w²,25h²])  (BoT-SORT init heuristic)

Inputs and outputs are always in [cx, cy, w, h] pixel format.
"""

from __future__ import annotations

import numpy as np


class KalmanTrack:
    """One Kalman-filtered track.

    Args:
        box_cxcywh:    Initial box [cx, cy, w, h] in pixels.
        Q_scale:       Process noise scale factor (config: kalman.Q_scale).
        R_scale:       Measurement noise scale factor (config: kalman.R_scale).
        track_id:      Unique integer ID (assigned by FLITTracker).
    """

    _next_id: int = 1

    def __init__(
        self,
        box_cxcywh: np.ndarray,
        Q_scale: float = 1.0,
        R_scale: float = 1.0,
        track_id: int | None = None,
    ) -> None:
        self.id = track_id if track_id is not None else KalmanTrack._next_id
        KalmanTrack._next_id += 1

        cx, cy, w, h = float(box_cxcywh[0]), float(box_cxcywh[1]), \
                       float(box_cxcywh[2]), float(box_cxcywh[3])

        # ── State ─────────────────────────────────────────────────────────
        self._x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # ── Transition F ──────────────────────────────────────────────────
        self._F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self._F[i, i + 4] = 1.0  # position += velocity * dt  (dt=1 sampled frame)

        # ── Observation H ─────────────────────────────────────────────────
        self._H = np.zeros((4, 8), dtype=np.float64)
        self._H[:4, :4] = np.eye(4)

        # ── Process noise Q ───────────────────────────────────────────────
        q_diag = np.array([1, 1, 1, 1, 0.01, 0.01, 0.0001, 0.0001], dtype=np.float64)
        self._Q = np.diag(q_diag * Q_scale)

        # ── Measurement noise R ───────────────────────────────────────────
        r_diag = np.array([1, 1, 10, 10], dtype=np.float64)
        self._R = np.diag(r_diag * R_scale)

        # ── Initial covariance P (BoT-SORT heuristic) ─────────────────────
        w2, h2 = w * w, h * h
        p_diag = np.array([2*w, 2*h, 2*w, 2*h,
                           25*w2, 25*h2, 25*w2, 25*h2], dtype=np.float64)
        self._P = np.diag(p_diag)

        # ── Track lifecycle ───────────────────────────────────────────────
        self.hits:              int = 1
        self.age:               int = 1
        self.time_since_update: int = 0
        self.score:           float = 1.0

    # ------------------------------------------------------------------
    # Kalman predict
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Kalman prediction step.

        Advances state by one sampled frame (dt=1).

        Returns:
            Predicted box [cx, cy, w, h] in pixels.
        """
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        # Clip w, h to stay positive
        self._x[2] = max(self._x[2], 1.0)
        self._x[3] = max(self._x[3], 1.0)
        self.age += 1
        self.time_since_update += 1
        return self._x[:4].copy()

    # ------------------------------------------------------------------
    # Kalman update
    # ------------------------------------------------------------------

    def update(self, box_cxcywh: np.ndarray, score: float = 1.0) -> None:
        """Kalman update step with a new detection.

        Args:
            box_cxcywh: Matched detection [cx, cy, w, h] in pixels.
            score:       Detection confidence score.
        """
        z = np.array(box_cxcywh[:4], dtype=np.float64)
        S = self._H @ self._P @ self._H.T + self._R       # innovation covariance
        K = self._P @ self._H.T @ np.linalg.inv(S)        # Kalman gain
        self._x = self._x + K @ (z - self._H @ self._x)  # state update
        I_KH = np.eye(8) - K @ self._H
        self._P = I_KH @ self._P                           # covariance update
        # Clip w, h
        self._x[2] = max(self._x[2], 1.0)
        self._x[3] = max(self._x[3], 1.0)
        self.hits += 1
        self.time_since_update = 0
        self.score = score

    # ------------------------------------------------------------------
    # Affine warp (CMC)
    # ------------------------------------------------------------------

    def warp(self, M: np.ndarray) -> None:
        """Apply a 2×3 affine matrix M to the predicted centre position.

        Only [cx, cy] is warped — w and h are unchanged because the affine
        captures camera translation/rotation, not object scale change.

        Design note: we apply M to the predicted centre (after predict()),
        before cost matrix computation.  Applying the forward transform
        (not its inverse) moves the prediction to align with the new frame.

        Args:
            M: (2, 3) affine matrix from CMC.apply()
        """
        cx, cy = self._x[0], self._x[1]
        pt = np.array([cx, cy, 1.0], dtype=np.float64)
        new_cxy = M @ pt          # (2,)
        self._x[0] = new_cxy[0]
        self._x[1] = new_cxy[1]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_box(self) -> np.ndarray:
        """Return current state as [cx, cy, w, h] in pixels."""
        return self._x[:4].copy()

    def get_box_xyxy(self) -> np.ndarray:
        """Return current state as [x1, y1, x2, y2] in pixels."""
        cx, cy, w, h = self._x[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float64)

    def __repr__(self) -> str:
        cx, cy, w, h = self._x[:4]
        return (f"KalmanTrack(id={self.id}, "
                f"cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}, "
                f"hits={self.hits}, tsu={self.time_since_update})")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_track(box_cxcywh: np.ndarray, cfg: dict, track_id: int | None = None) -> KalmanTrack:
    """Create a KalmanTrack from a config dict.

    Args:
        box_cxcywh: Initial box [cx, cy, w, h] pixels.
        cfg:        Full config dict.
        track_id:   Optional explicit ID.
    Returns:
        KalmanTrack instance.
    """
    kc = cfg.get("kalman", {})
    return KalmanTrack(
        box_cxcywh,
        Q_scale=float(kc.get("Q_scale", 1.0)),
        R_scale=float(kc.get("R_scale", 1.0)),
        track_id=track_id,
    )
