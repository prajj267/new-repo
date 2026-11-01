"""
FLIT tracker — Algorithm 1 from the paper.

Implements:
  1. Filter detections by confidence τ
  2. Kalman predict all tracks
  3. Apply CMC affine to predicted centres (if enabled)
  4. Build cost matrix (Eq. 10-11)
  5. Hungarian assignment (scipy)
  6. Update matched tracks
  7. Delete unmatched tracks where time_since_update > track_buffer
  8. Initialise new tracks for unmatched dets with score >= η
  9. Return confirmed tracks (hits >= min_hits)

All decisions not specified in the paper are config keys — see open_questions.md.

Frames with exist=0 have no GT box.  Any track output on those frames is a FP
in evaluation (stated in evaluate.py).

Stride semantics: the tracker treats consecutive sampled frames as dt=1.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracker.kalman import KalmanTrack, make_track
from src.tracker.cost import build_cost_matrix
from src.tracker.cmc import affines_to_dict


class FLITTracker:
    """FLIT multi-object tracker.

    Args:
        cfg:        Full config dict.
        affine_map: Optional pre-loaded affine dict {frame_idx: M(2x3)}.
                    If None, CMC is effectively disabled (identity used).
    """

    def __init__(
        self,
        cfg: dict,
        affine_map: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        tc = cfg.get("tracker", {})
        self._cfg        = cfg
        self._tau        = float(tc.get("tau",          0.01))
        self._eta        = float(tc.get("eta",          0.25))
        self._thresh     = float(tc.get("match_thresh", 0.8))
        self._buf        = int(tc.get("track_buffer",   0))
        self._min_hits   = int(tc.get("min_hits",       1))
        self._init_mode  = str(tc.get("init_mode", "detector"))
        self._cmc_on     = bool(cfg.get("cmc", {}).get("enabled", True))

        self._affine_map: Dict[int, np.ndarray] = affine_map or {}
        self._tracks: List[KalmanTrack] = []
        self.frame_count: int = 0

        # Reset class-level ID counter for reproducibility
        KalmanTrack._next_id = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset tracker state.  Call at the start of each new sequence."""
        self._tracks = []
        self.frame_count = 0
        KalmanTrack._next_id = 1

    def update(
        self,
        dets_xyxy: np.ndarray,
        scores: np.ndarray,
        frame_idx: int,
        gt_box_cxcywh: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process one sampled frame and return active tracks.

        Args:
            dets_xyxy:     (N, 4) detection boxes [x1, y1, x2, y2].
                           Empty (0, 4) array if no detections.
            scores:        (N,) detection confidence scores.
            frame_idx:     Raw frame index (used to look up CMC affine).
            gt_box_cxcywh: Only used when init_mode="first_frame_gt" on frame 0.

        Returns:
            (M, 6) ndarray: confirmed tracks [x1, y1, x2, y2, track_id, score].
            Shape (0, 6) if no confirmed tracks.
        """
        self.frame_count += 1

        # ── Step 1: filter detections by τ ────────────────────────────
        if len(dets_xyxy) > 0:
            mask = scores >= self._tau
            dets_xyxy = dets_xyxy[mask]
            scores    = scores[mask]

        # ── Step 2: Kalman predict ─────────────────────────────────────
        for trk in self._tracks:
            trk.predict()

        # ── Step 3: CMC warp ──────────────────────────────────────────
        if self._cmc_on and frame_idx in self._affine_map:
            M = self._affine_map[frame_idx]
            if not np.allclose(M, np.eye(2, 3)):
                for trk in self._tracks:
                    trk.warp(M)

        # ── First-frame GT init ───────────────────────────────────────
        if self.frame_count == 1 and self._init_mode == "first_frame_gt":
            if gt_box_cxcywh is not None:
                trk = make_track(gt_box_cxcywh, self._cfg)
                trk.score = 1.0
                self._tracks.append(trk)
            return self._get_output()

        # ── Steps 4-8: match → update → delete → init ─────────────────
        if len(self._tracks) == 0:
            # No existing tracks: init from all high-confidence detections
            for i in range(len(dets_xyxy)):
                if scores[i] >= self._eta:
                    cx, cy = (dets_xyxy[i, 0] + dets_xyxy[i, 2]) / 2, \
                             (dets_xyxy[i, 1] + dets_xyxy[i, 3]) / 2
                    w = dets_xyxy[i, 2] - dets_xyxy[i, 0]
                    h = dets_xyxy[i, 3] - dets_xyxy[i, 1]
                    trk = make_track(np.array([cx, cy, w, h]), self._cfg)
                    trk.score = float(scores[i])
                    self._tracks.append(trk)
            return self._get_output()

        if len(dets_xyxy) == 0:
            # No detections: all tracks age
            unmatched_trks = list(range(len(self._tracks)))
            matched, unmatched_dets = [], []
        else:
            # ── Step 4: cost matrix ────────────────────────────────────
            trk_xyxy = np.array([t.get_box_xyxy() for t in self._tracks])
            C = build_cost_matrix(trk_xyxy, dets_xyxy, scores, self._cfg)

            # ── Step 5: Hungarian assignment ──────────────────────────
            matched, unmatched_trks, unmatched_dets = \
                self._hungarian(C, self._thresh)

        # ── Step 6: update matched tracks ─────────────────────────────
        for t_idx, d_idx in matched:
            cx = (dets_xyxy[d_idx, 0] + dets_xyxy[d_idx, 2]) / 2
            cy = (dets_xyxy[d_idx, 1] + dets_xyxy[d_idx, 3]) / 2
            w  = dets_xyxy[d_idx, 2] - dets_xyxy[d_idx, 0]
            h  = dets_xyxy[d_idx, 3] - dets_xyxy[d_idx, 1]
            self._tracks[t_idx].update(
                np.array([cx, cy, w, h]),
                score=float(scores[d_idx]),
            )

        # ── Step 7: delete unmatched tracks ───────────────────────────
        # Algorithm 1: track_buffer=0 → delete immediately
        to_keep = []
        for i, trk in enumerate(self._tracks):
            if i in unmatched_trks:
                if trk.time_since_update > self._buf:
                    continue   # delete
            to_keep.append(trk)
        self._tracks = to_keep

        # ── Step 8: init new tracks for unmatched detections ──────────
        for d_idx in unmatched_dets:
            if scores[d_idx] >= self._eta:
                cx = (dets_xyxy[d_idx, 0] + dets_xyxy[d_idx, 2]) / 2
                cy = (dets_xyxy[d_idx, 1] + dets_xyxy[d_idx, 3]) / 2
                w  = dets_xyxy[d_idx, 2] - dets_xyxy[d_idx, 0]
                h  = dets_xyxy[d_idx, 3] - dets_xyxy[d_idx, 1]
                trk = make_track(np.array([cx, cy, w, h]), self._cfg)
                trk.score = float(scores[d_idx])
                self._tracks.append(trk)

        # ── Step 9: return confirmed tracks ───────────────────────────
        return self._get_output()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_output(self) -> np.ndarray:
        """Collect confirmed tracks as (M, 6) [x1,y1,x2,y2,id,score]."""
        out = []
        for trk in self._tracks:
            if trk.hits >= self._min_hits and trk.time_since_update == 0:
                box = trk.get_box_xyxy()
                out.append([*box, float(trk.id), float(trk.score)])
        if not out:
            return np.empty((0, 6), dtype=np.float64)
        return np.array(out, dtype=np.float64)

    @staticmethod
    def _hungarian(
        cost: np.ndarray,
        thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Run Hungarian algorithm and apply distance threshold.

        Args:
            cost:   (T, D) cost matrix
            thresh: maximum cost to accept a match
        Returns:
            (matches, unmatched_tracks, unmatched_dets)
            matches: list of (track_idx, det_idx)
        """
        T, D = cost.shape
        if T == 0 or D == 0:
            return [], list(range(T)), list(range(D))

        row_ind, col_ind = linear_sum_assignment(cost)

        matched       = []
        unmatched_trks = set(range(T))
        unmatched_dets = set(range(D))

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= thresh:
                matched.append((int(r), int(c)))
                unmatched_trks.discard(r)
                unmatched_dets.discard(c)

        return matched, sorted(unmatched_trks), sorted(unmatched_dets)

    @property
    def n_tracks(self) -> int:
        return len(self._tracks)
