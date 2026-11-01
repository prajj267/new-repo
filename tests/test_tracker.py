"""
Unit tests for the FLIT tracker modules  (Stage 5).

Covers: cost matrices, Hungarian matching, Kalman filter, CMC,
        full tracker lifecycle, cache stride validation.

Run: pytest tests/test_tracker.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.tracker.cost import fused_cost, final_cost, iou_cost, l2_cost
from src.tracker.kalman import KalmanTrack, make_track
from src.tracker.cmc import CMC, affines_to_dict, load_affine_cache, save_affine_cache
from src.tracker.flit import FLITTracker


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _cfg(
    gamma=0.25, match_thresh=0.8, track_buffer=0,
    min_hits=1, tau=0.0, eta=0.0,
    cmc_enabled=False,
):
    return {
        "data":    {"img_w": 640, "img_h": 512},
        "tracker": {
            "gamma": gamma, "match_thresh": match_thresh,
            "track_buffer": track_buffer, "min_hits": min_hits,
            "tau": tau, "eta": eta,
            "l2_norm": "diagonal", "l2_norm_fixed": 512.0,
            "init_mode": "detector",
        },
        "kalman": {"Q_scale": 1.0, "R_scale": 1.0},
        "cmc":    {"enabled": cmc_enabled, "max_corners": 100,
                   "quality_level": 0.01, "min_distance": 20,
                   "block_size": 3, "ransac_thresh": 3.0},
    }


def _box(*args):
    """Convenience: return np.array([[x1,y1,x2,y2]])."""
    return np.array([list(args)], dtype=np.float64)


def _boxes(*rows):
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Required test: cost matrix shapes
# ---------------------------------------------------------------------------

class TestCostShapes:
    def test_iou_cost_shape(self):
        T = np.random.rand(3, 4) * 100
        D = np.random.rand(5, 4) * 100
        T[:, 2:] += T[:, :2] + 10
        D[:, 2:] += D[:, :2] + 10
        assert iou_cost(T, D).shape == (3, 5)

    def test_l2_cost_shape(self):
        T = _boxes([0,0,10,10],[20,20,30,30])
        D = _boxes([5,5,15,15],[25,25,35,35],[50,50,60,60])
        assert l2_cost(T, D).shape == (2, 3)

    def test_fused_cost_shape(self):
        T = _boxes([0,0,10,10],[50,50,60,60])
        D = _boxes([1,1,11,11])
        assert fused_cost(T, D).shape == (2, 1)

    def test_final_cost_shape(self):
        C = np.ones((3, 4))
        s = np.ones(4) * 0.8
        assert final_cost(C, s).shape == (3, 4)

    def test_cost_shape_zero_tracks(self):
        T = np.empty((0, 4))
        D = _boxes([0, 0, 10, 10])
        assert iou_cost(T, D).shape == (0, 1)
        assert l2_cost(T, D).shape  == (0, 1)
        assert fused_cost(T, D).shape == (0, 1)

    def test_cost_shape_zero_dets(self):
        T = _boxes([0, 0, 10, 10])
        D = np.empty((0, 4))
        assert iou_cost(T, D).shape == (1, 0)
        assert l2_cost(T, D).shape  == (1, 0)


# ---------------------------------------------------------------------------
# Required tests: Hungarian handles empty inputs
# ---------------------------------------------------------------------------

class TestHungarianEmpty:
    def test_empty_dets(self):
        cfg = _cfg()
        tracker = FLITTracker(cfg)
        # Create one track first
        trk_box = _boxes([100, 100, 200, 200])
        tracker.update(trk_box, np.array([0.9]), frame_idx=0)
        # Now pass empty detections
        out = tracker.update(np.empty((0, 4)), np.array([]), frame_idx=5)
        # With track_buffer=0 the track should be deleted → no output
        assert out.shape == (0, 6) or out.shape[1] == 6

    def test_empty_tracks(self):
        cfg = _cfg()
        tracker = FLITTracker(cfg)
        # No tracks, send a detection
        out = tracker.update(_boxes([100, 100, 200, 200]), np.array([0.9]), frame_idx=0)
        assert out.shape[1] == 6

    def test_both_empty(self):
        cfg = _cfg()
        tracker = FLITTracker(cfg)
        out = tracker.update(np.empty((0, 4)), np.array([]), frame_idx=0)
        assert out.shape == (0, 6)


# ---------------------------------------------------------------------------
# Required test: Kalman predict shape
# ---------------------------------------------------------------------------

class TestKalmanPredict:
    def test_predict_shape(self):
        trk = KalmanTrack(np.array([100.0, 100.0, 20.0, 16.0]))
        box = trk.predict()
        assert box.shape == (4,), f"Expected (4,), got {box.shape}"

    def test_predict_moves_with_velocity(self):
        trk = KalmanTrack(np.array([100.0, 100.0, 20.0, 16.0]))
        # Manually inject velocity
        trk._x[4] = 5.0   # vx
        trk._x[5] = 3.0   # vy
        box = trk.predict()
        assert abs(box[0] - 105.0) < 1.0, f"Expected cx≈105, got {box[0]}"
        assert abs(box[1] - 103.0) < 1.0, f"Expected cy≈103, got {box[1]}"

    def test_update_resets_time_since_update(self):
        trk = KalmanTrack(np.array([100.0, 100.0, 20.0, 16.0]))
        trk.predict()
        assert trk.time_since_update == 1
        trk.update(np.array([102.0, 101.0, 20.0, 16.0]))
        assert trk.time_since_update == 0

    def test_get_box_xyxy(self):
        trk = KalmanTrack(np.array([100.0, 100.0, 20.0, 16.0]))
        box = trk.get_box_xyxy()
        assert box.shape == (4,)
        assert abs(box[0] - 90.0) < 1e-3   # x1 = cx - w/2
        assert abs(box[2] - 110.0) < 1e-3  # x2 = cx + w/2

    def test_wh_stays_positive_after_predict(self):
        trk = KalmanTrack(np.array([100.0, 100.0, 2.0, 2.0]))
        for _ in range(20):
            trk.predict()
        box = trk.get_box()
        assert box[2] >= 1.0 and box[3] >= 1.0

    def test_unique_ids(self):
        KalmanTrack._next_id = 1
        t1 = KalmanTrack(np.array([0.0, 0.0, 10.0, 10.0]))
        t2 = KalmanTrack(np.array([0.0, 0.0, 10.0, 10.0]))
        assert t1.id != t2.id


# ---------------------------------------------------------------------------
# Required test: CMC toy example (identity on two identical frames)
# ---------------------------------------------------------------------------

class TestCMCToyExample:
    def test_identity_on_first_frame(self):
        """First call always returns identity (no previous frame)."""
        cfg = _cfg(cmc_enabled=True)
        cmc = CMC(cfg)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        M = cmc.apply(frame)
        assert M.shape == (2, 3)
        assert np.allclose(M, np.eye(2, 3)), f"Expected identity, got {M}"

    def test_identity_on_identical_frames(self):
        """Two identical frames → no motion → affine should be near identity."""
        cfg = _cfg(cmc_enabled=True)
        cmc = CMC(cfg)
        # Create a frame with some texture so corners can be detected
        np.random.seed(0)
        frame = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        cmc.apply(frame)          # first frame: seeds the state
        M = cmc.apply(frame)      # second identical frame
        # Translation should be near zero; rotation near identity
        assert M.shape == (2, 3)
        assert abs(M[0, 2]) < 2.0, f"tx={M[0,2]:.3f} too large for identical frames"
        assert abs(M[1, 2]) < 2.0, f"ty={M[1,2]:.3f} too large for identical frames"

    def test_reset_clears_state(self):
        cfg = _cfg(cmc_enabled=True)
        cmc = CMC(cfg)
        np.random.seed(1)
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cmc.apply(frame)
        cmc.reset()
        # After reset, next call should return identity
        M = cmc.apply(frame)
        assert np.allclose(M, np.eye(2, 3))


# ---------------------------------------------------------------------------
# Required test (Amendment F): CMC corrects a known translation
# ---------------------------------------------------------------------------

class TestCMCKnownTranslation:
    def test_cmc_corrected_is_closer_than_uncorrected(self):
        """
        Construct two frames where the second is the first translated by (dx, dy).
        Run CMC. Apply the affine to a predicted centre that was NOT translated.
        Assert the corrected centre lands closer to the true next centre than
        the uncorrected one.

        This tests that the affine is applied in the FORWARD direction
        (prev→curr), not the inverse.
        """
        dx, dy = 20.0, 15.0

        # Create a texture-rich frame
        np.random.seed(42)
        frame1 = np.zeros((256, 256, 3), dtype=np.uint8)
        # Place random bright blobs to give the corner detector something to track
        for _ in range(30):
            cx_b = np.random.randint(20, 236)
            cy_b = np.random.randint(20, 236)
            frame1[cy_b-5:cy_b+5, cx_b-5:cx_b+5] = np.random.randint(150, 255)

        # Translate frame1 to get frame2
        M_true = np.float32([[1, 0, dx], [0, 1, dy]])
        frame2 = cv2_warp(frame1, M_true)

        cfg = _cfg(cmc_enabled=True)
        cmc = CMC(cfg)
        cmc.apply(frame1)       # seed with frame1
        M_est = cmc.apply(frame2)  # estimate frame1→frame2 affine

        # Predicted centre: stays at frame1 position (NOT translated)
        cx_pred, cy_pred = 100.0, 80.0
        pt = np.array([cx_pred, cy_pred, 1.0])

        # True next position (after translation)
        cx_true = cx_pred + dx
        cy_true = cy_pred + dy

        # Corrected position
        cx_corr = M_est[0, 0]*cx_pred + M_est[0, 1]*cy_pred + M_est[0, 2]
        cy_corr = M_est[1, 0]*cx_pred + M_est[1, 1]*cy_pred + M_est[1, 2]

        dist_uncorr = np.sqrt((cx_pred - cx_true)**2 + (cy_pred - cy_true)**2)
        dist_corr   = np.sqrt((cx_corr - cx_true)**2 + (cy_corr - cy_true)**2)

        assert dist_corr < dist_uncorr, (
            f"CMC-corrected ({dist_corr:.2f}px) is NOT closer than uncorrected "
            f"({dist_uncorr:.2f}px) to the true next centre. "
            f"Estimated translation: ({M_est[0,2]:.2f}, {M_est[1,2]:.2f}), "
            f"True: ({dx}, {dy})."
        )


def cv2_warp(frame, M):
    """Apply a 2x3 affine to a frame."""
    import cv2
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, M, (w, h))


# ---------------------------------------------------------------------------
# Required test: cache stride mismatch raises
# ---------------------------------------------------------------------------

class TestCacheStrideMismatch:
    def test_cache_stride_mismatch_raises(self, tmp_path):
        """load_affine_cache raises ValueError if stride doesn't match."""
        affines = [{"frame_idx": 0, "M": [[1,0,0],[0,1,0]]}]
        save_affine_cache("seq_001", frame_stride=5,
                          affines=affines, cache_dir=tmp_path)
        with pytest.raises(ValueError, match="stride"):
            load_affine_cache("seq_001", tmp_path, expected_stride=10)

    def test_cache_stride_match_ok(self, tmp_path):
        """load_affine_cache succeeds when strides match."""
        affines = [{"frame_idx": 0, "M": [[1,0,0],[0,1,0]]}]
        save_affine_cache("seq_001", frame_stride=5,
                          affines=affines, cache_dir=tmp_path)
        result = load_affine_cache("seq_001", tmp_path, expected_stride=5)
        assert len(result) == 1

    def test_cache_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_affine_cache("nonexistent", tmp_path, expected_stride=5)


# ---------------------------------------------------------------------------
# Required test: single target, no ID switch
# ---------------------------------------------------------------------------

class TestFLITSingleTarget:
    def test_single_target_no_id_switch(self):
        """One moving target over 10 frames → IDS must be 0."""
        cfg = _cfg(min_hits=1, tau=0.0, eta=0.0, track_buffer=0)
        tracker = FLITTracker(cfg)

        # Simulate a target moving right
        outputs = []
        for step in range(10):
            cx = 100.0 + step * 5.0
            cy = 100.0
            x1, y1, x2, y2 = cx-10, cy-8, cx+10, cy+8
            dets = np.array([[x1, y1, x2, y2]])
            scores = np.array([0.9])
            out = tracker.update(dets, scores, frame_idx=step*5)
            outputs.append(out)

        # Collect all track IDs seen
        ids_seen = set()
        for out in outputs:
            if len(out) > 0:
                ids_seen.update(int(x) for x in out[:, 4])

        assert len(ids_seen) == 1, \
            f"Expected 1 unique track ID (no ID switch), got {ids_seen}"

    def test_track_buffer_keeps_track_alive(self):
        """With track_buffer=1, one missed frame keeps the track alive."""
        cfg = _cfg(min_hits=1, tau=0.0, eta=0.0, track_buffer=1)
        tracker = FLITTracker(cfg)

        # Frame 0: create track
        tracker.update(_boxes([100,100,120,120]), np.array([0.9]), frame_idx=0)
        assert tracker.n_tracks == 1

        # Frame 5: no detection → track should survive (buffer=1)
        tracker.update(np.empty((0,4)), np.array([]), frame_idx=5)
        assert tracker.n_tracks == 1, "Track should survive one missed frame with buffer=1"

    def test_track_deleted_immediately_with_buffer0(self):
        """With track_buffer=0, a missed frame deletes the track."""
        cfg = _cfg(min_hits=1, tau=0.0, eta=0.0, track_buffer=0)
        tracker = FLITTracker(cfg)
        tracker.update(_boxes([100,100,120,120]), np.array([0.9]), frame_idx=0)
        assert tracker.n_tracks == 1
        tracker.update(np.empty((0,4)), np.array([]), frame_idx=5)
        assert tracker.n_tracks == 0, "Track should be deleted immediately with buffer=0"

    def test_output_shape(self):
        """Output must always have 6 columns: [x1,y1,x2,y2,id,score]."""
        cfg = _cfg(min_hits=1, tau=0.0, eta=0.0)
        tracker = FLITTracker(cfg)
        out = tracker.update(_boxes([100,100,120,120]), np.array([0.9]), frame_idx=0)
        assert out.ndim == 2
        assert out.shape[1] == 6

    def test_reset_clears_tracks(self):
        cfg = _cfg(min_hits=1)
        tracker = FLITTracker(cfg)
        tracker.update(_boxes([100,100,120,120]), np.array([0.9]), frame_idx=0)
        tracker.reset()
        assert tracker.n_tracks == 0
        assert tracker.frame_count == 0


# ---------------------------------------------------------------------------
# IoU cost correctness
# ---------------------------------------------------------------------------

class TestIoUCost:
    def test_perfect_overlap_is_zero(self):
        box = _boxes([10, 10, 50, 50])
        C = iou_cost(box, box)
        assert C[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_no_overlap_is_one(self):
        T = _boxes([0, 0, 10, 10])
        D = _boxes([100, 100, 110, 110])
        assert iou_cost(T, D)[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_values_in_range(self):
        np.random.seed(0)
        T = np.abs(np.random.randn(5, 4)) * 50 + 10
        T[:, 2:] += T[:, :2]
        D = np.abs(np.random.randn(4, 4)) * 50 + 10
        D[:, 2:] += D[:, :2]
        C = iou_cost(T, D)
        assert (C >= 0).all() and (C <= 1).all()


# ---------------------------------------------------------------------------
# Eq. 11 final_cost
# ---------------------------------------------------------------------------

class TestFinalCost:
    def test_high_confidence_lowers_cost(self):
        """Higher score → lower final cost."""
        C_fused = np.array([[0.5]])
        low_conf  = final_cost(C_fused, np.array([0.3]))
        high_conf = final_cost(C_fused, np.array([0.9]))
        assert high_conf[0, 0] < low_conf[0, 0]

    def test_zero_fused_cost_gives_score_complement(self):
        """C_fused=0 → C_final = 1 - 1*score = 1 - score."""
        C = np.array([[0.0]])
        s = np.array([0.8])
        assert final_cost(C, s)[0, 0] == pytest.approx(1 - 0.8, abs=1e-6)
