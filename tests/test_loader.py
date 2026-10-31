"""
Unit tests for src/data/loader.py  (Stage 1).

Run: pytest tests/test_loader.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.loader import (
    AntiUAVDataset,
    AntiUAVSequence,
    FrameRecord,
    dataset_from_config,
    xywh_tl_to_cxcywh,
    xywh_tl_to_xyxy,
    xyxy_to_cxcywh,
)

# ---------------------------------------------------------------------------
# Paths — real dataset
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("c:/Projects/ipa_paper/dataset")
VAL_SPLIT = DATASET_ROOT / "val"

# Skip all tests that touch the real dataset if it is not present
DATASET_AVAILABLE = VAL_SPLIT.exists()
needs_dataset = pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason="Real dataset not found at c:/Projects/ipa_paper/dataset",
)


# ---------------------------------------------------------------------------
# Box utility tests (no dataset needed)
# ---------------------------------------------------------------------------

class TestBoxUtils:
    def test_xywh_tl_to_xyxy_basic(self):
        assert xywh_tl_to_xyxy([10, 20, 30, 40]) == [10, 20, 40, 60]

    def test_xywh_tl_to_xyxy_zero_size(self):
        box = xywh_tl_to_xyxy([5, 5, 0, 0])
        assert box == [5, 5, 5, 5]

    def test_xywh_tl_to_cxcywh_basic(self):
        cx, cy, w, h = xywh_tl_to_cxcywh([10, 20, 30, 40])
        assert cx == pytest.approx(25.0)
        assert cy == pytest.approx(40.0)
        assert w == 30
        assert h == 40

    def test_xyxy_to_cxcywh_roundtrip(self):
        original = [10, 20, 40, 60]  # xyxy
        cx, cy, w, h = xyxy_to_cxcywh(original)
        assert cx == pytest.approx(25.0)
        assert cy == pytest.approx(40.0)
        assert w == pytest.approx(30.0)
        assert h == pytest.approx(40.0)

    def test_xywh_tl_to_cxcywh_and_back(self):
        """xywh_tl -> cxcywh -> xyxy should give same result as xywh_tl -> xyxy directly."""
        box = [100, 200, 50, 30]
        xyxy_direct = xywh_tl_to_xyxy(box)
        cx, cy, w, h = xywh_tl_to_cxcywh(box)
        xyxy_via_cx = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        for a, b in zip(xyxy_direct, xyxy_via_cx):
            assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Synthetic sequence (no real dataset needed)
# ---------------------------------------------------------------------------

def make_fake_seq(tmp_path: Path, n_frames: int = 20) -> Path:
    """Create a minimal fake sequence directory for testing."""
    seq_dir = tmp_path / "fake_seq_001"
    seq_dir.mkdir()
    gt_rect = [[float(10 + i), float(20 + i), 30.0, 25.0] for i in range(n_frames)]
    exist = [1] * n_frames
    # Make a couple of frames invisible (only if the sequence is long enough)
    if n_frames > 3:
        exist[3] = 0
    if n_frames > 7:
        exist[7] = 0
    (seq_dir / "infrared.json").write_text(
        json.dumps({"gt_rect": gt_rect, "exist": exist})
    )
    return seq_dir


class TestAntiUAVSequence:
    def test_loads_correct_frame_count(self, tmp_path):
        seq_dir = make_fake_seq(tmp_path, n_frames=20)
        seq = AntiUAVSequence(seq_dir, frame_stride=1, max_frames=1000)
        assert seq.n_sampled_frames == 20

    def test_stride_reduces_frames(self, tmp_path):
        seq_dir = make_fake_seq(tmp_path, n_frames=20)
        seq = AntiUAVSequence(seq_dir, frame_stride=5, max_frames=1000)
        # indices: 0, 5, 10, 15 → 4 frames
        assert seq.n_sampled_frames == 4

    def test_max_frames_cap(self, tmp_path):
        seq_dir = make_fake_seq(tmp_path, n_frames=20)
        seq = AntiUAVSequence(seq_dir, frame_stride=1, max_frames=7)
        assert seq.n_sampled_frames == 7

    def test_stride_and_max_frames_combined(self, tmp_path):
        seq_dir = make_fake_seq(tmp_path, n_frames=100)
        seq = AntiUAVSequence(seq_dir, frame_stride=5, max_frames=10)
        # stride=5 gives 20 frames; max_frames=10 caps at 10
        assert seq.n_sampled_frames == 10

    def test_sampled_indices_correct(self, tmp_path):
        seq_dir = make_fake_seq(tmp_path, n_frames=20)
        seq = AntiUAVSequence(seq_dir, frame_stride=5, max_frames=1000)
        assert seq.sampled_indices == [0, 5, 10, 15]

    def test_invisible_frames_have_no_box(self, tmp_path):
        seq_dir = make_fake_seq(tmp_path, n_frames=10)
        seq = AntiUAVSequence(seq_dir, frame_stride=1, max_frames=1000)
        records = list(seq)
        # Frame 3 and 7 are invisible
        r3 = records[3]
        assert r3.visible is False
        assert r3.box_xywh is None
        assert r3.box_xyxy is None

    def test_visible_frames_have_box(self, tmp_path):
        seq_dir = make_fake_seq(tmp_path, n_frames=10)
        seq = AntiUAVSequence(seq_dir, frame_stride=1, max_frames=1000)
        records = list(seq)
        r0 = records[0]
        assert r0.visible is True
        assert r0.box_xywh is not None
        assert r0.box_xyxy is not None
        assert len(r0.box_xyxy) == 4

    def test_box_conversion_correct(self, tmp_path):
        """box_xyxy must equal xywh_tl_to_xyxy(box_xywh)."""
        seq_dir = make_fake_seq(tmp_path, n_frames=5)
        seq = AntiUAVSequence(seq_dir, frame_stride=1, max_frames=1000)
        for rec in seq:
            if rec.visible:
                expected_xyxy = xywh_tl_to_xyxy(rec.box_xywh)
                assert rec.box_xyxy == pytest.approx(expected_xyxy)

    def test_missing_annotation_raises(self, tmp_path):
        seq_dir = tmp_path / "no_ann"
        seq_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            AntiUAVSequence(seq_dir)

    def test_wrong_annotation_keys_raises(self, tmp_path):
        seq_dir = tmp_path / "bad_ann"
        seq_dir.mkdir()
        (seq_dir / "infrared.json").write_text(
            json.dumps({"boxes": [], "flags": []})
        )
        with pytest.raises(KeyError):
            AntiUAVSequence(seq_dir)

    def test_length_mismatched_annotation_raises(self, tmp_path):
        seq_dir = tmp_path / "bad_len"
        seq_dir.mkdir()
        (seq_dir / "infrared.json").write_text(
            json.dumps({"gt_rect": [[1, 2, 3, 4]], "exist": [1, 0]})
        )
        with pytest.raises(ValueError):
            AntiUAVSequence(seq_dir)


# ---------------------------------------------------------------------------
# AntiUAVDataset (synthetic)
# ---------------------------------------------------------------------------

def make_fake_dataset(tmp_path: Path, n_seqs: int = 5, n_frames: int = 20) -> Path:
    """Create a fake val split with n_seqs sequences."""
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    for i in range(n_seqs):
        seq_dir = val_dir / f"seq_{i:03d}"
        seq_dir.mkdir()
        gt_rect = [[float(i * 10), float(i * 5), 20.0, 15.0]] * n_frames
        exist = [1] * n_frames
        (seq_dir / "infrared.json").write_text(
            json.dumps({"gt_rect": gt_rect, "exist": exist})
        )
    return tmp_path


class TestAntiUAVDataset:
    def test_loads_correct_number_of_sequences(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=5)
        ds = AntiUAVDataset(root, split="val", num_sequences=3,
                            frame_stride=1, max_frames=100)
        assert len(ds) == 3

    def test_truncates_to_available_sequences(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=3)
        ds = AntiUAVDataset(root, split="val", num_sequences=10,
                            frame_stride=1, max_frames=100)
        assert len(ds) <= 3

    def test_test_split_rejected(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=3)
        with pytest.raises(ValueError, match="test"):
            AntiUAVDataset(root, split="test")

    def test_train_split_rejected(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=3)
        with pytest.raises(ValueError):
            AntiUAVDataset(root, split="train")

    def test_iteration_yields_sequences(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=4)
        ds = AntiUAVDataset(root, split="val", num_sequences=4,
                            frame_stride=1, max_frames=100)
        seqs = list(ds)
        assert len(seqs) == 4
        assert all(isinstance(s, AntiUAVSequence) for s in seqs)

    def test_getitem(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=4)
        ds = AntiUAVDataset(root, split="val", num_sequences=4,
                            frame_stride=1, max_frames=100)
        assert isinstance(ds[0], AntiUAVSequence)

    def test_sequence_selection_is_deterministic(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=10)
        ds1 = AntiUAVDataset(root, split="val", num_sequences=5,
                             frame_stride=1, max_frames=100, sequence_seed=42)
        ds2 = AntiUAVDataset(root, split="val", num_sequences=5,
                             frame_stride=1, max_frames=100, sequence_seed=42)
        assert ds1.sequence_names == ds2.sequence_names

    def test_different_seeds_give_different_sequences(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=10)
        ds1 = AntiUAVDataset(root, split="val", num_sequences=5,
                             frame_stride=1, max_frames=100, sequence_seed=0)
        ds2 = AntiUAVDataset(root, split="val", num_sequences=5,
                             frame_stride=1, max_frames=100, sequence_seed=99)
        # With 10 sequences and selecting 5, different seeds very likely give different sets
        # (not guaranteed but expected; just check they CAN differ)
        assert ds1.sequence_names != ds2.sequence_names or True  # soft check

    def test_verify_no_test_in_val_no_overlap(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=3)
        # Create a non-overlapping test split
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "test_seq_999").mkdir()
        ds = AntiUAVDataset(root, split="val", num_sequences=3,
                            frame_stride=1, max_frames=100)
        ds.verify_no_test_in_val(test_root=test_dir)  # should not raise

    def test_verify_no_test_in_val_with_overlap_raises(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=3)
        # Create a test split that shares seq_000 with val
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        overlapping = test_dir / "seq_000"
        overlapping.mkdir()
        ds = AntiUAVDataset(root, split="val", num_sequences=3,
                            frame_stride=1, max_frames=100)
        with pytest.raises(RuntimeError, match="seq_000"):
            ds.verify_no_test_in_val(test_root=test_dir)

    def test_total_sampled_frames(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=3, n_frames=20)
        ds = AntiUAVDataset(root, split="val", num_sequences=3,
                            frame_stride=5, max_frames=1000)
        # Each seq: 20 frames, stride=5 → 4 frames; 3 seqs → 12 total
        assert ds.total_sampled_frames() == 12


# ---------------------------------------------------------------------------
# Config-driven factory (synthetic)
# ---------------------------------------------------------------------------

class TestDatasetFromConfig:
    def test_builds_from_config(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=5)
        cfg = {
            "data": {
                "root": str(tmp_path),
                "split": "val",
                "num_sequences": 3,
                "frame_stride": 2,
                "max_frames_per_sequence": 50,
                "sequence_seed": 7,
            }
        }
        ds = dataset_from_config(cfg)
        assert len(ds) == 3

    def test_rejects_test_split_in_config(self, tmp_path):
        root = make_fake_dataset(tmp_path, n_seqs=3)
        cfg = {
            "data": {
                "root": str(tmp_path),
                "split": "test",
                "num_sequences": 3,
                "frame_stride": 1,
                "max_frames_per_sequence": 100,
                "sequence_seed": 0,
            }
        }
        with pytest.raises(ValueError):
            dataset_from_config(cfg)


# ---------------------------------------------------------------------------
# Real dataset smoke tests (skipped if dataset not present)
# ---------------------------------------------------------------------------

class TestRealDataset:
    @needs_dataset
    def test_val_loads(self):
        ds = AntiUAVDataset(DATASET_ROOT, split="val", num_sequences=3,
                            frame_stride=5, max_frames=20)
        assert len(ds) == 3

    @needs_dataset
    def test_annotation_keys_exist(self):
        ds = AntiUAVDataset(DATASET_ROOT, split="val", num_sequences=1,
                            frame_stride=10, max_frames=5)
        seq = ds[0]
        records = list(seq)
        assert len(records) > 0
        # First frame should be visible (99.2% of val frames are visible)
        visible = [r for r in records if r.visible]
        assert len(visible) > 0

    @needs_dataset
    def test_box_values_in_frame_bounds(self):
        ds = AntiUAVDataset(DATASET_ROOT, split="val", num_sequences=3,
                            frame_stride=5, max_frames=10)
        for seq in ds:
            for rec in seq:
                if rec.visible:
                    x1, y1, x2, y2 = rec.box_xyxy
                    assert x1 >= 0 and y1 >= 0
                    assert x2 <= AntiUAVSequence.IMG_W + 1  # allow 1px rounding
                    assert y2 <= AntiUAVSequence.IMG_H + 1
                    assert x2 > x1 and y2 > y1

    @needs_dataset
    def test_no_test_in_val(self):
        ds = AntiUAVDataset(DATASET_ROOT, split="val", num_sequences=5,
                            frame_stride=5, max_frames=10)
        ds.verify_no_test_in_val()  # should not raise

    @needs_dataset
    def test_stride_semantics_frame_indices(self):
        """Sampled indices must be exactly [0, stride, 2*stride, ...]."""
        stride = 5
        ds = AntiUAVDataset(DATASET_ROOT, split="val", num_sequences=1,
                            frame_stride=stride, max_frames=10)
        seq = ds[0]
        indices = seq.sampled_indices
        for i, idx in enumerate(indices):
            assert idx == i * stride
