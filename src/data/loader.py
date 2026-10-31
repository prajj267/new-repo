"""
Anti-UAV dataset loader.

Confirmed dataset layout (inspected directly):
  dataset/{train,val,test}/<seq_name>/infrared.json   keys: gt_rect, exist
  dataset/{train,val,test}/<seq_name>/infrared.mp4    640x512 @ 20fps

Annotation format:
  gt_rect : list[list[float]]  [x, y, w, h] pixels, top-left origin
  exist   : list[int]          1=visible, 0=not visible

Frame stride semantics (design.md §1.3, Amendment C):
  With frame_stride=N the loader selects indices [0, N, 2N, ...].
  The tracker treats these as consecutive (dt=1 per sampled frame).
  CMC affines are computed between consecutive SAMPLED frames.

Split policy: only 'val' is used. 'test' is never read.
"""

from __future__ import annotations

import json
import random
from collections import namedtuple
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

FrameRecord = namedtuple("FrameRecord", ["frame_idx", "box_xywh", "box_xyxy", "visible"])


def xywh_tl_to_xyxy(box: List[float]) -> List[float]:
    """Convert [x, y, w, h] (top-left) to [x1, y1, x2, y2].

    Args:
        box: [x, y, w, h] in pixels, top-left origin.
    Returns:
        [x1, y1, x2, y2] in pixels.
    """
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xywh_tl_to_cxcywh(box: List[float]) -> List[float]:
    """Convert [x, y, w, h] (top-left) to [cx, cy, w, h] (centre format).

    Args:
        box: [x, y, w, h] in pixels, top-left origin.
    Returns:
        [cx, cy, w, h] in pixels, centre origin.
    """
    x, y, w, h = box
    return [x + w / 2.0, y + h / 2.0, w, h]


def xyxy_to_cxcywh(box: List[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h].

    Args:
        box: [x1, y1, x2, y2] in pixels.
    Returns:
        [cx, cy, w, h] in pixels.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return [x1 + w / 2.0, y1 + h / 2.0, w, h]


# ---------------------------------------------------------------------------
# AntiUAVSequence
# ---------------------------------------------------------------------------

class AntiUAVSequence:
    """One Anti-UAV infrared sequence.

    Loads annotation lazily. Frames are decoded on demand with OpenCV —
    no full extraction to disk. Only frames in the sampled index list are
    ever decoded.

    Args:
        seq_dir:               Path to sequence directory.
        frame_stride:          Use every Nth frame (default 1 = all frames).
        max_frames:            Cap on number of sampled frames (after striding).
        dataset_root:          Root passed through for error messages only.
    """

    IMG_W = 640
    IMG_H = 512

    def __init__(
        self,
        seq_dir: Path,
        frame_stride: int = 1,
        max_frames: int = 10_000,
        dataset_root: Optional[Path] = None,
    ) -> None:
        self.seq_dir = Path(seq_dir)
        self.name = self.seq_dir.name
        self.frame_stride = max(1, frame_stride)
        self.max_frames = max_frames
        self._root = dataset_root

        label_path = self.seq_dir / "infrared.json"
        if not label_path.exists():
            raise FileNotFoundError(
                f"Annotation not found: {label_path}\n"
                "Expected key: infrared.json with fields 'gt_rect' and 'exist'."
            )

        with open(label_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        # Validate confirmed field names
        for key in ("gt_rect", "exist"):
            if key not in ann:
                raise KeyError(
                    f"Expected key '{key}' in {label_path}. "
                    f"Found: {list(ann.keys())}. Update loader.py if field names differ."
                )

        self._gt_rect: List[List[float]] = ann["gt_rect"]
        self._exist: List[int] = ann["exist"]

        if len(self._gt_rect) != len(self._exist):
            raise ValueError(
                f"gt_rect length {len(self._gt_rect)} != exist length {len(self._exist)} "
                f"in {label_path}"
            )

        self._n_raw = len(self._exist)

        # Build sampled index list
        raw_indices = list(range(0, self._n_raw, self.frame_stride))
        self._sampled_indices: List[int] = raw_indices[: self.max_frames]

        # Lazy frame cache: raw_frame_idx -> np.ndarray (BGR)
        self._frame_cache: dict = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_raw_frames(self) -> int:
        """Total raw frames in the sequence (before striding)."""
        return self._n_raw

    @property
    def n_sampled_frames(self) -> int:
        """Number of frames that will be yielded after striding and capping."""
        return len(self._sampled_indices)

    @property
    def sampled_indices(self) -> List[int]:
        """Raw frame indices that are selected by the stride."""
        return list(self._sampled_indices)

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, raw_idx: int) -> Optional[np.ndarray]:
        """Decode and return a BGR frame by raw index. Cached after first decode.

        Args:
            raw_idx: Zero-based raw frame index.
        Returns:
            BGR ndarray (H, W, 3) or None if the video cannot be read.
        """
        if raw_idx in self._frame_cache:
            return self._frame_cache[raw_idx]

        video_path = self.seq_dir / "infrared.mp4"
        if not video_path.exists():
            return None

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        self._frame_cache[raw_idx] = frame
        return frame

    # ------------------------------------------------------------------
    # Annotation access
    # ------------------------------------------------------------------

    def get_record(self, raw_idx: int) -> FrameRecord:
        """Return the FrameRecord for a raw frame index.

        Args:
            raw_idx: Zero-based raw frame index.
        Returns:
            FrameRecord with box_xywh and box_xyxy set to None when not visible.
        """
        visible = bool(self._exist[raw_idx])
        raw_box = self._gt_rect[raw_idx]

        # Some annotations store [0,0,0,0] when not visible
        if visible and (raw_box is None or all(v == 0 for v in raw_box)):
            visible = False

        box_xywh = list(raw_box) if visible else None
        box_xyxy = xywh_tl_to_xyxy(raw_box) if visible else None

        return FrameRecord(
            frame_idx=raw_idx,
            box_xywh=box_xywh,
            box_xyxy=box_xyxy,
            visible=visible,
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[FrameRecord]:
        for idx in self._sampled_indices:
            yield self.get_record(idx)

    def __len__(self) -> int:
        return self.n_sampled_frames

    def __repr__(self) -> str:
        return (
            f"AntiUAVSequence(name={self.name!r}, "
            f"raw={self._n_raw}, sampled={self.n_sampled_frames}, "
            f"stride={self.frame_stride})"
        )


# ---------------------------------------------------------------------------
# AntiUAVDataset
# ---------------------------------------------------------------------------

class AntiUAVDataset:
    """All selected sequences for one split of Anti-UAV (infrared only).

    Selects num_sequences from the split, sorted by name, shuffled with
    sequence_seed, then truncated to num_sequences.

    Args:
        root:          Dataset root directory (contains train/, val/, test/).
        split:         Must be 'val'. 'test' is never allowed.
        num_sequences: How many sequences to use (default 15).
        frame_stride:  Use every Nth frame (default 5).
        max_frames:    Cap per sequence after striding (default 200).
        sequence_seed: Seed for sequence selection shuffle (default 42).
    """

    ALLOWED_SPLITS = ("val",)  # test is never read per project rules

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        num_sequences: int = 15,
        frame_stride: int = 5,
        max_frames: int = 200,
        sequence_seed: int = 42,
    ) -> None:
        if split not in self.ALLOWED_SPLITS:
            raise ValueError(
                f"split must be one of {self.ALLOWED_SPLITS}. "
                f"Got {split!r}. The test split is never read."
            )

        self.root = Path(root)
        self.split = split
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.num_sequences = num_sequences
        self.sequence_seed = sequence_seed

        self.split_dir = self.root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}\n"
                "Set data.root correctly in configs/default.yaml."
            )

        self._sequences: List[AntiUAVSequence] = self._load_sequences()

    def _load_sequences(self) -> List[AntiUAVSequence]:
        # Find all valid sequence dirs (contain infrared.json)
        all_dirs = sorted(
            d for d in self.split_dir.iterdir()
            if d.is_dir() and (d / "infrared.json").exists()
        )
        if not all_dirs:
            raise RuntimeError(
                f"No valid sequences found in {self.split_dir}. "
                "Each sequence dir must contain infrared.json."
            )

        # Seeded shuffle then truncate
        rng = random.Random(self.sequence_seed)
        shuffled = list(all_dirs)
        rng.shuffle(shuffled)
        selected = shuffled[: self.num_sequences]
        selected.sort()  # sort selected subset for determinism

        return [
            AntiUAVSequence(
                d,
                frame_stride=self.frame_stride,
                max_frames=self.max_frames,
                dataset_root=self.root,
            )
            for d in selected
        ]

    # ------------------------------------------------------------------

    @property
    def sequence_names(self) -> List[str]:
        return [s.name for s in self._sequences]

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> AntiUAVSequence:
        return self._sequences[idx]

    def __iter__(self) -> Iterator[AntiUAVSequence]:
        return iter(self._sequences)

    def __repr__(self) -> str:
        return (
            f"AntiUAVDataset(split={self.split!r}, "
            f"sequences={len(self._sequences)}, "
            f"frame_stride={self.frame_stride}, "
            f"max_frames={self.max_frames})"
        )

    def total_sampled_frames(self) -> int:
        """Total frames across all selected sequences after striding."""
        return sum(len(s) for s in self._sequences)

    # ------------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------------

    def verify_no_test_in_val(self, test_root: Optional[Path] = None) -> None:
        """Raise if any selected sequence name also appears in test.

        Args:
            test_root: Override for the test directory. Defaults to root/test.
        Raises:
            RuntimeError if overlap is found.
        """
        test_dir = (test_root or (self.root / "test"))
        if not test_dir.exists():
            return  # test dir not present; nothing to check

        test_names = {d.name for d in test_dir.iterdir() if d.is_dir()}
        overlap = set(self.sequence_names) & test_names
        if overlap:
            raise RuntimeError(
                f"DATA INTEGRITY ERROR: {len(overlap)} sequence(s) appear in both "
                f"val and test: {sorted(overlap)}"
            )


# ---------------------------------------------------------------------------
# Config-driven factory
# ---------------------------------------------------------------------------

def dataset_from_config(cfg: dict) -> AntiUAVDataset:
    """Build an AntiUAVDataset from a loaded config dict.

    Args:
        cfg: Dict loaded from configs/default.yaml.
    Returns:
        AntiUAVDataset for the configured val split.
    Raises:
        ValueError if cfg['data']['split'] != 'val'.
    """
    d = cfg["data"]
    split = d.get("split", "val")
    if split != "val":
        raise ValueError(
            f"Only 'val' split is allowed. Got {split!r}. "
            "Test split must never be read."
        )
    return AntiUAVDataset(
        root=d["root"],
        split="val",
        num_sequences=int(d.get("num_sequences", 15)),
        frame_stride=int(d.get("frame_stride", 5)),
        max_frames=int(d.get("max_frames_per_sequence", 200)),
        sequence_seed=int(d.get("sequence_seed", 42)),
    )
