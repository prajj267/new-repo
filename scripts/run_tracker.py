"""
Tracker ablation experiments — Stage 6.

Runs every variant from requirements §5.6 on the same 15 val sequences,
at IoU thresholds 0.5 and 0.25.  Records per-sequence and aggregate metrics.

Variants
--------
  detector_only   — count detections as hypotheses, no tracker
  iou_only        — gamma=0 (IoU cost only), no CMC
  l2_only         — gamma=1 (L2 cost only), no CMC
  hybrid          — gamma=0.25, no CMC
  hybrid_cmc      — gamma=0.25, CMC enabled  (full FLIT)
  gamma_sweep     — gamma in {0, 0.1, 0.25, 0.5, 0.75, 1.0}, no CMC
  l2_norm_fixed   — hybrid, l2_norm="fixed"
  thresh_sweep    — match_thresh in {0.5, 0.7, 0.8, 0.9}, hybrid
  buffer_sweep    — track_buffer in {0, 1, 2, 3}, hybrid

Outputs
-------
  results/tracker/ablations.csv   — one row per (variant, iou_thresh)
  results/tracker/ablations.md    — markdown table
  results/tracker/mota_ablation.png

Usage
-----
    python scripts/run_tracker.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import dataset_from_config
from src.evaluate import evaluate_dataset, evaluate_sequence
from src.tracker.cmc import affines_to_dict, load_affine_cache
from src.tracker.flit import FLITTracker


# ---------------------------------------------------------------------------
# Load detection cache
# ---------------------------------------------------------------------------

def load_det_cache(seq_name: str, cache_dir: Path, expected_stride: int) -> dict:
    p = cache_dir / f"{seq_name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Detection cache not found: {p}\n"
            "Run: python scripts/build_detection_cache.py --config configs/default.yaml"
        )
    with open(p) as f:
        data = json.load(f)
    cached = data.get("frame_stride")
    if cached != expected_stride:
        raise ValueError(
            f"Detection cache stride mismatch for {seq_name}: "
            f"cached={cached}, expected={expected_stride}."
        )
    return data


def det_cache_to_arrays(
    frames: list,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Convert cache frames list to {frame_idx: (boxes_xyxy, scores)}."""
    result = {}
    for f in frames:
        boxes = np.array(f["boxes"], dtype=np.float64) if f["boxes"] else np.empty((0, 4))
        scores = np.array(f["scores"], dtype=np.float64) if f["scores"] else np.empty(0)
        result[f["frame_idx"]] = (boxes, scores)
    return result


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def make_variants(base_cfg: dict) -> List[Tuple[str, dict]]:
    """Return list of (variant_name, cfg_override) pairs."""

    def override(base, **kwargs) -> dict:
        c = copy.deepcopy(base)
        for k, v in kwargs.items():
            parts = k.split(".")
            node = c
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = v
        return c

    variants: List[Tuple[str, dict]] = []

    # ── Core variants ─────────────────────────────────────────────────
    variants.append(("detector_only",  None))   # special: no tracker
    variants.append(("iou_only",       override(base_cfg,
                         **{"tracker.gamma": 0.0, "cmc.enabled": False})))
    variants.append(("l2_only",        override(base_cfg,
                         **{"tracker.gamma": 1.0, "cmc.enabled": False})))
    variants.append(("hybrid",         override(base_cfg,
                         **{"tracker.gamma": 0.25, "cmc.enabled": False})))
    variants.append(("hybrid_cmc",     override(base_cfg,
                         **{"tracker.gamma": 0.25, "cmc.enabled": True})))

    # ── Gamma sweep ───────────────────────────────────────────────────
    for g in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        name = f"gamma_{g:.2f}".replace(".", "p")
        variants.append((name, override(base_cfg,
            **{"tracker.gamma": g, "cmc.enabled": False})))

    # ── L2 norm: fixed ───────────────────────────────────────────────
    variants.append(("l2_norm_fixed",  override(base_cfg,
        **{"tracker.gamma": 0.25, "tracker.l2_norm": "fixed",
           "cmc.enabled": False})))

    # ── Match threshold sweep ─────────────────────────────────────────
    for thresh in [0.5, 0.7, 0.8, 0.9]:
        name = f"thresh_{thresh:.1f}".replace(".", "p")
        variants.append((name, override(base_cfg,
            **{"tracker.gamma": 0.25, "tracker.match_thresh": thresh,
               "cmc.enabled": False})))

    # ── Track buffer sweep ────────────────────────────────────────────
    for buf in [0, 1, 2, 3]:
        name = f"buffer_{buf}"
        variants.append((name, override(base_cfg,
            **{"tracker.gamma": 0.25, "tracker.track_buffer": buf,
               "cmc.enabled": False})))

    return variants


# ---------------------------------------------------------------------------
# Run one variant on all sequences
# ---------------------------------------------------------------------------

def run_variant(
    variant_name: str,
    cfg: dict | None,
    dataset,
    det_caches: Dict[str, Dict],
    affine_caches: Dict[str, dict],
    base_cfg: dict,
    iou_thresh: float,
) -> Dict:
    """Run one variant and return aggregate + per-seq metrics."""
    accs = []
    seq_names = []

    for seq in dataset:
        det_data = det_caches[seq.name]
        frame_map = det_cache_to_arrays(det_data["frames"])

        # Load affine map for this sequence (or empty dict if CMC disabled)
        use_cmc = (cfg is not None) and bool(cfg.get("cmc", {}).get("enabled", False))
        if use_cmc and seq.name in affine_caches:
            affine_map = affine_caches[seq.name]
        else:
            affine_map = {}

        gt_records = list(seq)   # FrameRecord objects

        # ── Detector-only baseline ──────────────────────────────────
        if variant_name == "detector_only":
            tracker_outputs = []
            for rec in gt_records:
                boxes, scores = frame_map.get(rec.frame_idx, (np.empty((0,4)), np.empty(0)))
                track_ids = np.arange(len(boxes), dtype=np.float64)
                tracker_outputs.append((rec.frame_idx, boxes, track_ids))
        else:
            # ── FLIT tracker ─────────────────────────────────────────
            tracker = FLITTracker(cfg, affine_map=affine_map)
            tracker.reset()
            tracker_outputs = []

            for rec in gt_records:
                boxes, scores = frame_map.get(rec.frame_idx, (np.empty((0,4)), np.empty(0)))
                out = tracker.update(boxes, scores, frame_idx=rec.frame_idx)

                if len(out) > 0:
                    trk_boxes = out[:, :4]
                    trk_ids   = out[:, 4]
                else:
                    trk_boxes = np.empty((0, 4))
                    trk_ids   = np.empty(0)

                tracker_outputs.append((rec.frame_idx, trk_boxes, trk_ids))

        acc = evaluate_sequence(gt_records, tracker_outputs, iou_thresh=iou_thresh)
        accs.append(acc)
        seq_names.append(seq.name)

    agg = evaluate_dataset(accs, seq_names)
    agg["variant"]    = variant_name
    agg["iou_thresh"] = iou_thresh

    # Per-sequence mean ± std
    motas = [agg["per_seq"][n]["mota"] for n in seq_names]
    agg["mota_mean"] = float(np.mean(motas))
    agg["mota_std"]  = float(np.std(motas))

    return agg


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "variant", "iou_thresh",
    "mota", "mota_mean", "mota_std",
    "idf1", "fp", "fn", "ids", "num_frames",
]


def save_csv(rows: List[Dict], path: Path, seq_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Tracker ablation results\n")
        f.write(f"# Sequences: {', '.join(seq_names)}\n")
        f.write(f"# Detector: gt_simulation (see design.md section 4.1)\n")
        f.write(f"# NOT comparable with Table 3 of the paper\n")
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def save_markdown(rows: List[Dict], path: Path, seq_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "> **Detector: GT simulation — results NOT comparable with Table 3 of the paper.**",
        f"> Sequences ({len(seq_names)}): {', '.join(seq_names)}",
        "",
        "| Variant | IoU | MOTA (pool) | MOTA mean±std | IDF1 | FP | FN | IDS |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} "
            f"| {r['iou_thresh']} "
            f"| {r['mota']:.4f} "
            f"| {r['mota_mean']:.4f}±{r['mota_std']:.4f} "
            f"| {r['idf1']:.4f} "
            f"| {r['fp']} "
            f"| {r['fn']} "
            f"| {r['ids']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(rows: List[Dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Only plot the core 5 variants × 2 iou_thresh for clarity
    core = ["detector_only", "iou_only", "l2_only", "hybrid", "hybrid_cmc"]
    colours = {
        "detector_only": "#aaaaaa",
        "iou_only":      "#1f77b4",
        "l2_only":       "#d62728",
        "hybrid":        "#2ca02c",
        "hybrid_cmc":    "#9467bd",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax_i, iou_t in enumerate([0.5, 0.25]):
        ax = axes[ax_i]
        subset = [r for r in rows if r["variant"] in core and r["iou_thresh"] == iou_t]
        names  = [r["variant"] for r in subset]
        motas  = [r["mota_mean"] for r in subset]
        stds   = [r["mota_std"] for r in subset]
        cols   = [colours.get(n, "#333333") for n in names]

        x = np.arange(len(names))
        bars = ax.bar(x, motas, yerr=stds, capsize=4, color=cols, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("MOTA (mean across sequences)")
        ax.set_title(f"IoU thresh = {iou_t}")
        ax.set_ylim(0, 1.05)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "FLIT tracker ablations — GT simulation detector\n"
        "NOT comparable with Table 3",
        fontsize=9,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out_dir", default="results/tracker")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    cache_det  = Path(base_cfg["paths"]["cache_detections"])
    cache_aff  = Path(base_cfg["paths"]["cache_affines"])
    frame_stride = int(base_cfg["data"]["frame_stride"])
    out_dir    = Path(args.out_dir)

    dataset = dataset_from_config(base_cfg)
    seq_names = dataset.sequence_names
    print(f"Dataset: {len(seq_names)} sequences")
    print(f"Sequences: {seq_names}\n")

    # Pre-load all caches
    print("Loading detection caches…")
    det_caches: Dict[str, dict] = {}
    for seq in dataset:
        det_caches[seq.name] = load_det_cache(seq.name, cache_det, frame_stride)

    print("Loading affine caches…")
    affine_caches: Dict[str, dict] = {}
    for seq in dataset:
        try:
            recs = load_affine_cache(seq.name, cache_aff, frame_stride)
            affine_caches[seq.name] = affines_to_dict(recs)
        except FileNotFoundError:
            affine_caches[seq.name] = {}

    variants = make_variants(base_cfg)
    iou_thresholds = [0.5, 0.25]

    print(f"\nRunning {len(variants)} variants × {len(iou_thresholds)} IoU thresholds…\n")

    all_rows: List[Dict] = []
    t0 = time.time()
    total = len(variants) * len(iou_thresholds)
    done = 0

    for variant_name, variant_cfg in variants:
        for iou_t in iou_thresholds:
            done += 1
            print(f"[{done:>3}/{total}] {variant_name:<22} iou={iou_t}…", end=" ", flush=True)
            t1 = time.time()
            result = run_variant(
                variant_name, variant_cfg, dataset,
                det_caches, affine_caches, base_cfg, iou_t,
            )
            elapsed = time.time() - t1
            print(
                f"MOTA={result['mota']:.4f} "
                f"({result['mota_mean']:.4f}±{result['mota_std']:.4f}) "
                f"FP={result['fp']} FN={result['fn']} IDS={result['ids']}  "
                f"[{elapsed:.1f}s]"
            )
            all_rows.append(result)

    total_t = time.time() - t0

    # Save outputs
    csv_path  = out_dir / "ablations.csv"
    md_path   = out_dir / "ablations.md"
    plot_path = out_dir / "mota_ablation.png"

    save_csv(all_rows, csv_path, seq_names)
    save_markdown(all_rows, md_path, seq_names)
    save_plot(all_rows, plot_path)

    print(f"\nTotal time: {total_t:.1f}s")
    print(f"Saved:\n  {csv_path}\n  {md_path}\n  {plot_path}\n")

    # ── Print core results summary ─────────────────────────────────
    core = ["detector_only", "iou_only", "l2_only", "hybrid", "hybrid_cmc"]
    print("=" * 75)
    print("  CORE RESULTS  —  GT simulation detector  —  NOT comparable with Table 3")
    print("=" * 75)
    for iou_t in iou_thresholds:
        print(f"\n  IoU threshold = {iou_t}")
        print(f"  {'Variant':<22} {'MOTA(pool)':>11} {'mean±std':>16} "
              f"{'IDF1':>7} {'FP':>7} {'FN':>7} {'IDS':>5}")
        print("  " + "-" * 73)
        for variant_name, _ in variants:
            if variant_name not in core:
                continue
            row = next((r for r in all_rows
                        if r["variant"] == variant_name
                        and r["iou_thresh"] == iou_t), None)
            if row is None:
                continue
            print(
                f"  {variant_name:<22} "
                f"{row['mota']:>11.4f} "
                f"{row['mota_mean']:>8.4f}±{row['mota_std']:.4f} "
                f"{row['idf1']:>7.4f} "
                f"{row['fp']:>7} "
                f"{row['fn']:>7} "
                f"{row['ids']:>5}"
            )
    print("=" * 75)

    # ── IDS observation ────────────────────────────────────────────
    ids_vals = [r["ids"] for r in all_rows
                if r["variant"] in core and r["iou_thresh"] == 0.5]
    if all(v <= 5 for v in ids_vals):
        print("\nNOTE: IDS ≈ 0 for all variants — expected for single-target data.")
        print("      MOTA/FP/FN differences and IDF1 are the meaningful signals.")

    # ── CMC assessment ─────────────────────────────────────────────
    hybrid_mota = next(
        (r["mota"] for r in all_rows
         if r["variant"] == "hybrid" and r["iou_thresh"] == 0.5), None)
    cmc_mota = next(
        (r["mota"] for r in all_rows
         if r["variant"] == "hybrid_cmc" and r["iou_thresh"] == 0.5), None)
    if hybrid_mota is not None and cmc_mota is not None:
        diff = cmc_mota - hybrid_mota
        if diff > 0.001:
            print(f"\nCMC: hybrid_cmc ({cmc_mota:.4f}) > hybrid ({hybrid_mota:.4f})"
                  f"  Δ={diff:+.4f}  CMC HELPS on this data.")
        elif diff < -0.001:
            print(f"\nCMC: hybrid_cmc ({cmc_mota:.4f}) < hybrid ({hybrid_mota:.4f})"
                  f"  Δ={diff:+.4f}  CMC does NOT help (mostly stationary camera).")
        else:
            print(f"\nCMC: hybrid_cmc ({cmc_mota:.4f}) ≈ hybrid ({hybrid_mota:.4f})"
                  f"  Δ={diff:+.4f}  CMC has negligible effect.")


if __name__ == "__main__":
    main()
