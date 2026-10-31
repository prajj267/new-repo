"""
Synthetic tiny-box regression benchmark  (Stage 3).

Tests the SSQ loss terms against CIoU by optimising predicted box parameters
directly with Adam, starting from noisy initialisations.

THIS IS NOT DETECTION AP. RESULTS CANNOT BE COMPARED WITH TABLE 2 OF THE PAPER.

The experiment measures how well each loss guides direct box-parameter
optimisation on CPU for tiny objects (4-32 px wide, ~1.2:1 aspect ratio).
It is a controlled proxy for the loss landscape, not an end-to-end detector.

Outputs
-------
  results/loss/benchmark.csv        — per-variant, per-seed summary statistics
  results/loss/benchmark.md         — markdown table
  results/loss/convergence.png      — mean IoU vs step (all variants, all seeds)
  results/loss/grad_vs_size.png     — mean |gradient| vs GT box width

Usage
-----
    python scripts/run_loss_benchmark.py --config configs/default.yaml
    python scripts/run_loss_benchmark.py --config configs/default.yaml --seeds 0 1 2 3 4
    python scripts/run_loss_benchmark.py --config configs/default.yaml --n_boxes 100 --steps 200
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.ssq import (
    _box_iou_elementwise,
    ciou_loss,
    compute_beta,
    quality_loss,
    scale_loss,
    shape_loss,
    ssq_loss,
)

DISCLAIMER = "NOT detection AP; cannot be compared with Table 2"

# ---------------------------------------------------------------------------
# Box generation
# ---------------------------------------------------------------------------

def generate_gt_boxes(
    n: int,
    seed: int,
    img_w: float = 640.0,
    img_h: float = 512.0,
    min_w: float = 4.0,
    max_w: float = 32.0,
    aspect_min: float = 1.0,
    aspect_max: float = 1.4,
) -> torch.Tensor:
    """Generate n random GT boxes in [cx, cy, w, h] format.

    Box sizes: width in [min_w, max_w], aspect ratio w/h in [aspect_min, aspect_max].
    This matches the paper's description of small UAV targets.

    Args:
        n:           number of boxes
        seed:        RNG seed for reproducibility
        img_w/img_h: canvas size (boxes stay within bounds)
        min_w/max_w: width range in pixels
        aspect_*:    aspect ratio range (w/h)
    Returns:
        Tensor (n, 4) [cx, cy, w, h], dtype float32
    """
    rng = np.random.default_rng(seed)
    w = rng.uniform(min_w, max_w, n).astype(np.float32)
    aspect = rng.uniform(aspect_min, aspect_max, n).astype(np.float32)
    h = (w / aspect).astype(np.float32)
    cx = rng.uniform(w / 2 + 1, img_w - w / 2 - 1, n).astype(np.float32)
    cy = rng.uniform(h / 2 + 1, img_h - h / 2 - 1, n).astype(np.float32)
    return torch.from_numpy(np.stack([cx, cy, w, h], axis=1))


def init_predictions(
    gt: torch.Tensor,
    seed: int,
    centre_sigma: float = 10.0,
    size_sigma: float = 5.0,
) -> torch.Tensor:
    """Initialise predictions as GT + Gaussian noise.

    Args:
        gt:            GT boxes (n, 4) [cx, cy, w, h]
        seed:          RNG seed
        centre_sigma:  std for centre noise (pixels)
        size_sigma:    std for size noise (pixels)
    Returns:
        Initial predictions (n, 4), requires_grad=True, all values > 0
    """
    torch.manual_seed(seed + 10_000)
    noise = torch.randn_like(gt) * torch.tensor(
        [centre_sigma, centre_sigma, size_sigma, size_sigma]
    )
    pred = (gt + noise).clamp(min=1.0)
    return pred.detach().requires_grad_(True)


# ---------------------------------------------------------------------------
# Loss functions keyed by variant name
# ---------------------------------------------------------------------------

def compute_loss(
    variant: str,
    pred: torch.Tensor,
    gt: torch.Tensor,
    step: int,
    total_steps: int,
    window: deque,
    cfg: dict,
) -> torch.Tensor:
    """Compute the mean loss for a given variant.

    Args:
        variant:     one of the VARIANTS keys
        pred:        predicted boxes (n, 4) [cx, cy, w, h]
        gt:          GT boxes (n, 4)
        step:        current optimisation step
        total_steps: total steps for r schedule (SSQ only)
        window:      sliding deque for beta (SSQ/quality only)
        cfg:         loaded config dict
    Returns:
        Scalar loss tensor
    """
    lc = cfg.get("loss", {})
    w_s = float(lc.get("w_s", 0.6))
    size_cov = float(lc.get("size_cov_scale", 1.0))
    sc = float(lc.get("shape_scale", 2.0))
    sig = float(lc.get("sigma", 2.0))
    eps = float(lc.get("eps", 1e-7))

    if variant == "ciou":
        return ciou_loss(pred, gt, eps=eps).mean()

    if variant == "scale_only":
        return scale_loss(pred, gt, w_s=w_s, size_cov_scale=size_cov, eps=eps).mean()

    if variant == "scale_shape":
        Ls = scale_loss(pred, gt, w_s=w_s, size_cov_scale=size_cov, eps=eps)
        Lsh = shape_loss(pred, gt, shape_scale=sc, eps=eps)
        return (Ls + Lsh).mean()

    if variant == "quality_only":
        with torch.no_grad():
            iou_val = _box_iou_elementwise(pred, gt, eps).mean().item()
        beta = compute_beta(iou_val, window, eps=eps)
        return quality_loss(pred, gt, beta=beta, sigma=sig, eps=eps).mean()

    if variant == "ssq":
        return ssq_loss(
            pred, gt, step=step, total_steps=total_steps, window=window,
            w_s=w_s, size_cov_scale=size_cov, shape_scale=sc, sigma=sig,
            eps=eps,
        ).mean()

    raise ValueError(f"Unknown variant: {variant!r}")


VARIANTS = ["ciou", "scale_only", "scale_shape", "quality_only", "ssq"]


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(
    variant: str,
    seed: int,
    n_boxes: int,
    n_steps: int,
    lr: float,
    cfg: dict,
) -> Dict:
    """Optimise predictions under one loss variant for one seed.

    Returns a dict with summary statistics and the IoU curve.
    """
    gt = generate_gt_boxes(n_boxes, seed=seed)
    pred = init_predictions(gt, seed=seed)
    gt_widths = gt[:, 2].numpy()  # for grad-vs-size analysis

    optimiser = torch.optim.Adam([pred], lr=lr)
    window: deque = deque(maxlen=100)

    iou_curve: List[float] = []
    grad_mags: List[Tuple[float, float]] = []  # (gt_w, |grad|) per step
    nan_count = 0
    steps_to_half: Optional[int] = None

    for step in range(n_steps):
        optimiser.zero_grad()
        loss = compute_loss(variant, pred, gt, step, n_steps, window, cfg)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            # Skip this step rather than crashing
            continue

        loss.backward()

        # Record gradient magnitude per box (before clipping)
        if pred.grad is not None:
            grad_norm = pred.grad.norm(dim=1).detach().numpy()  # (n,)
            for gw, gn in zip(gt_widths, grad_norm):
                grad_mags.append((float(gw), float(gn)))

        optimiser.step()
        # Keep w and h positive
        with torch.no_grad():
            pred[:, 2:].clamp_(min=0.5)

        with torch.no_grad():
            iou = _box_iou_elementwise(pred, gt).mean().item()
        iou_curve.append(iou)

        if steps_to_half is None and iou >= 0.5:
            steps_to_half = step + 1  # 1-indexed

    # Final IoU
    with torch.no_grad():
        final_iou = _box_iou_elementwise(pred, gt).mean().item()

    # Loss oscillation: std of IoU in last 50 steps
    last_50 = iou_curve[-50:] if len(iou_curve) >= 50 else iou_curve
    oscillation = float(np.std(last_50)) if last_50 else 0.0

    # Mean gradient per size bin
    grad_by_size: Dict[int, List[float]] = {}
    for gw, gn in grad_mags:
        bin_key = int(gw // 4) * 4  # 4px bins: 0-3, 4-7, 8-11, ...
        grad_by_size.setdefault(bin_key, []).append(gn)
    mean_grad_by_size = {k: float(np.mean(v)) for k, v in sorted(grad_by_size.items())}

    return {
        "variant": variant,
        "seed": seed,
        "final_iou": final_iou,
        "steps_to_half": steps_to_half,
        "nan_count": nan_count,
        "oscillation": oscillation,
        "iou_curve": iou_curve,
        "mean_grad_by_size": mean_grad_by_size,
    }


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------

def aggregate(results: List[Dict]) -> Dict:
    """Aggregate per-seed results into mean ± std."""
    final_ious = [r["final_iou"] for r in results]
    steps = [r["steps_to_half"] for r in results if r["steps_to_half"] is not None]
    nans = sum(r["nan_count"] for r in results)
    oscs = [r["oscillation"] for r in results]

    return {
        "variant": results[0]["variant"],
        "mean_final_iou": float(np.mean(final_ious)),
        "std_final_iou": float(np.std(final_ious)),
        "mean_steps_to_half": float(np.mean(steps)) if steps else float("nan"),
        "pct_reached_half": 100.0 * len(steps) / len(results),
        "total_nan_count": nans,
        "mean_oscillation": float(np.mean(oscs)),
        "n_seeds": len(results),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(
    all_results: Dict[str, List[Dict]],
    out_path: Path,
    n_steps: int,
) -> None:
    """Plot mean IoU vs step for all variants with ±1 std shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "ciou":         "#1f77b4",
        "scale_only":   "#ff7f0e",
        "scale_shape":  "#2ca02c",
        "quality_only": "#d62728",
        "ssq":          "#9467bd",
    }
    labels = {
        "ciou":         "CIoU (Eq.2 baseline)",
        "scale_only":   "Scale only (Eq.3)",
        "scale_shape":  "Scale + Shape (Eq.3+4)",
        "quality_only": "Quality only (Eq.5)",
        "ssq":          "Full SSQ (Eq.7-8)",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    steps_x = list(range(1, n_steps + 1))

    for variant in VARIANTS:
        runs = all_results.get(variant, [])
        if not runs:
            continue
        max_len = max(len(r["iou_curve"]) for r in runs)
        # Pad shorter curves with last value
        curves = []
        for r in runs:
            c = r["iou_curve"]
            if len(c) < max_len:
                c = c + [c[-1]] * (max_len - len(c))
            curves.append(c[:max_len])
        arr = np.array(curves)  # (n_seeds, steps)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        xs = steps_x[:max_len]
        col = colours[variant]
        ax.plot(xs, mean, label=labels[variant], color=col, linewidth=1.8)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=col)

    ax.set_xlabel("Optimisation step")
    ax.set_ylabel("Mean IoU")
    ax.set_title(f"Convergence — {DISCLAIMER}")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


def plot_grad_vs_size(
    all_results: Dict[str, List[Dict]],
    out_path: Path,
) -> None:
    """Plot mean |gradient| vs GT box width (binned) for all variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    colours = {
        "ciou": "#1f77b4", "scale_only": "#ff7f0e", "scale_shape": "#2ca02c",
        "quality_only": "#d62728", "ssq": "#9467bd",
    }

    # Collect all (bin, mean_grad) across seeds per variant
    variant_bin_data: Dict[str, Dict[int, List[float]]] = {}
    for variant in VARIANTS:
        runs = all_results.get(variant, [])
        bins: Dict[int, List[float]] = {}
        for r in runs:
            for b, v in r["mean_grad_by_size"].items():
                bins.setdefault(b, []).append(v)
        variant_bin_data[variant] = bins

    all_bins = sorted({
        b
        for vd in variant_bin_data.values()
        for b in vd
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in VARIANTS:
        bdata = variant_bin_data.get(variant, {})
        xs = [b + 2 for b in all_bins]  # centre of bin
        ys = [float(np.mean(bdata[b])) if b in bdata else float("nan")
              for b in all_bins]
        ax.plot(xs, ys, marker="o", markersize=4, label=variant,
                color=colours[variant], linewidth=1.5)

    ax.set_xlabel("GT box width (px)")
    ax.set_ylabel("Mean |gradient|")
    ax.set_title(f"Gradient magnitude vs box size — {DISCLAIMER}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_csv(summaries: List[Dict], path: Path) -> None:
    """Save benchmark summary to CSV with disclaimer in header comment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant", "mean_final_iou", "std_final_iou",
        "mean_steps_to_half", "pct_reached_half",
        "total_nan_count", "mean_oscillation", "n_seeds",
    ]
    with open(path, "w", newline="") as f:
        # CSV header comment
        f.write(f"# {DISCLAIMER}\n")
        f.write("# Columns: variant, mean/std final IoU, steps to IoU>=0.5, "
                "% seeds that reached 0.5, NaN count, oscillation (std last 50 steps)\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s[k] for k in fieldnames})


def save_markdown(summaries: List[Dict], path: Path) -> None:
    """Save benchmark summary to markdown table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"> **{DISCLAIMER}**\n",
        "",
        "| Variant | Mean IoU ± std | Steps to IoU≥0.5 | % reached 0.5 | NaN count | Oscillation |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        steps = (f"{s['mean_steps_to_half']:.0f}"
                 if not (isinstance(s["mean_steps_to_half"], float)
                         and np.isnan(s["mean_steps_to_half"]))
                 else "never")
        lines.append(
            f"| {s['variant']} "
            f"| {s['mean_final_iou']:.4f} ± {s['std_final_iou']:.4f} "
            f"| {steps} "
            f"| {s['pct_reached_half']:.0f}% "
            f"| {s['total_nan_count']} "
            f"| {s['mean_oscillation']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic tiny-box regression benchmark.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_boxes", type=int, default=500,
                   help="GT boxes per seed (default 500)")
    p.add_argument("--steps", type=int, default=500,
                   help="Adam steps per run (default 500)")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out_dir", default="results/loss")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_val = int(cfg.get("seed", 42))
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {DISCLAIMER}")
    print(f"{'='*60}")
    print(f"  n_boxes={args.n_boxes}  steps={args.steps}  "
          f"lr={args.lr}  seeds={args.seeds}")
    print(f"{'='*60}\n")

    t0 = time.time()
    all_results: Dict[str, List[Dict]] = {v: [] for v in VARIANTS}

    total_runs = len(VARIANTS) * len(args.seeds)
    done = 0
    for variant in VARIANTS:
        for seed in args.seeds:
            done += 1
            print(f"[{done:>2}/{total_runs}] variant={variant:<14} seed={seed} ...",
                  end=" ", flush=True)
            t1 = time.time()
            result = run_one(
                variant=variant,
                seed=seed,
                n_boxes=args.n_boxes,
                n_steps=args.steps,
                lr=args.lr,
                cfg=cfg,
            )
            elapsed = time.time() - t1
            print(f"final_iou={result['final_iou']:.4f}  "
                  f"steps_to_0.5={result['steps_to_half']}  "
                  f"nan={result['nan_count']}  "
                  f"({elapsed:.1f}s)")
            all_results[variant].append(result)

    total_elapsed = time.time() - t0
    print(f"\nTotal time: {total_elapsed:.1f}s\n")

    # Aggregate
    summaries = [aggregate(all_results[v]) for v in VARIANTS]

    # Save outputs
    csv_path = out_dir / "benchmark.csv"
    md_path  = out_dir / "benchmark.md"
    conv_path = out_dir / "convergence.png"
    grad_path = out_dir / "grad_vs_size.png"

    save_csv(summaries, csv_path)
    save_markdown(summaries, md_path)
    plot_convergence(all_results, conv_path, args.steps)
    plot_grad_vs_size(all_results, grad_path)

    print(f"Saved:\n  {csv_path}\n  {md_path}\n  {conv_path}\n  {grad_path}\n")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY  —  {DISCLAIMER}")
    print(f"{'='*60}")
    print(f"{'Variant':<16} {'Mean IoU':>10} {'±std':>8} {'Steps→0.5':>11} "
          f"{'% hit 0.5':>10} {'NaN':>5}")
    print("-" * 65)
    for s in summaries:
        steps_str = (
            f"{s['mean_steps_to_half']:>8.0f}"
            if not (isinstance(s["mean_steps_to_half"], float)
                    and np.isnan(s["mean_steps_to_half"]))
            else "   never"
        )
        print(
            f"{s['variant']:<16} "
            f"{s['mean_final_iou']:>10.4f} "
            f"{s['std_final_iou']:>8.4f} "
            f"{steps_str:>11} "
            f"{s['pct_reached_half']:>9.0f}% "
            f"{s['total_nan_count']:>5}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
