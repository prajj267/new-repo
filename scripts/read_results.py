"""Helper to print results for report writing."""
import csv

# ── Loss benchmark ──────────────────────────────────────────────────────────
print("=== LOSS BENCHMARK (seeds 0,1,2) ===")
with open("results/loss/benchmark.csv", encoding="utf-8") as f:
    lines = [l for l in f if not l.startswith("#")]
reader = csv.DictReader(lines)
for row in reader:
    iou = float(row["mean_final_iou"])
    std = float(row["std_final_iou"])
    steps = row["mean_steps_to_half"]
    pct = float(row["pct_reached_half"])
    print(f"  {row['variant']:<16} IoU={iou:.4f}+/-{std:.4f}  steps={steps}  hit50%={pct:.0f}%  nan={row['total_nan_count']}")

print()

# ── Tracker core variants ───────────────────────────────────────────────────
print("=== TRACKER CORE (IoU thr=0.5) ===")
with open("results/tracker/ablations.csv", encoding="utf-8") as f:
    lines = [l for l in f if not l.startswith("#")]
reader = csv.DictReader(lines)
core = ["detector_only", "iou_only", "l2_only", "hybrid", "hybrid_cmc"]
for row in reader:
    if row["variant"] in core and float(row["iou_thresh"]) == 0.5:
        print(f"  {row['variant']:<22} MOTA={float(row['mota']):.4f}  "
              f"mean={float(row['mota_mean']):.4f}+/-{float(row['mota_std']):.4f}  "
              f"IDF1={float(row['idf1']):.4f}  "
              f"FP={row['fp']}  FN={row['fn']}  IDS={row['ids']}")

print()
print("=== TRACKER CORE (IoU thr=0.25) ===")
with open("results/tracker/ablations.csv", encoding="utf-8") as f:
    lines = [l for l in f if not l.startswith("#")]
reader = csv.DictReader(lines)
for row in reader:
    if row["variant"] in core and float(row["iou_thresh"]) == 0.25:
        print(f"  {row['variant']:<22} MOTA={float(row['mota']):.4f}  "
              f"mean={float(row['mota_mean']):.4f}+/-{float(row['mota_std']):.4f}  "
              f"IDF1={float(row['idf1']):.4f}  "
              f"FP={row['fp']}  FN={row['fn']}  IDS={row['ids']}")
