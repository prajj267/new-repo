# Dist-Tracker CPU Reproduction

Partial reproduction of **Wang et al., "Dist-Tracker: A Small Object-aware Detector and
Tracker for UAV Tracking"**, CVPR Workshops 2025.

**Scope:** SSQ loss (Part A) + FLIT tracker with CMC (Part B). CPU-only.
**Out of scope:** YOLOv12 training (Table 2 AP). Detector is YOLOv5s/COCO (fallback).
See `docs/report.md` for full limitations.

---

## Setup

```
cd c:\Projects\ipa_paper\dist-tracker-repro-cpu
pip install -r requirements.txt
```

Verify:
```
python -c "import torch, cv2, scipy, motmetrics, yaml, tqdm, pandas, PIL, seaborn; print('OK')"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
Expected: `OK`, `2.2.2+cpu`, `False`.

---

## Reproduce every result

### Stage 1 — Visualise annotations (sanity check)
```
python src/data/visualize.py --config configs/default.yaml
# output: results/viz/*.jpg
```

### Stage 2 — Run loss unit tests
```
pytest tests/test_ssq_loss.py -v
# 35 passed
```

### Stage 3 — Synthetic tiny-box regression benchmark (Part A)
```
python scripts/run_loss_benchmark.py --config configs/default.yaml --seeds 0 1 2 3 4
# output: results/loss/benchmark.csv
#         results/loss/benchmark.md
#         results/loss/convergence.png
#         results/loss/grad_vs_size.png
# runtime: ~35s on CPU
```
> NOT detection AP. Cannot be compared with Table 2.

### Stage 4 — Build detection cache
```
python scripts/build_detection_cache.py --config configs/default.yaml
# output: cache/detections/<seq_name>.json  (one per sequence)
# runtime: <1s (GT simulation mode)
```

### Stage 5 — Build CMC cache, run tracker unit tests
```
python scripts/build_cmc_cache.py --config configs/default.yaml
# output: cache/affines/<seq_name>.json
# runtime: ~75s on CPU

pytest tests/test_tracker.py -v
# 32 passed
```

### Stage 6 — Tracker ablations (Part B)
```
python scripts/run_tracker.py --config configs/default.yaml
# output: results/tracker/ablations.csv
#         results/tracker/ablations.md
#         results/tracker/mota_ablation.png
# runtime: ~33s on CPU
```

### Run all unit tests
```
pytest tests/ -v
# 34 + 35 + 32 = 101 passed
```

---

## Config

All hyperparameters are in `configs/default.yaml`. Key settings:

| Key | Default | Effect |
|---|---|---|
| `data.num_sequences` | 15 | Sequences used (max 67 val) |
| `data.frame_stride` | 5 | Subsample every Nth frame |
| `detector.weights` | `yolov5s` | Swap to Anti-UAV baseline when available |
| `tracker.gamma` | 0.25 | IoU/L2 fusion weight |
| `cmc.enabled` | true | Camera motion compensation |

---

## Paper numbers (context only)

| Metric | Paper (val) |
|---|---|
| Table 3 baseline MOTA | 0.735 |
| Table 3 +CMC MOTA | 0.770 |
| Table 3 +L2 fusion MOTA | 0.775 |
| Test MOTA | 0.8132 (hidden test — not reproducible) |
| Table 2 AP50 SSQ | 87.8 (requires YOLOv12 training — not reproduced) |
