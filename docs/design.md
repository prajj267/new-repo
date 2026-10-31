# Design

Paper: Wang et al., "Dist-Tracker: A Small Object-aware Detector and Tracker for UAV Tracking",
CVPR Workshops 2025.

---

## 0. Repo layout

```
dist-tracker-repro-cpu/
  configs/
    default.yaml          ← single source of truth for all hyperparameters
  docs/
    design.md             ← this file
    requirements.md
    open_questions.md
    report.md
  src/
    data/
      loader.py           ← AntiUAVSequence, AntiUAVDataset
      visualize.py        ← draw boxes on sample frames
    losses/
      ssq.py              ← scale_loss, shape_loss, quality_loss, compute_beta, ssq_loss, ciou_loss
      baselines.py        ← nwd_loss, shape_iou_loss, wise_iou_loss  (optional)
    detector/
      yolov5_wrapper.py   ← thin wrapper; reads weights path from config
    tracker/
      kalman.py           ← KalmanTrack (BoT-SORT 8-state)
      cmc.py              ← CMC (sparse LK → RANSAC affine)
      cost.py             ← iou_cost, l2_cost, fused_cost, final_cost
      flit.py             ← FLITTracker (Algorithm 1)
    evaluate.py           ← MOTA/IDF1 via motmetrics
  scripts/
    build_detection_cache.py
    build_cmc_cache.py
    run_loss_benchmark.py
    run_tracker.py
  tests/
    test_ssq_loss.py
    test_tracker.py
    test_loader.py
  cache/
    detections/           ← <seq_name>.json  (includes frame_stride, detector_weights, torch_version)
    affines/              ← <seq_name>.json  (includes frame_stride)
  results/
    env_freeze.txt        ← pip freeze output from Stage 0
    loss/
    tracker/
    viz/
  paper/
    dist_tracker.pdf
  reference/
    yolov5/               ← vendored after first torch.hub.load (Amendment B)
    Baseline_code_1.zip   ← place here when available
  requirements.txt
  README.md
```

---

## 1. Data layer (`src/data/loader.py`)

### 1.1 Confirmed dataset layout
- Path: `dataset/{train,val,test}/<seq_name>/infrared.{json,mp4}`
- Annotation keys confirmed by inspection: `gt_rect` ([x,y,w,h] pixels, top-left), `exist` (0/1).
- Video: 640×512, 20 fps, ~1000 frames/sequence.
- Val: 67 sequences. Test: never read.

### 1.2 `AntiUAVSequence`
On construction: loads `infrared.json`, validates keys, builds a sampled index list
respecting `frame_stride` and `max_frames_per_sequence`. Frame images decoded **lazily**
with OpenCV on first access — no full extraction to disk.

### 1.3 Stride semantics (Amendment C)
With `frame_stride=5`, indices `[0, 5, 10, ...]` are selected. The tracker treats these
as consecutive frames (dt=1 per sampled frame). CMC affines are computed between
consecutive sampled frames (frame 0→5, 5→10, etc.), not between raw consecutive frames.
This is documented here and stated in the report.

### 1.4 `AntiUAVDataset`
Selects `num_sequences` from val, sorted → seeded-shuffle → truncate.
`verify_no_test_in_val()` asserts no sequence name overlap.

---

## 2. Part A — SSQ loss (`src/losses/ssq.py`)

All box tensors: shape `(..., 4)`, format `[cx, cy, w, h]`, pixel units (default).
Utility `xywh_tl_to_cxcywh` converts loader output.

### 2.1 `ciou_loss(p, t)` — Eq. 2 (baseline)
Standard CIoU. Reference: Zheng et al. 2020.

### 2.2 `scale_loss(p, t, cfg)` — Eq. 3

```
L_scale = (cx_p-cx_t)^2 + (cy_p-cy_t)^2
        + w_s * size_cov_scale * ((w_p-w_t)^2 + (h_p-h_t)^2)
```

Derivation: Eq. 3 is `||mu_p-mu_t||^2 + w_s * ||Sigma_p^(1/2)-Sigma_t^(1/2)||_F^2`
with `Sigma=diag(w^2,h^2)`, so `Sigma^(1/2)=diag(w,h)` and the Frobenius norm gives
`(w_p-w_t)^2+(h_p-h_t)^2`.

**Open question OQ-1** (Amendment D note): The NWD convention uses `Sigma=diag(w^2/4,h^2/4)`
which gives a factor of 1/4 on the size term. This implementation keeps factor=1 as the
default (matching the literal Eq. 3) and exposes it as `loss.size_cov_scale` (default 1.0,
set to 0.25 to match NWD). See `docs/open_questions.md`.

Optional wrappers (config keys):
- `use_normalised_coords`: divide cx,cy,w,h by 640,512 before computing.
- `use_nwd_exp`: return `1 - exp(-L_scale / C)` where C is `loss.nwd_exp_C` (default 25.0).

### 2.3 `shape_loss(p, t, cfg)` — Eq. 4

```
denom = w_t^scale + h_t^scale + eps
w_wt  = 2 * w_t^scale / denom
h_wt  = 2 * h_t^scale / denom
L_shape = w_wt * (|w_p-w_t| / (max(w_p,w_t)+eps))^2
        + h_wt * (|h_p-h_t| / (max(h_p,h_t)+eps))^2
```

**Open question OQ-3**: `scale` exponent — default 2.0.

### 2.4 `compute_beta(iou_val, window)` — Eq. 6

```python
# iou_val: Python float (already detached)
# window: collections.deque(maxlen=100)
window.append(iou_val)
mean_iou = sum(window) / len(window)
beta = (1 - iou_val) / (mean_iou + eps)   # Python float, no gradient
return beta
```

Beta is a Python float. It is never a tensor. The caller must pass `.item()` values.

### 2.5 `quality_loss(p, t, beta, cfg)` — Eq. 5

```
iou = box_iou(p, t)          # differentiable w.r.t. p
L_quality = beta / (sigma^(beta - 1 + eps)) * (1 - iou)^sigma
```

`beta` is a Python float (no gradient path through it).
Test: `quality_loss.backward()` then `assert beta.grad is None` — beta has no grad
because it is not a tensor.

**Open question OQ-5**: `sigma` — default 2.0.

### 2.6 `ssq_loss(p, t, step, total_steps, window, cfg)` — Eq. 7–8

```
r = step / total_steps
L_SSQ = 0.7*(1-r)*(scale_loss + shape_loss) + 0.3*(1+r)*quality_loss
```

Coefficients 0.7/0.3 and the `1±r` schedule are fixed by the paper. Not config keys.
At step=0: weights are 0.7 on (scale+shape), 0.3 on quality.
At step=total_steps: r=1, so 0.7*(0)=0 on (scale+shape), 0.3*(2)=0.6 on quality.
The function calls `compute_beta` internally and updates `window`.

---

## 3. Synthetic benchmark (`scripts/run_loss_benchmark.py`)

### 3.1 Box generation
```python
gt_w  = Uniform(4, 32);  gt_h = gt_w / Uniform(1.0, 1.4)
gt_cx = Uniform(gt_w/2, 640-gt_w/2);  gt_cy = Uniform(gt_h/2, 512-gt_h/2)
```
N=500 boxes, seeds 0–4. Same seeds for all loss variants.

### 3.2 Optimiser: Adam, lr=0.01, 500 steps. Identical for all variants.

### 3.3 Output files
- `results/loss/benchmark.csv` — header comment: "NOT detection AP; cannot be compared with Table 2"
- `results/loss/benchmark.md` — same note in text
- `results/loss/convergence.png` — title includes "NOT detection AP; cannot be compared with Table 2"
- `results/loss/grad_vs_size.png` — same title note

If SSQ does not beat CIoU, this is reported plainly. No post-hoc tuning on reported seeds.

---

## 4. Detector wrapper (`src/detector/yolov5_wrapper.py`) — Amendment B

### 4.1 Fallback detector
No `Baseline_code_1.zip` available. Using official `yolov5s.pt` (COCO-pretrained).
**Report disclaimer required:**
> "Detections are produced by YOLOv5s pretrained on COCO, not the Anti-UAV baseline model.
> Tracking results are therefore not directly comparable with Table 3 of the paper."

### 4.2 Offline / vendored mode
On first call, `torch.hub.load("ultralytics/yolov5", ...)` clones the repo to
`~/.cache/torch/hub/ultralytics_yolov5_master/`. After success, we copy it to
`reference/yolov5/`. Subsequent loads use:
```python
torch.hub.load("reference/yolov5", "custom",
               path=weights_path, source="local", force_reload=False)
```
This makes the pipeline fully offline after the first run.

### 4.3 PyTorch compatibility fix
```python
# trusted local weights — weights_only=False is safe here
model = torch.load(weights_path, map_location="cpu", weights_only=False)
```
This is applied inside the hub loading path by monkey-patching `torch.load` for the
duration of the hub call, then restoring the original. Documented here and in the report.

### 4.4 Recall gate (Amendment B)
After building cache for the first 3 sequences, `build_detection_cache.py` computes
recall at IoU 0.5 and 0.25 against GT and prints:
```
Recall@0.50: X.XXX  Recall@0.25: X.XXX  (N GT boxes, M detected)
```
If recall@0.25 < 0.05 (5%), the script prints the numbers and stops with a non-zero
exit code. It does not silently lower thresholds. The user must decide:
- Lower `detector.conf_thresh` (config key, default 0.25)
- Increase `detector.imgsz` (config key, default 640)
- Enable `detector.class_agnostic` (config key, default false)

### 4.5 Detection cache schema
```json
{
  "seq_name": "20190925_101846_1_4",
  "frame_stride": 5,
  "detector_weights": "yolov5s.pt",
  "torch_version": "2.2.2+cpu",
  "frames": [
    {"frame_idx": 0, "boxes": [[x1,y1,x2,y2]], "scores": [0.82]},
    ...
  ]
}
```

---

## 5. FLIT tracker

### 5.1 Kalman filter (`src/tracker/kalman.py`)

**State**: `[cx, cy, w, h, vx, vy, vw, vh]` (BoT-SORT, 8D).
**Observation**: `[cx, cy, w, h]` (4D).

**Why BoT-SORT**: paper references it; the 8-state formulation handles w/h velocity
which matters for UAVs that change apparent size as they move.

F = I_8 with F[0,4]=F[1,5]=F[2,6]=F[3,7]=1 (constant velocity).
H = [I_4 | 0_{4×4}].

**Q** (process noise): diagonal scales `[1,1,1,1,0.01,0.01,0.0001,0.0001]` × `kalman.Q_scale`.
**R** (measurement noise): diagonal scales `[1,1,10,10]` × `kalman.R_scale`.
**P_init**: diagonal `[2w, 2h, 2w, 2h, 25w², 25h², 25w², 25h²]` (BoT-SORT heuristic).

All scales are config keys.

### 5.2 CMC (`src/tracker/cmc.py`) — Amendment C

```
prev_sampled_frame → goodFeaturesToTrack → LK flow → estimateAffinePartial2D(RANSAC) → M(2×3)
```

Applied to predicted box centre only (w, h unchanged — affine is translation+rotation):
```python
new_cx, new_cy = M[:2,:2] @ [cx,cy] + M[:,2]
```

Cache records `frame_stride`; mismatch at load time raises `ValueError`.
Identity stored for frames where estimation fails.

**Unit test** (Amendment F): construct two frames where the second is the first translated
by (dx, dy). Run CMC. Apply the affine to a predicted centre that was NOT translated.
Assert that the corrected centre is within 2 px of the true translated centre, and that
the uncorrected centre is > |dx| px away. This tests warp direction — applying the
inverse transform is a common bug.

### 5.3 Cost matrix (`src/tracker/cost.py`)

```python
iou_cost(T_xyxy, D_xyxy)        → (T,D) ndarray, values in [0,1]
l2_cost(T_xyxy, D_xyxy, norm)   → (T,D) ndarray
fused_cost(T, D, gamma, norm)   → gamma*C_L2 + (1-gamma)*C_IoU   # Eq.10
final_cost(C_fused, scores)     → 1 - (1-C_fused)*scores[None,:] # Eq.11
```

`norm` config options: `"diagonal"` (÷√(640²+512²) ≈ 820) or `"fixed"` (÷512).

**Frames with exist=0**: GT has no box. Any hypothesis on those frames is FP.
This is stated in `evaluate.py` docstring and in the report.

### 5.4 `FLITTracker` (`src/tracker/flit.py`)

Algorithm 1 verbatim:
1. Filter detections by τ
2. Kalman predict
3. If CMC: apply cached affine to predicted centres
4. Build cost matrix (Eq. 10–11)
5. Hungarian assignment (scipy)
6. Kalman update matched tracks
7. Delete unmatched tracks where `time_since_update > track_buffer`
8. Init new tracks for unmatched detections above η
9. Return confirmed tracks (hits ≥ min_hits)

`init_mode`:
- `"detector"`: init from detector detections in frame 0.
- `"first_frame_gt"`: init from GT boxes in frame 0, then detector for subsequent frames.

### 5.5 Tracker experiments — Amendment F

Results include:
- Sequence list (which 15 sequences were used)
- Per-sequence MOTA, FP, FN, IDS at IoU 0.5 and 0.25
- Mean ± std across sequences
- "detector only" row as true baseline

IDS expected ≈ 0 for all variants (single-target data). MOTA/FP/FN and IDF1 are the
meaningful signals. Do not claim CMC helps unless numbers show it.

### 5.6 Evaluate (`src/evaluate.py`)

Wraps `motmetrics`. Docstring must state:
> "Frames where exist=0 have no GT box. Any tracker hypothesis on such a frame
> is counted as a false positive."

Returns dict with MOTA, IDF1, FP, FN, IDS. Called twice: iou_thresh=0.5 and 0.25.

---

## 6. Experiments (`scripts/run_tracker.py`)

Reads config, loads cached detections and affines, runs each variant, appends rows to
`results/tracker/ablations.csv`. Records the sequence list in the CSV header.

---

## 7. Dependency decisions — Amendment A

| Package | Installed version | Reason |
|---|---|---|
| `torch` | `2.14.1+cpu` | CPU-only; PyTorch CDN was unreachable so 2.2.2 could not be downloaded; 2.14.1 is already installed locally and is CPU-only (CUDA=False confirmed) |
| `torchvision` | `0.29.1+cpu` | Matches torch 2.14.1 |
| `opencv-python` | `4.10.0.84` | CMC optical flow |
| `scipy` | `1.17.1` | Hungarian assignment |
| `motmetrics` | `1.4.0` | MOTA, IDF1 |
| `numpy` | `1.26.4` | Compatible with torch 2.14.1 |
| `pandas` | `2.2.3` | Required by motmetrics |
| `matplotlib` | `3.10.9` | Plots |
| `seaborn` | `0.13.2` | Plots |
| `Pillow` | `10.4.0` | Image I/O |
| `PyYAML` | `6.0.1` | Config loading |
| `tqdm` | `4.70.1` | Progress bars |
| `requests` | `2.34.2` | torch.hub download |
| `pytest` | `8.3.4` | Unit tests |

Python version: 3.12.10 (confirmed at Stage 0).

**Note on torch version:** The spec pinned 2.2.2 (last before `weights_only=True` default).
torch 2.14.1 also defaults `weights_only=True`, so the explicit `weights_only=False` fix
in the detector wrapper is still required. Behaviour is identical to the spec intent.

**No Ultralytics package** — YOLOv5 loaded via `torch.hub.load`.

---

## 8. Known limitations (for the report) — Amendment G

1. **No YOLOv12 training**: Table 2 (detector AP) is not reproduced.
2. **YOLOv5s pretrained on COCO, not Anti-UAV**: IR thermal recall will be lower.
   This directly limits tracker MOTA comparability with Table 3.
3. **Single-target data**: IDS ≈ 0 for all variants. MOTA/FP/FN and IDF1 are the
   main signals. IDF1 is only weakly informative. State this explicitly.
4. **No hidden test set**: 81.32% test MOTA cannot be reproduced.
5. **Subset of val**: 15 sequences by default.
6. **Stride semantics**: Kalman dt=1 per sampled frame, not per raw frame (Amendment C).
7. **weights_only=False**: applied for trusted local YOLOv5 weights (PyTorch ≥ 2.6 compat).
