# Requirements

Paper: Wang et al., "Dist-Tracker: A Small Object-aware Detector and Tracker for UAV Tracking",
CVPR Workshops 2025.

---

## 1. Scope

### 1.1 In scope
- **Part A**: SSQ loss (Eq. 3–8) implemented in pure PyTorch (CPU), with unit tests and a
  synthetic tiny-box regression benchmark.
- **Part B**: FLIT tracker (Algorithm 1) with Kalman filter, camera motion compensation (CMC),
  fused cost matrix (Eq. 10–11), and Hungarian assignment. Tested on cached detections from
  the validation split.

### 1.2 Explicitly out of scope
- Training YOLOv12 (requires GPU, not available). The paper's Table 2 (detector AP) will NOT
  be reproduced. This must be stated plainly in the report and next to every benchmark table.
- The paper's hidden test-set score of 81.32% MOTA. Do not claim it.

---

## 2. Hard constraints

| Constraint | Detail |
|---|---|
| CPU only | No CUDA dependency anywhere. `torch.device("cpu")` throughout. |
| Runnable in minutes | Default config: ≤ 20 val sequences, frame_stride ≥ 5. Full val run optional. |
| No silent GPU fallback | If any step needs GPU, stop and report it. |
| Pinned dependencies | `requirements.txt` must pin every package to an exact version. |
| No test set | Never read dataset/test. Use only dataset/val. |

---

## 3. Data requirements

### 3.1 Dataset
- Anti-UAV300, infrared modality only. Ignore RGB (`visible.mp4`, `visible.json`).
- Layout confirmed from inspection:
  - `dataset/{train,val,test}/<seq_name>/infrared.mp4`
  - `dataset/{train,val,test}/<seq_name>/infrared.json`
- Annotation fields confirmed: `gt_rect` ([x, y, w, h] pixels, top-left origin), `exist` (0/1).
- Video: 640×512, 20 fps, ~1000 frames/sequence.
- Splits: train=160 seqs, val=67 seqs, test=91 seqs.

### 3.2 Split policy
- Use the **val** split for all experiments and tuning.
- The **test** split must never be touched or read.
- Split by sequence. No per-frame mixing across splits.

### 3.3 Subset config keys
| Key | Default | Meaning |
|---|---|---|
| `data.num_sequences` | 15 | How many val sequences to use |
| `data.frame_stride` | 5 | Use every Nth frame |
| `data.max_frames_per_sequence` | 200 | Cap per sequence after striding |
| `data.sequence_seed` | 42 | Seed for selecting the subset |

### 3.4 Stride semantics (Amendment C)
- With `frame_stride=5` the tracker sees every 5th raw frame as temporally consecutive.
- The Kalman filter dt is 1 per sampled frame (not per raw frame).
- CMC affines are computed between consecutive **sampled** frames, not raw neighbours.
- The CMC cache and detection cache both record `frame_stride`. At load time, if the
  cached stride does not match the config stride, raise an error and require a rebuild.

---

## 4. Part A — SSQ loss

### 4.1 Functions required (all in `src/losses/ssq.py`)

| Function | Equation | Notes |
|---|---|---|
| `scale_loss(p, t, cfg)` | Eq. 3 | Centre L2 + weighted size Frobenius term |
| `shape_loss(p, t, cfg)` | Eq. 4 | Aspect-ratio-weighted relative size mismatch |
| `quality_loss(p, t, beta, cfg)` | Eq. 5 | Beta-weighted IoU gap; sigma configurable |
| `compute_beta(iou_val, window)` | Eq. 6 | Sliding window mean; returns Python float |
| `ssq_loss(p, t, step, total_steps, window, cfg)` | Eq. 7–8 | Combines all three terms |
| `ciou_loss(p, t)` | Eq. 2 | Baseline comparison |

### 4.2 Configurable open questions (all in `configs/default.yaml`)

| Config key | Default | Open question |
|---|---|---|
| `loss.w_s` | 0.6 | Paper says "> 0.5" only |
| `loss.shape_scale` | 2.0 | Exponent in Eq. 4; not given |
| `loss.sigma` | 2.0 | Exponent in Eq. 5; not given |
| `loss.use_normalised_coords` | false | Pixel vs normalised in Eq. 3 |
| `loss.use_nwd_exp` | false | Whether to wrap Eq. 3 in NWD exponential |
| `loss.size_cov_scale` | 1.0 | Factor on (w_p-w_t)^2+(h_p-h_t)^2; 0.25 gives NWD convention |
| `loss.eps` | 1.0e-7 | Division guard |

### 4.3 Unit tests required (Amendment D)

| Test name | What it checks |
|---|---|
| `test_scale_matches_closed_form` | L_scale equals hand-computed value for a known pair |
| `test_shape_zero_identical_boxes` | L_shape == 0 when p == t |
| `test_no_nan_tiny_boxes` | no NaN/inf for w=1, h=1 boxes |
| `test_beta_is_detached` | `compute_beta` returns a Python float (not a tensor) |
| `test_window_maxlen` | after 150 calls, window length is exactly 100 |
| `test_ssq_weights_at_step_zero` | at step=0: total == 0.7*(Ls+Lsh) + 0.3*Lq within tol |
| `test_ssq_weights_at_final_step` | at step=total_steps: weights are 0 on (scale+shape), 0.6 on quality |
| `test_gradcheck_scale` | `torch.autograd.gradcheck` on scale_loss in float64 |
| `test_gradcheck_shape` | `torch.autograd.gradcheck` on shape_loss in float64 |
| `test_quality_no_grad_through_beta` | beta has no gradient after quality_loss backward |

### 4.4 Synthetic benchmark requirements (Amendment E)
- Generate N=500 random GT boxes (width 4–32 px, aspect ratio ~1.2:1) in 640×512.
- Initial predictions: GT + Gaussian noise (centre σ=10 px, size σ=5 px).
- Optimise with Adam for 500 steps under: CIoU, scale-only, scale+shape, quality-only, full SSQ.
- Report per loss variant and per seed (seeds 0–4):
  - Mean final IoU ± std
  - Convergence curve (IoU vs step), saved as `results/loss/convergence.png`
  - Steps to first reach IoU ≥ 0.5 (or "never")
  - Mean gradient magnitude as a function of GT box size (binned), saved as `results/loss/grad_vs_size.png`
  - NaN/inf count and loss oscillation (std of last 50 steps)
- Results table saved as `results/loss/benchmark.csv` and `results/loss/benchmark.md`.
- **The CSV header, benchmark.md, and both plot titles must contain the phrase:
  "NOT detection AP; cannot be compared with Table 2".**
- If SSQ does not beat CIoU, report that plainly with the numbers. Do not tune until
  it wins. Any tuning must use seeds disjoint from the reported seeds and be logged.

---

## 5. Part B — FLIT tracker

### 5.1 Detector (detection cache) — Amendment B

**Fallback (current):** Official pretrained `yolov5s.pt` from ultralytics/yolov5 GitHub
releases. This is a deliberate fallback. The report must state:
> "Detections are produced by YOLOv5s pretrained on COCO, not the Anti-UAV baseline model.
> Tracking results are therefore not directly comparable with Table 3 of the paper."

**Primary (when available):** Anti-UAV baseline weights from `reference/Baseline_code_1.zip`.
The weights path is a config key (`detector.weights`); swapping requires only changing config
and rebuilding the cache.

**Offline mode:** After first successful `torch.hub.load`, the cloned repo is vendored under
`reference/yolov5/`. Subsequent loads use `source="local"` pointing to that directory.
This makes the pipeline work without internet access after the first run.

**Recall check (Amendment B):** After building the cache for 2–3 sequences, compute recall
at IoU 0.5 and 0.25 against GT. If recall < 5% at IoU 0.25, STOP and report the numbers.
Do not silently lower thresholds. Propose options (lower conf, larger imgsz, class-agnostic).

Detection cache:
- One JSON file per sequence: `cache/detections/<seq_name>.json`
- Cache records `frame_stride`, `detector_weights`, `torch_version` in its header.
- Rebuild script: `scripts/build_detection_cache.py`
- Every tracker experiment reads from cache; never calls the detector directly.
- At load time, cached `frame_stride` must match config; otherwise raise and require rebuild.

### 5.2 Kalman filter
- State: [cx, cy, w, h, vx, vy, vw, vh] (BoT-SORT 8-state formulation).
- All Q, R diagonal scales exposed as config keys.

### 5.3 CMC — Amendment C
- Sparse Lucas-Kanade → RANSAC partial affine between consecutive **sampled** frames.
- Cached to `cache/affines/<seq_name>.json`, including `frame_stride` in the header.
- Stale cache (stride mismatch) raises an error.

### 5.4 Cost matrix (Eq. 10–11)
- `C_IoU = 1 - IoU`
- `C_L2`: normalisation options: `"diagonal"` (÷ sqrt(W²+H²)) or `"fixed"` (÷ 512)
- `C_IoU_L2 = gamma * C_L2 + (1-gamma) * C_IoU`, gamma=0.25
- `C_final = 1 - (1 - C_IoU_L2) * S_conf`
- Frames with `exist=0` have no GT box. Any tracker output on those frames counts as FP.
  This must be stated in `evaluate.py` docstring.

### 5.5 Assignment
- τ = 0.01 (config), η = 0.25 (config), match_thresh configurable (default 0.8)
- Default: track_buffer=0 (delete unmatched tracks immediately, Algorithm 1)

### 5.6 Tracker experiments — Amendment F
All on cached detections, same sequence list recorded in results:

| Variant | gamma | CMC |
|---|---|---|
| detector-only (no tracker) | — | — |
| IoU only | 0 | off |
| L2 only | 1 | off |
| hybrid | 0.25 | off |
| hybrid + CMC (full FLIT) | 0.25 | on |
| gamma sweep | {0,0.1,0.25,0.5,0.75,1.0} | off |
| L2-norm: diagonal vs fixed | 0.25 | off |
| match_thresh sweep | {0.5,0.7,0.8,0.9} | off |
| track_buffer sweep | {0,1,2,3} | off |

- Report mean ± std across sequences (not only pooled), at IoU 0.5 AND 0.25.
- IDS is expected ≈ 0 for all variants (single-target data). State this in the report.
  MOTA/FP/FN differences and IDF1 are the main signal.
- Do not claim CMC helps unless numbers show it. A null or negative result is acceptable.
- Unit test (Amendment F): with CMC enabled and a known synthetic whole-image translation,
  the CMC-corrected prediction must land closer to the true next box than uncorrected.

### 5.7 Metrics
- MOTA, IDF1, FP, FN, IDS via `motmetrics`
- Reported at IoU threshold **0.5 AND 0.25** for every variant
- Results table: `results/tracker/ablations.csv` and `results/tracker/ablations.md`

---

## 6. Deliverables checklist

- [ ] `src/` — all source modules
- [ ] `configs/default.yaml` — all hyperparameters as config keys
- [ ] `scripts/` — `build_detection_cache.py`, `build_cmc_cache.py`,
      `run_loss_benchmark.py`, `run_tracker.py`
- [ ] `tests/` — unit tests covering every requirement in §4 and §5
- [ ] `docs/open_questions.md` — all ambiguities with chosen defaults
- [ ] `docs/report.md` — completed report including all limitations from design §8
- [ ] `results/loss/` — benchmark CSV, plots
- [ ] `results/tracker/` — ablation CSV, plots
- [ ] `README.md` — exact commands to reproduce every table and plot
- [ ] `requirements.txt` — pinned versions, CPU PyTorch only, all transitive deps
- [ ] `results/env_freeze.txt` — output of `pip freeze` at Stage 0
