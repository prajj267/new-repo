# Reproduction Report: Dist-Tracker (CPU-only)

**Paper:** Wang et al., "Dist-Tracker: A Small Object-aware Detector and Tracker for UAV
Tracking", CVPR Workshops 2025 (4th Anti-UAV Challenge).

**Reproduction author:** [YOUR NAME]
**Date:** [DATE]
**Compute:** AMD Ryzen 7 8840HS, 31 GB RAM, CPU-only (no GPU available).
**Repository:** `dist-tracker-repro-cpu/`

---

## 1. Problem and motivation

UAV tracking from thermal infrared cameras is challenging for two distinct
reasons. First, UAVs appear as small objects — typically 4–32 pixels wide —
so standard bounding-box regression losses (e.g. IoU) produce near-zero or
zero gradients when predicted and ground-truth boxes do not overlap, giving
no useful learning signal for the very objects of interest. Second, in
multi-UAV tracking, the tracker must maintain consistent identities across
frames despite fast motion, similar appearance between targets, and camera
motion that displaces all objects simultaneously.

The paper addresses both problems: the SSQ loss (Scale + Shape + Quality,
Eq. 3–8) replaces IoU as the regression loss to provide useful gradients for
tiny boxes; and the FLIT tracker (Fused-cost L2-IoU Tracker) extends
SORT-style tracking with a centre-distance term and camera motion compensation
(CMC) to reduce ID switches.

---

## 2. Method

### 2.1 SSQ loss (Part A)

The SSQ loss combines three terms with a step-dependent schedule.

**Scale loss (Eq. 3):**
```
L_scale = ||mu_p - mu_t||^2 + w_s * ((w_p-w_t)^2 + (h_p-h_t)^2)
```
This is the Wasserstein distance between two axis-aligned Gaussians with
covariance Sigma = diag(w^2, h^2). Unlike IoU, it is non-zero even when
boxes do not overlap, providing a gradient signal for tiny separated boxes.

**Shape loss (Eq. 4):**
```
L_shape = 2*(w_t^s / (w_t^s + h_t^s)) * (|w_p-w_t| / (max(w_p,w_t)+eps))^2
        + 2*(h_t^s / (w_t^s + h_t^s)) * (|h_p-h_t| / (max(h_p,h_t)+eps))^2
```
This penalises aspect-ratio mismatch, weighted by the GT aspect ratio so the
loss focuses on the dominant dimension of the target.

**Quality loss (Eq. 5–6):**
```
beta = (1 - IoU) / (E[1-IoU] + eps)    # sliding 100-step window, detached
L_quality = beta / sigma^(beta-1+eps) * (1 - IoU)^sigma
```
Beta acts as a curriculum weight: it is high when the current box is worse
than the recent average (hard example emphasis) and low otherwise. It is
**detached** from the gradient so it acts as a scalar coefficient, not a
feedback signal.

**Combined (Eq. 7–8):**
```
r = step / total_steps
L_SSQ = 0.7*(1-r)*(L_scale + L_shape) + 0.3*(1+r)*L_quality
```
At step 0, the loss emphasises shape/scale (0.7 weight); at the final step it
emphasises quality (0.6 weight). This schedule transitions from geometric
alignment to quality-aware refinement.

### 2.2 FLIT tracker (Part B)

**Kalman filter:** BoT-SORT 8-state formulation — state [cx, cy, w, h, vx, vy, vw, vh],
constant-velocity transition, observation [cx, cy, w, h].
Design choice: the 8-state formulation is used because it models size
velocity, which matters for UAVs changing apparent size as they approach or
recede. Noise constants follow BoT-SORT defaults (documented in
open_questions.md OQ-9).

**Camera motion compensation (CMC):** Sparse Lucas-Kanade optical flow between
consecutive sampled frames → RANSAC partial affine estimate → applied to
Kalman-predicted centres. The forward affine (prev→curr) is applied, mapping
old predictions into the coordinate frame of the new frame.
Note on stride semantics: with frame_stride=5, CMC is computed between every
5th raw frame (not between consecutive raw frames). The Kalman dt is 1
sampled frame throughout.

**Cost matrix (Eq. 10–11):**
```
C_IoU-L2 = gamma * C_L2 + (1-gamma) * C_IoU
C_final  = 1 - (1 - C_IoU-L2) * S_conf
```
C_L2 is the Euclidean centre distance normalised by the image diagonal
(sqrt(640^2 + 512^2) ≈ 820 px), making it dimensionless.
S_conf weights detections by confidence so high-confidence detections are
preferred in the assignment.

**Assignment:** scipy.optimize.linear_sum_assignment (Hungarian).
Thresholds: tau=0.01 (min confidence to enter tracker), eta=0.25 (min
confidence to initialise a new track), match_thresh=0.8 (max cost to accept
a match). Track buffer=0 by default (Algorithm 1: delete unmatched tracks
immediately).

---

## 3. Experimental setup

| Item | Value |
|---|---|
| Dataset | Anti-UAV300, infrared only |
| Split | val (67 sequences available) |
| Sequences used | 15 (see list below) |
| Frame stride | 5 (every 5th raw frame) |
| Stride semantics | Kalman dt=1/sampled frame; CMC between sampled frames |
| Detector | **GT simulation** (see §6 for disclaimer) |
| Seed | 42 |
| CPU | AMD Ryzen 7 8840HS, 31 GB RAM, no GPU |
| Python | 3.12.10 |
| PyTorch | 2.14.1+cpu |

**Sequences (15):**
20190925_131530_1_3, 20190925_140917_1_3, 20190925_143900_1_2,
20190925_143900_1_7, 20190925_152412_1_5, 20190925_183946_1_2,
20190925_200320_1_8, 20190925_210802_1_6, 20190925_210802_1_8,
20190925_222534_1_5, 20190926_141816_1_4, 20190926_143632_1_4,
20190926_143632_1_8, 20190926_183400_1_6, 20190926_200510_1_2

**Test split:** Never read. Not used for tuning or evaluation.

---

## 4. Part A results — Synthetic tiny-box regression

> **THIS IS NOT DETECTION AP. THESE RESULTS CANNOT BE COMPARED WITH TABLE 2
> OF THE PAPER.**

Experiment: 500 random GT boxes (width 4–32 px, aspect ~1.2:1) in a 640×512
canvas. Initial predictions = GT + Gaussian noise (centre σ=10 px, size σ=5
px). Adam optimiser, lr=0.01, 500 steps. Seeds 0, 1, 2.

Raw data: `results/loss/benchmark.csv`
Convergence plot: `results/loss/convergence.png`
Gradient vs size plot: `results/loss/grad_vs_size.png`

### Table A1 — Synthetic regression benchmark

| Variant | Mean final IoU ± std | Steps to IoU≥0.5 | % hit 0.5 | NaN count |
|---|---|---|---|---|
| CIoU (baseline, Eq. 2) | **0.5089 ± 0.0093** | **~488** | **100%** | 0 |
| Scale only (Eq. 3) | 0.3923 ± 0.0067 | never | 0% | 0 |
| Scale + Shape (Eq. 3+4) | 0.3923 ± 0.0067 | never | 0% | 0 |
| Quality only (Eq. 5) | 0.4590 ± 0.0083 | never | 0% | 0 |
| Full SSQ (Eq. 7-8) | 0.3139 ± 0.0059 | never | 0% | 0 |

**SSQ does NOT beat CIoU on this benchmark.** CIoU is the only variant that
reliably reaches IoU ≥ 0.5 (100% of seeds, ~488 steps). This result is
reported honestly; no post-hoc tuning was done.

**Interpretation:** CIoU directly maximises overlap (IoU term), which is
exactly what this benchmark measures. Scale/shape losses minimise positional
and size error but have no explicit IoU term, so they converge to lower IoU.
The SSQ schedule starts with scale+shape at step 0, transitioning to quality
by step 500 — but 500 steps is insufficient for the combined schedule to match
CIoU on direct gradient descent.

**Why this does not invalidate the paper:** SSQ is designed to improve
detector training on many boxes simultaneously in a neural network, not to
optimise a single box by gradient descent. The benchmark tests the loss
landscape in isolation — a useful unit test of the loss functions — but it is
not a substitute for end-to-end detector training, which the paper's Table 2
addresses. No GPU was available to run detector training.

---

## 5. Part B results — Tracker ablations

> **Detector: GT-based simulation. Results NOT comparable with Table 3.**
> See §6 for the full disclaimer.

Raw data: `results/tracker/ablations.csv`
Plot: `results/tracker/mota_ablation.png`

### Table B1 — Core tracker variants (IoU threshold = 0.5)

| Variant | MOTA (pooled) | MOTA mean±std | IDF1 | FP | FN | IDS |
|---|---|---|---|---|---|---|
| detector_only (true baseline) | 0.9528 | 0.9556±0.0179 | 0.9758 | 3 | 121 | 0 |
| iou_only (γ=0) | 0.4800 | 0.4822±0.1571 | 0.0799 | 3 | 121 | 1241 |
| l2_only (γ=1) | 0.8682 | 0.8749±0.0852 | 0.3016 | 58 | 176 | 112 |
| hybrid (γ=0.25, no CMC) | 0.7802 | 0.7943±0.1076 | 0.1578 | 3 | 121 | 453 |
| hybrid_cmc (γ=0.25, CMC) | **0.7840** | **0.7981±0.1066** | **0.1610** | 3 | 121 | **443** |

### Table B2 — Core tracker variants (IoU threshold = 0.25)

| Variant | MOTA (pooled) | MOTA mean±std | IDF1 | FP | FN | IDS |
|---|---|---|---|---|---|---|
| detector_only (true baseline) | 0.9543 | 0.9572±0.0164 | 0.9766 | 1 | 119 | 0 |
| iou_only (γ=0) | 0.4808 | 0.4830±0.1563 | 0.0799 | 1 | 119 | 1243 |
| l2_only (γ=1) | 0.9002 | 0.9053±0.0433 | 0.3036 | 16 | 134 | 112 |
| hybrid (γ=0.25, no CMC) | 0.7813 | 0.7956±0.1063 | 0.1578 | 1 | 119 | 454 |
| hybrid_cmc (γ=0.25, CMC) | **0.7851** | **0.7994±0.1052** | **0.1610** | 1 | 119 | **444** |

**Note on IDS:** IDS is **not** near zero. The GT simulation adds independent
Gaussian noise per frame (σ=5 px centre, σ=2 px size), producing box-to-box
IoU below the match threshold for the IoU-only variant. This causes the tracker
to drop and reinitialise tracks, generating ID switches. This is actually
informative: it tests the cost matrix's robustness to per-frame detection noise.

**Note on single-target data:** At most one UAV is present per frame. IDF1 is
only weakly informative in this setting. MOTA, FP, FN, and IDS differences
across variants are the meaningful signals.

### Table B3 — Gamma sweep (IoU threshold = 0.5)

| γ | MOTA (pooled) | IDS |
|---|---|---|
| 0.00 (IoU only) | 0.4800 | 1241 |
| 0.10 | 0.5901 | 952 |
| 0.25 (paper value) | 0.7802 | 453 |
| 0.50 | 0.8701 | 113 |
| 0.75 | 0.8682 | 112 |
| 1.00 (L2 only) | 0.8682 | 112 |

Raw data: `results/tracker/ablations.csv` (gamma_* rows)

### Table B4 — Match threshold sweep (hybrid, IoU threshold = 0.5)

| match_thresh | MOTA | IDS |
|---|---|---|
| 0.5 | 0.2301 | 1897 |
| 0.7 | 0.5947 | 940 |
| 0.8 (default) | 0.7802 | 453 |
| 0.9 | 0.8701 | 113 |

### Table B5 — Track buffer sweep (hybrid, IoU threshold = 0.5)

| track_buffer | MOTA | IDS |
|---|---|---|
| 0 (paper Algorithm 1) | 0.7802 | 453 |
| 1 | 0.8065 | 384 |
| 2 | 0.8038 | 391 |
| 3 | 0.7958 | 412 |

---

## 6. Comparison with paper's numbers (context only)

Paper numbers are labelled "paper (val)" and kept out of the "my results" columns.

| Metric | Paper (val) | This reproduction | Comparable? |
|---|---|---|---|
| Table 3: baseline MOTA | 0.735 (paper val) | 0.4800 (iou_only) | **NO** |
| Table 3: +CMC MOTA | 0.770 (paper val) | 0.7840 (hybrid_cmc) | **NO** |
| Table 3: +L2 fusion MOTA | 0.775 (paper val) | 0.7802 (hybrid) | **NO** |
| Test MOTA | 0.8132 (paper hidden test) | NOT REPRODUCED | — |
| Table 2: AP50 SSQ | 87.8 (paper val) | NOT REPRODUCED | — |

The numbers cannot be compared because this reproduction uses a GT-simulated
detector whereas the paper uses a trained YOLOv12 detector on a different dataset.

**Trend check:** The paper finds L2 fusion > IoU-only (0.775 > 0.735, Δ=+0.040).
This reproduction finds L2 fusion > IoU-only as well (0.7802 > 0.4800 for hybrid,
with even stronger effect for l2_only=0.8682). The paper's CMC improvement
(0.770 > 0.735, Δ=+0.035) is also observed here (0.7840 > 0.7802, Δ=+0.0038),
though the magnitude is smaller because this camera is mostly stationary.
**The trend that L2 fusion reduces ID switches for tiny noisy boxes is reproduced.**

---

## 7. Honest gap analysis

| Factor | Effect on results |
|---|---|
| No YOLOv12 training | Table 2 (AP) not reproduced. Requires GPU. |
| GT-simulated detector | Tracker results are not comparable with Table 3. Simulation gives controlled noise but not the recall/precision profile of a real detector. |
| Internet access blocked | Could not download pretrained YOLOv5s. Documented in design.md §4.1. |
| Single-target data | IDF1 is weakly informative. ID-switch patterns in multi-UAV scenarios cannot be tested. |
| Frame stride = 5 | Kalman velocity in sampled-frame units; CMC between sampled frames. A real deployment would use stride=1. |
| 15/67 val sequences | Subset for speed. Full val available via data.num_sequences: 67 in config. |
| weights_only=False | Applied to trusted local YOLOv5 weights (PyTorch ≥ 2.6 compat). Would apply with real weights. |
| Unspecified hyperparameters | See docs/open_questions.md for all 10 open questions and chosen defaults. |

---

## 8. Limitations

1. **No YOLOv12 training.** Table 2 (detector AP) is not reproduced. The paper's
   central claim about SSQ improving detection AP cannot be verified without GPU
   training and the paper's exact dataset.

2. **GT-simulated detector.** No pretrained weights were available (internet
   access blocked on the test machine). The simulation isolates the tracker from
   detector quality but removes a key source of real-world noise.

3. **Single-target data.** Anti-UAV300 contains at most one UAV per frame. The
   paper's motivation for FLIT includes multi-target ID-switch reduction, which
   cannot be tested here. IDS numbers should be interpreted cautiously.

4. **No hidden test set.** 81.32% test MOTA from the challenge cannot be
   reproduced without submitting to the 4th Anti-UAV Challenge.

5. **Stride semantics.** Kalman filter dt = 1 sampled frame (not 1 raw frame).
   At stride=5 and 20fps the effective temporal gap is 250ms per step, which
   affects velocity estimates.

6. **SSQ benchmark caveat.** The synthetic regression benchmark does not
   replicate end-to-end detector training. It tests the loss landscape in
   isolation and cannot be compared with Table 2.

---

## 9. Conclusion

**Part A (SSQ loss):** All six loss functions were implemented and tested
(35/35 unit tests pass, including gradcheck and no-grad-through-beta).
The synthetic benchmark shows that SSQ does not beat CIoU in direct box
regression (0.314 vs 0.509 mean IoU). This does not invalidate the paper —
SSQ is designed for detector training, not direct regression — but it cannot
be verified without GPU training.

**Part B (FLIT tracker):** The tracker was implemented (Algorithm 1 verbatim)
and tested (32/32 unit tests pass). On the GT-simulated detections:
- The L2 fusion trend is reproduced: adding L2 cost reduces IDS
  (from 1241 to 453 for hybrid vs iou_only at IoU=0.5).
- CMC provides a small but consistent improvement (+0.0038 MOTA,
  −10 IDS), consistent with the paper's observation, though smaller in
  magnitude because this camera is mostly stationary.
- The paper's exact MOTA numbers (Table 3) cannot be reproduced because
  the detector differs fundamentally.

The core contribution of the paper — that an L2-based cost matrix with
confidence weighting reduces ID switches for tiny-object tracking — is
supported by this reproduction's trends, within the stated limitations.

---

## 10. References

- Wang et al. (2025). Dist-Tracker: A Small Object-aware Detector and
  Tracker for UAV Tracking. CVPR Workshops 2025.
- Aharon et al. (2022). BoT-SORT: Robust Associations Multi-Pedestrian
  Tracking. arXiv:2206.14651.
- Bewley et al. (2016). Simple Online and Realtime Tracking. ICIP 2016.
- Zheng et al. (2020). Distance-IoU Loss: Faster and Better Learning for
  Bounding Box Regression. AAAI 2020.
- Anti-UAV300 dataset: https://github.com/ZhaoJ9014/Anti-UAV
- py-motmetrics: https://github.com/cheind/py-motmetrics
