# Reproduction Report: Dist-Tracker (CPU-only)

**Paper:** Wang et al., "Dist-Tracker: A Small Object-aware Detector and Tracker for UAV
Tracking", CVPR Workshops 2025.

**Author of this reproduction:** [YOUR NAME]
**Date:** [DATE]
**Compute:** AMD Ryzen 7 8840HS, 31 GB RAM, CPU-only (no GPU available).

---

## 1. Problem and motivation

<!-- Write in your own words. Cover:
     - Why tracking UAVs is hard (small size, thermal IR, fast motion)
     - Why standard IoU-based losses fail for tiny objects
     - What the paper contributes: SSQ loss + FLIT tracker with CMC
-->

*[TODO]*

---

## 2. Method

### 2.1 SSQ loss (Part A)

*(In your own words, referencing the equations.)*

**Scale loss (Eq. 3):** ...

**Shape loss (Eq. 4):** ...

**Quality loss (Eq. 5–6):** ...

**Combined SSQ (Eq. 7–8):** ...

### 2.2 FLIT tracker (Part B)

**Kalman filter:** BoT-SORT 8-state formulation [cx,cy,w,h,vx,vy,vw,vh]. ...

**CMC:** Sparse Lucas-Kanade optical flow → RANSAC partial affine. Applied to predicted
box centres between consecutive **sampled** frames (stride=5). See §3 for stride semantics.

**Cost matrix (Eq. 10–11):** ...

**Assignment:** Hungarian with confidence weighting. τ=0.01, η=0.25.

---

## 3. Experimental setup

| Item | Value |
|---|---|
| Dataset | Anti-UAV300, IR only |
| Split used | val (67 sequences available; 15 used by default) |
| Sequences (names) | *[filled by run_tracker.py]* |
| Frame stride | 5 (every 5th raw frame) |
| Stride semantics | Kalman dt=1 per sampled frame; CMC between sampled frames |
| Detector | YOLOv5s pretrained on COCO (fallback; see §6) |
| Seed | 42 |
| CPU | AMD Ryzen 7 8840HS, no GPU |

**Test split:** Never read. Not used for tuning or evaluation.

---

## 4. Part A Results — Synthetic tiny-box regression

> **This is NOT detection AP. These results cannot be compared with Table 2 of the paper.**

*[Table from results/loss/benchmark.md — filled after Stage 3]*

Convergence plot: `results/loss/convergence.png`
Gradient vs size plot: `results/loss/grad_vs_size.png`

---

## 5. Part B Results — Tracker ablations

*[Table from results/tracker/ablations.md — filled after Stage 6]*

Results at IoU 0.5 and IoU 0.25.

**Note on IDS:** The data is single-target (at most one UAV per frame). ID switches are
expected ≈ 0 for all variants regardless of tracker quality. MOTA/FP/FN differences and
IDF1 are the meaningful signals. IDF1 is only weakly informative in single-target settings.

**Note on CMC:** *[State honestly whether CMC helped based on numbers.]*

---

## 6. Honest gap analysis

| Factor | Effect on results |
|---|---|
| No YOLOv12 training | Table 2 (detector AP) not reproduced |
| YOLOv5s COCO weights, not Anti-UAV | Recall on IR thermal UAVs is lower; MOTA not comparable with Table 3 paper (val) |
| Single-target data | IDS ≈ 0 always; tracker ID-switch advantage untestable |
| Frame stride = 5 | Kalman dt in sampled-frame units; velocity estimates differ from paper |
| CPU only, 15 sequences | Subset of val; full val run available via config |
| Unspecified hyperparameters | See docs/open_questions.md |

---

## 7. Paper numbers for context (not targets)

| Metric | Paper (val) | This reproduction |
|---|---|---|
| Table 3: baseline MOTA | 0.735 (paper val) | *[see results]* |
| Table 3: +CMC MOTA | 0.770 (paper val) | *[see results]* |
| Table 3: +L2 fusion MOTA | 0.775 (paper val) | *[see results]* |
| Test MOTA | 0.8132 (paper hidden test) | NOT REPRODUCED |
| Table 2: AP50 SSQ | 87.8 (paper val) | NOT REPRODUCED — no YOLOv12 training |

---

## 8. Limitations

1. **No YOLOv12 training.** Table 2 (AP) is not reproduced. Requires GPU.
2. **YOLOv5s COCO detector.** Not the paper's Anti-UAV baseline. Recall on IR thermal frames
   is lower. This propagates directly to all MOTA numbers.
3. **Single-target data.** IDS ≈ 0 for every variant. ID-switch advantages of FLIT cannot
   be verified on this data.
4. **No hidden test set.** 81.32% test MOTA is not reproducible.
5. **Stride semantics.** Kalman velocity in sampled-frame units, not raw-frame units.
6. **weights_only=False.** Applied for trusted local YOLOv5 weights (PyTorch ≥ 2.6 compat fix).
7. **Subset of val.** 15/67 sequences. Full run available via `data.num_sequences: 67`.

---

## 9. References

- Wang et al. (2025). Dist-Tracker. CVPR Workshops 2025.
- Aharon et al. (2022). BoT-SORT. arXiv:2206.14651.
- Redmon & Farhadi (2016). YOLOv5s. ultralytics/yolov5.
- Bewley et al. (2016). SORT. ICIP 2016.
- Anti-UAV300: https://github.com/ZhaoJ9014/Anti-UAV
