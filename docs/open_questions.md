# Open Questions

Every ambiguity in the paper is listed here with the chosen default, the config key
that controls it, and the rationale.

---

## OQ-1  Value of `w_s` in scale loss (Eq. 3)

**Paper says:** "w_s > 0.5" only.
**Chosen default:** 0.6
**Config key:** `loss.w_s`
**Rationale:** Smallest round value strictly above 0.5.
**Status:** Untested — sweep {0.6, 0.7, 0.8} if scale loss underperforms.

---

## OQ-1b  Frobenius norm factor in scale loss — this repo vs NWD convention

**Issue:** Eq. 3 uses `Sigma = diag(w^2, h^2)` so `Sigma^(1/2) = diag(w, h)` and
the Frobenius norm of the difference is `(w_p-w_t)^2 + (h_p-h_t)^2`.
The NWD paper uses `Sigma = diag(w^2/4, h^2/4)` which introduces a factor of 1/4.
**Chosen default:** factor = 1.0 (literal Eq. 3 of this paper).
**Config key:** `loss.size_cov_scale` (set to 0.25 to match NWD convention).
**Status:** Default matches paper Eq. 3 literally.

---

## OQ-2  Pixel vs normalised coordinates in scale loss

**Paper says:** Does not specify. Loss could be computed in pixels or in [0,1].
**Chosen default:** pixel coordinates (no normalisation).
**Config key:** `loss.use_normalised_coords` (false = pixels, true = normalised).
**Rationale:** Pixels are the natural unit for small-object UAV boxes.
  Normalised mode is available as a configurable alternative.
**Status:** Both modes implemented; pixel is the default.

---

## OQ-2b  NWD-style exponential wrapping of scale loss

**Paper says:** Not mentioned. NWD wraps a Wasserstein term in `1 - exp(-D/C)`.
**Chosen default:** off.
**Config key:** `loss.use_nwd_exp` (false), `loss.nwd_exp_C` (25.0).
**Status:** Implemented; off by default.

---

## OQ-3  Exponent `scale` in shape loss (Eq. 4)

**Paper says:** Not given. The term uses `w_t^scale` and `h_t^scale`.
**Chosen default:** 2.0 (squared, consistent with MSE-style penalties).
**Config key:** `loss.shape_scale`
**Status:** Untested — try {1.0, 2.0, 3.0} if shape term is ineffective.

---

## OQ-4  Sliding window implementation for beta (Eq. 6)

**Paper says:** "E[1-IoU] is the mean over a 100-iteration sliding window."
**Chosen default:** `collections.deque(maxlen=100)`.
  - The window is per-optimisation-run (not shared across loss variants in the benchmark).
  - `iou_val` passed to `compute_beta` must be a Python float (already detached).
**Config key:** none — window size is fixed at 100 by the paper.
**Status:** Confirmed; covered by `test_window_maxlen`.

---

## OQ-5  Sigma in quality loss (Eq. 5)

**Paper says:** Not given. Sigma is the exponent on `(1-IoU)^sigma`.
**Chosen default:** 2.0.
**Config key:** `loss.sigma`
**Status:** Untested — try {1.5, 2.0, 2.5}.

---

## OQ-6  L2 distance normalisation in tracker (Eq. 10)

**Paper says:** "normalised Euclidean distance between box centres" — no denominator given.
**Chosen default:** divide by image diagonal `sqrt(640^2 + 512^2) ≈ 820 px`.
**Config key:** `tracker.l2_norm` ("diagonal" | "fixed"), `tracker.l2_norm_fixed` (512).
**Rationale:** Image diagonal makes the cost dimensionless and resolution-independent.
**Status:** Both options implemented; diagonal is default.

---

## OQ-7  Hungarian match-distance threshold

**Paper says:** Not specified.
**Chosen default:** 0.8 (cost matrix values are in [0,1] range after normalisation).
**Config key:** `tracker.match_thresh`
**Status:** Swept in ablation: {0.5, 0.7, 0.8, 0.9}.

---

## OQ-8  Track buffer (unmatched track retention)

**Paper Algorithm 1:** Unmatched tracks deleted immediately → track_buffer = 0.
**Chosen default:** 0 (paper-exact).
**Config key:** `tracker.track_buffer`
**Status:** Ablation runs {0, 1, 2, 3}.

---

## OQ-9  Kalman filter noise constants

**Paper says:** References BoT-SORT formulation but gives no specific Q or R values.
**Chosen defaults:** Q diagonal scales [1,1,1,1,0.01,0.01,0.0001,0.0001] × Q_scale.
  R diagonal scales [1,1,10,10] × R_scale. (Standard BoT-SORT values.)
**Config keys:** `kalman.Q_scale` (1.0), `kalman.R_scale` (1.0).
**Status:** Standard values; document if changed.

---

## OQ-10  Stride semantics

**Issue:** With frame_stride=5, consecutive sampled frames are 5 raw frames apart.
  The Kalman filter dt is 1 per sampled frame (effectively assuming unit time steps).
  CMC affines are computed between consecutive sampled frames.
**Consequence:** Kalman velocity estimates are in units of "pixels per sampled frame",
  not "pixels per raw frame". This is documented in design.md §1.3 and the report.
**Config key:** `data.frame_stride`
**Status:** Stated in design, report, and evaluate.py docstring.
