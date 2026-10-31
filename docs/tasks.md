# Tasks

Build order is fixed. Each stage has an acceptance test.
**Do not start a stage until the previous stage's acceptance test passes and you have said "next".**

---

## Stage 0 — Repo scaffold and environment

**Plan**
- Create all directories, `configs/default.yaml`, `requirements.txt`, `.gitignore`, `README.md`.
- Verify Python version has a torch 2.2.2 CPU wheel (3.8–3.12).
- Install all pinned packages.
- Save `pip freeze` to `results/env_freeze.txt`.
- `git init` and initial commit.

**Acceptance test**
```
python -c "import torch, cv2, scipy, motmetrics, yaml, tqdm, pandas, PIL, seaborn; print('OK')"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
Must print `OK`, then `2.2.2+cpu` (or `2.2.2`), then `False`.
`results/env_freeze.txt` must exist.

---

## Stage 1 — Data loader and visualisation

**Plan**
- `src/data/loader.py`: `AntiUAVSequence`, `AntiUAVDataset`, `verify_no_test_in_val`.
- `src/data/visualize.py`: draw GT boxes on 20 random frames, save to `results/viz/`.
- `tests/test_loader.py`: frame count, box conversion, split disjoint check.

**Acceptance test**
```
pytest tests/test_loader.py -v
python src/data/visualize.py --config configs/default.yaml
```
Inspect `results/viz/*.jpg` — boxes must sit on the UAV. No box on invisible frames.

---

## Stage 2 — SSQ loss and unit tests

**Plan**
- `src/losses/ssq.py`: all six functions.
- `tests/test_ssq_loss.py`: all ten tests from requirements §4.3.

**Acceptance test**
```
pytest tests/test_ssq_loss.py -v
```
All ten tests must pass. Tests include:
- `test_scale_matches_closed_form`
- `test_shape_zero_identical_boxes`
- `test_no_nan_tiny_boxes`
- `test_beta_is_detached`
- `test_window_maxlen`
- `test_ssq_weights_at_step_zero` — total == 0.7*(Ls+Lsh)+0.3*Lq within tol
- `test_ssq_weights_at_final_step` — 0 on (scale+shape), 0.6 on quality
- `test_gradcheck_scale`
- `test_gradcheck_shape`
- `test_quality_no_grad_through_beta`

---

## Stage 3 — Synthetic tiny-box regression benchmark

**Plan**
- `scripts/run_loss_benchmark.py`: box generation, Adam optimisation, save all outputs.

**Acceptance test**
```
python scripts/run_loss_benchmark.py --config configs/default.yaml --seeds 0 1 2
```
- Finishes in < 5 minutes on CPU.
- Produces: `benchmark.csv`, `benchmark.md`, `convergence.png`, `grad_vs_size.png`.
- CSV header, markdown, and both plot titles contain "NOT detection AP; cannot be compared with Table 2".
- If SSQ does not beat CIoU: reported plainly in output and markdown. No tuning.

---

## Stage 4 — Detector wrapper and detection cache

**Plan**
- `src/detector/yolov5_wrapper.py`: hub load, offline vendor, weights_only fix.
- `scripts/build_detection_cache.py`: run detector, recall gate after first 3 seqs, save cache.

**Acceptance test**
```
python scripts/build_detection_cache.py --config configs/default.yaml
```
- Prints `Recall@0.50: X.XXX  Recall@0.25: X.XXX` after first 3 sequences.
- If recall@0.25 < 0.05: stops with non-zero exit, prints proposed options. Does NOT continue.
- If recall OK: builds full cache; one JSON per sequence in `cache/detections/`.
- Each JSON contains `frame_stride`, `detector_weights`, `torch_version`.
- `reference/yolov5/` exists after the run (vendored repo).

---

## Stage 5 — FLIT tracker and CMC, with unit tests

**Plan**
- `src/tracker/kalman.py`, `cmc.py`, `cost.py`, `flit.py`, `src/evaluate.py`.
- `scripts/build_cmc_cache.py`: compute and cache affines.
- `tests/test_tracker.py`: all unit tests.

**Build CMC cache first:**
```
python scripts/build_cmc_cache.py --config configs/default.yaml
```

**Acceptance test**
```
pytest tests/test_tracker.py -v
```
All tests must pass. Required tests:
- `test_cost_shapes`
- `test_hungarian_empty_dets`
- `test_hungarian_empty_tracks`
- `test_kalman_predict_shape`
- `test_cmc_toy_example` — identity affine on two identical frames
- `test_cmc_known_translation` — CMC-corrected centre is closer than uncorrected (Amendment F)
- `test_flit_single_target_no_id_switch`
- `test_cache_stride_mismatch_raises`

---

## Stage 6 — Tracker experiments and ablations

**Plan**
- `scripts/run_tracker.py`: all variants in requirements §5.6, per-sequence + aggregate metrics.
- Produces `ablations.csv`, `ablations.md`, `mota_ablation.png`.

**Acceptance test**
```
python scripts/run_tracker.py --config configs/default.yaml
```
- Finishes in < 15 minutes for 15 sequences.
- CSV has one row per (variant, iou_thresh) combination.
- CSV header records which sequences were used.
- Results at IoU 0.5 AND 0.25 for every variant.
- "detector only" row present.
- Mean ± std across sequences present.
- Report honestly: state whether CMC helps or not based on the numbers.

---

## Stage 7 — Optional: synthetic multi-target stress test

Skip unless told to proceed. Clearly labelled optional in the report.

---

## Stage 8 — Report

**Plan**
- Fill in `docs/report.md`.
- Every table cell references a file in `results/`.
- Must include all items from design §8 (limitations).
- Must state stride semantics (Amendment C).
- Must include detector fallback disclaimer.
- Must include weights_only note.
- Must state plainly which paper results were NOT reproduced.

**Acceptance test** (manual)
- Every number in results tables has a raw log file in `results/`.
- "NOT detection AP" appears next to the synthetic benchmark table.
- "paper (val)" labels every paper number cited.
- No paper number appears in a "my results" column.

---

## Task summary

| Stage | Key files | Est. CPU time | Blocking |
|---|---|---|---|
| 0 | scaffold, config, requirements | 5 min install | yes |
| 1 | loader, visualise | 2 min | yes |
| 2 | ssq.py, test_ssq_loss.py | 1 min | yes |
| 3 | run_loss_benchmark.py | < 5 min | yes |
| 4 | yolov5_wrapper.py, cache script | < 10 min | yes |
| 5 | kalman, cmc, cost, flit, tests | 2 min | yes |
| 6 | run_tracker.py, ablations | < 15 min | yes |
| 7 | synthetic stress test | < 5 min | no (optional) |
| 8 | report.md | manual | no |
