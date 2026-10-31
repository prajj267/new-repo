# Project Rules

These rules apply to every Kiro interaction in this project.

## Process
1. Work one stage at a time, in order. After each stage run its acceptance test,
   paste the real output, then STOP and wait for "next".
2. State each stage's plan and its acceptance test BEFORE writing any code.
3. If something requires a GPU, say so explicitly and propose a CPU alternative.
   Never silently work around it.

## Integrity
4. Never report a metric you did not compute yourself. Keep raw logs, fixed seeds,
   and saved configs alongside every result.
5. Never tune on the test folder. Never read dataset/test at all. Use only dataset/val.
   State clearly in the report whether any tuning was done on the validation subset.
6. Never copy the paper's numbers into a "your results" column. Label paper numbers
   "paper (val)" or "paper (test)" everywhere.
7. Never invent or hard-code results. Every number in the report must trace to a
   file in results/.

## Design
8. Every unspecified hyperparameter must be a config key in `configs/default.yaml`
   with the chosen default documented in `docs/open_questions.md`.
9. Loss terms must be implemented as separate, individually testable functions.
10. The tracker cost matrix computation must be separate from the assignment step.
11. The CMC affine must be computed once per frame and cached to disk so all
    ablations share the same affine. The cache must record frame_stride; a stale
    cache (wrong stride) must be detected and rejected at load time.
12. The detection cache must be built once and read by every tracker experiment;
    provide a rebuild script. Cache files must record detector_weights and
    torch_version.

## Code quality
13. Every function must have a docstring stating inputs, outputs, and units
    (pixels vs normalised).
14. No silent NaN/inf: raise or log a clear error with context.
15. CPU PyTorch only. No CUDA dependency anywhere in requirements or code.

## Reproducibility
16. Set random seeds (Python, NumPy, PyTorch) from a config value in every script.
17. Commit after every completed stage.
18. Every result table must be reproducible by running a single command listed in
    README.md.
19. Pin every package in requirements.txt to an exact version, including pandas,
    Pillow, requests, seaborn, and any package transitively required by torch.hub
    yolov5 or motmetrics.

## Communication
20. If Kiro makes a design choice not in the paper, state it explicitly and add it
    to `docs/open_questions.md`.
21. If any equation or hyperparameter is ambiguous, pick the config-key default
    from the spec, record it in open_questions.md, and continue. Ask only if it
    blocks progress.
