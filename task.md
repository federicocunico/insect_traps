Read throughfully my paper in -2026-PLoS-Scaphoideus-Titanus-Detection folder. That is, a overleaf project fro PloS. I want to write a paper and include the experimental results i obtained recently (see CRON.md, experiment_results.xlsx and generate_results*.py). I do want to adjust the paper, by improving it, and writing a good solid experimental section. Do this in two steps: first, act as a strong reviewer with a sub-agent (with effort and intelligence high, don't use simple models like Haiku, prefer Opus) that finds out existing issues, then, act as the researcher (again, with a subagent, here i want opus for a good wriring and coherence and scientific value) by including the new experimental results, and by writing a convincing narrative that strenghten the paper. Finally re-run the reviewer's subagent issues, trying to fixing them. Those that can't be fixed must be produced in a report file. THe report file will have the issues raised, and how they were solved. 

Do not invent new citations, if you need to add new citations, propose them in the final report specifying which paper co cite and where should be added (section number, paragraph).

---

# Context

This section records (a) what was done in the working sessions that produced the
current `experiment_results.*` files, and (b) how the experiments are structured
and why — so a follow-up agent can write the paper's experimental section with an
accurate mental model. Treat code as the source of truth; verify file:line claims.

## What was done in these sessions

1. **Built the Excel exporter** `generate_results_xlsx.py`. It reuses the parsers
   in `generate_results_tables.py` (so numbers match the `.txt`/`.tex` exactly)
   and writes `experiment_results.xlsx`: an Overview sheet, one sheet per
   experiment (EXP1–EXP5) with mAP + classification metrics, and a "Confusion
   Matrices" sheet embedding representative normalized matrices.

2. **Backfilled EXP3 cross-domain P/R/F1.** The original cross runs only stored
   `Cross mAP50/mAP50-95`. The trained weights (`cross_*/fold_0/train/weights/best.pt`)
   and the target-domain val split (`cross_*/test_eval/data.yaml`) survived, so
   `backfill_cross_metrics.py` re-validates `best.pt` on the target dataset to
   recover the operating-point metrics (no retraining). Recomputed cross mAP50
   matched the stored value within tolerance, confirming fidelity.

3. **Added two EXP3 cross pairs:** `hi_res_low_res → literature` and the reverse
   (×2 models = 4 runs). NOTE the dataset choice: `combined` = Hi-Res + Low-Res +
   Literature (so it LEAKS literature into training); `hi_res_low_res` = the two
   novel datasets only. The clean cross test against Literature therefore uses
   `hi_res_low_res`, not `combined`. Updated the pair lists in
   `run_experiments.py`, `detector/experiments/experiment_runner.py`,
   `run_all_atomic.sh`, and the `TOTAL_EXPECTED` count (322 → 326).

4. **Fixed a delete→retrain loop for Faster R-CNN exp5.** `backfill_done_files.py`
   used to unconditionally delete every fasterrcnn exp5 `done.txt`, and the
   watchdog entrypoint runs that backfill on *every* relaunch — so corrected runs
   were wiped and retrained endlessly. The real P/R fix is metric-side, not
   checkpoint-side (a 0.5 score threshold in `FasterRCNNTrainer._evaluate()`), so
   the deletion block was removed and the markers are now rebuilt from the
   existing checkpoints by `recompute_fasterrcnn_done.py` (re-validate `best.pt`,
   no retraining).

5. **Regenerated** `experiment_results.{txt,tex,xlsx}`. Final state: **326/326**
   experiments complete (EXP1 90, EXP2 120, EXP3 16, EXP4 20, EXP5 80).

**Known caveat for the paper:** the Faster R-CNN exp5 "Time (min)" column is **0**.
The original per-fold training times lived only in the deleted `done.txt` and are
unrecoverable; all accuracy metrics (mAP/P/R/F1) are correct (recomputed from
checkpoints). Do not report fasterrcnn wall-clock training time from these tables.

## Experimental design & logic

Task: single-class detection of *Scaphoideus titanus* on sticky traps. Three
datasets differ by acquisition method:
- **Hi-Res** — high-resolution close captures.
- **Low-Res** — field / lower-resolution captures (largest, slowest to train).
- **Literature** — Checola et al. (2024); labels filtered to the target class.
- Derived combinations: **hi_res_low_res** (novel datasets only) and **combined**
  (all three).

The five experiments build on each other:
- **EXP1 — Intra-dataset baselines.** 5-fold CV per dataset × 6 YOLO variants
  (YOLOv5s/m, YOLOv8s/m, YOLO11s/m) at 1024 px. Establishes per-dataset/per-model
  reference points so later changes can be attributed to a cause.
- **EXP2 — Resolution sweep.** 5-fold CV at 512/640/768/1024 px on each dataset
  with YOLOv8s and YOLO11s. Quantifies the accuracy–resolution trade-off / minimum
  viable resolution.
- **EXP3 — Cross-dataset generalization.** Train on dataset A (fold-0 model),
  evaluate on dataset B's val split. Measures the domain gap between acquisition
  methods; a large drop motivates mixed training. Reports train-domain and
  target-domain (cross) mAP + P/R/F1.
- **EXP4 — Combination strategies.** 5-fold CV training on `hi_res_low_res` and on
  `combined`. Tests whether pooling heterogeneous data beats any single dataset.
- **EXP5 — Alternative architectures.** Faster R-CNN (ResNet-50 FPN v2) and
  RT-DETR-L on Hi-Res and Low-Res over the same 512–1024 sweep at 5-fold CV, to
  check that YOLO-family conclusions generalize across a two-stage CNN and a
  transformer detector.

**Metrics.** mAP@50, mAP@50-95, mAP@75, precision, recall, F1. YOLO and RT-DETR use
Ultralytics (P/R/F1 at the best-F1 confidence). Faster R-CNN uses a custom
evaluator: mAP via torchmetrics (integrates over confidence), but P/R/F1 computed
only on detections with confidence ≥ 0.5 — without this threshold the unbounded
low-confidence proposals collapse precision and make P/R incomparable across
architectures. Aggregates are reported as mean ± std over folds.

## Execution infrastructure (how results are produced)

- `run_all_atomic.sh [DEVICE]` — atomic, idempotent runner. Each experiment writes
  a `runs/experiments/<exp_name>/done.txt` on success; existing `done.txt` means
  skip (crash-safe resume). `<exp_name>` encodes group/dataset/model/img/fold or
  `..._<train>_to_<test>_<model>` for cross.
- `run_single_experiment.py` — runs one fold or one cross experiment; writes the
  full metric set into `done.txt`.
- `done.txt` is the unit of truth for the tables. A JSON cache lives at
  `runs/experiments/.cache/experiments_cache.json`.
- `tmux_experiments_entrypoint.sh` — 4-phase pipeline in a detached tmux session
  `insect_exps`: (1) activate conda, (2) `backfill_done_files.py`, (3)
  `run_all_atomic.sh`, (4) `generate_results_tables.py`.
- `experiment_watchdog.sh` — cron every 5 min; relaunches the pipeline if the tmux
  session is down and work remains (`done.txt` count < `TOTAL_EXPECTED=326`). See
  `CRON.md`.
- Table generation: `generate_results_tables.py` → `experiment_results.txt` +
  `.tex`; `generate_results_xlsx.py` → `experiment_results.xlsx`. Regenerate after
  any `done.txt` change.

**Environment gotcha:** torch CUDA indices are REVERSED from `nvidia-smi` on this
box — `cuda:1` is the RTX 5090, `cuda:0` is the RTX 4090 (often shared). `--device`
/ cron `DEVICE=` are torch indices (currently `DEVICE=1` → 5090). Verify by GPU
name, not nvidia-smi index.

## For the paper task specifically

- Pull all numbers from `experiment_results.xlsx` / `.txt` (already regenerated and
  consistent). Do not re-run training to "refresh" them.
- The EXP3 narrative should distinguish the leakage-free `hi_res_low_res ↔
  literature` comparison from the (leaky) `combined`-includes-literature case.
- Faster R-CNN training-time is unavailable (see caveat); discuss efficiency using
  RT-DETR/YOLO timings or omit fasterrcnn timing.

