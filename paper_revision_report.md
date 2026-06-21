# Paper Revision Report — *Multi-Camera and Smartphone Detection of* Scaphoideus titanus *(PLoS)*

**Date:** 2026-06-21
**Scope:** Two-step revision of the Overleaf project in `-2026-PLoS-Scaphoideus-Titanus-Detection/`.
**Process:** (1) An independent Opus reviewer produced a 23-issue major-revision review (`review_issues.md`). (2) The new experimental results (`experiment_results.txt`) were integrated and the narrative strengthened. (3) The reviewer was re-run to verify fixes against the revised files and the ground-truth data.

**Constraints honored:**
- **Training-time columns removed** from all tables (author instruction). Inference time is **not stored** anywhere in the run artifacts (`done.txt`/`results.csv` only contain training time and per-epoch metrics), so it was **not** invented or reported — see Open Item O7.
- **No citations were invented.** Duplicate bib keys were consolidated in `\cite` usage only; needed-but-missing citations are listed in the *Proposed Citations* section with exact locations.
- **No data was fabricated.** All five tables were regenerated verbatim from `experiment_results.txt` and verified line-by-line in the second reviewer pass. The one instance count that could be computed and verified from the on-disk labels (Hi-Res = 2,782) was filled; the Low-Res count could not be verified against the stated image count and was left flagged (see O1).

**Second-pass verdict:** of 23 issues — **11 Fixed, 9 Partial (honestly disclosed / safest defensible edit made), 3 Not fixed (require author input or new analysis).** All table numbers verified to match the data exactly.

---

## Verification of the experimental data integration

| Experiment | Change applied | Verified |
|---|---|---|
| EXP1 (Table 1) | Dropped training-time column; added Precision/Recall/F1 | matches `experiment_results.txt` |
| EXP2 (Table 2) | Dropped time; added P/R/F1 | matches |
| EXP3 (Table 3) | Added leakage-free `Hi-Res+Low-Res ↔ Literature` rows + cross P/R/F1; "—" where only cross mAP@50 was stored; clarified `Src mAP@50` | matches |
| EXP4 (Table 4) | Dropped time; added P/R/F1; relabeled "All Combined" vs "Hi-Res+Low-Res" with leakage note | matches |
| EXP5 (Table 5) | **Replaced stale 3-fold@640 table** with the 5-fold full resolution sweep (512/640/768/1024) for Faster R-CNN and RT-DETR; added P/R/F1 with cross-architecture caveat (`†`) | matches |

Key data facts established from the code/labels (source of truth = code, per task):
- Splits: image-level `StratifiedKFold` (`detector/datasets/data_loader.py`), `n_splits=5`, `shuffle=True`, `random_state=42`, stratified by insect presence/absence, with a held-out test partition. **Not** trap-grouped — disclosed as a limitation.
- Training: 100 epochs, patience 20 (`run_experiments.py`); YOLO/RT-DETR via Ultralytics from COCO-pretrained weights; Faster R-CNN = torchvision `fasterrcnn_resnet50_fpn_v2`, COCO-pretrained, SGD lr 0.005 / mom 0.9 / wd 5e-4, StepLR (`detector/experiments/experiment_runner.py`).
- Faster R-CNN P/R/F1 use a fixed conf ≥ 0.5 operating point, unlike the best-F1 point used by Ultralytics — hence the cross-architecture P/R caveat.
- Hi-Res labels: 1,760 files / **2,782** single-class instances (matches stated image count). Literature is **2-class** (1,327 / 1,505), confirming Issue 9. Low-Res on-disk `merged` = 3,800 imgs / 13,142 inst, which does **not** match the paper's stated 5,610 images → not filled (O1).

---

## Per-issue resolution log

### Critical
- **Issue 1 — Placeholders/TODOs.** *Partial → mostly Fixed.* Removed `\fc`, `\textcolor{red}`, `LOREM`/`IPSUM` macros, the acks `\todo`, and the "TO BE COMPLETELY REVISED" conclusion (fully rewritten). Filled Hi-Res instance count (2,782). Box dimensions de-flagged ("approximately"). **Remaining:** Low-Res instance count, samples figure, code/data URL — genuine data gaps (O1–O3), kept as explicit `\todo`s.
- **Issue 2 — "YOLO pipeline" mischaracterization.** *Fixed.* Abstract, intro (3rd contribution), method overview, and conclusion now describe a single-stage/two-stage/transformer benchmark.
- **Issue 3 — Title "Diagnostics" / abstract omits smartphone.** *Partial.* Abstract rewritten to feature the Low-Res/smartphone dataset and the controlled-vs-uncontrolled comparison. **Title not changed** (author decision, O4): no diagnostic/operating-point analysis exists to back the word "Diagnostics".
- **Issue 4 — EXP5 protocol mismatch.** *Fixed.* `04_exps.tex` and `05_results.tex` + Table 5 now state 5-fold CV over the 512/640/768/1024 sweep; numbers regenerated.
- **Issue 5 — Cross-architecture P/R not comparable.** *Fixed.* Evaluation Metrics states the per-detector operating-point rule; EXP5 text and Table 5 carry the `†` caveat; mAP designated the primary cross-architecture metric.
- **Issue 6 — Combined-dataset leakage.** *Fixed.* Experiments intro, EXP3, and EXP4 make the "All Combined = in-domain vs Hi-Res+Low-Res = held-out" distinction explicit; EXP3 reports the leakage-free pairs.

### Major
- **Issue 7 — Hyperbolic language.** *Fixed.* EXP3 prose neutralized; the one surviving "catastrophic" in EXP4 was also replaced ("severe").
- **Issue 8 — "Minimum optimal resolution" undelivered.** *Partial.* Rationale softened to "point of diminishing returns"; EXP2 now names 1024 px as the operating resolution and 768 px as the accuracy/cost compromise, and quantifies the shrinking increments. A μm/pixel-on-target threshold analysis was **not** added (O5).
- **Issue 9 — Single-class vs discrimination claim.** *Partial.* Now explicitly framed as single-class detection (abstract, method annotation, Evaluation Metrics, conclusion); mAP described as localization not discrimination. **Remaining:** the Literature 2-class→1-class reduction must be described by the authors (O6); some intro motivation language about morphological discrimination remains as motivation.
- **Issue 10 — Duplicate bib keys.** *Fixed (usage).* All `\cite` now use canonical CamelCase keys (`Checola2024GrapevinePest`, `Goncalves2022EdgeViticulture`, `ortis2025new`); the lowercase duplicates are no longer cited and will not appear in the reference list. (Dead entries left physically in `.bib`, harmless; optional cleanup O8.)
- **Issue 11 — Trap-grouping / split protocol.** *Partial (disclosed).* Added a "Cross-Validation Protocol" subsection describing the actual image-level stratified split (seed 42, held-out test) and the corrected "no same-batch trap" claim, with trap-grouping stated as a limitation in the Conclusions. Not eliminated (would require re-running with grouped folds, O9).
- **Issue 12 — Reproducibility.** *Partial.* Implementation Details now lists epochs/patience/pretraining/optimizer/schedule/augmentation. "Fully reproducible" over-claim removed. **Remaining:** code/data availability URL (O3); training seed and NMS/IoU thresholds not documented.
- **Issue 13 — No significance testing.** *Partial.* "Best"/"consistently outperformed" softened; within-1-std differences explicitly called ties; bolding annotated as ties. No formal paired test added (O9).
- **Issue 14 — Recall-vs-strict-IoU conclusion mismatch.** *Fixed.* Conclusion now attributes the Hi-Res advantage to strict-IoU mAP@50-95; AP@0.5 = mAP@50 clarified; no unreported mAP@75.
- **Issue 15 — Table 5 stale numbers / efficiency claims.** *Fixed.* Regenerated from the 5-fold sweep; all efficiency/training-time claims dropped.
- **Issue 16 — Laekeman μm/pixel dangling promise.** *Not fixed.* Requires a new effective-resolution analysis (O5).

### Minor
- **Issue 17 — PLoS figure/table compliance.** *Partial.* Removed colored `\todo` from the Table 1 cell (now plain `(to be reported)`); the reminder lives in the (non-cell) table note. **Remaining:** the multi-panel `subfigure` figure must be externalized/combined per PLoS rules at submission (O10).
- **Issue 18 — Image-/trap-count inconsistency.** *Not fixed.* Needs the distinct physical-trap count and per-modality breakdown to reconcile 1,760 vs 5,610 (O1).
- **Issue 19 — JPEG/8-bit vs fine-morphology.** *Fixed.* Acknowledged in the imaging-system paragraph.
- **Issue 20 — EXP3 missing P/R/F1 + Train column.** *Fixed.* Columns added; caption explains `Src mAP@50`.
- **Issue 21 — Abstract "significantly".** *Fixed.* Replaced with the measured mAP@50-95 effect.
- **Issue 22 — Wording/tense nits.** *Partial.* Fixed: Italian note, `$\sim$`, `Fig~\ref` spacing, "apples-to-apples"→"paired", "small-object-optimized" claim, ortis key. Cosmetic leftovers: the "(TrapScan?)" comment and "TrapScan" appearing only in Methods (O11, trivial).
- **Issue 23 — "Fully reproducible / field-deployable" over-claim.** *Fixed.* Removed; framed as future work; box described as a benchtop instrument.

### New defects found in pass 2 and fixed
- Broken `\ref{fig:samples}` (no label) → reworded to avoid the dangling reference; figure kept as a flagged `\todo`.
- Red `\todo` text inside a Table 1 cell (PLoS colored-cell violation) → replaced with plain text; reminder moved to the table note.
- EXP3 rounding "0.66/0.63/0.66" → corrected precision to 0.68.

---

## Open items requiring author input (cannot be fixed without you)

- **O1 — Low-Res annotated-instance count and distinct trap counts.** On-disk Low-Res (`merged`: 3,800 imgs / 13,142 instances) does not match the manuscript's 5,610 images, so neither the instance count nor the Hi-Res(440 traps)↔Low-Res reconciliation could be filled safely. Please report the Low-Res instance count and the number of distinct physical traps per modality. (Table 1 `\todo`s.)
- **O2 — Dataset samples figure (`fig:samples`).** Needs to be produced and given a `\label{fig:samples}`, after which restore the in-text `Fig~\ref`.
- **O3 — Code/data availability URLs.** Required for the reproducibility claim and PLoS data-availability policy. (`04_exps.tex` `\todo`.)
- **O4 — Title "Diagnostics".** Either soften the title (e.g., drop "Diagnostics") or add a genuine smartphone diagnostic/operating-point analysis; editorial decision.
- **O5 — Effective μm/pixel-on-target analysis (Issue 16).** Recommended new analysis tying the EXP2 resolution sweep to pixels-on-insect / μm-per-pixel vs the Laekeman threshold; would convert EXP2 into a transferable finding.
- **O6 — Literature 2-class → 1-class reduction (Issue 9).** State how the second class (*Orientus ishidae*) was handled (discarded vs. treated as background); this materially affects the cross-dataset numbers.
- **O7 — Inference time.** Not stored in any artifact; reporting it would require re-benchmarking `best.pt` on fixed hardware (out of scope here, and the task excluded training time). Omitted by design.
- **O8 — `.bib` cleanup (optional).** The now-uncited duplicate entries (`checola2024novel`, `goncalves2022edge`, `ortis2025newtools`) can be deleted; note `checola2024novel` lists the wrong first name ("Giulia" vs correct "Giorgio").
- **O9 — Stricter protocol (optional, Issues 11/13).** Trap-grouped CV and paired significance tests would require re-running; currently disclosed as limitations.
- **O10 — Figure externalization (Issue 17).** Combine the multi-panel acquisition figure into a single image file at submission.
- **O11 — "TrapScan" naming (trivial).** Introduce the box name in the abstract/intro or drop it; remove the "(TrapScan?)" comment.

---

## Proposed citations (NOT added — propose key, source, and exact location)

No bib entries were invented. The following citations are genuinely needed or recommended; please add the bibentry and the `\cite`:

1. **RT-DETR (REQUIRED).** RT-DETR-L is used in Experiment 5 with no citation (the `.bib` has none).
   - **Where:** `04_exps.tex`, *Experiments* — Experiment 5 bullet, at "a transformer-based detector (RT-DETR-L)"; and optionally the *Related Works* model overview and `05_results.tex` EXP5 paragraph.
   - **Source:** Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection," CVPR 2024 (the RT-DETR paper).

2. **Ultralytics YOLOv5/v8/v11 (RECOMMENDED).** The six variants (YOLOv5s … YOLO11m) are currently attributed only to the original YOLO citation (`Redmon2016YOLO`), which is YOLOv1 and does not cover the Ultralytics releases actually used.
   - **Where:** `04_exps.tex` Experiment 1 bullet (at "six YOLO variants …") and the method overview.
   - **Source:** Jocher et al., *Ultralytics YOLO* (software/release, e.g., the Ultralytics YOLOv8 release) — a software citation, not a paper.

3. **(Optional) COCO / mAP metric.** If a reference for the mAP@[.5:.95] protocol is desired in *Evaluation Metrics*, cite Lin et al., "Microsoft COCO," ECCV 2014. Not strictly required.

---

## Addendum (2026-06-21) — EXP3 cross-domain P/R/F1 fully populated

The "—" entries in the Cross-Dataset Generalization table (12 single-domain pairs) were **not** a data-loss issue: every `cross_*/fold_0/.../best.pt` and `cross_*/test_eval/data.yaml` survived. The dashes existed only because the February runs (`run_single_experiment.py`) logged just `Cross mAP50/mAP50-95`, while the cross operating-point P/R/F1 were added to the logger only for the June combined runs.

`backfill_cross_metrics.py` was run (env `aiprah5090`, RTX 5090 pinned via `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`) to re-validate each surviving `best.pt` on the target-domain validation split — **inference only, no retraining**. It updated all 12 remaining `exp3_*/done.txt` files; no `mAP50` drift warnings were emitted, confirming the recomputed evaluations reproduce the stored `Cross mAP@50` exactly, so the new P/R/F1 are on the same footing as the published cross mAP.

- The cross P/R/F1 are reported at the Ultralytics best-F1 confidence and are comparable across all EXP3 rows.
- For targets in the 2-class Literature set, only the target *S. titanus* class is scored (non-target labels excluded), matching how the existing cross mAP@50 was computed.
- `experiment_results.{txt,tex,xlsx}` were regenerated; **Table 3 in the paper now shows complete Cross P/R/F1 for all 16 pairs**, the caption was updated, and the narrative now notes the precision-collapse failure mode (hard transfers keep moderate recall but drop precision, e.g. Literature→Hi-Res: R 0.46 / P 0.10).

This resolves the EXP3 portion of Issue 20. Note the working directory was renamed from `-2026-...` to `2026-PLoS-Scaphoideus-Titanus-Detection/` (leading dash removed); all edits are intact in the renamed directory.

## Files changed
`00_abstract.tex`, `01_introduction.tex`, `03_method.tex`, `04_exps.tex`, `05_results.tex`, `06_conclusions.tex`, `main.tex` (macro/acks cleanup). Reviewer artifacts: `review_issues.md` (pass-1 review). This report: `paper_revision_report.md`.
