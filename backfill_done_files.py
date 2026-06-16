#!/usr/bin/env python
"""
Backfill done.txt markers with the full metric set from the experiment cache.

The atomic runner historically recorded only mAP@50 / mAP@50-95 in each
experiment's done.txt, even though the trainers compute (and cache) Precision,
Recall, F1 and mAP@75 as well. This script rewrites every existing done.txt from
the JSON cache (runs/experiments/.cache/experiments_cache.json) so the result
tables gain P/R/F1 *without any retraining*.

Special handling:
  - Faster R-CNN exp5 runs used a broken (threshold-free) Precision/Recall
    computation. Their done.txt files are *removed* so the atomic runner reruns
    them with the corrected metric and the new resolution sweep.

Run from the repo root:  python backfill_done_files.py [--dry-run]
"""
import argparse
import copy
import json
import re
from pathlib import Path

from detector.experiments.experiment_runner import (
    ExperimentConfig,
    ExperimentCache,
    MODEL_CONFIGS,
)
from run_single_experiment import write_done_file

EXPERIMENTS_DIR = Path("runs/experiments")
SEED = 42

# Variable model suffixes use [a-z0-9]+ (no underscore) so an optional _img<size>
# or _fold<k> suffix is not greedily swallowed into the model token.
_MODEL_ALT = r"yolov?5[a-z]*|yolov?8[a-z]*|yolo11[a-z]*|fasterrcnn_[a-z0-9]+|rtdetr_[a-z0-9]+"
FOLD_RE = re.compile(
    rf"(exp\d+)_(.+?)_({_MODEL_ALT})(?:_img(\d+))?_fold(\d+)$"
)
CROSS_RE = re.compile(
    rf"(exp\d+)_(.+?)_to_(.+?)_({_MODEL_ALT})$"
)


def reconstruct_config(name: str, dataset: str, model: str, img_size: int, fold: int):
    """Rebuild an ExperimentConfig so its hash matches the cached entry."""
    if model not in MODEL_CONFIGS:
        return None
    model_config = copy.deepcopy(MODEL_CONFIGS[model])
    model_config.img_size = img_size
    return ExperimentConfig(
        name=name,
        dataset=dataset,
        model=model_config,
        fold=fold,
        seed=SEED,
    )


def main():
    parser = argparse.ArgumentParser(description="Backfill done.txt from the experiment cache")
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing")
    args = parser.parse_args()

    if not EXPERIMENTS_DIR.exists():
        print(f"No experiments directory at {EXPERIMENTS_DIR}")
        return

    cache = ExperimentCache(EXPERIMENTS_DIR / ".cache")

    n_backfilled = n_removed = n_missing = n_skipped = 0

    for entry in sorted(EXPERIMENTS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        done_file = entry / "done.txt"
        if not done_file.exists():
            continue

        name = entry.name

        # Remove broken Faster R-CNN exp5 markers so they are rerun with the fix.
        if name.startswith("exp5_") and "fasterrcnn" in name:
            print(f"[REMOVE] {name} (broken Faster R-CNN P/R metric -> will rerun)")
            if not args.dry_run:
                done_file.unlink()
            n_removed += 1
            continue

        fold_m = FOLD_RE.match(name)
        cross_m = CROSS_RE.match(name)

        if fold_m:
            group, dataset, model, img, fold = fold_m.groups()
            img_size = int(img) if img else 1024
            config = reconstruct_config(name, dataset, model, img_size, int(fold))
            extra_lines = None
        elif cross_m:
            group, train_ds, test_ds, model = cross_m.groups()
            # Cross runs were cached under the internal "cross_<train>_to_<test>_<model>" name.
            cache_name = f"cross_{train_ds}_to_{test_ds}_{model}"
            config = reconstruct_config(cache_name, train_ds, model, 1024, 0)
            extra_lines = None  # filled after fetching metrics below
        else:
            print(f"[SKIP]   {name} (unrecognized name pattern)")
            n_skipped += 1
            continue

        if config is None:
            print(f"[SKIP]   {name} (unknown model)")
            n_skipped += 1
            continue

        result = cache.get(config)
        if result is None:
            print(f"[MISS]   {name} (no cache entry; leaving done.txt untouched)")
            n_missing += 1
            continue

        if cross_m:
            m = result.metrics
            extra_lines = [
                f"Train mAP50: {m.get('mAP50', 0):.4f}",
                f"Cross mAP50: {m.get('cross_mAP50', 0):.4f}",
                f"Cross mAP50-95: {m.get('cross_mAP50-95', 0):.4f}",
            ]
            # cross P/R/F1 only exist for runs produced after the runner fix
            if "cross_precision" in m:
                extra_lines += [
                    f"Cross precision: {m.get('cross_precision', 0):.4f}",
                    f"Cross recall: {m.get('cross_recall', 0):.4f}",
                    f"Cross f1: {m.get('cross_f1', 0):.4f}",
                ]

        print(f"[OK]     {name}: P={result.metrics.get('precision', 0):.3f} "
              f"R={result.metrics.get('recall', 0):.3f} F1={result.metrics.get('f1', 0):.3f}")
        if not args.dry_run:
            write_done_file(done_file, result, extra_lines=extra_lines)
        n_backfilled += 1

    print("\n" + "=" * 60)
    print(f"Backfilled: {n_backfilled}")
    print(f"Removed (rerun): {n_removed}")
    print(f"Missing cache:  {n_missing}")
    print(f"Skipped:        {n_skipped}")
    if args.dry_run:
        print("(dry-run: no files were modified)")


if __name__ == "__main__":
    main()
