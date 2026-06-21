#!/usr/bin/env python
"""
Regenerate the Faster R-CNN exp5 done.txt markers *from the existing checkpoints*
— no retraining.

Background: the Faster R-CNN exp5 runs originally used a broken (threshold-free)
Precision/Recall. The fix lives in FasterRCNNTrainer._evaluate() (a 0.5 score
threshold), which operates on the trained weights — it never needed retraining.
A stale block in backfill_done_files.py used to *delete* these done.txt on every
watchdog relaunch, which forced an endless delete -> retrain loop. With that block
removed, this script rebuilds the markers correctly by re-validating best.pt.

Each marker gets the corrected mAP / Precision / Recall / F1. The original wall
clock training time is not recoverable (it was only ever stored in the deleted
done.txt), so Training time is written as 0 with an explanatory note.

Run with:  ~/miniconda3/envs/aiprah5090/bin/python recompute_fasterrcnn_done.py
           [--device 1] [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from detector.experiments.experiment_runner import (
    MODEL_CONFIGS,
    get_trainer,
    get_safe_batch_size,
)
from run_single_experiment import write_done_file

EXPERIMENTS_DIR = Path("runs/experiments")
NAME_RE = re.compile(
    r"^exp5_(hi_res|low_res)_fasterrcnn_resnet50(?:_img(\d+))?_fold(\d+)$"
)
NOTE = "Note: metrics recomputed from checkpoint (validation only, not retrained)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1,
                    help="CUDA index (1 = RTX 5090 in this box's torch ordering)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = []
    for d in sorted(EXPERIMENTS_DIR.glob("exp5_*fasterrcnn*")):
        if not d.is_dir():
            continue
        if (d / "done.txt").exists():
            continue
        m = NAME_RE.match(d.name)
        if not m:
            print(f"[SKIP] {d.name}: name not recognized")
            continue
        targets.append((d, m))

    print(f"{len(targets)} Faster R-CNN exp5 dirs missing done.txt.\n")
    done = failed = 0
    for d, m in targets:
        dataset, img, fold = m.group(1), m.group(2), int(m.group(3))
        img_size = int(img) if img else 1024
        best_pt = d / f"fold_{fold}" / "train" / "best.pt"
        data_yaml = d / f"fold_{fold}" / "data.yaml"

        if not best_pt.exists() or not data_yaml.exists():
            print(f"[MISS] {d.name}: missing best.pt or data.yaml -> skip")
            failed += 1
            continue

        config = copy.deepcopy(MODEL_CONFIGS["fasterrcnn_resnet50"])
        config.img_size = img_size
        config.batch_size = get_safe_batch_size(
            "fasterrcnn_resnet50", img_size, default=config.batch_size
        )

        try:
            trainer = get_trainer(config, args.device)
            metrics = trainer.validate(best_pt, data_yaml)
        except Exception as e:
            print(f"[FAIL] {d.name}: {e!r}")
            failed += 1
            continue

        result = SimpleNamespace(
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            training_time=0.0,
        )
        print(f"[OK]   {d.name}: mAP50={metrics.get('mAP50', 0):.4f} "
              f"P={metrics.get('precision', 0):.4f} R={metrics.get('recall', 0):.4f} "
              f"F1={metrics.get('f1', 0):.4f}")
        if not args.dry_run:
            write_done_file(d / "done.txt", result, extra_lines=[NOTE])
        done += 1

    print(f"\nRegenerated {done} done.txt ; failed {failed}.")
    if args.dry_run:
        print("(dry-run: nothing written)")


if __name__ == "__main__":
    main()
