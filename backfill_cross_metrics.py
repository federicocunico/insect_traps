#!/usr/bin/env python
"""
Backfill Cross precision / recall / F1 for the EXP3 cross-dataset runs.

The original cross runs were produced before the runner recorded target-domain
P/R/F1, so ``exp3_*/done.txt`` only carries Cross mAP50 / mAP50-95. The trained
weights and the target-domain val split survive under ``cross_*/`` though, so we
simply re-validate ``best.pt`` on ``test_eval/data.yaml`` — exactly what the
runner does — and append the operating-point metrics.

Idempotent: skips an EXP3 done.txt that already has a "Cross precision" line
(unless --force). The recomputed Cross mAP50 is checked against the stored value
as a sanity guard.

Run with:  ~/miniconda3/envs/aiprah5090/bin/python backfill_cross_metrics.py
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from ultralytics import YOLO

EXPERIMENTS_DIR = Path("runs/experiments")


def read_done(path: Path) -> dict:
    data = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data


def imgsz_from_args(cross_dir: Path, default: int = 1024) -> int:
    args = cross_dir / "fold_0" / "train" / "args.yaml"
    if args.exists():
        m = re.search(r"^imgsz:\s*(\d+)", args.read_text(), re.MULTILINE)
        if m:
            return int(m.group(1))
    return default


def append_cross_metrics(done_path: Path, p: float, r: float, f1: float, force: bool):
    text = done_path.read_text()
    if "Cross precision" in text and not force:
        return False
    # strip any existing cross-PRF lines (when --force re-running)
    kept = [
        ln for ln in text.splitlines()
        if not ln.startswith(("Cross precision", "Cross recall", "Cross f1"))
    ]
    kept += [f"Cross precision: {p:.4f}", f"Cross recall: {r:.4f}", f"Cross f1: {f1:.4f}"]
    done_path.write_text("\n".join(kept) + "\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if present")
    ap.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = ap.parse_args()

    cross_dirs = sorted(EXPERIMENTS_DIR.glob("cross_*_to_*"))
    print(f"Found {len(cross_dirs)} cross runs.\n")

    updated = 0
    for cdir in cross_dirs:
        exp3_dir = EXPERIMENTS_DIR / f"exp3_{cdir.name[len('cross_'):]}"
        done = exp3_dir / "done.txt"
        weights = cdir / "fold_0" / "train" / "weights" / "best.pt"
        data_yaml = cdir / "test_eval" / "data.yaml"

        if not done.exists():
            print(f"  ! {exp3_dir.name}: no exp3 done.txt — skip")
            continue
        if "Cross precision" in done.read_text() and not args.force:
            print(f"  = {exp3_dir.name}: already has Cross P/R/F1 — skip")
            continue
        if not weights.exists() or not data_yaml.exists():
            print(f"  ! {cdir.name}: missing weights or test yaml — skip")
            continue

        imgsz = imgsz_from_args(cdir)
        model = YOLO(str(weights))
        res = model.val(
            data=str(data_yaml), imgsz=imgsz, split="val",
            verbose=False, plots=False, save_json=False, device=args.device,
        )
        b = res.box
        p, r = float(b.mp), float(b.mr)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        stored = read_done(done).get("Cross mAP50")
        flag = ""
        if stored is not None:
            try:
                if abs(float(stored) - float(b.map50)) > 0.02:
                    flag = f"  [WARN mAP50 drift: stored={stored} recomputed={b.map50:.4f}]"
            except ValueError:
                pass

        append_cross_metrics(done, p, r, f1, args.force)
        updated += 1
        print(f"  + {exp3_dir.name}: P={p:.4f} R={r:.4f} F1={f1:.4f} "
              f"(mAP50 {b.map50:.4f}){flag}")

    print(f"\nUpdated {updated} exp3 done.txt files.")


if __name__ == "__main__":
    main()
