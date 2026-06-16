#!/usr/bin/env python
"""
Generate results tables from completed atomic experiments.
Reads done.txt files from runs/experiments/ and produces:
  - LaTeX tables (experiment_results.tex)
  - Pretty-printed psql-format tables (experiment_results.txt)

Handles incomplete experiments gracefully by skipping missing data.
"""
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate


EXPERIMENTS_DIR = Path("runs/experiments")

# ── Descriptions & rationale ────────────────────────────────────────────────

EXPERIMENT_META = {
    "exp1": {
        "title": "Intra-Dataset Baseline Performance",
        "description": (
            "5-fold cross-validation on each dataset independently with all six "
            "YOLO model variants. Establishes per-dataset, per-model baselines."
        ),
        "rationale": (
            "Provides a controlled reference for every subsequent comparison. "
            "Without reliable single-dataset baselines we cannot attribute "
            "performance changes to dataset mixing, resolution, or domain shift."
        ),
    },
    "exp2": {
        "title": "Resolution Impact Analysis",
        "description": (
            "5-fold CV at four image resolutions (512, 640, 768, 1024) on each "
            "dataset with YOLOv8s and YOLO11s."
        ),
        "rationale": (
            "Quantifies the accuracy--resolution trade-off. Determines whether "
            "the higher cost of high-resolution acquisition is justified and "
            "identifies the minimum resolution for reliable detection."
        ),
    },
    "exp3": {
        "title": "Cross-Dataset Generalization",
        "description": (
            "Train on one dataset, evaluate on another (6 directional pairs) "
            "with YOLOv8s and YOLO11s.  No fold CV on the target domain."
        ),
        "rationale": (
            "Measures domain gap between acquisition methods. A large "
            "performance drop signals that domain-specific adaptation or "
            "mixed training is necessary for deployment."
        ),
    },
    "exp4": {
        "title": "Dataset Combination Strategies",
        "description": (
            "Train on combined datasets (hi_res + low_res, or all combined) "
            "with 5-fold CV using YOLOv8s and YOLO11s."
        ),
        "rationale": (
            "Tests whether pooling heterogeneous data improves generalization "
            "beyond what any single dataset achieves alone, guiding data "
            "collection strategy."
        ),
    },
    "exp5": {
        "title": "Alternative Architectures",
        "description": (
            "Evaluate Faster R-CNN (ResNet-50) and RT-DETR-L on hi_res and "
            "low_res datasets with 5-fold CV over the full resolution sweep "
            "(512/640/768/1024 px), matching the YOLO evaluation protocol."
        ),
        "rationale": (
            "Validates that conclusions drawn from the YOLO family generalize "
            "across fundamentally different detector architectures (two-stage "
            "and transformer-based)."
        ),
    },
}


# ── Parsing helpers ─────────────────────────────────────────────────────────

def parse_done_file(path: Path) -> dict | None:
    """Parse a done.txt and return a dict of key-value pairs."""
    if not path.exists():
        return None
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                # Try to parse as float
                try:
                    val = float(val.rstrip("s"))
                except ValueError:
                    pass
                data[key] = val
    return data


def scan_experiments(base_dir: Path) -> list[dict]:
    """Scan all experiment directories and return parsed results."""
    results = []
    if not base_dir.exists():
        return results

    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        done_file = entry / "done.txt"
        parsed = parse_done_file(done_file)
        if parsed is None:
            continue

        name = entry.name
        record = {"dir_name": name}

        # Determine experiment group
        m = re.match(r"(exp\d+)_", name)
        if not m:
            continue
        record["group"] = m.group(1)

        # Parse fold experiments: exp{N}_{dataset}_{model}[_img{size}]_fold{k}
        # Variable model suffixes use [a-z0-9]+ (no underscore) so an optional
        # _img<size> / _fold<k> suffix is not swallowed into the model token.
        fold_match = re.match(
            r"(exp\d+)_(.+?)_(yolov?5[a-z]*|yolov?8[a-z]*|yolo11[a-z]*|fasterrcnn_[a-z0-9]+|rtdetr_[a-z0-9]+)"
            r"(?:_img(\d+))?_fold(\d+)$",
            name,
        )
        if fold_match:
            record["type"] = "fold"
            record["dataset"] = fold_match.group(2)
            record["model"] = fold_match.group(3)
            record["img_size"] = int(fold_match.group(4)) if fold_match.group(4) else 1024
            record["fold"] = int(fold_match.group(5))
            record["mAP50"] = parsed.get("mAP50")
            record["mAP50-95"] = parsed.get("mAP50-95")
            record["mAP75"] = parsed.get("mAP75")
            record["precision"] = parsed.get("precision")
            record["recall"] = parsed.get("recall")
            record["f1"] = parsed.get("f1")
            record["training_time"] = parsed.get("Training time")
            results.append(record)
            continue

        # Parse cross experiments: exp{N}_{train}_to_{test}_{model}
        cross_match = re.match(
            r"(exp\d+)_(.+?)_to_(.+?)_(yolov?5[a-z]*|yolov?8[a-z]*|yolo11[a-z]*|fasterrcnn_[a-z0-9]+|rtdetr_[a-z0-9]+)$",
            name,
        )
        if cross_match:
            record["type"] = "cross"
            record["train_dataset"] = cross_match.group(2)
            record["test_dataset"] = cross_match.group(3)
            record["model"] = cross_match.group(4)
            record["train_mAP50"] = parsed.get("Train mAP50")
            record["cross_mAP50"] = parsed.get("Cross mAP50")
            record["cross_mAP50-95"] = parsed.get("Cross mAP50-95")
            # Train-domain operating-point metrics (standard keys in done.txt)
            record["train_precision"] = parsed.get("precision")
            record["train_recall"] = parsed.get("recall")
            record["train_f1"] = parsed.get("f1")
            # Target-domain operating-point metrics (only present in newer runs)
            record["cross_precision"] = parsed.get("Cross precision")
            record["cross_recall"] = parsed.get("Cross recall")
            record["cross_f1"] = parsed.get("Cross f1")
            record["training_time"] = parsed.get("Training time")
            results.append(record)
            continue

    return results


# ── Aggregation helpers ─────────────────────────────────────────────────────

def mean_std_str(values, fmt=".4f"):
    """Return 'mean ± std' string from a list of floats."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "—"
    m = np.mean(vals)
    s = np.std(vals)
    return f"{m:{fmt}} ± {s:{fmt}}"


LATEX_MISSING = "$-$"  # LaTeX-safe placeholder for missing values (avoids unicode em-dash)


def mean_std_latex(values, fmt=".4f"):
    """Return '$mean \\pm std$' string for LaTeX."""
    vals = [v for v in values if v is not None]
    if not vals:
        return LATEX_MISSING
    m = np.mean(vals)
    s = np.std(vals)
    return f"${m:{fmt}} \\pm {s:{fmt}}$"


def fmt_time(values):
    """Return mean training time in minutes."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "—"
    m = np.mean(vals) / 60.0
    return f"{m:.1f}"


DATASET_DISPLAY = {
    "hi_res": "Hi-Res",
    "low_res": "Low-Res",
    "literature": "Literature",
    "hi_res_low_res": "Hi-Res + Low-Res",
    "combined": "All Combined",
}

MODEL_DISPLAY = {
    "yolov5s": "YOLOv5s",
    "yolov5m": "YOLOv5m",
    "yolov8s": "YOLOv8s",
    "yolov8m": "YOLOv8m",
    "yolo11s": "YOLO11s",
    "yolo11m": "YOLO11m",
    "fasterrcnn_resnet50": "FasterRCNN-R50",
    "rtdetr_l": "RT-DETR-L",
}


# ── Table builders ──────────────────────────────────────────────────────────

def _collect_fold_groups(fold_recs, key_fn):
    """Aggregate per-fold metric lists keyed by ``key_fn(record)``."""
    groups = defaultdict(lambda: {
        "mAP50": [], "mAP50-95": [], "precision": [], "recall": [], "f1": [], "time": []
    })
    for r in fold_recs:
        g = groups[key_fn(r)]
        g["mAP50"].append(r.get("mAP50"))
        g["mAP50-95"].append(r.get("mAP50-95"))
        g["precision"].append(r.get("precision"))
        g["recall"].append(r.get("recall"))
        g["f1"].append(r.get("f1"))
        g["time"].append(r.get("training_time"))
    return groups


def build_exp1_table(records: list[dict], latex: bool = False) -> pd.DataFrame | None:
    """EXP1: rows = (Dataset, Model), cols = mAP50, mAP50-95, P, R, F1, Time, Folds."""
    fold_recs = [r for r in records if r["group"] == "exp1" and r["type"] == "fold"]
    if not fold_recs:
        return None

    ms = mean_std_latex if latex else mean_std_str
    groups = _collect_fold_groups(fold_recs, lambda r: (r["dataset"], r["model"]))

    rows = []
    for (ds, model), vals in sorted(groups.items()):
        rows.append({
            "Dataset": DATASET_DISPLAY.get(ds, ds),
            "Model": MODEL_DISPLAY.get(model, model),
            "mAP@50": ms(vals["mAP50"]),
            "mAP@50-95": ms(vals["mAP50-95"]),
            "Precision": ms(vals["precision"]),
            "Recall": ms(vals["recall"]),
            "F1": ms(vals["f1"]),
            "Time (min)": fmt_time(vals["time"]),
            "Folds": len([v for v in vals["mAP50"] if v is not None]),
        })

    return pd.DataFrame(rows)


def build_exp1_latex(records: list[dict]) -> pd.DataFrame | None:
    """EXP1 LaTeX variant."""
    return build_exp1_table(records, latex=True)


def build_exp2_table(records: list[dict], latex: bool = False) -> pd.DataFrame | None:
    """EXP2: rows = (Dataset, Model, ImgSize), cols = mAP50, mAP50-95, P, R, F1."""
    fold_recs = [r for r in records if r["group"] == "exp2" and r["type"] == "fold"]
    if not fold_recs:
        return None

    ms = mean_std_latex if latex else mean_std_str
    groups = _collect_fold_groups(fold_recs, lambda r: (r["dataset"], r["model"], r["img_size"]))

    rows = []
    for (ds, model, img_size), vals in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        rows.append({
            "Dataset": DATASET_DISPLAY.get(ds, ds),
            "Model": MODEL_DISPLAY.get(model, model),
            "Img Size": img_size,
            "mAP@50": ms(vals["mAP50"]),
            "mAP@50-95": ms(vals["mAP50-95"]),
            "Precision": ms(vals["precision"]),
            "Recall": ms(vals["recall"]),
            "F1": ms(vals["f1"]),
            "Time (min)": fmt_time(vals["time"]),
            "Folds": len([v for v in vals["mAP50"] if v is not None]),
        })

    return pd.DataFrame(rows)


def build_exp3_table(records: list[dict], latex: bool = False) -> pd.DataFrame | None:
    """EXP3: rows = (Train→Test, Model), cols = Train/Cross mAP50 + P/R/F1."""
    cross_recs = [r for r in records if r["group"] == "exp3" and r["type"] == "cross"]
    if not cross_recs:
        return None

    def scalar(v):
        if v is None:
            return LATEX_MISSING if latex else "—"
        return f"${v:.4f}$" if latex else f"{v:.4f}"

    rows = []
    for r in sorted(cross_recs, key=lambda x: (x["train_dataset"], x["test_dataset"], x["model"])):
        t_time = r.get("training_time")
        time_str = f"{t_time / 60:.1f}" if t_time is not None else "—"

        rows.append({
            "Train": DATASET_DISPLAY.get(r["train_dataset"], r["train_dataset"]),
            "Test": DATASET_DISPLAY.get(r["test_dataset"], r["test_dataset"]),
            "Model": MODEL_DISPLAY.get(r["model"], r["model"]),
            "Train mAP@50": scalar(r.get("train_mAP50")),
            "Train P": scalar(r.get("train_precision")),
            "Train R": scalar(r.get("train_recall")),
            "Train F1": scalar(r.get("train_f1")),
            "Cross mAP@50": scalar(r.get("cross_mAP50")),
            "Cross P": scalar(r.get("cross_precision")),
            "Cross R": scalar(r.get("cross_recall")),
            "Cross F1": scalar(r.get("cross_f1")),
            "Time (min)": time_str,
        })

    return pd.DataFrame(rows)


def build_exp4_table(records: list[dict], latex: bool = False) -> pd.DataFrame | None:
    """EXP4: rows = (Dataset, Model), cols = mAP50, mAP50-95, P, R, F1."""
    fold_recs = [r for r in records if r["group"] == "exp4" and r["type"] == "fold"]
    if not fold_recs:
        return None

    ms = mean_std_latex if latex else mean_std_str
    groups = _collect_fold_groups(fold_recs, lambda r: (r["dataset"], r["model"]))

    rows = []
    for (ds, model), vals in sorted(groups.items()):
        rows.append({
            "Dataset": DATASET_DISPLAY.get(ds, ds),
            "Model": MODEL_DISPLAY.get(model, model),
            "mAP@50": ms(vals["mAP50"]),
            "mAP@50-95": ms(vals["mAP50-95"]),
            "Precision": ms(vals["precision"]),
            "Recall": ms(vals["recall"]),
            "F1": ms(vals["f1"]),
            "Time (min)": fmt_time(vals["time"]),
            "Folds": f"{len([v for v in vals['mAP50'] if v is not None])}/5",
        })

    return pd.DataFrame(rows)


def build_exp5_table(records: list[dict], latex: bool = False) -> pd.DataFrame | None:
    """EXP5: rows = (Dataset, Model, ImgSize), cols = mAP50, mAP50-95, P, R, F1.

    Alternative architectures are now evaluated over the same resolution sweep as
    YOLO (512/640/768/1024) at 5-fold CV, so the table is keyed by resolution.
    """
    fold_recs = [r for r in records if r["group"] == "exp5" and r["type"] == "fold"]
    if not fold_recs:
        return None

    ms = mean_std_latex if latex else mean_std_str
    groups = _collect_fold_groups(fold_recs, lambda r: (r["dataset"], r["model"], r["img_size"]))

    rows = []
    for (ds, model, img_size), vals in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        rows.append({
            "Dataset": DATASET_DISPLAY.get(ds, ds),
            "Model": MODEL_DISPLAY.get(model, model),
            "Img Size": img_size,
            "mAP@50": ms(vals["mAP50"]),
            "mAP@50-95": ms(vals["mAP50-95"]),
            "Precision": ms(vals["precision"]),
            "Recall": ms(vals["recall"]),
            "F1": ms(vals["f1"]),
            "Time (min)": fmt_time(vals["time"]),
            "Folds": f"{len([v for v in vals['mAP50'] if v is not None])}/5",
        })

    return pd.DataFrame(rows)


# ── LaTeX rendering ─────────────────────────────────────────────────────────

# Columns treated as identifiers (left-aligned); everything else is numeric (right-aligned).
_ID_COLUMNS = {"Dataset", "Model", "Train", "Test", "Img Size", "Folds"}


def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert a DataFrame to a full LaTeX table environment.

    Wide tables are wrapped in an ``adjustbox`` so they shrink to fit the text
    width (without enlarging narrow tables). Numeric columns are right-aligned.
    Requires: \\usepackage{booktabs} and \\usepackage{adjustbox}.
    """
    col_fmt = "".join("l" if c in _ID_COLUMNS else "r" for c in df.columns)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")
    # Header
    header = " & ".join(f"\\textbf{{{c}}}" for c in df.columns)
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    # Rows
    prev_dataset = None
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        # Add midrule between dataset groups
        cur_dataset = row.get("Dataset", None)
        if prev_dataset is not None and cur_dataset != prev_dataset:
            lines.append(r"\midrule")
        prev_dataset = cur_dataset
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── psql-style rendering ───────────────────────────────────────────────────

def df_to_psql(df: pd.DataFrame) -> str:
    """Render DataFrame in psql-style table format."""
    return tabulate(df, headers="keys", tablefmt="psql", showindex=False)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Scanning experiments directory …")
    records = scan_experiments(EXPERIMENTS_DIR)
    print(f"Found {len(records)} completed experiment results.\n")

    if not records:
        print("No completed experiments found. Exiting.")
        sys.exit(1)

    # Count per group
    group_counts = defaultdict(int)
    for r in records:
        group_counts[r["group"]] += 1
    for g in sorted(group_counts):
        print(f"  {g}: {group_counts[g]} results")
    print()

    # ── Build tables ────────────────────────────────────────────────────────

    tex_sections = []
    txt_sections = []

    def add_section(exp_key, builder_txt, builder_latex, caption, label):
        meta = EXPERIMENT_META[exp_key]
        header_txt = (
            f"{'=' * 80}\n"
            f"{exp_key.upper()}: {meta['title']}\n"
            f"{'=' * 80}\n\n"
            f"Description: {meta['description']}\n\n"
            f"Rationale:   {meta['rationale']}\n"
        )
        header_tex = (
            f"% ── {exp_key.upper()}: {meta['title']} "
            + "─" * max(0, 60 - len(exp_key) - len(meta['title']))
            + "\n"
            f"% Description: {meta['description']}\n"
            f"% Rationale:   {meta['rationale']}\n"
        )

        df_txt = builder_txt(records)
        df_tex = builder_latex(records)

        if df_txt is None or df_txt.empty:
            header_txt += "\n  *** No completed results yet ***\n"
            txt_sections.append(header_txt)
            tex_sections.append(header_tex + f"% No completed results yet.\n")
            return

        txt_sections.append(header_txt + "\n" + df_to_psql(df_txt) + "\n")
        tex_sections.append(header_tex + "\n" + df_to_latex(df_tex, caption, label) + "\n")

    # EXP1
    add_section(
        "exp1",
        build_exp1_table,
        lambda recs: build_exp1_latex(recs),
        "Intra-dataset baseline performance (5-fold CV, mean $\\pm$ std).",
        "tab:exp1",
    )

    # EXP2
    add_section(
        "exp2",
        lambda recs: build_exp2_table(recs, latex=False),
        lambda recs: build_exp2_table(recs, latex=True),
        "Resolution impact analysis (5-fold CV, mean $\\pm$ std).",
        "tab:exp2",
    )

    # EXP3
    add_section(
        "exp3",
        lambda recs: build_exp3_table(recs, latex=False),
        lambda recs: build_exp3_table(recs, latex=True),
        "Cross-dataset generalization (train $\\rightarrow$ test).",
        "tab:exp3",
    )

    # EXP4
    add_section(
        "exp4",
        lambda recs: build_exp4_table(recs, latex=False),
        lambda recs: build_exp4_table(recs, latex=True),
        "Dataset combination strategies (5-fold CV, mean $\\pm$ std).",
        "tab:exp4",
    )

    # EXP5
    add_section(
        "exp5",
        lambda recs: build_exp5_table(recs, latex=False),
        lambda recs: build_exp5_table(recs, latex=True),
        "Alternative architectures over the resolution sweep (5-fold CV, mean $\\pm$ std).",
        "tab:exp5",
    )

    # ── Write outputs ───────────────────────────────────────────────────────

    # Pretty-print text
    txt_path = Path("experiment_results.txt")
    with open(txt_path, "w") as f:
        f.write("EXPERIMENT RESULTS — Insect Trap Detection\n")
        f.write(f"Generated from: {EXPERIMENTS_DIR}\n")
        f.write(f"Total completed experiments: {len(records)}\n")
        f.write("=" * 80 + "\n\n")
        for section in txt_sections:
            f.write(section + "\n\n")
    print(f"Written: {txt_path}")

    # LaTeX
    tex_path = Path("experiment_results.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated experiment results tables\n")
        f.write("% Requires: \\usepackage{booktabs} and \\usepackage{adjustbox}\n\n")
        for section in tex_sections:
            f.write(section + "\n\n")
    print(f"Written: {tex_path}")

    # Also print to stdout
    print("\n")
    for section in txt_sections:
        print(section)
        print()


if __name__ == "__main__":
    main()
