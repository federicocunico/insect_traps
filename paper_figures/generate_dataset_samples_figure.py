#!/usr/bin/env python
"""
Generate the 3x3 dataset-samples figure for the paper.

Layout (3 rows x 3 columns):

    col 0 = Hi-Res      col 1 = Low-Res     col 2 = Literature
    row 0 = example 1
    row 1 = example 2
    row 2 = example 3

Selection criteria the figure is meant to illustrate:
  Hi-Res     : clear, well-defined insects (anatomy visible)
               row0 -> one big insect, centered
               row1 -> insect on the side / off-center
               row2 -> another clear insect
  Low-Res    : in-field scenario, varying insect counts
               row0 -> many insects
               row1 -> few insects (but not too few)
               row2 -> medium count
  Literature : any sample is fine

Two modes
---------
  Scan / explore candidates (prints ranked candidates + their stable index):
      python generate_dataset_samples_figure.py --scan

  Generate the figure (uses the SELECTION indices below):
      python generate_dataset_samples_figure.py

The SELECTION dict holds the *stable index* of the chosen image inside the
deterministic, name-sorted master list of each dataset. Tune those three
numbers per dataset to swap samples in/out. Run --scan to discover which
indices match each criterion.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_ROOT = PROJECT_ROOT / "detector" / "data"
OUTPUT_DIR = HERE  # paper_figures/

# Each dataset maps to a list of (images_dir, labels_dir) pairs. The master
# list of samples is the name-sorted union of all image/label pairs found.
#
# keep_classes: which YOLO class ids to KEEP. Hi-Res / Low-Res are single-class
# (0 = S. titanus). The Literature dataset (Checola et al.) is MULTI-class:
#   class 0 = Scaphoideus titanus  (what we want)
#   class 1 = other insect species (must be discarded)
# The experiment pipeline keeps only class 0 for literature
# (DatasetManager._filter_label_class0), so we do the same here to make sure
# every literature box/crop/count refers ONLY to S. titanus.
DATASETS = {
    "hi_res": {
        "title": "Hi-Res",
        "keep_classes": None,  # single class, keep all
        "pairs": [
            (DATA_ROOT / "hi_res" / "images" / "train", DATA_ROOT / "hi_res" / "labels" / "train"),
            (DATA_ROOT / "hi_res" / "images" / "val", DATA_ROOT / "hi_res" / "labels" / "val"),
            (DATA_ROOT / "hi_res" / "images" / "test", DATA_ROOT / "hi_res" / "labels" / "test"),
        ],
    },
    "low_res": {
        "title": "Low-Res",
        "keep_classes": None,  # single class, keep all
        "pairs": [
            (DATA_ROOT / "low_res" / "merged" / "images", DATA_ROOT / "low_res" / "merged" / "labels"),
        ],
    },
    "literature": {
        "title": "Literature",
        "keep_classes": {0},  # keep ONLY S. titanus, drop other species
        "pairs": [
            (DATA_ROOT / "InsectDetectionDataset" / "images", DATA_ROOT / "InsectDetectionDataset" / "labels"),
        ],
    },
}


def keep_for(ds_key):
    """Class ids to keep for a dataset (None = all)."""
    return DATASETS[ds_key].get("keep_classes")

COLUMN_ORDER = ["hi_res", "low_res", "literature"]

# --------------------------------------------------------------------------- #
# >>> TUNE HERE <<<  stable indices into each dataset's name-sorted master list
# (run with --scan to find indices matching each criterion described above)
# --------------------------------------------------------------------------- #
# Matched-count layout: each ROW targets the same insect count across the three
# datasets (+-3) so detail can be compared at equal density.
#   row0 = FEW (min 3) | row1 = MEDIUM | row2 = MANY
# Literature counts are S. titanus only (class 0).
#
# Two variants are rendered into two separate files:
#
# 1) SELECTION_LANDSCAPE -> dataset_samples.{png,pdf}
#    Counts ~5 / 8 / 10. Every image is LANDSCAPE so the grid is clean and the
#    titanus stay clearly visible (literature landscape tops out at ~10 titanus).
SELECTION_LANDSCAPE = {
    "hi_res": [547, 841, 858],       # counts: 5, 8, 10
    "low_res": [3440, 1453, 1879],   # counts: 5, 8, 10
    "literature": [546, 544, 591],   # titanus counts: 5, 8, 10
}

# 2) SELECTION_HIGHCOUNT -> dataset_samples_highcount.{png,pdf}
#    Counts ~6 / 14 / 22 (bigger few->many spread). Literature has no landscape
#    image with 14/22 titanus, so its medium/many rows are PORTRAIT full traps
#    (titanus appear small, grid is less uniform).
SELECTION_HIGHCOUNT = {
    "hi_res": [422, 990, 711],       # counts: 6, 14, 22
    "low_res": [3087, 2269, 2313],   # counts: 6, 14, 22
    "literature": [545, 382, 89],    # titanus counts: 6, 14, 21
}

# Kept for the --scan helper / backwards compat.
SELECTION = SELECTION_LANDSCAPE

# Draw bounding boxes over insects to make them easy to see in the figure.
DRAW_BOXES = True
BOX_COLOR = "#00FF66"
BOX_LW = 1.2

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# --------------------------------------------------------------------------- #
# Crop figure (figure 2): 3x3 of single-insect close-ups to show detail
# differences between datasets. Each cell is a tight crop around ONE bbox.
#
# >>> TUNE HERE <<<  each entry is (image_index, box_index): the box_index-th
# bounding box of the image at <image_index> in the dataset's master list.
# Run with --scan-crops to find the largest/clearest insects per dataset.
# --------------------------------------------------------------------------- #
CROP_SELECTION = {
    "hi_res": [(912, 1), (1044, 3), (599, 0)],
    "low_res": [(1066, 0), (2639, 1), (1113, 1)],
    "literature": [(447, 0), (457, 0), (454, 0)],
}

# Fraction of the box size added as context padding around each crop.
# Small value -> tight, extreme zoom onto the insect (maximizes visible detail
# differences between datasets). Increase for more surrounding context.
CROP_PAD = 0.05
# Draw the bbox outline inside the crop too (set False for clean crops).
CROP_DRAW_BOX = False


# --------------------------------------------------------------------------- #
# Data loading / stats
# --------------------------------------------------------------------------- #
def build_master_list(pairs):
    """Return name-sorted list of (image_path, label_path) for labeled images."""
    items = []
    for images_dir, labels_dir in pairs:
        if not images_dir.exists():
            continue
        for img in images_dir.iterdir():
            if img.suffix not in IMAGE_EXTS:
                continue
            lbl = labels_dir / (img.stem + ".txt")
            if lbl.exists():
                items.append((img, lbl))
    items.sort(key=lambda t: t[0].name)
    return items


def read_boxes(label_path, keep=None):
    """Read YOLO-normalized boxes -> list of (xc, yc, w, h).

    keep: optional set of class ids to keep (None = all). For multi-class
    datasets (Literature) pass {0} to keep only S. titanus.
    """
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cls, xc, yc, w, h = parts[:5]
            if keep is not None and int(float(cls)) not in keep:
                continue
            boxes.append((float(xc), float(yc), float(w), float(h)))
    return boxes


def image_stats(label_path, keep=None):
    """Compute per-image stats used for ranking candidates."""
    boxes = read_boxes(label_path, keep)
    n = len(boxes)
    if n == 0:
        return {"n": 0, "max_area": 0.0, "max_center_dist": 0.0, "biggest_dist": 0.0}
    areas = [w * h for (_, _, w, h) in boxes]
    max_i = int(np.argmax(areas))
    # distance of the biggest box from image center, normalized (0..~0.7)
    bx, by = boxes[max_i][0], boxes[max_i][1]
    biggest_dist = float(np.hypot(bx - 0.5, by - 0.5))
    return {
        "n": n,
        "max_area": float(areas[max_i]),
        "biggest_dist": biggest_dist,
    }


# --------------------------------------------------------------------------- #
# Scan / candidate ranking
# --------------------------------------------------------------------------- #
def scan(master_lists, top=8):
    def fmt(idx, items, stats):
        img = items[idx][0]
        s = stats[idx]
        return (f"    idx={idx:<5d} n={s['n']:<3d} area={s['max_area']:.4f} "
                f"dist={s['biggest_dist']:.3f}  {img.name}")

    # ---- Hi-Res ----
    items = master_lists["hi_res"]
    stats = [image_stats(l, keep_for("hi_res")) for _, l in items]
    print("\n========== HI-RES candidates ==========")

    # row0: big + centered, ideally a single clear insect
    centered = [i for i in range(len(items)) if stats[i]["n"] >= 1]
    centered.sort(key=lambda i: (stats[i]["max_area"] - 1.5 * stats[i]["biggest_dist"]), reverse=True)
    print("\n  [row0] BIG & CENTERED (large area, small dist):")
    for i in centered[:top]:
        print(fmt(i, items, stats))

    # row1: clear insect off to the side (decent area, large dist)
    sides = [i for i in range(len(items)) if stats[i]["max_area"] >= 0.004]
    sides.sort(key=lambda i: stats[i]["biggest_dist"], reverse=True)
    print("\n  [row1] OFF-CENTER / ON THE SIDE (decent area, large dist):")
    for i in sides[:top]:
        print(fmt(i, items, stats))

    # row2: another clear single insect (big area, few boxes)
    clear = [i for i in range(len(items)) if stats[i]["n"] in (1, 2)]
    clear.sort(key=lambda i: stats[i]["max_area"], reverse=True)
    print("\n  [row2] CLEAR SINGLE INSECT (large area, 1-2 boxes):")
    for i in clear[:top]:
        print(fmt(i, items, stats))

    # ---- Low-Res ----
    items = master_lists["low_res"]
    stats = [image_stats(l, keep_for("low_res")) for _, l in items]
    counts = np.array([s["n"] for s in stats])
    print("\n========== LOW-RES candidates ==========")
    print(f"  count distribution: min={counts.min()} max={counts.max()} "
          f"median={int(np.median(counts))} mean={counts.mean():.1f}")

    many = np.argsort(-counts)[:top]
    print("\n  [row0] MANY insects (highest count):")
    for i in many:
        print(fmt(int(i), items, stats))

    med_val = int(np.median(counts[counts > 0]))
    medium = sorted(range(len(items)), key=lambda i: abs(stats[i]["n"] - med_val))
    medium = [i for i in medium if stats[i]["n"] > 0][:top]
    print(f"\n  [row2] MEDIUM count (~{med_val}):")
    for i in medium:
        print(fmt(i, items, stats))

    few = [i for i in range(len(items)) if 2 <= stats[i]["n"] <= 4]
    few.sort(key=lambda i: stats[i]["n"])
    print("\n  [row1] FEW insects (2-4, not too few):")
    for i in few[:top]:
        print(fmt(i, items, stats))

    # ---- Literature ----  (S. titanus only)
    items = master_lists["literature"]
    stats = [image_stats(l, keep_for("literature")) for _, l in items]
    print("\n========== LITERATURE candidates (S. titanus / class 0 only) ==========")
    have = [i for i in range(len(items)) if stats[i]["n"] >= 1]
    have.sort(key=lambda i: stats[i]["n"], reverse=True)
    for i in have[:top]:
        print(fmt(i, items, stats))
    print()


# --------------------------------------------------------------------------- #
# Figure rendering
# --------------------------------------------------------------------------- #
def load_rgb(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render(master_lists, selection=SELECTION, basename="dataset_samples"):
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 8.2))

    for c, ds_key in enumerate(COLUMN_ORDER):
        items = master_lists[ds_key]
        title = DATASETS[ds_key]["title"]
        for r in range(n_rows):
            ax = axes[r, c]
            idx = selection[ds_key][r]
            img_path, lbl_path = items[idx]
            img = load_rgb(img_path)
            H, W = img.shape[:2]
            ax.imshow(img)

            if DRAW_BOXES:
                for (xc, yc, bw, bh) in read_boxes(lbl_path, keep_for(ds_key)):
                    x = (xc - bw / 2) * W
                    y = (yc - bh / 2) * H
                    ax.add_patch(Rectangle((x, y), bw * W, bh * H,
                                           fill=False, edgecolor=BOX_COLOR, lw=BOX_LW))

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=15, fontweight="bold", pad=8)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01,
                        wspace=0.03, hspace=0.03)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{basename}.png"
    pdf_path = OUTPUT_DIR / f"{basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  {png_path}\n  {pdf_path}")

    # Report exactly which files were used (for the paper / reproducibility).
    print(f"\nSelected samples ({basename}):")
    for ds_key in COLUMN_ORDER:
        items = master_lists[ds_key]
        for r, idx in enumerate(selection[ds_key]):
            n = len(read_boxes(items[idx][1], keep_for(ds_key)))
            print(f"  {ds_key:<11s} row{r} idx={idx:<5d} count={n:<3d} {items[idx][0].name}")


# --------------------------------------------------------------------------- #
# Crop figure: single-insect close-ups
# --------------------------------------------------------------------------- #
def crop_box(img, box, pad=CROP_PAD):
    """Crop a square region around a YOLO-normalized box, with context padding."""
    H, W = img.shape[:2]
    xc, yc, bw, bh = box
    # square side in pixels = longest box edge * (1 + 2*pad)
    side = max(bw * W, bh * H) * (1 + 2 * pad)
    cx, cy = xc * W, yc * H
    half = side / 2
    x0 = int(max(0, cx - half)); x1 = int(min(W, cx + half))
    y0 = int(max(0, cy - half)); y1 = int(min(H, cy + half))
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def scan_crops(master_lists, top=12):
    """Rank individual insect bboxes by absolute pixel area (more px = more detail)."""
    for ds_key in COLUMN_ORDER:
        items = master_lists[ds_key]
        print(f"\n========== {DATASETS[ds_key]['title']} largest single insects ==========")
        keep = keep_for(ds_key)
        cand = []
        for idx, (img_path, lbl) in enumerate(items):
            boxes = read_boxes(lbl, keep)
            # need image size to rank by pixels; read W,H cheaply from the label-less stat
            for bi, b in enumerate(boxes):
                cand.append((idx, bi, b))
        # rank by normalized area (proxy); refine with pixels for the printed top
        cand.sort(key=lambda t: t[2][2] * t[2][3], reverse=True)
        printed = 0
        for idx, bi, b in cand:
            img = cv2.imread(str(items[idx][0]))
            if img is None:
                continue
            H, W = img.shape[:2]
            pw, ph = int(b[2] * W), int(b[3] * H)
            print(f"  (idx={idx}, box={bi})  px={pw}x{ph}  area_norm={b[2]*b[3]:.4f}  {items[idx][0].name}")
            printed += 1
            if printed >= top:
                break


def render_crops(master_lists):
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.4, 8.8))

    for c, ds_key in enumerate(COLUMN_ORDER):
        items = master_lists[ds_key]
        title = DATASETS[ds_key]["title"]
        for r in range(n_rows):
            ax = axes[r, c]
            idx, bi = CROP_SELECTION[ds_key][r]
            img_path, lbl_path = items[idx]
            img = load_rgb(img_path)
            boxes = read_boxes(lbl_path, keep_for(ds_key))
            box = boxes[bi]
            crop, (x0, y0, x1, y1) = crop_box(img, box)
            ax.imshow(crop)

            if CROP_DRAW_BOX:
                H, W = img.shape[:2]
                xc, yc, bw, bh = box
                bx = (xc - bw / 2) * W - x0
                by = (yc - bh / 2) * H - y0
                ax.add_patch(Rectangle((bx, by), bw * W, bh * H,
                                       fill=False, edgecolor=BOX_COLOR, lw=BOX_LW))

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=15, fontweight="bold", pad=8)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01,
                        wspace=0.04, hspace=0.04)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "dataset_insect_crops.png"
    pdf_path = OUTPUT_DIR / "dataset_insect_crops.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  {png_path}\n  {pdf_path}")

    print("\nSelected insect crops:")
    for ds_key in COLUMN_ORDER:
        items = master_lists[ds_key]
        for r, (idx, bi) in enumerate(CROP_SELECTION[ds_key]):
            print(f"  {ds_key:<11s} row{r} (idx={idx}, box={bi}) {items[idx][0].name}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true",
                    help="Print ranked candidate indices per criterion and exit.")
    ap.add_argument("--scan-crops", action="store_true",
                    help="Print largest single-insect bboxes (idx, box) per dataset and exit.")
    ap.add_argument("--crops-only", action="store_true",
                    help="Only render the insect-crops figure.")
    ap.add_argument("--samples-only", action="store_true",
                    help="Only render the full-image samples figure.")
    ap.add_argument("--top", type=int, default=8, help="How many candidates to print in --scan.")
    args = ap.parse_args()

    master_lists = {k: build_master_list(v["pairs"]) for k, v in DATASETS.items()}
    for k, lst in master_lists.items():
        print(f"{k}: {len(lst)} labeled samples")

    if args.scan:
        scan(master_lists, top=args.top)
        return
    if args.scan_crops:
        scan_crops(master_lists, top=max(args.top, 12))
        return

    if not args.crops_only:
        # two matched-count variants in separate files
        render(master_lists, SELECTION_LANDSCAPE, "dataset_samples")
        render(master_lists, SELECTION_HIGHCOUNT, "dataset_samples_highcount")
    if not args.samples_only:
        render_crops(master_lists)


if __name__ == "__main__":
    main()
