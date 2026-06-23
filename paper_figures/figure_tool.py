#!/usr/bin/env python
"""
All-in-one tool to explore dataset insect crops and build the paper figures.

It computes per-bounding-box metrics for every dataset:
  - sharp    : Laplacian variance of the crop (focus / detail)
  - area_px  : insect bounding-box area in pixels (bigger = more detail)
  - nn_px    : distance to the nearest other insect (isolation)
  - isolated : True if no other insect falls inside the crop window
  - n        : number of insects in the source image

You can sort all datasets by any of these to maximize focus + insect size,
browse the crops side by side, pick 3 per dataset, and export the 3x3
"detail comparison" figure (PNG + PDF).

Run the interactive app:
    conda activate aiprah5090
    streamlit run paper_figures/figure_tool.py

Pre-build the metrics cache from the CLI (recommended once, it's the slow part):
    python paper_figures/figure_tool.py --build-cache
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse the loaders / crop / render logic from the figure script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_dataset_samples_figure as g

CACHE_DIR = g.HERE / ".metrics_cache"
SORT_KEYS = {
    "focus x size (sharp*sqrt area)": lambda r: r["sharp"] * np.sqrt(max(r["area_px"], 1)),
    "sharpness (Laplacian)": lambda r: r["sharp"],
    "isolation (nn distance)": lambda r: r["nn_px"],
    "insect area (px)": lambda r: r["area_px"],
    "insects in image (n)": lambda r: r["n"],
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(master_items, pad=g.CROP_PAD, keep=None, progress=None):
    """Compute per-box metrics for one dataset's master list.

    keep: class ids to keep (None = all). Box indices in the output are over
    the KEPT boxes only, so they match read_boxes(..., keep)[box].
    """
    rows = []
    total = len(master_items)
    for idx, (img_p, lbl_p) in enumerate(master_items):
        boxes = g.read_boxes(lbl_p, keep)
        if boxes:
            img = cv2.imread(str(img_p))
            if img is not None:
                H, W = img.shape[:2]
                cxs = np.array([b[0] * W for b in boxes])
                cys = np.array([b[1] * H for b in boxes])
                for bi, b in enumerate(boxes):
                    w_px, h_px = b[2] * W, b[3] * H
                    max_dim = max(w_px, h_px)
                    d = np.hypot(cxs - b[0] * W, cys - b[1] * H)
                    d[bi] = np.inf
                    nn = float(d.min()) if len(boxes) > 1 else 1e9
                    crop, _ = g.crop_box(img, b, pad)
                    if crop.size:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    else:
                        sharp = 0.0
                    rows.append({
                        "idx": idx, "box": bi, "name": img_p.name, "n": len(boxes),
                        "area_px": float(w_px * h_px), "max_dim_px": float(max_dim),
                        "nn_px": nn, "sharp": sharp,
                    })
        if progress is not None:
            progress(idx + 1, total)
    return rows


def cache_path(ds_key):
    return CACHE_DIR / f"{ds_key}.json"


def load_cache(ds_key):
    p = cache_path(ds_key)
    if p.exists():
        return json.loads(p.read_text())
    return None


def save_cache(ds_key, rows):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(ds_key).write_text(json.dumps(rows))


def is_isolated(row, pad=g.CROP_PAD):
    side = row["max_dim_px"] * (1 + 2 * pad)
    return row["nn_px"] > side * 0.75


# --------------------------------------------------------------------------- #
# CLI cache builder
# --------------------------------------------------------------------------- #
def build_cache_cli():
    master_lists = {k: g.build_master_list(v["pairs"]) for k, v in g.DATASETS.items()}
    for k, items in master_lists.items():
        print(f"Building cache for {k} ({len(items)} images)...")

        def prog(i, t):
            if i % 200 == 0 or i == t:
                print(f"  {i}/{t}", end="\r", flush=True)

        rows = compute_metrics(items, keep=g.keep_for(k), progress=prog)
        save_cache(k, rows)
        print(f"\n  saved {len(rows)} boxes -> {cache_path(k)}")


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #
def run_app():
    import streamlit as st

    st.set_page_config(page_title="Insect crop figure tool", layout="wide")
    st.title("🪲 Dataset insect-crop figure tool")

    master_lists = {k: g.build_master_list(v["pairs"]) for k, v in g.DATASETS.items()}

    # session state for selections: {ds_key: [(idx, box), ...]}
    if "sel" not in st.session_state:
        st.session_state.sel = {k: list(g.CROP_SELECTION[k]) for k in g.COLUMN_ORDER}

    # ---- sidebar controls ----
    sb = st.sidebar
    sb.header("Controls")
    pad = sb.slider("Crop padding (zoom)", 0.0, 1.0, float(g.CROP_PAD), 0.05,
                    help="Lower = tighter zoom onto the insect.")
    sort_label = sb.selectbox("Sort datasets by", list(SORT_KEYS.keys()))
    only_isolated = sb.checkbox("Only isolated insects (clean background)", value=True)
    min_area = sb.number_input("Min insect area (px)", 0, 2_000_000, 3000, step=1000)
    top_n = sb.slider("Crops shown per dataset", 6, 500, 18, 6)
    n_cols = sb.slider("Gallery columns", 3, 8, 6)

    if sb.button("⚙️ (Re)build metrics cache"):
        prog = st.progress(0.0, text="Computing metrics...")
        for k, items in master_lists.items():
            rows = compute_metrics(items, pad=pad, keep=g.keep_for(k),
                                   progress=lambda i, t, k=k: prog.progress(
                                       i / t, text=f"{k}: {i}/{t}"))
            save_cache(k, rows)
        prog.empty()
        st.success("Cache rebuilt.")

    sort_key = SORT_KEYS[sort_label]

    # ---- per-dataset tabs ----
    tabs = st.tabs([g.DATASETS[k]["title"] for k in g.COLUMN_ORDER])
    for tab, ds_key in zip(tabs, g.COLUMN_ORDER):
        with tab:
            rows = load_cache(ds_key)
            if rows is None:
                st.warning("No cache yet. Click **(Re)build metrics cache** in the sidebar.")
                continue

            cand = [r for r in rows if r["area_px"] >= min_area]
            if only_isolated:
                cand = [r for r in cand if is_isolated(r, pad)]
            cand.sort(key=sort_key, reverse=True)
            cand = cand[:top_n]

            st.caption(f"{len(cand)} candidates shown · current picks: "
                       f"{st.session_state.sel[ds_key]}")

            items = master_lists[ds_key]
            cols = st.columns(n_cols)
            for i, r in enumerate(cand):
                col = cols[i % n_cols]
                img = g.load_rgb(items[r["idx"]][0])
                crop, _ = g.crop_box(
                    img, g.read_boxes(items[r["idx"]][1], g.keep_for(ds_key))[r["box"]], pad)
                with col:
                    st.image(crop, use_container_width=True)
                    st.caption(f"idx={r['idx']} box={r['box']}\n\n"
                               f"sharp={r['sharp']:.0f} · area={int(r['area_px'])} · "
                               f"nn={int(r['nn_px'])} · n={r['n']}")
                    if st.button("➕ pick", key=f"{ds_key}_{r['idx']}_{r['box']}"):
                        pick = (r["idx"], r["box"])
                        sel = list(st.session_state.sel[ds_key])
                        if pick not in sel:
                            sel = (sel + [pick])[-3:]  # keep last 3
                            st.session_state.sel[ds_key] = sel
                            # keep the "Selected picks" text box in sync (a keyed
                            # text_input ignores value= after creation, so we must
                            # write its session_state directly)
                            st.session_state[f"txt_{ds_key}"] = ",".join(
                                f"{i}:{b}" for i, b in sel)
                        st.rerun()

    # ---- selection editor + render ----
    st.divider()
    st.subheader("Selected picks (3 per dataset, top→bottom rows)")
    cols = st.columns(3)
    for col, ds_key in zip(cols, g.COLUMN_ORDER):
        with col:
            st.markdown(f"**{g.DATASETS[ds_key]['title']}**")
            tkey = f"txt_{ds_key}"
            if tkey not in st.session_state:
                st.session_state[tkey] = ",".join(
                    f"{i}:{b}" for i, b in st.session_state.sel[ds_key])
            txt = st.text_input(
                f"{ds_key} picks as idx:box,idx:box,idx:box",
                key=tkey,
            )
            try:
                parsed = [tuple(int(x) for x in p.split(":")) for p in txt.split(",") if p.strip()]
                st.session_state.sel[ds_key] = parsed[:3]
            except ValueError:
                st.error("Format must be idx:box,idx:box,idx:box")

    if st.button("🖼️ Render 3×3 detail comparison", type="primary"):
        # temporarily apply selections + pad, then reuse the script's renderer
        g.CROP_SELECTION = {k: st.session_state.sel[k] for k in g.COLUMN_ORDER}
        g.CROP_PAD = pad
        g.render_crops(master_lists)
        st.success("Saved dataset_insect_crops.png / .pdf in paper_figures/")
        st.image(str(g.OUTPUT_DIR / "dataset_insect_crops.png"))

    st.caption("Tip: paste the picks above into CROP_SELECTION in "
               "generate_dataset_samples_figure.py to bake them in.")


# --------------------------------------------------------------------------- #
def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _in_streamlit():
    run_app()
elif __name__ == "__main__":
    if "--build-cache" in sys.argv:
        build_cache_cli()
    else:
        print(__doc__)
