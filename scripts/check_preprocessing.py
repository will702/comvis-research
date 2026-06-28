"""
check_preprocessing.py
-----------------------
Visual diagnostic for the full preprocessing + feature-extraction pipeline.

For each acne class it samples N images from data_split/train, runs every
pipeline step, then saves two output artefacts into diagnostics/:

  1. diagnostics/grid_<class>.png
       A 5-column visual grid per sampled image:
         [original | skin mask | skin overlay | lesion mask | lesion overlay + labels]

  2. diagnostics/stats_summary.png
       Box plots of key numeric features (lesion_count, total_area,
       lesion_density, skin coverage %) broken down by class — reveals
       whether the feature distributions actually separate the classes.

A text report is also printed to stdout with:
  - Face-detection success / fallback rate per class
  - Skin-coverage statistics (mean ± std)
  - Lesion-count statistics
  - How many images produced zero lesions (potential false-negative / bad mask)

Usage
-----
    python -m scripts.check_preprocessing            # 8 samples per class
    python -m scripts.check_preprocessing --n 16    # 16 samples per class
    python -m scripts.check_preprocessing --class acne3_1024 --n 20
"""

import argparse
import os
import random
import sys
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")          # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Make sure the project root is on the path when run as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import BASE_DIR, CLASSES
from src.preprocessing import FaceProcessor
from src.features import FeatureExtractor

DIAG_DIR = os.path.join(BASE_DIR, "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

# ── colour overlays ────────────────────────────────────────────────────────────
SKIN_COLOUR   = (0, 200, 80)     # green
LESION_COLOUR = (0, 0, 255)      # red (BGR)


# ── helpers ───────────────────────────────────────────────────────────────────

def _overlay(img_bgr, mask, colour_bgr, alpha=0.45):
    """Blend a binary mask onto a BGR image."""
    out = img_bgr.copy()
    out[mask == 255] = (
        np.array(img_bgr[mask == 255], dtype=np.float32) * (1 - alpha)
        + np.array(colour_bgr, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return out


def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _mask_to_rgb(mask):
    """Convert a single-channel mask to a 3-channel grey image."""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def _build_lesion_mask_and_labels(img_bgr, skin_mask):
    """
    Re-run just the lesion-segmentation part of FeatureExtractor so we can
    produce a coloured lesion mask and per-lesion bounding boxes.
    Returns (red_mask, lesion_stats_list, thr_cr, thr_a).
    """
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a_ch    = lab[:, :, 1]
    ycrcb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr_ch   = ycrcb[:, :, 1]

    a_skin  = a_ch[skin_mask == 255]
    cr_skin = cr_ch[skin_mask == 255]

    if len(a_skin) == 0:
        h, w = skin_mask.shape
        return np.zeros((h, w), dtype=np.uint8), [], 0, 0

    mu_a,  std_a  = float(np.mean(a_skin)),  float(np.std(a_skin))
    mu_cr, std_cr = float(np.mean(cr_skin)), float(np.std(cr_skin))

    thr_cr = max(145, mu_cr + 1.5 * std_cr)
    thr_a  = max(133, mu_a  + 1.2 * std_a)

    _, mask_cr = cv2.threshold(cr_ch, thr_cr, 255, cv2.THRESH_BINARY)
    _, mask_a  = cv2.threshold(a_ch,  thr_a,  255, cv2.THRESH_BINARY)
    red_mask   = cv2.bitwise_and(cv2.bitwise_or(mask_cr, mask_a), skin_mask)

    red_mask = cv2.morphologyEx(
        red_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    red_mask = cv2.morphologyEx(
        red_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    skin_px  = int(np.sum(skin_mask == 255))
    min_area = max(20, int(0.0001 * skin_px))

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(red_mask)
    lesion_stats = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            lesion_stats.append({
                "area":     area,
                "x":        stats[i, cv2.CC_STAT_LEFT],
                "y":        stats[i, cv2.CC_STAT_TOP],
                "w":        stats[i, cv2.CC_STAT_WIDTH],
                "h":        stats[i, cv2.CC_STAT_HEIGHT],
                "cx":       int(centroids[i][0]),
                "cy":       int(centroids[i][1]),
            })

    return red_mask, lesion_stats, thr_cr, thr_a


def _annotate_lesion_overlay(img_bgr, skin_mask, lesion_stats, lesion_mask):
    """Draw the lesion overlay with bounding boxes and a count badge."""
    vis = _overlay(img_bgr, lesion_mask, LESION_COLOUR, alpha=0.55)
    for ls in lesion_stats:
        cv2.rectangle(vis,
                      (ls["x"], ls["y"]),
                      (ls["x"] + ls["w"], ls["y"] + ls["h"]),
                      (0, 255, 255), 1)
    # count badge
    txt = f"lesions: {len(lesion_stats)}"
    cv2.rectangle(vis, (0, 0), (160, 22), (0, 0, 0), -1)
    cv2.putText(vis, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ── per-class visual grid ─────────────────────────────────────────────────────

def make_class_grid(cls_name, sample_paths, processor, extractor, out_path):
    """
    Rows = images, Cols = [original | skin_mask | skin_overlay |
                            lesion_mask | lesion_overlay+labels]
    """
    n  = len(sample_paths)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.02})
    if n == 1:
        axes = axes[np.newaxis, :]   # keep 2-D indexing consistent

    col_titles = [
        "Original (512×512)",
        "Skin mask",
        "Skin overlay",
        "Lesion mask (red_mask)",
        "Lesion overlay + boxes",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, pad=4)

    for row, img_path in enumerate(sample_paths):
        fname = os.path.basename(img_path)
        is_aug = fname.startswith("aug_")

        img_resized, skin_mask = processor.preprocess_image(img_path)

        # ── fallback handling ─────────────────────────────────────────────
        if img_resized is None or skin_mask is None:
            for col in range(5):
                axes[row, col].set_visible(False)
            axes[row, 0].set_visible(True)
            axes[row, 0].text(0.5, 0.5, f"LOAD FAILED\n{fname}",
                              ha="center", va="center", color="red",
                              transform=axes[row, 0].transAxes)
            continue

        skin_coverage = float(np.sum(skin_mask == 255)) / (512 * 512) * 100

        lesion_mask, lesion_stats, thr_cr, thr_a = _build_lesion_mask_and_labels(
            img_resized, skin_mask)

        features     = extractor.extract(img_resized, skin_mask)
        lesion_count = int(features[0])
        total_area   = int(features[1])
        density      = float(features[6])
        # Global skin redness — indices 38-41 in the 42-dim vector
        mu_a   = float(features[38])
        mu_cr  = float(features[40])

        # col 0 — original
        ax = axes[row, 0]
        ax.imshow(_bgr_to_rgb(img_resized))
        lbl = f"{'[AUG] ' if is_aug else ''}{fname[:30]}"
        ax.set_ylabel(lbl, fontsize=6, rotation=0, labelpad=120, va="center")

        # col 1 — skin mask (grayscale)
        axes[row, 1].imshow(skin_mask, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_xlabel(f"coverage: {skin_coverage:.1f}%", fontsize=7)

        # col 2 — skin overlay
        axes[row, 2].imshow(_bgr_to_rgb(_overlay(img_resized, skin_mask, SKIN_COLOUR)))
        axes[row, 2].set_xlabel(
            f"mu_a*={mu_a:.1f}  mu_Cr={mu_cr:.1f}", fontsize=7)

        # col 3 — lesion (red_mask) binary
        axes[row, 3].imshow(lesion_mask, cmap="hot", vmin=0, vmax=255)
        axes[row, 3].set_xlabel(
            f"thr_Cr={thr_cr:.0f}  thr_a*={thr_a:.0f}", fontsize=7)

        # col 4 — lesion overlay + bounding boxes
        vis = _annotate_lesion_overlay(img_resized, skin_mask, lesion_stats, lesion_mask)
        axes[row, 4].imshow(_bgr_to_rgb(vis))
        mean_sz = total_area / max(lesion_count, 1)
        axes[row, 4].set_xlabel(
            f"count={lesion_count}  area={total_area}  mean_sz={mean_sz:.0f}px",
            fontsize=7)

        for col in range(5):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle(f"Preprocessing diagnostic — {cls_name}  ({n} samples)",
                 fontsize=12, y=1.002)
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── per-class statistics ──────────────────────────────────────────────────────

def collect_stats(cls_name, sample_paths, processor, extractor):
    """Returns a dict with per-image diagnostic values for the class."""
    records = []
    face_ok = 0
    no_face = 0

    for img_path in sample_paths:
        img_resized, skin_mask = processor.preprocess_image(img_path)
        if img_resized is None or skin_mask is None:
            continue

        # Use the processor's full cascade chain (same as training pipeline)
        gray    = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g_clahe = clahe.apply(gray)
        if processor._detect_face(g_clahe) is not None:
            face_ok += 1
        else:
            no_face += 1

        skin_px      = int(np.sum(skin_mask == 255))
        skin_cov_pct = skin_px / (512 * 512) * 100

        features     = extractor.extract(img_resized, skin_mask)
        lesion_count = int(features[0])
        total_area   = int(features[1])
        density      = float(features[6])
        mean_circ    = float(features[7])
        # Global skin redness — indices 38-41 in the 42-dim vector
        mu_a   = float(features[38])
        std_a  = float(features[39])
        mu_cr  = float(features[40])
        std_cr = float(features[41])
        mean_lesion_sz = total_area / max(lesion_count, 1)

        records.append({
            "lesion_count":    lesion_count,
            "total_area":      total_area,
            "mean_lesion_size": mean_lesion_sz,
            "skin_coverage":   skin_cov_pct,
            "lesion_density":  density,
            "mean_circularity": mean_circ,
            "mu_a":            mu_a,
            "std_a":           std_a,
            "mu_cr":           mu_cr,
            "std_cr":          std_cr,
            "zero_lesions":    lesion_count == 0,
        })

    return {
        "cls":         cls_name,
        "records":     records,
        "face_ok":     face_ok,
        "no_face":     no_face,
    }


def print_text_report(all_stats):
    SEP = "─" * 70
    print(f"\n{SEP}")
    print("  PREPROCESSING DIAGNOSTIC REPORT")
    print(SEP)

    for s in all_stats:
        cls  = s["cls"]
        recs = s["records"]
        n    = len(recs)
        if n == 0:
            print(f"\n{cls}: no records")
            continue

        counts   = [r["lesion_count"]    for r in recs]
        areas    = [r["total_area"]       for r in recs]
        sz_mean  = [r["mean_lesion_size"] for r in recs]
        covs     = [r["skin_coverage"]    for r in recs]
        dens     = [r["lesion_density"]   for r in recs]
        circs    = [r["mean_circularity"] for r in recs]
        mu_as    = [r["mu_a"]             for r in recs]
        std_as   = [r["std_a"]            for r in recs]
        mu_crs   = [r["mu_cr"]            for r in recs]
        std_crs  = [r["std_cr"]           for r in recs]
        zero_n   = sum(r["zero_lesions"]  for r in recs)

        face_rate = s["face_ok"] / (s["face_ok"] + s["no_face"]) * 100

        print(f"\n{'━'*40}")
        print(f"  {cls}  (n={n})")
        print(f"{'━'*40}")
        print(f"  Face detection success : {s['face_ok']}/{n}  ({face_rate:.0f}%)")
        print(f"  Fallback (no face)     : {s['no_face']}/{n}")
        print(f"  Zero-lesion images     : {zero_n}/{n}  "
              f"({'⚠ high — possible bad mask' if zero_n/n > 0.3 else 'OK'})")
        print(f"  Skin coverage  mean±std: {np.mean(covs):.1f}% ± {np.std(covs):.1f}%  "
              f"[min {np.min(covs):.1f}%  max {np.max(covs):.1f}%]")
        print(f"  --- Lesion features ---")
        print(f"  Lesion count   mean±std: {np.mean(counts):.1f} ± {np.std(counts):.1f}  "
              f"[min {int(np.min(counts))}  max {int(np.max(counts))}]")
        print(f"  Total area     mean±std: {np.mean(areas):.0f} ± {np.std(areas):.0f}")
        print(f"  Mean lesion sz mean±std: {np.mean(sz_mean):.0f} ± {np.std(sz_mean):.0f} px  "
              f"← acne3 cysts should be largest")
        print(f"  Lesion density mean±std: {np.mean(dens):.6f} ± {np.std(dens):.6f}")
        print(f"  Mean circularity  mean : {np.mean(circs):.3f}  "
              f"(1.0=perfect circle, lower=elongated)")
        print(f"  --- Global skin redness (NEW) ---")
        print(f"  mu_a*  (LAB a*)  mean±std: {np.mean(mu_as):.2f} ± {np.std(mu_as):.2f}  "
              f"← higher = more red/inflamed")
        print(f"  std_a* (LAB a*)  mean±std: {np.mean(std_as):.2f} ± {np.std(std_as):.2f}  "
              f"← higher = more uneven redness")
        print(f"  mu_Cr  (YCrCb)   mean±std: {np.mean(mu_crs):.2f} ± {np.std(mu_crs):.2f}")
        print(f"  std_Cr (YCrCb)   mean±std: {np.mean(std_crs):.2f} ± {np.std(std_crs):.2f}")

    print(f"\n{SEP}\n")


# ── summary box-plot figure ───────────────────────────────────────────────────

def make_stats_plot(all_stats, out_path):
    metrics = [
        ("lesion_count",    "Lesion count"),
        ("mean_lesion_size","Mean lesion size (px)  ← key for acne3"),
        ("total_area",      "Total lesion area (px)"),
        ("mu_a",            "mu_a* — global skin redness (LAB)  ← NEW"),
        ("mu_cr",           "mu_Cr — global skin redness (YCrCb)  ← NEW"),
        ("mean_circularity","Mean lesion circularity"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5))
    colours = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    for ax, (key, label), colour in zip(axes, metrics, colours):
        data_per_class = []
        xlabels        = []
        for s in all_stats:
            vals = [r[key] for r in s["records"]]
            if vals:
                data_per_class.append(vals)
                # shorten label: acne0_1024 → acne0
                xlabels.append(s["cls"].replace("_1024", ""))

        bplot = ax.boxplot(data_per_class, patch_artist=True, notch=False,
                           medianprops={"color": "black", "linewidth": 1.5})
        for patch, c in zip(bplot["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(xlabels) + 1))
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Feature distribution by acne class (train samples)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocessing diagnostic tool")
    parser.add_argument("--n",     type=int, default=8,
                        help="Number of images to sample per class (default: 8)")
    parser.add_argument("--class", dest="cls_filter", default=None,
                        help="Only inspect this class, e.g. acne3_1024")
    parser.add_argument("--seed",  type=int, default=0,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_dir  = os.path.join(BASE_DIR, "data_split", "train")
    if not os.path.exists(train_dir):
        print("ERROR: data_split/train not found. Run src/data_prep.py first.")
        sys.exit(1)

    processor = FaceProcessor()
    extractor = FeatureExtractor()

    target_classes = (
        [args.cls_filter] if args.cls_filter else CLASSES
    )

    all_stats = []

    for cls_name in target_classes:
        cls_dir = os.path.join(train_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"WARNING: {cls_dir} not found, skipping.")
            continue

        all_imgs = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.endswith(".jpg")
        ]
        if not all_imgs:
            print(f"WARNING: no .jpg files in {cls_dir}")
            continue

        # Sample: prefer a mix of original and augmented images
        originals = [p for p in all_imgs if not os.path.basename(p).startswith("aug_")]
        augmented = [p for p in all_imgs if os.path.basename(p).startswith("aug_")]

        n_orig = min(max(args.n // 2, 1), len(originals))
        n_aug  = min(args.n - n_orig,    len(augmented))
        sample = random.sample(originals, n_orig) + (
            random.sample(augmented, n_aug) if n_aug > 0 else []
        )
        sample = sample[:args.n]   # cap to requested N
        random.shuffle(sample)

        print(f"\n[{cls_name}]  {len(all_imgs)} total images, "
              f"sampling {len(sample)} "
              f"({n_orig} original + {n_aug} augmented) ...")

        # Visual grid
        grid_path = os.path.join(DIAG_DIR, f"grid_{cls_name}.png")
        make_class_grid(cls_name, sample, processor, extractor, grid_path)

        # Collect stats over ALL images (not just the visual sample) for the report
        # but cap to 100 to avoid excessive runtime
        stat_sample = all_imgs if len(all_imgs) <= 100 else random.sample(all_imgs, 100)
        print(f"  Collecting stats from {len(stat_sample)} images ...")
        stats = collect_stats(cls_name, stat_sample, processor, extractor)
        all_stats.append(stats)

    # Text report
    print_text_report(all_stats)

    # Summary box-plot
    if len(all_stats) > 1:
        plot_path = os.path.join(DIAG_DIR, "stats_summary.png")
        print("Generating stats summary plot ...")
        make_stats_plot(all_stats, plot_path)

    print(f"\nAll diagnostics saved to: {DIAG_DIR}/")


if __name__ == "__main__":
    main()
