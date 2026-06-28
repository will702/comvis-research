"""
TracedPipeline — parameterized re-run of preprocessing + feature extraction
that captures every intermediate image and value for the playground.

Mirrors src/preprocessing.py + src/features.py exactly at default params.
Guard: backend/test_pipeline.py asserts numerical equivalence with the originals.
"""
import os
import sys
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from dataclasses import dataclass, field
from typing import Optional

# Make src/ importable for cascade loading
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.params import PipelineParams, DEFAULT_PARAMS
from backend.encode import (
    to_png_b64, overlay_mask, draw_face_box, draw_blobs,
    lbp_hist_to_png, redness_heatmap, colorize_mask,
)


# ── Cascade detection helpers (shared across calls) ──────────────────────────
_face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
_eye_cascade     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
_nose_cascade    = cv2.CascadeClassifier(os.path.join(_ROOT, "haarcascade_mcs_nose.xml"))


def _detect_face(gray_clahe: np.ndarray):
    """4-attempt cascade fallback chain, returns (x,y,w,h) or None."""
    attempts = [
        (_face_cascade,     1.05, 4, (90,  90)),
        (_face_cascade,     1.10, 3, (70,  70)),
        (_face_cascade_alt, 1.05, 4, (90,  90)),
        (_face_cascade_alt, 1.10, 3, (60,  60)),
    ]
    for i, (cascade, scale, neighbors, min_size) in enumerate(attempts):
        faces = cascade.detectMultiScale(gray_clahe, scaleFactor=scale,
                                         minNeighbors=neighbors, minSize=min_size)
        if len(faces) > 0:
            best = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            return tuple(best), i  # (bbox, attempt_idx)
    return None, -1


# ── Feature name labels ───────────────────────────────────────────────────────
FEATURE_NAMES = (
    # 8 structural
    ["lesion_count", "total_area", "intensity_mean", "intensity_std",
     "area_max", "area_std", "lesion_density", "mean_circularity"]
    # 27 LBP  (R=1 bins 0-8, R=2 bins 0-8, R=3 bins 0-8)
    + [f"lbp_r{r}_b{b}" for r in (1, 2, 3) for b in range(9)]
    # 3 GLCM
    + ["glcm_contrast", "glcm_homogeneity", "glcm_energy"]
    # 4 global redness
    + ["mu_a", "std_a", "mu_cr", "std_cr"]
)

FEATURE_GROUPS = {
    "Structural (8)":    list(range(0, 8)),
    "LBP R=1 (9)":       list(range(8, 17)),
    "LBP R=2 (9)":       list(range(17, 26)),
    "LBP R=3 (9)":       list(range(26, 35)),
    "GLCM (3)":          list(range(35, 38)),
    "Global Redness (4)": list(range(38, 42)),
}

FEATURE_DESCRIPTIONS = {
    "lesion_count":    "Number of qualifying lesion blobs after all shape/contrast filters",
    "total_area":      "Sum of pixel areas of all kept lesion blobs",
    "intensity_mean":  "Mean LAB a* value across lesion blob pixels (redness intensity)",
    "intensity_std":   "Std-dev of per-blob mean a* — how uniform is lesion redness",
    "area_max":        "Area (px²) of the largest single lesion blob",
    "area_std":        "Std-dev of blob areas — lesion size variability",
    "lesion_density":  "lesion_count / skin_pixel_count — how 'crowded' the acne is",
    "mean_circularity":"Mean 4π·A/P² of kept blobs; cysts (acne3) are rounder than papules",
    "glcm_contrast":   "GLCM contrast — local intensity variance (Σ(i-j)²)",
    "glcm_homogeneity":"GLCM homogeneity — inverse contrast; smooth skin → high",
    "glcm_energy":     "GLCM energy — orderliness of texture; uniform → high",
    "mu_a":   "Mean LAB a* over ALL skin pixels — global face redness level",
    "std_a":  "Std-dev of LAB a* over skin — redness variability",
    "mu_cr":  "Mean YCrCb Cr over skin pixels — chroma-red global level",
    "std_cr": "Std-dev of YCrCb Cr over skin",
}
for r in (1, 2, 3):
    for b in range(9):
        FEATURE_DESCRIPTIONS[f"lbp_r{r}_b{b}"] = (
            f"LBP (R={r}) uniform histogram bin {b} — "
            f"{'uniform pattern' if b < 8 else 'non-uniform catch-all'} frequency on skin"
        )


def run(img_bgr: np.ndarray, params: PipelineParams = DEFAULT_PARAMS) -> dict:
    """
    Full traced pipeline. Returns a dict of all intermediates (b64 PNGs),
    feature vector, feature breakdown, blob list, and threshold values.
    """
    p = params

    # ── Stage 1: Resize ───────────────────────────────────────────────────
    resized = cv2.resize(img_bgr, (512, 512))

    # ── Stage 2: Grayscale ────────────────────────────────────────────────
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # ── Stage 3: CLAHE ────────────────────────────────────────────────────
    clahe_obj   = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(p.clahe_tile, p.clahe_tile))
    gray_clahe  = clahe_obj.apply(gray)

    # ── Stage 4: Face detection ───────────────────────────────────────────
    face_box, cascade_attempt = _detect_face(gray_clahe)

    # ── Stage 5: ROI mask (eyes / nose / lips blacked out) ───────────────
    roi_mask = np.ones((512, 512), dtype=np.uint8) * 255
    roi_regions = []  # for viz

    if face_box is not None:
        fx, fy, fw, fh = face_box
        face_roi = gray_clahe[fy:fy + fh, fx:fx + fw]

        eyes = _eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=6)
        nose = _nose_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=6)

        for (ex, ey, ew, eh) in eyes:
            ex += fx; ey += fy
            roi_mask[max(0, ey - 6):min(512, ey + eh + 6),
                     max(0, ex - 8):min(512, ex + ew + 8)] = 0
            roi_regions.append({"type": "eye", "bbox": [int(ex - 8), int(ey - 6), int(ew + 16), int(eh + 12)]})

        for (nx, ny, nw, nh) in nose:
            nx += fx; ny += fy
            roi_mask[max(0, ny - 5):min(512, ny + nh + 5),
                     max(0, nx - 5):min(512, nx + nw + 5)] = 0
            roi_regions.append({"type": "nose", "bbox": [int(nx - 5), int(ny - 5), int(nw + 10), int(nh + 10)]})

        lip_y1 = int(fy + 0.68 * fh); lip_y2 = int(fy + 0.90 * fh)
        lip_x1 = int(fx + 0.20 * fw); lip_x2 = int(fx + 0.80 * fw)
        roi_mask[max(0, lip_y1):min(512, lip_y2),
                 max(0, lip_x1):min(512, lip_x2)] = 0
        roi_regions.append({"type": "lips", "bbox": [int(lip_x1), int(lip_y1), int(lip_x2 - lip_x1), int(lip_y2 - lip_y1)]})

    # ── Stage 6: YCrCb skin segmentation ─────────────────────────────────
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
    skin_raw = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8),
                                  np.array([255, 173, 127], dtype=np.uint8))
    skin_raw = cv2.morphologyEx(skin_raw, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.skin_close, p.skin_close)))
    skin_raw = cv2.morphologyEx(skin_raw, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.skin_open,  p.skin_open)))

    # ── Stage 7: Combined mask ────────────────────────────────────────────
    skin_mask = cv2.bitwise_and(skin_raw, roi_mask)
    masked_face = cv2.bitwise_and(resized, resized, mask=skin_mask)

    # ── Feature channels ──────────────────────────────────────────────────
    lab        = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    a_channel  = lab[:, :, 1]
    cr_channel = ycrcb[:, :, 1]

    # ── Stage 8: Global skin redness stats + adaptive thresholds ─────────
    a_skin  = a_channel[skin_mask == 255]
    cr_skin = cr_channel[skin_mask == 255]
    skin_pixel_count = len(a_skin)

    if skin_pixel_count == 0:
        # fallback: return zeros
        vec = np.zeros(42)
        return _package_zero_result(resized, gray, gray_clahe, skin_mask, params)

    mu_a   = float(np.mean(a_skin));  std_a  = float(np.std(a_skin))
    mu_cr  = float(np.mean(cr_skin)); std_cr = float(np.std(cr_skin))

    thr_cr = max(p.cr_floor, mu_cr + p.cr_k * std_cr)
    thr_a  = max(p.a_floor,  mu_a  + p.a_k  * std_a)

    # ── Stage 9: Red mask + morphology ───────────────────────────────────
    _, mask_cr = cv2.threshold(cr_channel, thr_cr, 255, cv2.THRESH_BINARY)
    _, mask_a  = cv2.threshold(a_channel,  thr_a,  255, cv2.THRESH_BINARY)
    red_mask   = cv2.bitwise_and(cv2.bitwise_and(mask_cr, mask_a), skin_mask)
    red_mask   = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.red_open,  p.red_open)))
    red_mask   = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.red_close, p.red_close)))

    # ── Stage 10: Connected components + shape filter ─────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask)
    min_area  = max(20, int(0.0001 * skin_pixel_count))
    max_area  = int(p.max_area_pct * skin_pixel_count)
    surr_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    lesion_count = 0
    lesion_areas = []
    lesion_circs = []
    loc_intensities = []
    blobs = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bx   = stats[i, cv2.CC_STAT_LEFT]
        by   = stats[i, cv2.CC_STAT_TOP]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh_  = stats[i, cv2.CC_STAT_HEIGHT]

        reason = None
        if area < min_area:   reason = "too_small"
        elif area > max_area: reason = "too_large"
        else:
            aspect = max(bw, bh_) / max(min(bw, bh_), 1)
            if aspect > p.max_aspect: reason = f"aspect>{p.max_aspect:.1f}"
            else:
                fill = area / max(bw * bh_, 1)
                if fill < p.min_fill: reason = f"fill<{p.min_fill:.2f}"
                else:
                    comp_mask = (labels == i).astype(np.uint8) * 255
                    dilated   = cv2.dilate(comp_mask, surr_kern, iterations=1)
                    surround  = cv2.bitwise_and(
                        cv2.subtract(dilated, comp_mask),
                        (skin_mask > 0).astype(np.uint8) * 255,
                    )
                    surr_vals = a_channel[surround > 0]
                    blob_vals = a_channel[comp_mask > 0]
                    local_contrast = 0.0
                    if len(surr_vals) >= 10:
                        local_contrast = float(np.mean(blob_vals)) - float(np.mean(surr_vals))
                        if local_contrast < p.min_local_contrast:
                            reason = f"contrast<{p.min_local_contrast:.1f}"

                    if reason is None:
                        # compute circularity
                        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        circ = 0.0
                        if contours:
                            perim = cv2.arcLength(contours[0], True)
                            if perim > 0:
                                circ = min(1.0, 4 * np.pi * area / (perim ** 2))

                        lesion_count += 1
                        lesion_areas.append(float(area))
                        lesion_circs.append(circ)
                        pxls = a_channel[labels == i]
                        if len(pxls) > 0:
                            loc_intensities.append(float(np.mean(pxls)))

                        blobs.append({
                            "bbox": [int(bx), int(by), int(bw), int(bh_)],
                            "area": int(area), "aspect": float(max(bw, bh_) / max(min(bw, bh_), 1)),
                            "fill": float(area / max(bw * bh_, 1)),
                            "local_contrast": float(local_contrast),
                            "circularity": float(circ),
                            "kept": True, "reject_reason": None,
                        })
                        continue

        blobs.append({
            "bbox": [int(bx), int(by), int(bw), int(bh_)],
            "area": int(area), "aspect": 0.0, "fill": 0.0,
            "local_contrast": 0.0, "circularity": 0.0,
            "kept": False, "reject_reason": reason,
        })

    # ── Stage 11: Structural features ─────────────────────────────────────
    total_area       = float(sum(lesion_areas))            if lesion_areas            else 0.0
    intensity_mean   = float(np.mean(loc_intensities))     if loc_intensities         else 0.0
    intensity_std    = float(np.std(loc_intensities))      if len(loc_intensities) > 1 else 0.0
    area_max         = float(max(lesion_areas))            if lesion_areas            else 0.0
    area_std         = float(np.std(lesion_areas))         if len(lesion_areas) > 1   else 0.0
    lesion_density   = float(lesion_count) / skin_pixel_count
    mean_circ        = float(np.mean(lesion_circs))        if lesion_circs            else 0.0

    struct = [float(lesion_count), total_area, intensity_mean, intensity_std,
              area_max, area_std, lesion_density, mean_circ]

    # ── Stage 12: Multi-scale LBP ─────────────────────────────────────────
    gray_full = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    lbp_features = []
    lbp_maps     = []
    lbp_hists    = []

    for r in (1, 2, 3):
        lbp_img  = local_binary_pattern(gray_full, 8, r, method="uniform")
        lbp_skin = lbp_img[skin_mask == 255]
        if len(lbp_skin) > 0:
            hist, _ = np.histogram(lbp_skin, bins=np.arange(0, 11), range=(0, 10))
            hist = hist.astype(float)
            hist /= hist.sum() + 1e-7
        else:
            hist = np.zeros(10)
        lbp_features.extend(hist[:9].tolist())

        # Normalize LBP image for visualization
        lbp_vis = (lbp_img / lbp_img.max() * 255).astype(np.uint8) if lbp_img.max() > 0 else lbp_img.astype(np.uint8)
        lbp_maps.append(to_png_b64(lbp_vis))
        lbp_hists.append(lbp_hist_to_png(hist[:9], r))

    # ── Stage 13: GLCM ────────────────────────────────────────────────────
    rows = np.any(skin_mask == 255, axis=1)
    cols = np.any(skin_mask == 255, axis=0)
    glcm_vals = [0.0, 0.0, 0.0]
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        gc = gray_full[rmin:rmax + 1, cmin:cmax + 1].copy()
        mc = skin_mask[rmin:rmax + 1, cmin:cmax + 1]
        gc[mc != 255] = 0
        gq = (gc // 4).astype(np.uint8)
        glcm = graycomatrix(gq, distances=[1],
                             angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                             levels=64, symmetric=True, normed=True)
        glcm_vals = [
            float(graycoprops(glcm, "contrast").mean()),
            float(graycoprops(glcm, "homogeneity").mean()),
            float(graycoprops(glcm, "energy").mean()),
        ]

    # ── Stage 14: Global redness ──────────────────────────────────────────
    global_redness = [mu_a, std_a, mu_cr, std_cr]

    # ── Assemble feature vector ───────────────────────────────────────────
    vec = np.array(struct + lbp_features + glcm_vals + global_redness, dtype=float)

    # ── Build overlay images ──────────────────────────────────────────────
    face_annot   = draw_face_box(resized, face_box)
    # Draw ROI exclusion regions
    for rr in roi_regions:
        rx, ry, rw, rh = rr["bbox"]
        col = {"eye": (200, 60, 60), "nose": (60, 60, 200), "lips": (180, 60, 180)}[rr["type"]]
        cv2.rectangle(face_annot, (rx, ry), (rx + rw, ry + rh), col, 2)
        cv2.putText(face_annot, rr["type"], (rx, max(ry - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    blob_img     = draw_blobs(overlay_mask(resized, red_mask, (0, 60, 220), 0.3), blobs)
    skin_ov      = overlay_mask(resized, skin_mask, (0, 180, 100), 0.5)
    roi_ov       = overlay_mask(resized, roi_mask,  (100, 180, 255), 0.3)
    a_heat       = redness_heatmap(a_channel, skin_mask)
    cr_heat      = redness_heatmap(cr_channel, skin_mask)

    # Feature breakdown dict for inspector
    feat_list = [
        {"idx": i, "name": FEATURE_NAMES[i], "value": float(vec[i]),
         "description": FEATURE_DESCRIPTIONS.get(FEATURE_NAMES[i], "")}
        for i in range(42)
    ]
    groups = {g: idxs for g, idxs in FEATURE_GROUPS.items()}

    return {
        # Stage images (b64 PNG)
        "img_resized":    to_png_b64(resized),
        "img_gray":       to_png_b64(gray),
        "img_clahe":      to_png_b64(gray_clahe),
        "img_face_box":   to_png_b64(face_annot),
        "img_roi_mask":   to_png_b64(roi_ov),
        "img_skin_raw":   to_png_b64(overlay_mask(resized, skin_raw, (0, 200, 100), 0.5)),
        "img_skin_mask":  to_png_b64(skin_ov),
        "img_masked_face": to_png_b64(masked_face),
        "img_red_mask":   to_png_b64(overlay_mask(resized, red_mask, (0, 60, 220), 0.5)),
        "img_blobs":      to_png_b64(blob_img),
        "img_lbp_r1":     lbp_maps[0],
        "img_lbp_r2":     lbp_maps[1],
        "img_lbp_r3":     lbp_maps[2],
        "img_lbp_hist_r1": lbp_hists[0],
        "img_lbp_hist_r2": lbp_hists[1],
        "img_lbp_hist_r3": lbp_hists[2],
        "img_a_heat":     to_png_b64(a_heat),
        "img_cr_heat":    to_png_b64(cr_heat),
        # Metadata
        "face_detected":      face_box is not None,
        "cascade_attempt":    int(cascade_attempt),
        "face_box":           list(face_box) if face_box else None,
        "roi_regions":        roi_regions,
        "skin_pixel_count":   int(skin_pixel_count),
        "thr_cr":             float(thr_cr),
        "thr_a":              float(thr_a),
        "mu_cr":              float(mu_cr), "std_cr": float(std_cr),
        "mu_a":               float(mu_a),  "std_a":  float(std_a),
        "blobs":              blobs,
        "lesion_count":       int(lesion_count),
        "kept_blobs":         int(lesion_count),
        "rejected_blobs":     int(len(blobs) - lesion_count),
        "glcm": {"contrast": glcm_vals[0], "homogeneity": glcm_vals[1], "energy": glcm_vals[2]},
        # Feature vector
        "feature_vector":     vec.tolist(),
        "features":           feat_list,
        "feature_groups":     groups,
    }


def _package_zero_result(resized, gray, gray_clahe, skin_mask, params):
    """Minimal result when skin segmentation finds no pixels."""
    return {k: "" for k in [
        "img_resized", "img_gray", "img_clahe", "img_face_box",
        "img_roi_mask", "img_skin_raw", "img_skin_mask", "img_masked_face",
        "img_red_mask", "img_blobs", "img_lbp_r1", "img_lbp_r2", "img_lbp_r3",
        "img_lbp_hist_r1", "img_lbp_hist_r2", "img_lbp_hist_r3",
        "img_a_heat", "img_cr_heat",
    ]} | {
        "face_detected": False, "cascade_attempt": -1, "face_box": None, "roi_regions": [],
        "skin_pixel_count": 0, "thr_cr": 0.0, "thr_a": 0.0,
        "mu_cr": 0.0, "std_cr": 0.0, "mu_a": 0.0, "std_a": 0.0,
        "blobs": [], "lesion_count": 0, "kept_blobs": 0, "rejected_blobs": 0,
        "glcm": {"contrast": 0.0, "homogeneity": 0.0, "energy": 0.0},
        "feature_vector": [0.0] * 42,
        "features": [{"idx": i, "name": FEATURE_NAMES[i], "value": 0.0, "description": ""} for i in range(42)],
        "feature_groups": FEATURE_GROUPS,
        "img_resized": to_png_b64(resized),
        "img_gray": to_png_b64(gray),
        "img_clahe": to_png_b64(gray_clahe),
    }
