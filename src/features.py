import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


class FeatureExtractor:
    P = 8
    R_SCALES = [1, 2, 3]   # multi-scale LBP radii
    LBP_BINS = P + 2        # 10 bins per scale (uniform LBP)
    # After normalization hist sums to 1.0, so the last bin is linearly
    # determined by the other 9.  We drop it to avoid a redundant dimension.
    LBP_BINS_USED = LBP_BINS - 1   # 9 independent bins per scale

    # Feature layout (42 dims total):
    #    8  structural  — lesion_count, total_area, intensity_mean/std,
    #                     area_max/std, lesion_density, mean_circularity
    #                     (intensity_max removed: equals mean when count=1)
    #                     (area_min removed: always near min_area threshold)
    #   27  LBP         — 3 radii × 9 bins  (last dependent bin dropped)
    #    3  GLCM        — contrast, homogeneity, energy
    #                     (dissimilarity removed: ~0.95 correlation with contrast)
    #    4  global skin — mu_a, std_a, mu_cr, std_cr   ← overall face redness
    #   HSV saturation removed: redundant with global skin redness (LAB/YCrCb)
    FEATURE_DIM = 8 + LBP_BINS_USED * len(R_SCALES) + 3 + 4  # 42

    # ── lesion filter thresholds ─────────────────────────────────────────
    # Skin folds / wrinkles are elongated (high aspect ratio) and/or very
    # sparse (low fill ratio).  Real lesions are roughly circular blobs.
    _MAX_ASPECT_RATIO  = 3.5    # width/height or height/width > 3.5  → reject
    _MIN_FILL_RATIO    = 0.25   # area / bounding-box area   < 0.25   → reject
    # A single acne lesion should not exceed ~4 % of the total skin area.
    # Larger blobs are continuous redness zones (cheek flush, lips, shadows).
    _MAX_AREA_SKIN_PCT = 0.04   # area / skin_pixel_count    > 4 %    → reject
    # How many LAB a* units a blob must be ABOVE its immediate surroundings.
    # Faint scars/hyperpigmentation blend gradually into surrounding skin;
    # active inflamed acne has a sharp, localised redness spike.
    _MIN_LOCAL_CONTRAST = 1.5   # mean_blob_a* - mean_surround_a* < 1.5 → reject

    def extract(self, img, skin_mask):
        """
        Returns a 42-dim feature vector.

        Key structural change vs previous version
        ------------------------------------------
        1. Shape filter on connected components: blobs that are too elongated
           (aspect_ratio > 3.5) or too hollow (fill_ratio < 0.25) are discarded
           before counting — this removes skin-fold / wrinkle false positives
           that inflate lesion_count in acne0/1 images.

        2. Global skin redness features (mu_a, std_a, mu_cr, std_cr): these
           capture the *overall* redness of the skin region, independent of
           the adaptive lesion threshold.  In acne0 the whole face is neutral;
           in acne3 the whole face is inflamed — this signal was entirely
           absent before and is the primary separator for severe classes.

        3. mean_circularity of surviving lesions: acne3 cysts are rounder than
           acne1 papules / acne0 false-positive folds.
        """
        if img is None or skin_mask is None:
            return np.zeros(self.FEATURE_DIM)

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab      = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1]

        ycrcb      = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        cr_channel = ycrcb[:, :, 1]

        # ── global skin statistics ───────────────────────────────────────
        # Computed BEFORE lesion segmentation — these reflect the ambient
        # redness of the whole face, not just detected lesion blobs.
        a_skin  = a_channel[skin_mask == 255]
        cr_skin = cr_channel[skin_mask == 255]
        skin_pixel_count = len(a_skin)

        if skin_pixel_count == 0:
            return np.zeros(self.FEATURE_DIM)

        mu_a   = float(np.mean(a_skin))
        std_a  = float(np.std(a_skin))
        mu_cr  = float(np.mean(cr_skin))
        std_cr = float(np.std(cr_skin))

        # ── adaptive lesion segmentation ─────────────────────────────────
        thr_cr = max(145, mu_cr + 1.5 * std_cr)
        thr_a  = max(133, mu_a  + 1.2 * std_a)

        _, mask_cr = cv2.threshold(cr_channel, thr_cr, 255, cv2.THRESH_BINARY)
        _, mask_a  = cv2.threshold(a_channel,  thr_a,  255, cv2.THRESH_BINARY)
        # AND instead of OR: require both channels elevated simultaneously.
        # Active inflamed acne (haemoglobin-rich) is elevated in both Cr and
        # a*.  Post-inflammatory hyperpigmentation (brown scars) and lip
        # redness tend to be elevated in only ONE channel — those are filtered.
        red_mask   = cv2.bitwise_and(cv2.bitwise_and(mask_cr, mask_a), skin_mask)

        red_mask = cv2.morphologyEx(
            red_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        red_mask = cv2.morphologyEx(
            red_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        # ── connected components + shape filter ──────────────────────────
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask)
        min_area = max(20, int(0.0001 * skin_pixel_count))

        lesion_count       = 0
        lesion_areas       = []
        lesion_circularities = []
        localized_intensities = []

        max_area   = int(self._MAX_AREA_SKIN_PCT * skin_pixel_count)
        surr_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            # ── size filter: upper bound ──────────────────────────────────
            # Blobs larger than 4 % of the skin area are diffuse redness
            # zones (cheek flush, lip region) not individual acne lesions.
            if area > max_area:
                continue

            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            # ── shape filter 1: aspect ratio ─────────────────────────────
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > self._MAX_ASPECT_RATIO:
                continue

            # ── shape filter 2: bounding-box fill ratio ───────────────────
            fill = area / max(bw * bh, 1)
            if fill < self._MIN_FILL_RATIO:
                continue

            # ── local contrast check ──────────────────────────────────────
            # Compare the blob's mean a* to an annular ring around it.
            # Faint scars / hyperpigmentation blend gradually into surrounding
            # skin (low local contrast); active acne has a sharp redness spike.
            comp_mask = (labels == i).astype(np.uint8) * 255
            dilated   = cv2.dilate(comp_mask, surr_kernel, iterations=1)
            surround  = cv2.bitwise_and(
                dilated - comp_mask,
                (skin_mask > 0).astype(np.uint8) * 255,
            )
            blob_a_vals = a_channel[comp_mask > 0]
            surr_a_vals = a_channel[surround  > 0]
            if len(surr_a_vals) >= 10:
                local_contrast = float(np.mean(blob_a_vals)) - float(np.mean(surr_a_vals))
                if local_contrast < self._MIN_LOCAL_CONTRAST:
                    continue   # not significantly redder than surroundings

            # ── circularity (4π·A / P²) ───────────────────────────────────
            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circularity = 0.0
            if contours:
                perim = cv2.arcLength(contours[0], True)
                if perim > 0:
                    circularity = min(1.0, 4 * np.pi * area / (perim ** 2))

            lesion_count += 1
            lesion_areas.append(float(area))
            lesion_circularities.append(circularity)

            pixels = a_channel[labels == i]
            if len(pixels) > 0:
                localized_intensities.append(float(np.mean(pixels)))

        # ── structural features ───────────────────────────────────────────
        # intensity_max removed: equals intensity_mean when lesion_count==1
        #   (common in acne0/1), so it's redundant for most samples.
        # area_min removed: always near the min_area filter threshold (~20px),
        #   giving near-zero variance across the dataset.
        total_area       = float(sum(lesion_areas))        if lesion_areas            else 0.0
        intensity_mean   = float(np.mean(localized_intensities)) if localized_intensities  else 0.0
        intensity_std    = float(np.std(localized_intensities))  if len(localized_intensities) > 1 else 0.0
        area_max         = float(max(lesion_areas))        if lesion_areas            else 0.0
        area_std         = float(np.std(lesion_areas))     if len(lesion_areas) > 1   else 0.0
        lesion_density   = float(lesion_count) / skin_pixel_count
        mean_circularity = float(np.mean(lesion_circularities)) if lesion_circularities else 0.0

        base_features = [
            float(lesion_count), total_area,
            intensity_mean, intensity_std,
            area_max, area_std,
            lesion_density, mean_circularity,
        ]

        # ── multi-scale LBP (skin pixels only) ───────────────────────────
        # Last bin of each normalized histogram is dropped: it is fully
        # determined by the other 9 (sum-to-1 constraint), so it carries
        # zero additional information and only adds noise.
        lbp_features = []
        for r in self.R_SCALES:
            lbp      = local_binary_pattern(gray, self.P, r, method="uniform")
            lbp_skin = lbp[skin_mask == 255]
            if len(lbp_skin) > 0:
                hist, _ = np.histogram(
                    lbp_skin,
                    bins=np.arange(0, self.P + 3),
                    range=(0, self.P + 2))
                hist = hist.astype(float)
                hist /= hist.sum() + 1e-7
            else:
                hist = np.zeros(self.LBP_BINS)
            lbp_features.extend(hist[: self.LBP_BINS_USED].tolist())  # drop last bin

        # ── GLCM texture ──────────────────────────────────────────────────
        glcm_features = self._compute_glcm(gray, skin_mask)

        # ── global skin redness (appended last) ───────────────────────────
        # mu_a / mu_cr capture the *ambient* inflammatory level of the whole
        # face — this is the signal the adaptive threshold was masking out.
        # HSV saturation removed: redundant with LAB a* and YCrCb Cr which are
        # more specific to redness and already cover the same color information.
        global_skin = [mu_a, std_a, mu_cr, std_cr]

        return np.array(
            base_features + lbp_features + glcm_features + global_skin,
            dtype=float,
        )

    # ── GLCM helper ───────────────────────────────────────────────────────
    def _compute_glcm(self, gray, skin_mask):
        rows = np.any(skin_mask == 255, axis=1)
        cols = np.any(skin_mask == 255, axis=0)
        if not rows.any() or not cols.any():
            return [0.0, 0.0, 0.0]

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        gray_crop        = gray[rmin:rmax + 1, cmin:cmax + 1].copy()
        mask_crop        = skin_mask[rmin:rmax + 1, cmin:cmax + 1]
        gray_crop[mask_crop != 255] = 0

        gray_q = (gray_crop // 4).astype(np.uint8)
        glcm   = graycomatrix(
            gray_q,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=64, symmetric=True, normed=True)

        # dissimilarity removed: ~0.95 correlation with contrast (both measure
        # local intensity variation with different weighting — Σ(i-j)² vs Σ|i-j|).
        return [
            float(graycoprops(glcm, 'contrast').mean()),
            float(graycoprops(glcm, 'homogeneity').mean()),
            float(graycoprops(glcm, 'energy').mean()),
        ]
