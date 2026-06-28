import cv2
import numpy as np
import os
from src.config import BASE_DIR


class FaceProcessor:
    def __init__(self):
        # Primary cascade — general frontal face
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Secondary cascade — alt2 is trained differently and catches faces
        # the default cascade misses (e.g. slightly tilted, close-up, acne skin)
        self.face_cascade_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.nose_cascade = cv2.CascadeClassifier(
            os.path.join(BASE_DIR, 'haarcascade_mcs_nose.xml'))

    # ── face detection with cascade fallback chain ────────────────────────
    def _detect_face(self, gray_clahe):
        """
        Try four detection attempts in order of strictness.
        Returns (x, y, w, h) of the largest detected face, or None.

        Why a chain instead of a single call:
        - minNeighbors=5 + large minSize rejects many valid close-up acne photos.
        - Relaxing gradually lets us catch more faces before giving up entirely.
        - Using a second cascade (alt2) covers faces that the default model
          misses due to skin texture differences caused by heavy acne.
        """
        attempts = [
            # (cascade,               scaleFactor, minNeighbors, minSize)
            (self.face_cascade,      1.05,        4,            (90,  90)),
            (self.face_cascade,      1.10,        3,            (70,  70)),
            (self.face_cascade_alt,  1.05,        4,            (90,  90)),
            (self.face_cascade_alt,  1.10,        3,            (60,  60)),
        ]
        for cascade, scale, neighbors, min_size in attempts:
            faces = cascade.detectMultiScale(
                gray_clahe,
                scaleFactor=scale,
                minNeighbors=neighbors,
                minSize=min_size,
            )
            if len(faces) > 0:
                return tuple(sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0])
        return None

    # ── main pipeline ─────────────────────────────────────────────────────
    def preprocess_image(self, img_path):
        """
        1. Resize to 512×512
        2. CLAHE
        3. ROI mask  — face cascade chain → eye/nose suppression
                      → center-crop fallback if all cascades fail
        4. YCrCb skin segmentation
        """
        img = cv2.imread(img_path)
        if img is None:
            return None, None

        img_resized = cv2.resize(img, (512, 512))

        gray       = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        roi_mask = np.ones((512, 512), dtype=np.uint8) * 255

        face = self._detect_face(gray_clahe)

        if face is not None:
            fx, fy, fw, fh = face
            face_roi = gray_clahe[fy:fy + fh, fx:fx + fw]

            eyes = self.eye_cascade.detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=6)
            nose = self.nose_cascade.detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=6)

            for (ex, ey, ew, eh) in eyes:
                ex += fx;  ey += fy
                roi_mask[max(0, ey - 6) : min(512, ey + eh + 6),
                          max(0, ex - 8) : min(512, ex + ew + 8)] = 0

            for (nx, ny, nw, nh) in nose:
                nx += fx;  ny += fy
                roi_mask[max(0, ny - 5) : min(512, ny + nh + 5),
                          max(0, nx - 5) : min(512, nx + nw + 5)] = 0

            # ── lip / mouth exclusion ─────────────────────────────────────
            lip_y1 = int(fy + 0.68 * fh)
            lip_y2 = int(fy + 0.90 * fh)
            lip_x1 = int(fx + 0.20 * fw)
            lip_x2 = int(fx + 0.80 * fw)
            roi_mask[max(0, lip_y1):min(512, lip_y2),
                     max(0, lip_x1):min(512, lip_x2)] = 0

        # ── YCrCb skin segmentation ───────────────────────────────────────
        ycrcb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(
            ycrcb,
            np.array([0,   133,  77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8),
        )

        skin_mask = cv2.morphologyEx(
            skin_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        skin_mask = cv2.morphologyEx(
            skin_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        return img_resized, cv2.bitwise_and(skin_mask, roi_mask)
