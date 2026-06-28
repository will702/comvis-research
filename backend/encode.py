"""
Helpers: numpy arrays → base64-encoded PNG strings for the JSON response.
"""
import base64
import io
import cv2
import numpy as np
from PIL import Image


def to_png_b64(arr: np.ndarray) -> str:
    """BGR or grayscale ndarray → base64 PNG string."""
    if arr is None:
        return ""
    if arr.dtype != np.uint8:
        # Normalize floats to [0,255]
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L")
    else:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def colorize_mask(mask: np.ndarray, color_bgr=(0, 200, 100)) -> np.ndarray:
    """Binary mask → colored BGR image (for overlays)."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == 255] = color_bgr[::-1]  # store as BGR → convert to RGB later
    out[mask == 255] = color_bgr
    return out


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color_bgr=(0, 200, 100), alpha=0.45) -> np.ndarray:
    """Blend a binary mask onto a BGR image."""
    out = img_bgr.copy()
    colored = np.zeros_like(img_bgr)
    colored[mask == 255] = color_bgr
    mask3 = (mask == 255)[:, :, np.newaxis]
    out = np.where(mask3, (out * (1 - alpha) + colored * alpha).astype(np.uint8), out)
    return out


def draw_face_box(img_bgr: np.ndarray, face_box) -> np.ndarray:
    """Draw face bounding box + labeled ROI exclusions on a copy of the image."""
    out = img_bgr.copy()
    if face_box is None:
        return out
    fx, fy, fw, fh = face_box
    cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (50, 220, 50), 2)
    cv2.putText(out, "face", (fx, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 1)
    return out


def draw_blobs(img_bgr: np.ndarray, blobs: list) -> np.ndarray:
    """Draw kept (green) and rejected (red) lesion blobs with reason labels."""
    out = img_bgr.copy()
    for b in blobs:
        x, y, w, h = b["bbox"]
        color = (0, 210, 0) if b["kept"] else (0, 60, 220)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
        label = "" if b["kept"] else b.get("reject_reason", "")[:12]
        if label:
            cv2.putText(out, label, (x, max(y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return out


def lbp_hist_to_png(hist: np.ndarray, radius: int, width=280, height=100) -> str:
    """Render a small LBP histogram bar chart as a base64 PNG."""
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 245
    n = len(hist)
    bar_w = width // n
    max_v = hist.max() if hist.max() > 0 else 1.0
    colors = [(99, 102, 241), (236, 72, 153), (16, 185, 129)]  # indigo/pink/emerald
    c = colors[min(radius - 1, 2)]
    for i, v in enumerate(hist):
        bh = int(v / max_v * (height - 20))
        x0 = i * bar_w + 2
        x1 = x0 + bar_w - 3
        y0 = height - 10 - bh
        cv2.rectangle(canvas, (x0, y0), (x1, height - 10), c[::-1], -1)
    cv2.putText(canvas, f"LBP R={radius}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    return to_png_b64(canvas)


def redness_heatmap(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Colormap a single-channel redness image masked to skin pixels."""
    masked = np.zeros_like(channel)
    masked[mask == 255] = channel[mask == 255]
    cm = cv2.applyColorMap(masked, cv2.COLORMAP_JET)
    cm[mask != 255] = [20, 20, 20]
    return cm
