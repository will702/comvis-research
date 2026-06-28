"""
Guard: TracedPipeline at default params must reproduce FeatureExtractor.extract()
within numerical tolerance. Run: .venv/bin/python -m pytest backend/test_pipeline.py -v
"""
import os
import sys
import glob
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.preprocessing import FaceProcessor
from src.features import FeatureExtractor
from backend.pipeline import run as traced_run
from backend.params import DEFAULT_PARAMS

fp = FaceProcessor()
fe = FeatureExtractor()

# Pick 4 test images across classes
_TEST_IMAGES = sorted(
    glob.glob(os.path.join(_ROOT, "data_split_6535", "test", "**", "*.jpg"), recursive=True)
)[:4]


@pytest.mark.parametrize("path", _TEST_IMAGES)
def test_vector_matches_original(path):
    """TracedPipeline(default) ≈ FeatureExtractor.extract() for same image."""
    import cv2
    img = cv2.imread(path)
    assert img is not None, f"Cannot read {path}"

    # Original pipeline
    img_r, mask = fp.preprocess_image(path)
    orig_vec = fe.extract(img_r, mask)

    # Traced pipeline
    trace = traced_run(img, DEFAULT_PARAMS)
    traced_vec = np.array(trace["feature_vector"])

    assert len(traced_vec) == 42, f"Expected 42 dims, got {len(traced_vec)}"
    assert np.allclose(orig_vec, traced_vec, atol=1e-5), (
        f"Mismatch for {os.path.basename(path)}:\n"
        f"  orig  : {orig_vec}\n"
        f"  traced: {traced_vec}\n"
        f"  diff  : {np.abs(orig_vec - traced_vec)}"
    )


def test_vector_length():
    """Sanity: always 42 dims even on a blank image."""
    import cv2
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    trace = traced_run(blank, DEFAULT_PARAMS)
    assert len(trace["feature_vector"]) == 42
