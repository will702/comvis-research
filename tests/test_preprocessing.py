import cv2
import os
import numpy as np
# pyrefly: ignore [missing-import]
import pytest
from src.preprocessing import FaceProcessor

def test_preprocess_image(tmp_path):
    # Create a dummy image and save it to tmp_path
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    # Give it a generic color
    img[:] = (200, 200, 200) 
    
    dummy_img_path = str(tmp_path / "dummy.jpg")
    cv2.imwrite(dummy_img_path, img)
    
    processor = FaceProcessor()
    img_resized, skin_mask = processor.preprocess_image(dummy_img_path)
    
    assert img_resized is not None
    assert img_resized.shape == (512, 512, 3)
    assert skin_mask is not None
    assert skin_mask.shape == (512, 512)
    assert skin_mask.dtype == np.uint8
