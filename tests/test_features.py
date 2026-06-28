import numpy as np
import cv2
import pytest
from src.features import FeatureExtractor

def test_feature_vector_shape():
    # Create a dummy image and skin_mask
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    skin_mask = np.zeros((512, 512), dtype=np.uint8)
    
    # Simulate some skin area
    skin_mask[100:400, 100:400] = 255
    
    extractor = FeatureExtractor()
    features = extractor.extract(img, skin_mask)
    
    assert isinstance(features, np.ndarray)
    assert len(features) == FeatureExtractor.FEATURE_DIM
