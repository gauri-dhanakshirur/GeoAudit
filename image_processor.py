import cv2
import numpy as np
from skimage.morphology import skeletonize

def occlusion_aware_clean(raw_mask):
    # Thresholding to binary
    binary = (raw_mask > 0.3).astype(np.uint8) * 255
    
    # Dilation to bridge small gaps and identify road interiors
    kernel_dil = np.ones((3,3), np.uint8)
    thick = cv2.dilate(binary, kernel_dil, iterations=1)
    
    # Morphological closing to smooth parts not visible due to occlusion
    kernel_close = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, kernel_close)
    
    # Opening to remove noise
    kernel_open = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

def get_skeleton(cleaned_mask):
    # Ensures topological correctness for vector conversion
    return skeletonize(cleaned_mask > 0).astype(np.uint8) * 255