"""Image preprocessing: Resizing to 224x224 and Logarithmic Illumination Enhancement.
Paper Reference: Section "Image preprocessing", Eqs. 2, 7, 8 (Pages 7-9).
"""

import os
from typing import Tuple, Union
import cv2
import numpy as np


def load_image_rgb(image_path: str) -> np.ndarray:
    """Load an image from disk and convert to RGB format."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image to target (height, width) matching Eq. 2 of the paper."""
    h, w = target_size
    # cv2.resize expects (width, height)
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized


def apply_logarithmic_enhancement(
    image: np.ndarray,
    c: float = 2.0
) -> np.ndarray:
    """Apply the paper's logarithmic illumination enhancement.
    
    Eq. 7: DtNorm = Dt / 255
    Eq. 8: DtLog = c * log(1 + DtNorm), with c = 2.0
    
    Args:
        image: RGB uint8 image array with shape (H, W, 3).
        c: Scaling constant (default 2.0 as specified in paper Eq. 8).
        
    Returns:
        Enhanced float32 array in range [0, c * ln(2)] ≈ [0, 1.3863].
    """
    if image.dtype != np.uint8 and image.max() > 1.0:
        norm = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint8:
        norm = image.astype(np.float32) / 255.0
    else:
        norm = image.astype(np.float32)

    norm = np.clip(norm, 0.0, 1.0)
    # Using np.log1p for numerical accuracy: log(1 + x)
    enhanced = c * np.log1p(norm)
    return enhanced.astype(np.float32)


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    c: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Complete image loading, resizing, and logarithmic enhancement pipeline.
    
    Returns:
        (resized_rgb_uint8, enhanced_float32)
    """
    raw_rgb = load_image_rgb(image_path)
    resized = resize_image(raw_rgb, target_size=target_size)
    enhanced = apply_logarithmic_enhancement(resized, c=c)
    return resized, enhanced

