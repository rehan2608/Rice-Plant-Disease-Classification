"""Handcrafted Feature Extraction: Color Correlogram, Multi-Level MCLBP, and Feature Fusion.
Paper Reference: Section "Feature extraction", Eqs. 9-13 (Pages 9-13).

- Color Correlogram (CC): 16 quantized colors, spatial distance D=1, joint probability matrix (256-D).
- Multi-Level MCLBP: 3x3 patch (P=8, R=1), 256 bins per channel (R, G, B), L x L grid blocks (L^2 * 768-D).
- Feature Fusion: Horizontal vector concatenation [CC + MCLBP] yielding 256 + (L^2 * 768) dimensions.
"""

import hashlib
import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


def quantize_16_colors(rgb_img: np.ndarray) -> np.ndarray:
    """Quantize RGB image into 16 discrete color bins matching paper's 16-bin design.
    4 bins on Green, 2 bins on Red, 2 bins on Blue:
    Red: [0..127]->0, [128..255]->1 (2 levels)
    Green: [0..63]->0, [64..127]->1, [128..191]->2, [192..255]->3 (4 levels)
    Blue: [0..127]->0, [128..255]->1 (2 levels)
    Resulting index in [0, 15].
    """
    if rgb_img.dtype != np.uint8 and rgb_img.max() <= 1.5:
        # If float normalized or enhanced, scale to [0, 255]
        scaled = np.clip(rgb_img * (255.0 / max(1e-5, rgb_img.max())), 0, 255).astype(np.uint8)
    else:
        scaled = np.clip(rgb_img, 0, 255).astype(np.uint8)

    r = (scaled[:, :, 0] // 128).astype(np.int32)
    g = (scaled[:, :, 1] // 64).astype(np.int32)
    b = (scaled[:, :, 2] // 128).astype(np.int32)
    
    color_map = (r * 8 + g * 2 + b).astype(np.int32)
    return color_map


def compute_color_correlogram(
    rgb_img: np.ndarray,
    distance: int = 1,
    normalize: bool = True
) -> np.ndarray:
    """Compute Color Correlogram table according to Eq. 9.
    Indexed by color pairs (c_i, c_j) at spatial distance D.
    
    Returns:
        1D array of shape (256,) representing the 16x16 joint co-occurrence probabilities.
    """
    c_map = quantize_16_colors(rgb_img)
    h, w = c_map.shape
    d = distance

    # 8-connected neighborhood at distance d (Chebyshev distance)
    shifts = [
        (-d, -d), (-d, 0), (-d, d),
        (0, -d),           (0, d),
        (d, -d),  (d, 0),  (d, d)
    ]
    
    co_occurrence = np.zeros(256, dtype=np.float64)
    
    for dy, dx in shifts:
        y1_s = max(0, -dy); y1_e = min(h, h - dy)
        x1_s = max(0, -dx); x1_e = min(w, w - dx)
        y2_s = max(0, dy);  y2_e = min(h, h + dy)
        x2_s = max(0, dx);  x2_e = min(w, w + dx)
        
        c1 = c_map[y1_s:y1_e, x1_s:x1_e].ravel()
        c2 = c_map[y2_s:y2_e, x2_s:x2_e].ravel()
        
        pairs = c1 * 16 + c2
        counts = np.bincount(pairs, minlength=256)
        co_occurrence += counts

    if normalize:
        total = co_occurrence.sum()
        if total > 0:
            co_occurrence = co_occurrence / total

    return co_occurrence.astype(np.float32)


def compute_channel_lbp(channel: np.ndarray) -> np.ndarray:
    """Compute classical 3x3 LBP with P=8, R=1 on a single 2D channel array (Eq. 11)."""
    h, w = channel.shape
    center = channel[1:-1, 1:-1]
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    
    # 8-neighbor offsets and corresponding bit weights 2^p
    # Order: (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)
    shifts = [
        (-1, -1, 0), (-1, 0, 1), (-1, 1, 2), (0, 1, 3),
        (1, 1, 4),   (1, 0, 5),  (1, -1, 6), (0, -1, 7)
    ]
    
    for dy, dx, bit in shifts:
        neigh = channel[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        lbp |= ((neigh >= center).astype(np.uint8) << bit)
        
    return lbp


def extract_mclbp_multilevel(
    img_rgb: np.ndarray,
    level: int = 5,
    l1_normalize_blocks: bool = True
) -> np.ndarray:
    """Extract Multi-Level Multi-Channel Local Binary Pattern (MCLBP) features.
    
    Paper Reference:
    - Eq. 10: Decomposes image into N = L^2 equal blocks.
    - Eq. 11: 256-bin LBP histogram per channel.
    - Eq. 12: Concatenates across Red, Green, Blue channels (768 features per block).
    
    Args:
        img_rgb: RGB image (uint8 or float) with shape (H, W, 3).
        level: Multi-level grid count L (yields L^2 blocks).
        l1_normalize_blocks: Whether each block's 768-D histogram is L1 normalized.
        
    Returns:
        1D array of shape (L^2 * 768,).
    """
    if level < 1:
        raise ValueError(f"MCLBP level must be >= 1, got {level}")
        
    h, w, c = img_rgb.shape
    block_h = h // level
    block_w = w // level
    
    # Precompute full LBP map for all 3 color channels
    lbp_maps = []
    for ch in range(3):
        channel_data = img_rgb[:, :, ch]
        lbp_map = compute_channel_lbp(channel_data)
        lbp_maps.append(lbp_map)
        
    # Valid inner LBP region has shape (h-2, w-2)
    inner_h, inner_w = h - 2, w - 2
    
    block_descriptors = []
    for bi in range(level):
        for bj in range(level):
            y_s = max(0, bi * block_h - 1)
            y_e = min(inner_h, (bi + 1) * block_h - 1)
            x_s = max(0, bj * block_w - 1)
            x_e = min(inner_w, (bj + 1) * block_w - 1)
            
            ch_hists = []
            for ch in range(3):
                sub_lbp = lbp_maps[ch][y_s:y_e, x_s:x_e]
                hist, _ = np.histogram(sub_lbp, bins=256, range=(0, 256))
                hist = hist.astype(np.float64)
                ch_hists.append(hist)
                
            block_vec = np.concatenate(ch_hists)  # Shape (768,)
            if l1_normalize_blocks:
                total = block_vec.sum()
                if total > 0:
                    block_vec = block_vec / total
            block_descriptors.append(block_vec)
            
    mclbp_feature_vector = np.concatenate(block_descriptors)
    return mclbp_feature_vector.astype(np.float32)


def fuse_features(cc_vec: np.ndarray, mclbp_vec: np.ndarray) -> np.ndarray:
    """Concatenate CC and MCLBP feature vectors matching Eq. 13 of the paper.
    
    FE = [gamma(D) + LBP(RGB)]
    
    Returns:
        1D array of shape (256 + L^2 * 768,).
    """
    return np.concatenate([cc_vec, mclbp_vec]).astype(np.float32)


def extract_fused_features_for_image(
    image_rgb: np.ndarray,
    level: int = 5,
    cc_distance: int = 1,
    l1_normalize_blocks: bool = True
) -> np.ndarray:
    """End-to-end extraction and fusion of CC and MCLBP features for a single image."""
    cc = compute_color_correlogram(image_rgb, distance=cc_distance, normalize=True)
    mclbp = extract_mclbp_multilevel(image_rgb, level=level, l1_normalize_blocks=l1_normalize_blocks)
    fused = fuse_features(cc, mclbp)
    return fused


def get_feature_cache_path(
    cache_dir: str,
    image_path: str,
    level: int,
    cc_distance: int = 1,
    c_const: float = 2.0
) -> str:
    """Generate deterministic cache file path based on image content & parameters."""
    with open(image_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    key = f"{file_hash}_L{level}_D{cc_distance}_c{c_const:.1f}.npy"
    return os.path.join(cache_dir, key)

