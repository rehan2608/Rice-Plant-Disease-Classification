"""Unit tests for feature extraction: Color Correlogram, MCLBP, and Feature Fusion."""

import numpy as np
import pytest

from src.features import (
    compute_color_correlogram,
    extract_fused_features_for_image,
    extract_mclbp_multilevel,
    fuse_features,
    quantize_16_colors
)


def test_quantize_16_colors():
    img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    cmap = quantize_16_colors(img)
    assert cmap.shape == (50, 50)
    assert cmap.min() >= 0
    assert cmap.max() <= 15


def test_color_correlogram_shape_and_sum():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    cc = compute_color_correlogram(img, distance=1, normalize=True)
    assert cc.shape == (256,), f"Expected shape (256,), got {cc.shape}"
    assert np.isclose(cc.sum(), 1.0, atol=1e-5), f"Correlogram must sum to 1.0, got {cc.sum()}"
    assert np.all(cc >= 0), "Probabilities must be non-negative"
    assert not np.isnan(cc).any(), "Found NaNs in CC"


def test_mclbp_multilevel_dimensions():
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    for level in [1, 2, 3, 4, 5, 6, 7]:
        expected_dim = (level ** 2) * 768
        mclbp = extract_mclbp_multilevel(img, level=level, l1_normalize_blocks=True)
        assert mclbp.shape == (expected_dim,), f"For Level {level}, expected {expected_dim}, got {mclbp.shape[0]}"
        assert not np.isnan(mclbp).any(), f"Found NaNs in MCLBP level {level}"
        assert not np.isinf(mclbp).any(), f"Found Infs in MCLBP level {level}"


def test_feature_fusion_exact_dimensions():
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    expected_fused_dims = {
        1: 256 + 768,        # 1,024
        2: 256 + 3072,       # 3,328
        3: 256 + 6912,       # 7,168
        4: 256 + 12288,      # 12,544
        5: 256 + 19200,      # 19,456
    }
    for level, expected_total in expected_fused_dims.items():
        fused = extract_fused_features_for_image(img, level=level, cc_distance=1)
        assert fused.shape == (expected_total,), f"Expected {expected_total} for Level {level}, got {fused.shape[0]}"
        assert not np.isnan(fused).any()
        assert not np.isinf(fused).any()

