"""Unit tests for preprocessing: Resizing and Logarithmic Illumination Enhancement."""

import numpy as np
import pytest

from src.preprocessing import (
    apply_logarithmic_enhancement,
    resize_image
)


def test_resize_image():
    dummy = np.zeros((100, 150, 3), dtype=np.uint8)
    resized = resize_image(dummy, target_size=(224, 224))
    assert resized.shape == (224, 224, 3), f"Expected (224, 224, 3), got {resized.shape}"


def test_logarithmic_enhancement_mathematical_fidelity():
    # Value 0 should map to 2 * ln(1 + 0) = 0
    zero_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    enh_zero = apply_logarithmic_enhancement(zero_arr, c=2.0)
    assert np.allclose(enh_zero, 0.0), f"Expected 0.0, got max {enh_zero.max()}"

    # Value 255 should map to 2 * ln(1 + 1) = 2 * ln(2) ≈ 1.386294
    max_arr = np.full((10, 10, 3), 255, dtype=np.uint8)
    enh_max = apply_logarithmic_enhancement(max_arr, c=2.0)
    expected_max = 2.0 * np.log(2.0)
    assert np.allclose(enh_max, expected_max, atol=1e-5), f"Expected {expected_max}, got {enh_max.max()}"


def test_logarithmic_enhancement_monotonicity():
    # Grayscale ramp from 0 to 255
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
    ramp_rgb = np.repeat(ramp, 3, axis=2)
    enhanced = apply_logarithmic_enhancement(ramp_rgb, c=2.0)[:, :, 0].flatten()

    # Verify strictly monotonic increasing
    diffs = np.diff(enhanced)
    assert np.all(diffs > 0), "Enhancement function must be strictly monotonically increasing"
    # Verify contrast expansion for low intensities: diffs should be larger at lower intensities
    assert diffs[0] > diffs[-1], "Log transformation must expand lower intensities more than higher intensities"

