"""Unit tests for fixed augmentation policy: operations, bounds, and training isolation."""

import numpy as np
import pandas as pd
import pytest

from src.augmentation import (
    apply_flip,
    apply_rotation,
    apply_translation,
    generate_balanced_augmentations,
    transform_image
)


def test_augmentation_operations():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Test translation within paper range [-5, 15]
    t = apply_translation(img, dx=10, dy=-3)
    assert t.shape == img.shape
    
    # Test flip
    f = apply_flip(img, horizontal=True, vertical=True)
    assert f.shape == img.shape
    
    # Test rotation within paper range [-10, 10]
    r = apply_rotation(img, angle_deg=7.5)
    assert r.shape == img.shape
    
    # Compound transform
    c = transform_image(img, dx=5, dy=5, angle_deg=-4.0, h_flip=True, v_flip=False)
    assert c.shape == img.shape


def test_augmentation_lineage_and_bounds():
    dummy_df = pd.DataFrame([
        {
            "file_name": "sample1.jpg",
            "resolved_path": "nonexistent.jpg",
            "canonical_label": "BL",
            "leakage_group_id": "grp_100",
            "split": "train"
        },
        {
            "file_name": "sample2.jpg",
            "resolved_path": "nonexistent.jpg",
            "canonical_label": "BL",
            "leakage_group_id": "grp_101",
            "split": "train"
        }
    ])
    
    # Request target of 5 samples for BL (needs 3 augmentations)
    aug_df = generate_balanced_augmentations(
        dummy_df,
        target_samples_per_class=5,
        output_dir="Data/Augmented",
        seed=42,
        save_to_disk=False
    )
    
    assert len(aug_df) == 3
    for _, row in aug_df.iterrows():
        assert -10.0 <= row["aug_rotation_deg"] <= 10.0, "Rotation angle out of bounds"
        assert -5 <= row["aug_dx"] <= 15, "dx out of bounds"
        assert -5 <= row["aug_dy"] <= 15, "dy out of bounds"
        assert row["canonical_label"] == "BL"
        assert row["is_augmented"] is True
        assert row["leakage_group_id"] in ["grp_100", "grp_101"], "Must inherit group id from source"

