"""Unit and integration tests for single-image inference pipeline."""

import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.infer import (
    ALLOWED_EXTENSIONS,
    DISEASE_METADATA,
    load_and_validate_rgb,
    predict_rice_disease,
    validate_image_file
)


@pytest.fixture
def real_test_image():
    """Return path to an existing untouched test set image."""
    path = "Data/Raw/Data - 1/Test_Data_SVM/blast/blast_30_output.jpg"
    assert os.path.isfile(path), f"Test fixture image not found: {path}"
    return path


def test_allowed_extensions():
    assert ".jpg" in ALLOWED_EXTENSIONS
    assert ".jpeg" in ALLOWED_EXTENSIONS
    assert ".png" in ALLOWED_EXTENSIONS


def test_disease_metadata_integrity():
    expected_classes = {"BL", "BLB", "BS", "SB", "TU"}
    assert set(DISEASE_METADATA.keys()) == expected_classes
    for cls, meta in DISEASE_METADATA.items():
        assert "full_name" in meta
        assert "scientific_name" in meta
        assert "pathogen_type" in meta
        assert "description" in meta


def test_validate_image_file_nonexistent():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_image_file("nonexistent_rice_image.jpg")


def test_validate_image_file_unsupported_ext(tmp_path):
    bad_file = tmp_path / "sample.bmp"
    bad_file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        validate_image_file(str(bad_file))


def test_load_and_validate_grayscale_rejected(tmp_path):
    gray_img = np.zeros((100, 100), dtype=np.uint8)
    gray_path = str(tmp_path / "gray.png")
    cv2.imwrite(gray_path, gray_img)
    
    with pytest.raises(ValueError, match="single-channel grayscale"):
        load_and_validate_rgb(gray_path)


def test_load_and_validate_tiny_image_rejected(tmp_path):
    tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
    tiny_path = str(tmp_path / "tiny.png")
    cv2.imwrite(tiny_path, tiny_img)
    
    with pytest.raises(ValueError, match="too small"):
        load_and_validate_rgb(tiny_path)


def test_successful_inference_on_real_image(real_test_image):
    result = predict_rice_disease(
        image_path=real_test_image,
        model_path="outputs/final_model/rice_svm_model.pkl"
    )
    
    # Check structure
    assert result["image_path"] == real_test_image
    assert result["predicted_class"] in DISEASE_METADATA
    assert result["full_disease_name"] == DISEASE_METADATA[result["predicted_class"]]["full_name"]
    assert result["feature_dimension"] == 7168
    assert result["mclbp_level"] == 3
    assert result["model_kernel"] == "rbf"
    
    # Check decision scores
    scores = result["decision_scores"]
    assert set(scores.keys()) == {"BL", "BLB", "BS", "SB", "TU"}
    for sc in scores.values():
        assert isinstance(sc, float)
        assert not np.isnan(sc)
        
    # Check ranking
    ranked = result["ranked_classes"]
    assert len(ranked) == 5
    assert ranked[0]["class"] == result["predicted_class"]
    assert ranked[0]["rank"] == 1
    # Check scores are sorted descending
    for i in range(len(ranked) - 1):
        assert ranked[i]["decision_score"] >= ranked[i + 1]["decision_score"]

