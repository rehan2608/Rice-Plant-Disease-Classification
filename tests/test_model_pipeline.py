"""Integration test for model fitting, prediction, and metric computation."""

import os
import numpy as np
import pytest

from src.evaluate import compute_all_metrics
from src.model import RicePlantClassifier


def test_model_fit_predict_and_metrics(tmp_path):
    classes = ["BL", "BLB", "BS", "SB", "TU"]
    rng = np.random.RandomState(42)
    
    # 50 samples, 1024 features (level 1 fused: 256 + 768) in [0, 1]
    X_train = rng.uniform(0.0, 1.0, size=(50, 1024)).astype(np.float32)
    y_train = np.array(rng.choice(classes, size=50))
    
    X_val = rng.uniform(0.0, 1.0, size=(20, 1024)).astype(np.float32)
    y_val = np.array(rng.choice(classes, size=20))
    
    # Fit Linear One-vs-Rest SVM
    clf = RicePlantClassifier(kernel="linear", c_param=1.0, standardize=True, seed=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    assert len(y_pred) == 20
    assert set(y_pred).issubset(set(classes))
    
    # Test metrics computation
    metrics = compute_all_metrics(y_val, y_pred, class_names=classes)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert len(metrics["per_class"]) == 5
    
    # Test serialization
    save_file = str(tmp_path / "test_model.pkl")
    clf.save(save_file)
    assert os.path.exists(save_file)
    
    loaded_clf = RicePlantClassifier.load(save_file)
    y_pred_loaded = loaded_clf.predict(X_val)
    assert np.array_equal(y_pred, y_pred_loaded)
