"""Tests for data-leakage safeguards, partition isolation, and cross-validation integrity."""

import os
import pandas as pd
import pytest

from src.cross_validation import build_group_aware_folds
from src.dataset import load_manifest, validate_dataset_integrity


def test_split_manifests_isolation():
    train_df = load_manifest("audit/train_manifest.csv", data_raw_root="Data/Raw")
    val_df = load_manifest("audit/val_manifest.csv", data_raw_root="Data/Raw")
    test_df = load_manifest("audit/test_manifest.csv", data_raw_root="Data/Raw")

    checks = validate_dataset_integrity(train_df, val_df, test_df)
    for check_name, status in checks.items():
        assert status, f"Integrity check failed: {check_name}"


def test_group_aware_cross_validation_folds():
    train_df = load_manifest("audit/train_manifest.csv", data_raw_root="Data/Raw")
    folds = build_group_aware_folds(train_df, n_splits=5, seed=42)
    
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
    
    groups = train_df["leakage_group_id"].values
    labels = train_df["canonical_label"].values
    
    for fold_idx, (t_idx, v_idx) in enumerate(folds):
        # 1. Total samples in train + held-out must equal 4,642
        assert len(t_idx) + len(v_idx) == len(train_df)
        
        # 2. Index sets must be disjoint
        assert len(set(t_idx) & set(v_idx)) == 0
        
        # 3. Leakage groups must be strictly disjoint (zero leakage)
        train_groups = set(groups[t_idx])
        val_groups = set(groups[v_idx])
        assert len(train_groups & val_groups) == 0, f"Fold {fold_idx + 1} has leaky group overlap!"
        
        # 4. Held-out fold must contain all 5 classes
        held_out_classes = set(labels[v_idx])
        assert len(held_out_classes) == 5, f"Fold {fold_idx + 1} missing classes in held-out: {held_out_classes}"

