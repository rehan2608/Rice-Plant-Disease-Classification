"""Dataset loading, path resolution, and integrity verification.
Preserves strict-clean manifests and enforces zero cross-partition leakage.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


EXPECTED_CLASSES = sorted(["BL", "BLB", "BS", "SB", "TU"])
EXPECTED_COUNTS = {
    "train": 4642,
    "val": 995,
    "test": 995,
    "total": 6632
}
EXPECTED_TRAIN_CLASS_COUNTS = {
    "BL": 390,
    "BLB": 1320,
    "BS": 1232,
    "SB": 392,
    "TU": 1308
}


def resolve_image_path(relative_path: str, data_raw_root: str = "Data/Raw") -> str:
    """Normalize path separators and resolve relative path against data raw root."""
    normalized_rel = relative_path.replace("\\", "/").strip()
    full_path = os.path.join(data_raw_root, normalized_rel)
    return os.path.normpath(full_path)


def load_manifest(manifest_path: str, data_raw_root: str = "Data/Raw") -> pd.DataFrame:
    """Load a manifest CSV, resolve filepaths, and validate structure."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    
    required_cols = ["relative_path", "canonical_label", "sha256", "leakage_group_id"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Manifest {manifest_path} missing required column: {col}")

    # Resolve local physical path
    df["resolved_path"] = df["relative_path"].apply(lambda p: resolve_image_path(p, data_raw_root))
    
    # Validate labels
    invalid_labels = set(df["canonical_label"]) - set(EXPECTED_CLASSES)
    if invalid_labels:
        raise ValueError(f"Manifest contains unrecognized class labels: {invalid_labels}")

    return df


def verify_file_existence(df: pd.DataFrame) -> Tuple[int, List[str]]:
    """Check that all resolved image files exist on disk."""
    missing = []
    for path in df["resolved_path"]:
        if not os.path.isfile(path):
            missing.append(path)
    return len(df) - len(missing), missing


def validate_dataset_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, bool]:
    """Run comprehensive integrity checks across partitions."""
    checks = {}

    # 1. Image counts
    checks["train_count_valid"] = (len(train_df) == EXPECTED_COUNTS["train"])
    checks["val_count_valid"] = (len(val_df) == EXPECTED_COUNTS["val"])
    checks["test_count_valid"] = (len(test_df) == EXPECTED_COUNTS["test"])
    checks["total_count_valid"] = (len(train_df) + len(val_df) + len(test_df) == EXPECTED_COUNTS["total"])

    # 2. Train class counts
    actual_train_counts = train_df["canonical_label"].value_counts().to_dict()
    checks["train_class_counts_valid"] = (actual_train_counts == EXPECTED_TRAIN_CLASS_COUNTS)

    # 3. Cryptographic hash isolation (no exact duplicates across splits)
    train_sha = set(train_df["sha256"])
    val_sha = set(val_df["sha256"])
    test_sha = set(test_df["sha256"])
    checks["no_sha_train_val"] = (len(train_sha & val_sha) == 0)
    checks["no_sha_train_test"] = (len(train_sha & test_sha) == 0)
    checks["no_sha_val_test"] = (len(val_sha & test_sha) == 0)

    # 4. Leakage group isolation (no near duplicates across splits)
    train_groups = set(train_df["leakage_group_id"].dropna())
    val_groups = set(val_df["leakage_group_id"].dropna())
    test_groups = set(test_df["leakage_group_id"].dropna())
    checks["no_group_train_val"] = (len(train_groups & val_groups) == 0)
    checks["no_group_train_test"] = (len(train_groups & test_groups) == 0)
    checks["no_group_val_test"] = (len(val_groups & test_groups) == 0)

    # 5. Physical file existence
    _, train_missing = verify_file_existence(train_df)
    _, val_missing = verify_file_existence(val_df)
    _, test_missing = verify_file_existence(test_df)
    checks["all_train_files_exist"] = (len(train_missing) == 0)
    checks["all_val_files_exist"] = (len(val_missing) == 0)
    checks["all_test_files_exist"] = (len(test_missing) == 0)

    all_passed = all(checks.values())
    if not all_passed:
        failed = [k for k, v in checks.items() if not v]
        raise AssertionError(f"Dataset integrity checks failed: {failed}")

    return checks

