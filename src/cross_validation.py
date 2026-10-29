"""Group-Aware Stratified 5-Fold Cross-Validation for Configuration Selection.
Enforces strict zero-leakage constraints:
- Evaluates strictly on the 4,642 original training images.
- Group-aware partitioning on `leakage_group_id` prevents duplicate/near-duplicate leakage across folds.
- Fixed validation and test sets remain completely outside the CV loop.
- Augmentation is applied exclusively to the fold-training portion.
- Feature scalers are fitted exclusively on the fold-training portion.
- Saves intermediate results after each candidate configuration.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.augmentation import generate_balanced_augmentations
from src.evaluate import compute_all_metrics
from src.features import extract_fused_features_for_image
from src.model import RicePlantClassifier
from src.preprocessing import preprocess_image


def build_group_aware_folds(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build stratified, group-aware cross-validation folds."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros(len(train_df))
    y = train_df["canonical_label"].values
    groups = train_df["leakage_group_id"].values
    
    folds = list(sgkf.split(X_dummy, y, groups))
    
    # Verify group isolation across every fold
    for fold_idx, (t_idx, v_idx) in enumerate(folds):
        t_groups = set(groups[t_idx])
        v_groups = set(groups[v_idx])
        overlap = t_groups & v_groups
        if overlap:
            raise AssertionError(f"Fold {fold_idx + 1} has {len(overlap)} overlapping leakage groups!")
            
    return folds


def precompute_original_features(
    df: pd.DataFrame,
    level: int,
    cc_distance: int = 1,
    log_c: float = 2.0,
    cache_dir: str = "outputs/feature_cache"
) -> np.ndarray:
    """Precompute and cache features for the 4,642 original images deterministically."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"orig_train_L{level}_D{cc_distance}_c{log_c:.1f}.npy")
    
    if os.path.exists(cache_file):
        print(f"Loading cached original features from {cache_file}...")
        return np.load(cache_file)
        
    print(f"Extracting features for {len(df)} original training images (Level {level})...")
    X_list = []
    t0 = time.time()
    for idx, row in df.iterrows():
        path = row["resolved_path"]
        _, enhanced = preprocess_image(path, target_size=(224, 224), c=log_c)
        fused = extract_fused_features_for_image(
            enhanced,
            level=level,
            cc_distance=cc_distance,
            l1_normalize_blocks=True
        )
        X_list.append(fused)
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1} / {len(df)} images ({time.time() - t0:.1f}s)...")
            
    X_mat = np.vstack(X_list).astype(np.float32)
    np.save(cache_file, X_mat)
    print(f"Cached features saved to {cache_file} (shape: {X_mat.shape}).")
    return X_mat


def run_single_fold(
    train_df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    level: int,
    kernel: str,
    orig_features: np.ndarray,
    c_param: float = 1.0,
    log_c: float = 2.0,
    seed: int = 42,
    augmented_dir: str = "Data/Augmented"
) -> Dict:
    """Execute evaluation on a single fold obeying all leakage safeguards."""
    # 1. Isolate fold train and fold held-out
    fold_train_orig = train_df.iloc[train_indices].copy().reset_index(drop=True)
    fold_val = train_df.iloc[val_indices].copy().reset_index(drop=True)
    
    X_train_orig = orig_features[train_indices]
    y_train_orig = fold_train_orig["canonical_label"].values
    
    X_val = orig_features[val_indices]
    y_val = fold_val["canonical_label"].values
    
    # 2. Augment ONLY the fold-training subset to balance classes
    max_count = fold_train_orig["canonical_label"].value_counts().max()
    fold_aug_dir = os.path.join(augmented_dir, f"cv_fold_aug_seed_{seed}")
    
    fold_aug_df = generate_balanced_augmentations(
        fold_train_orig,
        target_samples_per_class=max_count,
        output_dir=fold_aug_dir,
        seed=seed,
        save_to_disk=True
    )
    
    if len(fold_aug_df) > 0:
        # Extract features for generated augmentations
        aug_feats = []
        for idx, row in fold_aug_df.iterrows():
            _, enh = preprocess_image(row["resolved_path"], target_size=(224, 224), c=log_c)
            fused = extract_fused_features_for_image(enh, level=level, cc_distance=1, l1_normalize_blocks=True)
            aug_feats.append(fused)
        X_aug = np.vstack(aug_feats).astype(np.float32)
        y_aug = fold_aug_df["canonical_label"].values
        
        X_train = np.vstack([X_train_orig, X_aug])
        y_train = np.concatenate([y_train_orig, y_aug])
    else:
        X_train = X_train_orig
        y_train = y_train_orig
        
    # 3. Fit One-vs-Rest SVM (scaler is fitted strictly on fold-train features)
    model = RicePlantClassifier(kernel=kernel, c_param=c_param, standardize=True, seed=seed)
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    # 4. Predict on untouched held-out fold
    infer_start = time.time()
    y_pred = model.predict(X_val)
    infer_time = time.time() - infer_start
    
    classes = sorted(list(set(train_df["canonical_label"])))
    metrics = compute_all_metrics(y_val, y_pred, class_names=classes)
    
    return {
        "metrics": metrics,
        "feature_dim": int(X_train.shape[1]),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "train_time_sec": train_time,
        "infer_time_sec": infer_time
    }


def evaluate_configuration_5fold(
    train_df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    level: int,
    kernel: str,
    orig_features: np.ndarray,
    c_param: float = 1.0,
    log_c: float = 2.0,
    seed: int = 42,
    augmented_dir: str = "Data/Augmented",
    output_dir: str = "outputs/cv_results"
) -> Dict:
    """Run full 5-fold cross validation for a specific (level, kernel) configuration.
    Saves intermediate result immediately upon completion.
    """
    os.makedirs(output_dir, exist_ok=True)
    intermediate_path = os.path.join(output_dir, f"intermediate_L{level}_{kernel}.json")
    
    if os.path.exists(intermediate_path):
        print(f"Loading existing intermediate result from {intermediate_path}...")
        with open(intermediate_path, "r") as f:
            return json.load(f)
            
    fold_results = []
    macro_f1s = []
    accuracies = []
    bal_accuracies = []
    
    for fold_idx, (t_idx, v_idx) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/5 (Train: {len(t_idx)}, Val: {len(v_idx)})...")
        res = run_single_fold(
            train_df=train_df,
            train_indices=t_idx,
            val_indices=v_idx,
            level=level,
            kernel=kernel,
            orig_features=orig_features,
            c_param=c_param,
            log_c=log_c,
            seed=seed + fold_idx,
            augmented_dir=augmented_dir
        )
        fold_results.append(res)
        macro_f1s.append(res["metrics"]["macro_f1"])
        accuracies.append(res["metrics"]["accuracy"])
        bal_accuracies.append(res["metrics"]["balanced_accuracy"])
        print(f"    Fold {fold_idx + 1} Result -> Macro-F1: {res['metrics']['macro_f1']:.2f}%, Accuracy: {res['metrics']['accuracy']:.2f}%")
        
    summary = {
        "mclbp_level": level,
        "svm_kernel": kernel,
        "c_param": c_param,
        "feature_dim": fold_results[0]["feature_dim"],
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "std_macro_f1": float(np.std(macro_f1s)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_balanced_accuracy": float(np.mean(bal_accuracies)),
        "std_balanced_accuracy": float(np.std(bal_accuracies)),
        "fold_results": fold_results
    }
    
    # Save intermediate result
    with open(intermediate_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved intermediate result to {intermediate_path}")
    
    return summary
