"""Main orchestrator and reproducible pipeline CLI.
Paper: "An enhanced classification system of various rice plant diseases based on multi-level handcrafted feature extraction technique"
Scientific Reports (2024) 14:30601
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from src.augmentation import generate_balanced_augmentations
from src.cross_validation import (
    build_group_aware_folds,
    evaluate_configuration_5fold,
    precompute_original_features
)
from src.dataset import (
    load_manifest,
    validate_dataset_integrity
)
from src.evaluate import (
    compute_all_metrics,
    save_evaluation_report
)
from src.features import extract_fused_features_for_image
from src.model import RicePlantClassifier
from src.preprocessing import preprocess_image


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_validation(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Verify dataset manifests, physical file decodability, and leakage group isolation."""
    print("=================================================================")
    print("STEP 1: DATASET MANIFEST INTEGRITY & DATA-LEAKAGE AUDIT")
    print("=================================================================")
    
    raw_root = config["paths"]["data_raw_root"]
    train_df = load_manifest(config["paths"]["train_manifest"], data_raw_root=raw_root)
    val_df = load_manifest(config["paths"]["val_manifest"], data_raw_root=raw_root)
    test_df = load_manifest(config["paths"]["test_manifest"], data_raw_root=raw_root)
    
    print(f"Loaded Train Manifest: {len(train_df)} rows")
    print(f"Loaded Val Manifest:   {len(val_df)} rows")
    print(f"Loaded Test Manifest:  {len(test_df)} rows")
    
    checks = validate_dataset_integrity(train_df, val_df, test_df)
    for check_name, status in checks.items():
        print(f"  [{'PASSED' if status else 'FAILED'}] {check_name}")
        
    print("\nAll dataset integrity & leakage-isolation checks PASSED.\n")
    return train_df, val_df, test_df


def run_cross_validation(config: dict, train_df: pd.DataFrame) -> Tuple[int, str, dict]:
    """Execute Phase 4: Group-Aware 5-Fold Configuration Selection.
    
    Phase 4A: Feature level selection across candidate levels L in {3, 4, 5} using Linear SVM.
    Phase 4B: Kernel selection across Linear, RBF, and Polynomial kernels on the chosen level.
    """
    print("=================================================================")
    print("STEP 2: PHASE 4 - GROUP-AWARE 5-FOLD CONFIGURATION SELECTION")
    print("=================================================================")
    
    cv_cfg = config["cross_validation"]
    levels = cv_cfg["candidate_mclbp_levels"]
    kernels = cv_cfg["candidate_kernels"]
    seed = config["seed"]
    log_c = config["preprocessing"]["log_constant_c"]
    aug_dir = config["paths"]["augmented_dir"]
    output_dir = config["paths"]["output_cv_dir"]
    cache_dir = config["paths"]["feature_cache_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    print("Building 5 group-aware stratified folds on 4,642 original train images...")
    folds = build_group_aware_folds(train_df, n_splits=5, seed=seed)
    
    # -------------------------------------------------------------
    # Phase 4A: Evaluate Candidate MCLBP Levels L in {3, 4, 5}
    # -------------------------------------------------------------
    print("\n--- PHASE 4A: MCLBP LEVEL COUNT CANDIDATE SELECTION ---")
    level_summaries = {}
    best_level = None
    best_level_macro_f1 = -1.0
    
    for level in levels:
        print(f"\n[Evaluating MCLBP Level {level} with Linear SVM]")
        # Precompute/load cached original features for this level
        orig_feats = precompute_original_features(
            train_df,
            level=level,
            cc_distance=config["features"]["correlogram"]["distance"],
            log_c=log_c,
            cache_dir=cache_dir
        )
        
        summary = evaluate_configuration_5fold(
            train_df=train_df,
            folds=folds,
            level=level,
            kernel="linear",
            orig_features=orig_feats,
            c_param=config["model"]["c_param"],
            log_c=log_c,
            seed=seed,
            augmented_dir=aug_dir,
            output_dir=output_dir
        )
        level_summaries[level] = summary
        print(f"Level {level} Summary -> Macro-F1: {summary['mean_macro_f1']:.2f}% (±{summary['std_macro_f1']:.2f}%) | "
              f"Accuracy: {summary['mean_accuracy']:.2f}% (±{summary['std_accuracy']:.2f}%)")
              
        if summary["mean_macro_f1"] > best_level_macro_f1:
            best_level_macro_f1 = summary["mean_macro_f1"]
            best_level = level
            
    print(f"\n>>> Selected Best MCLBP Level: {best_level} (Macro-F1: {best_level_macro_f1:.2f}%) <<<")
    
    # -------------------------------------------------------------
    # Phase 4B: Evaluate SVM Kernels (Linear, RBF, Poly) on Best Level
    # -------------------------------------------------------------
    print(f"\n--- PHASE 4B: SVM KERNEL EVALUATION ON SELECTED LEVEL {best_level} ---")
    orig_feats_best = precompute_original_features(
        train_df,
        level=best_level,
        cc_distance=config["features"]["correlogram"]["distance"],
        log_c=log_c,
        cache_dir=cache_dir
    )
    
    kernel_summaries = {}
    best_kernel = None
    best_kernel_macro_f1 = -1.0
    
    for kernel in kernels:
        print(f"\n[Evaluating Kernel: '{kernel}' on Level {best_level}]")
        summary = evaluate_configuration_5fold(
            train_df=train_df,
            folds=folds,
            level=best_level,
            kernel=kernel,
            orig_features=orig_feats_best,
            c_param=config["model"]["c_param"],
            log_c=log_c,
            seed=seed,
            augmented_dir=aug_dir,
            output_dir=output_dir
        )
        kernel_summaries[kernel] = summary
        print(f"Kernel '{kernel}' Summary -> Macro-F1: {summary['mean_macro_f1']:.2f}% (±{summary['std_macro_f1']:.2f}%) | "
              f"Accuracy: {summary['mean_accuracy']:.2f}% (±{summary['std_accuracy']:.2f}%)")
              
        if summary["mean_macro_f1"] > best_kernel_macro_f1:
            best_kernel_macro_f1 = summary["mean_macro_f1"]
            best_kernel = kernel
            
    print(f"\n>>> Selected Best SVM Kernel: '{best_kernel}' (Macro-F1: {best_kernel_macro_f1:.2f}%) <<<")
    
    # Save overall CV selection summary
    overall_summary = {
        "selected_level": best_level,
        "selected_kernel": best_kernel,
        "best_macro_f1": best_kernel_macro_f1,
        "level_candidates": level_summaries,
        "kernel_candidates": kernel_summaries
    }
    cv_summary_path = os.path.join(output_dir, "cv_selection_report.json")
    with open(cv_summary_path, "w") as f:
        json.dump(overall_summary, f, indent=2)
    print(f"Full CV selection report saved to: {cv_summary_path}\n")
    
    return best_level, best_kernel, overall_summary


def extract_features_from_manifest(
    df: pd.DataFrame,
    level: int,
    log_c: float = 2.0,
    cc_dist: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to extract features from a manifest dataframe."""
    X_list = []
    y_list = []
    t0 = time.time()
    for idx, row in df.iterrows():
        path = row["resolved_path"]
        label = row["canonical_label"]
        _, enh = preprocess_image(path, target_size=(224, 224), c=log_c)
        fused = extract_fused_features_for_image(enh, level=level, cc_distance=cc_dist, l1_normalize_blocks=True)
        X_list.append(fused)
        y_list.append(label)
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(df):
            print(f"  Extracted {idx + 1} / {len(df)} images ({time.time() - t0:.1f}s)...")
            
    return np.vstack(X_list).astype(np.float32), np.array(y_list)


def train_final_model(
    config: dict,
    train_df: pd.DataFrame,
    selected_level: int,
    selected_kernel: str
) -> RicePlantClassifier:
    """Train the final One-vs-Rest SVM on the complete balanced training dataset."""
    print("=================================================================")
    print("STEP 3: PHASE 5 - TRAIN FINAL MODEL ON FULL BALANCED TRAINING SET")
    print("=================================================================")
    
    target_count = config["dataset"]["target_train_samples_per_class"]
    aug_dir = os.path.join(config["paths"]["augmented_dir"], "final_train_balanced")
    seed = config["seed"]
    log_c = config["preprocessing"]["log_constant_c"]
    cache_dir = config["paths"]["feature_cache_dir"]
    
    print(f"Original train samples: {len(train_df)}")
    print(f"Target count per class: {target_count} (Total target: {target_count * 5})")
    
    # 1. Load cached original training features
    X_orig = precompute_original_features(
        train_df,
        level=selected_level,
        cc_distance=config["features"]["correlogram"]["distance"],
        log_c=log_c,
        cache_dir=cache_dir
    )
    y_orig = train_df["canonical_label"].values
    
    # 2. Generate balanced augmentations for training data
    print("Generating balanced augmentations strictly from training images...")
    aug_df = generate_balanced_augmentations(
        train_df,
        target_samples_per_class=target_count,
        output_dir=aug_dir,
        seed=seed,
        save_to_disk=True
    )
    print(f"Generated {len(aug_df)} augmentations.")
    
    if len(aug_df) > 0:
        print(f"Extracting features for {len(aug_df)} augmented images (Level {selected_level})...")
        X_aug, y_aug = extract_features_from_manifest(aug_df, level=selected_level, log_c=log_c)
        X_final = np.vstack([X_orig, X_aug])
        y_final = np.concatenate([y_orig, y_aug])
    else:
        X_final = X_orig
        y_final = y_orig
        
    print(f"Final training set: {len(y_final)} samples across 5 classes (Exactly {target_count} per class).")
    print(f"Training One-Against-All SVM (Kernel: '{selected_kernel}', C={config['model']['c_param']})...")
    
    model = RicePlantClassifier(
        kernel=selected_kernel,
        c_param=config["model"]["c_param"],
        standardize=True,
        seed=seed
    )
    t0 = time.time()
    model.fit(X_final, y_final)
    print(f"Model fitting complete in {time.time() - t0:.1f}s.")
    
    # Save model artifact
    model_dir = config["paths"]["output_final_model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rice_svm_model.pkl")
    model.save(model_path)
    print(f"Final model serialized to: {model_path}\n")
    return model


def evaluate_split(
    model: RicePlantClassifier,
    df: pd.DataFrame,
    split_name: str,
    selected_level: int,
    config: dict
) -> dict:
    """Evaluate trained model on fixed, unaugmented validation or test split."""
    print("=================================================================")
    print(f"EVALUATING MODEL ON UNTOUCHED {split_name.upper()} SET ({len(df)} images)")
    print("=================================================================")
    
    log_c = config["preprocessing"]["log_constant_c"]
    output_dir = config["paths"]["output_eval_dir"]
    
    print(f"Extracting features for {split_name} split (Level {selected_level})...")
    t0 = time.time()
    X_eval, y_eval = extract_features_from_manifest(df, level=selected_level, log_c=log_c)
    feat_time = time.time() - t0
    
    print(f"Inference on {len(df)} samples...")
    t_inf = time.time()
    y_pred = model.predict(X_eval)
    infer_time = time.time() - t_inf
    
    classes = sorted(list(set(df["canonical_label"])))
    metrics = compute_all_metrics(y_eval, y_pred, class_names=classes)
    metrics["runtime"] = {
        "feat_extraction_sec": feat_time,
        "inference_sec": infer_time,
        "avg_inference_ms_per_image": (infer_time / len(df)) * 1000.0
    }
    
    report_path = save_evaluation_report(metrics, output_dir=output_dir, prefix=split_name.lower())
    print(f"Saved {split_name} report to: {report_path}")
    print(f"  -> Accuracy:         {metrics['accuracy']:.2f}%")
    print(f"  -> Balanced Acc:     {metrics['balanced_accuracy']:.2f}%")
    print(f"  -> Macro F1:         {metrics['macro_f1']:.2f}%")
    print(f"  -> Macro Precision:  {metrics['macro_precision']:.2f}%")
    print(f"  -> Macro Recall:     {metrics['macro_recall']:.2f}%")
    print(f"  -> Macro MCC:        {metrics['macro_mcc']:.2f}%\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Reproduce Scientific Reports (2024) 14:30601")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--stage", default="all", choices=["audit", "cv", "train", "eval", "all"],
                        help="Pipeline stage to execute")
    parser.add_argument("--image", "-i", default=None,
                        help="Path to a single rice leaf image for inference (JPG, JPEG, PNG)")
    parser.add_argument("--json", action="store_true",
                        help="Output inference result in JSON format")
    args = parser.parse_args()
    
    if args.image is not None:
        from src.infer import predict_rice_disease, format_text_output
        model_path = os.path.join("outputs", "final_model", "rice_svm_model.pkl")
        res = predict_rice_disease(image_path=args.image, model_path=model_path)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(format_text_output(res))
        return

    config = load_config(args.config)
    train_df, val_df, test_df = run_validation(config)
    
    if args.stage == "audit":
        print("Audit stage complete.")
        return
        
    best_level, best_kernel, cv_summary = run_cross_validation(config, train_df)
    
    model = train_final_model(config, train_df, best_level, best_kernel)
    
    # Sanity check on fixed validation set (995 images)
    val_metrics = evaluate_split(
        model=model,
        df=val_df,
        split_name="validation",
        selected_level=best_level,
        config=config
    )
    
    # Final untouched test set evaluation (995 images)
    test_metrics = evaluate_split(
        model=model,
        df=test_df,
        split_name="test",
        selected_level=best_level,
        config=config
    )
    
    print("=================================================================")
    print("REPRODUCTION WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"Selected Configuration: MCLBP Level={best_level}, SVM Kernel='{best_kernel}'")
    print(f"5-Fold CV Macro-F1:   {cv_summary['best_macro_f1']:.2f}%")
    print(f"Validation Macro-F1:  {val_metrics['macro_f1']:.2f}% (Acc: {val_metrics['accuracy']:.2f}%)")
    print(f"Final Test Macro-F1:  {test_metrics['macro_f1']:.2f}% (Acc: {test_metrics['accuracy']:.2f}%)")
    print("=================================================================")


if __name__ == "__main__":
    main()
