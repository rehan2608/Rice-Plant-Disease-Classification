"""Evaluation metrics, confusion matrix, and reporting.
Paper Reference: Section "Performance evaluation metrics", Eqs. 17-22 (Pages 14-15).
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


def compute_binary_confusion_components(
    cm: np.ndarray,
    class_idx: int
) -> Tuple[int, int, int, int]:
    """Extract TP, FP, FN, TN for a specific class from a multiclass confusion matrix."""
    tp = cm[class_idx, class_idx]
    fp = cm[:, class_idx].sum() - tp
    fn = cm[class_idx, :].sum() - tp
    tn = cm.sum() - (tp + fp + fn)
    return int(tp), int(fp), int(fn), int(tn)


def compute_mcc_binary(tp: int, fp: int, fn: int, tn: int) -> float:
    """Compute Matthews Correlation Coefficient matching Eq. 22."""
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0.0:
        return 0.0
    return float(num / den)


def compute_all_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    class_names: List[str]
) -> Dict:
    """Compute complete set of evaluation metrics matching Eqs. 17-22 of the paper."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    macro_prec = float(precision_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0))
    
    weighted_prec = float(precision_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    per_class = {}
    specificities = []
    mccs = []
    
    for idx, cls in enumerate(class_names):
        tp, fp, fn, tn = compute_binary_confusion_components(cm, idx)
        
        # Eq. 18 Precision
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        # Eq. 19 Recall / Sensitivity
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        # Eq. 20 F1-Score
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        # Eq. 21 Specificity (TN / (TN + FP))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        # Eq. 22 MCC
        mcc = compute_mcc_binary(tp, fp, fn, tn)
        
        specificities.append(spec)
        mccs.append(mcc)
        
        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": prec * 100.0,
            "recall": rec * 100.0,
            "f1_score": f1 * 100.0,
            "specificity": spec * 100.0,
            "mcc": mcc * 100.0,
            "support": int(tp + fn)
        }
        
    macro_specificity = float(np.mean(specificities))
    macro_mcc = float(np.mean(mccs))
    
    metrics = {
        "accuracy": acc * 100.0,
        "balanced_accuracy": bal_acc * 100.0,
        "macro_precision": macro_prec * 100.0,
        "macro_recall": macro_rec * 100.0,
        "macro_f1": macro_f1 * 100.0,
        "macro_specificity": macro_specificity * 100.0,
        "macro_mcc": macro_mcc * 100.0,
        "weighted_precision": weighted_prec * 100.0,
        "weighted_recall": weighted_rec * 100.0,
        "weighted_f1": weighted_f1 * 100.0,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": class_names
    }
    return metrics


def save_evaluation_report(
    metrics: Dict,
    output_dir: str,
    prefix: str = "eval"
) -> str:
    """Save metrics JSON and readable markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save JSON
    json_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    # 2. Save Markdown Report
    md_path = os.path.join(output_dir, f"{prefix}_report.md")
    lines = [
        f"# Evaluation Report: {prefix.upper()}",
        "",
        "## Overall Summary Metrics",
        "",
        f"- **Accuracy**: {metrics['accuracy']:.2f}%",
        f"- **Balanced Accuracy**: {metrics['balanced_accuracy']:.2f}%",
        f"- **Macro F1-Score**: {metrics['macro_f1']:.2f}%",
        f"- **Macro Precision**: {metrics['macro_precision']:.2f}%",
        f"- **Macro Recall (Sensitivity)**: {metrics['macro_recall']:.2f}%",
        f"- **Macro Specificity**: {metrics['macro_specificity']:.2f}%",
        f"- **Macro MCC**: {metrics['macro_mcc']:.2f}%",
        "",
        "## Per-Class Performance Breakdown",
        "",
        "| Class | Support | Precision (%) | Recall (%) | F1-Score (%) | Specificity (%) | MCC (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for cls, stat in metrics["per_class"].items():
        lines.append(
            f"| **{cls}** | {stat['support']} | {stat['precision']:.2f} | {stat['recall']:.2f} | "
            f"{stat['f1_score']:.2f} | {stat['specificity']:.2f} | {stat['mcc']:.2f} |"
        )
        
    lines.extend([
        "",
        "## Confusion Matrix",
        "",
        f"Row = True Label, Column = Predicted Label. Classes: {metrics['classes']}",
        "",
        "```"
    ])
    
    cm = np.array(metrics["confusion_matrix"])
    lines.append(str(cm))
    lines.extend(["```", ""])
    
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
        
    return md_path

