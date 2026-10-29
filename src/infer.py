"""Single-image inference pipeline for rice plant disease classification.
Reproduces the paper's exact feature extraction, scaling, and classification steps:
1. Input validation & loading (JPG, JPEG, PNG, 3-channel color).
2. Resizing to 224x224 & Logarithmic Illumination Enhancement (c=2.0).
3. 256-D Color Correlogram + 6,912-D Level 3 MCLBP feature fusion (7,168-D).
4. Feature standardization via fitted StandardScaler from training.
5. One-Against-All RBF SVM prediction and decision scores.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.features import extract_fused_features_for_image
from src.model import RicePlantClassifier
from src.preprocessing import preprocess_image


# Verified agronomic and taxonomic disease metadata
DISEASE_METADATA = {
    "BL": {
        "full_name": "Blast",
        "scientific_name": "Magnaporthe oryzae (syn. Pyricularia oryzae)",
        "pathogen_type": "Fungus",
        "description": "Fungal disease causing diamond-shaped or spindle lesions with gray centers on rice leaves."
    },
    "BLB": {
        "full_name": "Bacterial Leaf Blight",
        "scientific_name": "Xanthomonas oryzae pv. oryzae",
        "pathogen_type": "Bacterium",
        "description": "Bacterial vascular disease producing yellow-orange wavy lesions progressing down the leaf blades."
    },
    "BS": {
        "full_name": "Brown Spot",
        "scientific_name": "Bipolaris oryzae (syn. Helminthosporium oryzae)",
        "pathogen_type": "Fungus",
        "description": "Fungal disease exhibiting small, oval to circular brown spots with lighter brownish centers."
    },
    "SB": {
        "full_name": "Sheath Blight",
        "scientific_name": "Rhizoctonia solani (teleomorph: Thanatephorus cucumeris)",
        "pathogen_type": "Fungus",
        "description": "Soilborne fungal disease forming snake-skin like, irregular grayish-green water-soaked lesions."
    },
    "TU": {
        "full_name": "Tungro",
        "scientific_name": "Rice tungro bacilliform virus (RTBV) & Rice tungro spherical virus (RTSV)",
        "pathogen_type": "Virus complex",
        "description": "Leafhopper-vectored viral disease causing leaf yellowing, stunting, and reduced tillering."
    }
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Global cache for the loaded model to avoid redundant disk I/O on repeated calls
_CACHED_MODEL: Optional[RicePlantClassifier] = None
_CACHED_MODEL_PATH: Optional[str] = None


def get_model(model_path: str = "outputs/final_model/rice_svm_model.pkl") -> RicePlantClassifier:
    """Load and cache the trained RicePlantClassifier artifact."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    if _CACHED_MODEL is None or _CACHED_MODEL_PATH != model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Trained model artifact not found at: {model_path}")
        _CACHED_MODEL = RicePlantClassifier.load(model_path)
        _CACHED_MODEL_PATH = model_path
    return _CACHED_MODEL


def validate_image_file(image_path: str) -> Path:
    """Validate that the input image path exists, has supported format, and is readable."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image file does not exist: {image_path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a regular file: {image_path}")
        
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported formats: {sorted(list(ALLOWED_EXTENSIONS))}"
        )
        
    return path


def load_and_validate_rgb(image_path: str) -> np.ndarray:
    """Load image from disk and strictly validate dimensions and color channels."""
    path = validate_image_file(image_path)
    
    # Read image using OpenCV
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"Failed to decode image. File may be corrupted or not a valid image: {image_path}")
        
    # Check channel count
    if bgr.ndim == 2:
        raise ValueError(
            f"Incompatible image format: Image '{path.name}' is single-channel grayscale (2D). "
            f"The pipeline requires 3-channel color (RGB) images."
        )
        
    if bgr.ndim == 3 and bgr.shape[2] == 1:
        raise ValueError(
            f"Incompatible image format: Image '{path.name}' has only 1 channel. "
            f"The pipeline requires 3-channel color (RGB) images."
        )
        
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        # Convert BGRA to BGR
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    elif bgr.ndim == 3 and bgr.shape[2] != 3:
        raise ValueError(
            f"Incompatible image format: Image '{path.name}' has unexpected number of channels: {bgr.shape[2]}."
        )
        
    # Verify dimensions are viable
    if bgr.shape[0] < 16 or bgr.shape[1] < 16:
        raise ValueError(
            f"Image dimensions ({bgr.shape[1]}x{bgr.shape[0]}) are too small. Minimum required: 16x16 pixels."
        )
        
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def predict_rice_disease(
    image_path: str,
    model_path: str = "outputs/final_model/rice_svm_model.pkl",
    level: int = 3,
    cc_distance: int = 1,
    log_c: float = 2.0
) -> Dict:
    """Execute end-to-end inference on a single rice leaf image.
    
    Pipeline Steps:
    1. Validate input path, file integrity, and 3-channel RGB representation.
    2. Resize to 224x224 & apply logarithmic illumination enhancement (c=2.0).
    3. Extract 256-D Color Correlogram features.
    4. Extract 6,912-D Level 3 MCLBP features (3x3 grid * 768 bins/block).
    5. Concatenate to 7,168-D fused feature vector.
    6. Transform using the trained StandardScaler embedded in the saved model.
    7. Predict class and compute per-class SVM decision scores.
    
    Note:
        Decision scores are raw One-Against-All SVM hyperplane margins,
        where higher positive scores indicate stronger separation.
        They are not calibrated probabilities.
        
    Returns:
        Structured dictionary containing predicted class, full disease name,
        pathogen metadata, decision scores, and ranked class list.
    """
    # 1. Validate & load image
    rgb_raw = load_and_validate_rgb(image_path)
    
    # 2. Preprocess: 224x224 resize and logarithmic enhancement
    # (Matches training pipeline: src.preprocessing.preprocess_image)
    resized_rgb, enhanced_img = preprocess_image(
        image_path=image_path,
        target_size=(224, 224),
        c=log_c
    )
    
    # 3-5. Feature extraction & fusion (Matches src.features)
    fused_features = extract_fused_features_for_image(
        image_rgb=enhanced_img,
        level=level,
        cc_distance=cc_distance,
        l1_normalize_blocks=True
    )
    
    # Verify exact feature dimensionality
    expected_dim = 256 + (level ** 2) * 768  # 7,168 for level 3
    if fused_features.shape[0] != expected_dim:
        raise ValueError(
            f"Extracted feature dimension mismatch: Expected {expected_dim}, got {fused_features.shape[0]}"
        )
        
    # 6. Load model and verify scaler
    clf = get_model(model_path)
    if clf.scaler is None or clf.scaler.n_features_in_ != expected_dim:
        raise ValueError(
            f"Model scaler feature dimension mismatch: Expected {expected_dim}, "
            f"got {getattr(clf.scaler, 'n_features_in_', None)}"
        )
        
    # 7. Predict & compute decision function
    X = np.expand_dims(fused_features, axis=0)  # Shape (1, 7168)
    pred_label = str(clf.predict(X)[0])
    raw_scores = clf.decision_function(X)[0]
    
    # Build per-class scores and ranking
    class_names = [str(c) for c in clf.classes_]
    scores_dict = {}
    ranked_list = []
    
    for cls_name, score in zip(class_names, raw_scores):
        sc_val = float(score)
        scores_dict[cls_name] = sc_val
        meta = DISEASE_METADATA.get(cls_name, {})
        ranked_list.append({
            "class": cls_name,
            "full_name": meta.get("full_name", cls_name),
            "decision_score": sc_val
        })
        
    # Sort ranking by decision score descending
    ranked_list.sort(key=lambda item: item["decision_score"], reverse=True)
    for rank_idx, item in enumerate(ranked_list, start=1):
        item["rank"] = rank_idx
        
    winning_meta = DISEASE_METADATA.get(pred_label, {})
    
    result = {
        "image_path": str(image_path),
        "predicted_class": pred_label,
        "full_disease_name": winning_meta.get("full_name", pred_label),
        "scientific_name": winning_meta.get("scientific_name", "Unknown"),
        "pathogen_type": winning_meta.get("pathogen_type", "Unknown"),
        "description": winning_meta.get("description", ""),
        "decision_score": float(scores_dict[pred_label]),
        "decision_scores": scores_dict,
        "ranked_classes": ranked_list,
        "feature_dimension": int(expected_dim),
        "mclbp_level": int(level),
        "model_kernel": str(clf.kernel),
        "note": "Decision scores represent raw One-vs-Rest SVM margin distances (not probabilities)."
    }
    
    return result


def format_text_output(result: Dict) -> str:
    """Format inference results into a clear, readable text report."""
    lines = [
        "=================================================================",
        "RICE PLANT DISEASE INFERENCE RESULT",
        "=================================================================",
        f"Input Image:      {result['image_path']}",
        f"Predicted Class:  {result['predicted_class']} ({result['full_disease_name']})",
        f"Pathogen:         {result['scientific_name']} [{result['pathogen_type']}]",
        f"Top Margin Score: {result['decision_score']:+.4f}",
        f"Feature Vector:   {result['feature_dimension']} dimensions (CC: 256 + L{result['mclbp_level']} MCLBP: 6912)",
        f"Model:            One-vs-Rest SVM ({result['model_kernel'].upper()} kernel)",
        "",
        "Disease Description:",
        f"  {result['description']}",
        "",
        "Ranked Class Margins (Raw SVM Decision Scores):",
        "  Rank | Class | Disease Name            | Decision Score | Status",
        "  -----+-------+-------------------------+----------------+-----------"
    ]
    
    for item in result["ranked_classes"]:
        status = ">>> PREDICTED <<<" if item["class"] == result["predicted_class"] else ""
        lines.append(
            f"   {item['rank']:2d}  | {item['class']:5s} | {item['full_name']:23s} | {item['decision_score']:+14.4f} | {status}"
        )
        
    lines.extend([
        "",
        "Note: Decision scores are uncalibrated geometric margins relative to",
        "      the One-vs-Rest separating hyperplanes. Scores > 0 indicate positive",
        "      class membership; the highest score denotes the predicted disease.",
        "================================================================="
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Predict rice plant disease from a single leaf image (JPG, JPEG, PNG)."
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input rice leaf image file."
    )
    parser.add_argument(
        "--model", "-m",
        default="outputs/final_model/rice_svm_model.pkl",
        help="Path to the trained One-vs-Rest SVM model artifact."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON."
    )
    args = parser.parse_args()
    
    try:
        result = predict_rice_disease(image_path=args.image, model_path=args.model)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_text_output(result))
    except Exception as exc:
        print(f"Error during inference: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

