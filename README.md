# Reproduction: Multi-Level Handcrafted Feature Classification for Rice Diseases

Reproducing:  
*“An enhanced classification system of various rice plant diseases based on multi-level handcrafted feature extraction technique”*  
**Scientific Reports (2024) 14:30601** | [https://doi.org/10.1038/s41598-024-81143-1](https://doi.org/10.1038/s41598-024-81143-1)

---

## 1. Quick Start & Execution

### Environment Setup
```bash
pip install -r requirements.txt
```

### Run Integrity & Property Tests
```bash
pytest -v
```

### Run Full Reproduction Pipeline
```bash
# End-to-end: Audit -> Group-Aware 5-Fold CV -> Final Training -> Val & Test Evaluation
python -m src.main --stage all
```

Or execute individual pipeline stages:
```bash
python -m src.main --stage audit  # Manifest & leakage audit only
```

---

## 2. Single-Image Inference Pipeline

To classify a single rice leaf image (`.jpg`, `.jpeg`, or `.png`):

```bash
# Text output with ranked class margins & disease description
python -m src.infer --image "path/to/rice_leaf.jpg"

# Or via main CLI
python -m src.main --image "path/to/rice_leaf.jpg"

# Machine-readable JSON output
python -m src.infer --image "path/to/rice_leaf.jpg" --json
```

The pipeline automatically:
1. Validates image integrity and 3-channel RGB color format.
2. Applies 224x224 resizing and logarithmic illumination enhancement ($c=2.0$).
3. Extracts the 256-D Color Correlogram and 6,912-D Level 3 MCLBP feature vectors (total: 7,168 dimensions).
4. Standardizes features using the exact `StandardScaler` fitted on training data.
5. Predicts the disease using the trained One-vs-Rest RBF SVM and reports raw hyperplane margin decision scores.

---

## 2. Key Experimental Findings

1. **Configuration Selection**:
   - **MCLBP Level Count**: Candidate evaluation across $L \in \{3, 4, 5\}$ identified **Level 3 ($7,168$ features)** as optimal (**88.97% Macro-F1**, **90.15% Accuracy**). Higher level counts ($L=4, 5$) over-parameterized minority disease classes (`BL` and `SB`).
   - **SVM Kernel Comparison**: On the selected Level 3 features, **RBF kernel** achieved **89.53% Macro-F1** (±0.99%) and **90.54% Accuracy** (±1.00%), outperforming Linear SVM (88.97% Macro-F1) and Polynomial SVM (78.75% Macro-F1).
2. **Final Evaluation Results**:
   - **Held-Out Validation Set (995 images)**: **92.36% Macro-F1**, **93.07% Accuracy**, **94.10% Balanced Accuracy**.
   - **Untouched Final Test Set (995 images)**: **91.47% Macro-F1**, **91.66% Accuracy**, **92.97% Balanced Accuracy**, **89.37% MCC**.

---

## 3. Directory Structure & Key Artifacts

```
Project -single classifier/
├── config.yaml                              # Central configuration & hyperparameters
├── requirements.txt                         # Dependencies
├── pytest.ini                               # Pytest configuration
├── RESEARCH_FIDELITY.md                     # Comprehensive paper fidelity audit & results
├── audit/                                   # Audited strict-clean manifests & split reports
│   ├── train_manifest.csv                   # 4,642 images
│   ├── val_manifest.csv                     # 995 images
│   └── test_manifest.csv                    # 995 images
├── Data/
│   ├── Raw/                                 # 6,632 original images
│   └── Augmented/                           # Balanced training-only augmentations
├── src/
│   ├── dataset.py                           # Manifest loader & split integrity validator
│   ├── preprocessing.py                     # Resizing & log illumination enhancement
│   ├── augmentation.py                      # Fixed policy & deterministic train balancing
│   ├── features.py                          # 16-bin CC, Multi-Level MCLBP & feature fusion
│   ├── model.py                             # One-vs-Rest SVM (Linear, RBF, Polynomial)
│   ├── cross_validation.py                  # Group-aware stratified 5-fold CV
│   ├── evaluate.py                          # Multiclass metrics & reporting
│   └── main.py                              # Reproducible CLI entry point
├── tests/                                   # 19 unit & property tests
└── outputs/
    ├── datasets/
    │   ├── final_train_balanced_complete/   # 6,600 balanced images (1,320 per class across BL, BLB, BS, SB, TU)
    │   ├── validation/                      # 995 untouched validation images organized by class (BL, BLB, BS, SB, TU)
    │   ├── test/                            # 995 untouched test images organized by class (BL, BLB, BS, SB, TU)
    │   ├── final_train_balanced_complete_manifest.csv
    │   ├── validation_manifest.csv
    │   └── test_manifest.csv
    ├── cv_results/                          # 5-fold CV reports & candidate intermediate checkpoints
    ├── final_model/rice_svm_model.pkl       # Serialized trained One-vs-Rest SVM model
    ├── evaluation/test_report.md            # Final test evaluation metrics & confusion matrix
    └── feature_cache/                       # Cached deterministic features
```
