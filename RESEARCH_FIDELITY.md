# RESEARCH FIDELITY & REPRODUCTION REPORT

**Target Paper**: *“An enhanced classification system of various rice plant diseases based on multi-level handcrafted feature extraction technique”*  
**Citation**: Scientific Reports (2024) 14:30601 | [https://doi.org/10.1038/s41598-024-81143-1](https://doi.org/10.1038/s41598-024-81143-1)  
**Dataset Context**: Strict-clean 5-class rice leaf disease dataset (6,632 images: BL: 558, BLB: 1,886, BS: 1,760, SB: 560, TU: 1,868).

---

## 1. Executive Summary of Reproduction Results

| Experimental Phase | Configuration / Model | Primary Selection Metric (Macro-F1) | Accuracy | Balanced Accuracy | Macro Specificity | Macro MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CV Level Selection ($L=3$)** | CC + MCLBP $L=3$, Linear SVM | 88.97% (±1.56%) | 90.15% (±1.13%) | 90.29% (±1.20%) | 97.43% | 87.05% |
| **CV Level Selection ($L=4$)** | CC + MCLBP $L=4$, Linear SVM | 87.52% (±1.34%) | 88.82% (±1.44%) | 88.94% (±1.45%) | 97.09% | 85.31% |
| **CV Level Selection ($L=5$)** | CC + MCLBP $L=5$, Linear SVM | 86.58% (±1.14%) | 88.00% (±0.99%) | 88.22% (±0.98%) | 96.88% | 84.28% |
| **CV Kernel Search (Linear)** | CC + MCLBP $L=3$, Linear SVM | 88.97% (±1.56%) | 90.15% (±1.13%) | 90.29% (±1.20%) | 97.43% | 87.05% |
| **CV Kernel Search (RBF)** | CC + MCLBP $L=3$, RBF SVM ($C=1.0$) | **89.53% (±0.99%)** | **90.54% (±1.00%)** | **90.87% (±1.01%)** | **97.57%** | **87.69%** |
| **CV Kernel Search (Poly)** | CC + MCLBP $L=3$, Poly SVM ($d=3$) | 78.75% (±1.22%) | 81.26% (±1.26%) | 81.33% (±1.28%) | 95.14% | 75.31% |
| **Final Validation Set (995 img)** | Selected Model ($L=3$, RBF SVM) | **92.36%** | **93.07%** | **94.10%** | **98.26%** | **90.73%** |
| **Final Untouched Test Set (995 img)** | Selected Model ($L=3$, RBF SVM) | **91.47%** | **91.66%** | **92.97%** | **97.84%** | **89.37%** |

---

## 2. Methodology Fidelity Audit: Paper vs. Adaptation

| Pipeline Component | Paper Reference & Formula | Paper Specification | Reproduction Implementation | Fidelity Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Image Resizing** | Page 7, Section "Image resizing", Eq. 2 | $224 \times 224$ pixels | Resized to $(224, 224, 3)$ RGB via bilinear interpolation. | **Direct Reproduction** |
| **Logarithmic Illumination Enhancement** | Pages 8–9, Section "Rice image enhancement", Eqs. 7–8 | $D_{tNorm} = D_t / 255$, $D_{tLog} = c \cdot \log(1 + D_{tNorm})$ with $c = 2$ | Normalization to $[0, 1]$, then $2.0 \cdot \ln(1 + x)$ via `np.log1p(x)`. | **Direct Reproduction** (Natural log $\ln$ adapted as standard) |
| **Data Augmentation** | Pages 7–8, Section "Data augmentation", Eqs. 3–6 | Translation $[-5, 15]$ px, Rotation $[-10, 10]^\circ$, Flips (H/V) | Applied strictly to training partitions to balance classes to 1,320 images/class. | **Adapted Policy** (Documented ranges directly reproduced; training balancing is adaptation) |
| **Color Correlogram (CC)** | Page 9, Section "Color correlogram", Eq. 9 | 16 color bins, Eq. 9 joint co-occurrence table | 16-color RGB quantization, Chebyshev distance $D=1$, $16 \times 16 = 256$-D normalized probability vector. | **Adapted Implementation** ($D=1$ and Chebyshev metric are adaptations) |
| **Multi-Level MCLBP** | Pages 10–12, Eqs. 10–12, Figs. 5–7 | $3 \times 3$ patch ($P=8, R=1$), 256 bins/channel RGB ($768$ features/block), $L \times L$ grid | Extracted across $L^2$ blocks ($L^2 \times 768$ features). Blocks L1-normalized for scale compatibility. | **Direct Reproduction** |
| **Feature Fusion** | Pages 12–13, Section "Features fusion", Eq. 13 | $FE = [\gamma(D)_{c_{ii}, c_{jj}} + LBP(RGB)]$ | Horizontal vector concatenation $[CC + MCLBP]$ yielding $256 + (L^2 \times 768)$ features. | **Direct Reproduction** |
| **Multiclass SVM Strategy** | Pages 13–14, 21, 27, Eq. 14, Table 11 | One-Against-All (One-vs-Rest); Linear, RBF, Polynomial tested | One-vs-Rest SVM wrapper evaluating Linear, RBF, and Poly kernels. Standard default $C=1.0$ used. | **Direct Reproduction** ($C=1.0$ is adaptation) |
| **Kernel Comparison Outcome** | Page 21 & Table 11 | Linear kernel reported best on authors' datasets (99.53% DS1, 99.40% DS2) | On our strict-clean 5-class dataset, **RBF kernel** outperformed Linear by +0.56% Macro-F1 (89.53% vs. 88.97%). | **Documented Empirical Deviation** |
| **Cross-Validation Safeguards** | Pages 6–7, Section "Dataset acquisition" | 5 equal folds | Group-aware stratified 5-fold CV on training data only (`leakage_group_id`). Validation & test sets untouched. | **Leakage-Safe Adaptation** |

---

## 3. Configuration Selection Analysis

### 3.1 MCLBP Level Count Selection ($L \in \{3, 4, 5\}$)
- **Level 3 (7,168 features)** achieved **88.97% Macro-F1** and **90.15% Accuracy**.
- **Level 4 (12,544 features)** achieved **87.52% Macro-F1** and **88.82% Accuracy**.
- **Level 5 (19,456 features)** achieved **86.58% Macro-F1** and **88.00% Accuracy**.

*Analysis*: In the original paper, the authors reported Level 5 optimal for DS1/DS3 and Level 4 for DS2. On our strict-clean combined 5-class dataset, **Level 3** proved superior. Because our dataset has smaller sample sizes for classes `BL` (390 train images) and `SB` (392 train images), the 19,456-dimensional feature vector of Level 5 suffers from mild over-parameterization and higher variance compared to the 7,168-dimensional representation of Level 3.

### 3.2 SVM Kernel Selection
- **RBF Kernel**: **89.53% Macro-F1**, **90.54% Accuracy** (±0.99% std across folds).
- **Linear Kernel**: **88.97% Macro-F1**, **90.15% Accuracy** (±1.56% std across folds).
- **Polynomial Kernel**: **78.75% Macro-F1**, **81.26% Accuracy** (±1.22% std across folds).

*Analysis*: While the paper concluded that Linear SVM was optimal on its benchmark datasets, empirical evaluation on our group-isolated 5-class dataset demonstrates that **RBF SVM** yields both higher Macro-F1 and lower cross-fold variance.

---

## 4. Final Test Set Performance Breakdown (Untouched 995 Images)

- **Overall Accuracy**: **91.66%**
- **Balanced Accuracy**: **92.97%**
- **Macro F1-Score**: **91.47%**
- **Macro Specificity**: **97.84%**
- **Macro MCC**: **89.37%**

| Disease Class | Support | Precision (%) | Recall (%) | F1-Score (%) | Specificity (%) | MCC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BL (Blast)** | 84 | 80.41% | 92.86% | **86.19%** | 97.91% | 85.08% |
| **BLB (Bacterial Leaf Blight)** | 283 | 89.36% | 89.05% | **89.20%** | 95.79% | 84.92% |
| **BS (Brown Spot)** | 264 | 97.52% | 89.39% | **93.28%** | 99.18% | 91.15% |
| **SB (Sheath Blight)** | 84 | 91.30% | 100.00% | **95.45%** | 99.12% | 95.13% |
| **TU (Tungro)** | 280 | 92.91% | 93.57% | **93.24%** | 97.20% | 90.58% |

### Confusion Matrix on Test Set
```
Classes: ['BL', 'BLB', 'BS', 'SB', 'TU']
True \ Pred    BL   BLB    BS    SB    TU
BL             78     2     3     1     0
BLB             8   252     3     1    19
BS             10    17   236     0     1
SB              0     0     0    84     0
TU              1    11     0     6   262
```

---

## 5. Artifact Locations

- **Cross-Validation Selection Report**: [`outputs/cv_results/cv_selection_report.json`](outputs/cv_results/cv_selection_report.json)
- **Serialized Final Model**: [`outputs/final_model/rice_svm_model.pkl`](outputs/final_model/rice_svm_model.pkl)
- **Validation Evaluation Report**: [`outputs/evaluation/validation_report.md`](outputs/evaluation/validation_report.md)
- **Final Test Evaluation Report**: [`outputs/evaluation/test_report.md`](outputs/evaluation/test_report.md)
- **Feature Cache**: [`outputs/feature_cache/`](outputs/feature_cache/)
