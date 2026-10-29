# Evaluation Report: VALIDATION

## Overall Summary Metrics

- **Accuracy**: 93.07%
- **Balanced Accuracy**: 94.10%
- **Macro F1-Score**: 92.36%
- **Macro Precision**: 91.12%
- **Macro Recall (Sensitivity)**: 94.10%
- **Macro Specificity**: 98.26%
- **Macro MCC**: 90.73%

## Per-Class Performance Breakdown

| Class | Support | Precision (%) | Recall (%) | F1-Score (%) | Specificity (%) | MCC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BL** | 84 | 75.96 | 94.05 | 84.04 | 97.26 | 82.97 |
| **BLB** | 283 | 92.42 | 90.46 | 91.43 | 97.05 | 88.08 |
| **BS** | 264 | 94.61 | 86.36 | 90.30 | 98.22 | 87.17 |
| **SB** | 84 | 94.38 | 100.00 | 97.11 | 99.45 | 96.88 |
| **TU** | 280 | 98.24 | 99.64 | 98.94 | 99.30 | 98.52 |

## Confusion Matrix

Row = True Label, Column = Predicted Label. Classes: ['BL', 'BLB', 'BS', 'SB', 'TU']

```
[[ 79   1   3   1   0]
 [  8 256  10   4   5]
 [ 16  20 228   0   0]
 [  0   0   0  84   0]
 [  1   0   0   0 279]]
```
