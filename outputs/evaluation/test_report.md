# Evaluation Report: TEST

## Overall Summary Metrics

- **Accuracy**: 91.66%
- **Balanced Accuracy**: 92.97%
- **Macro F1-Score**: 91.47%
- **Macro Precision**: 90.30%
- **Macro Recall (Sensitivity)**: 92.97%
- **Macro Specificity**: 97.84%
- **Macro MCC**: 89.37%

## Per-Class Performance Breakdown

| Class | Support | Precision (%) | Recall (%) | F1-Score (%) | Specificity (%) | MCC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BL** | 84 | 80.41 | 92.86 | 86.19 | 97.91 | 85.08 |
| **BLB** | 283 | 89.36 | 89.05 | 89.20 | 95.79 | 84.92 |
| **BS** | 264 | 97.52 | 89.39 | 93.28 | 99.18 | 91.15 |
| **SB** | 84 | 91.30 | 100.00 | 95.45 | 99.12 | 95.13 |
| **TU** | 280 | 92.91 | 93.57 | 93.24 | 97.20 | 90.58 |

## Confusion Matrix

Row = True Label, Column = Predicted Label. Classes: ['BL', 'BLB', 'BS', 'SB', 'TU']

```
[[ 78   2   3   1   0]
 [  8 252   3   1  19]
 [ 10  17 236   0   1]
 [  0   0   0  84   0]
 [  1  11   0   6 262]]
```
