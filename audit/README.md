# Dataset Audit & Split Manifests

This directory contains the verified, leakage-free data splits and taxonomy mappings for the 5-class rice plant disease dataset.

---

## 1. Quick Summary of What the Audit Achieved

- **Source Dataset**: Combined foliar rice disease benchmarks (6,632 raw images).
- **Integrity & Cleaning**: Removed unreadable files, resolved ambiguous classes, and grouped near-duplicate images (Hamming perceptual hash $\le 10$) into atomic leakage groups so duplicate images never cross train/val/test splits.
- **Reproducible Stratified Split**: Fixed 70% Train, 15% Validation, and 15% Test partitions.

```
Total Clean Images: 6,632
├── Train: 4,642 images (~70%)
├── Validation: 995 images (~15%)
└── Test:       995 images (~15%)
```

---

## 2. Active Files in this Directory

| File | Purpose |
| :--- | :--- |
| [`split_summary.csv`](split_summary.csv) | High-level 1-page table showing image and group counts per class across splits. |
| [`train_manifest.csv`](train_manifest.csv) | Master training set manifest (4,642 images with image paths and labels). |
| [`val_manifest.csv`](val_manifest.csv) | Held-out validation manifest (995 images). |
| [`test_manifest.csv`](test_manifest.csv) | Locked, untouched final test manifest (995 images). |
| [`label_mapping_config.json`](label_mapping_config.json) | Taxonomy map mapping folder names to standard codes: `BL`, `BLB`, `BS`, `SB`, `TU`. |
| [`split_validation_report.md`](split_validation_report.md) | Audit proof confirming 0% cross-split data leakage and perfect class stratification. |
| `archive/` | Historical diagnostic dumps (duplicate lists, candidate comparisons, pre-clean scans). |

---

## 3. Disease Class Taxonomy

- **BL**: Rice Blast (*Magnaporthe oryzae*)
- **BLB**: Bacterial Leaf Blight (*Xanthomonas oryzae*)
- **BS**: Brown Spot (*Bipolaris oryzae*)
- **SB**: Sheath Blight (*Rhizoctonia solani*)
- **TU**: Tungro Disease (*RTBV & RTSV*)
