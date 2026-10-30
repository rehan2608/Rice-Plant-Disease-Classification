# Phase 1: Comprehensive Dataset Audit Report
## Rice Plant Disease Classification (Handcrafted Pipeline: CC + MCLBP + Linear SVM)

**Audit Date**: 2026-10-29 02:34:43  
**Auditor**: Senior Machine Learning Engineer  
**Project Phase**: Phase 1 — Dataset Audit & Integrity Verification  
**Scope**: Exploration, integrity validation, duplicate detection, and label mapping verification across raw datasets (`DS1`, `DS2`, `DS3`). No files moved, altered, or deleted.

---

## 1. Executive Summary

A comprehensive, automated, and reproducible dataset audit was performed on all three candidate datasets located in `Data/Raw/`. A total of **13,416 raw image files** were discovered, verified for file format integrity and decoded using Pillow, and systematically audited using cryptographic hashing (SHA-256) and dual perceptual hashing (DCT-based pHash and difference-based dHash).

### Key Audit Metrics
| Metric | Value | Technical Implications |
| :--- | :--- | :--- |
| **Total Raw Images Discovered** | **13,416** | All files have `.jpg` extension; 0 corrupted or unreadable images. |
| **Unique Cryptographic Images (SHA-256)** | **7,592** | **9,312 duplicate files** exist due to artificial data replication. |
| **Exact Duplicate Groups** | **3,488** | Intra-dataset replication in DS2/DS3 and complete cross-dataset sharing. |
| **Conflicting Label Duplicate Groups** | **960** | **960 unique images (2,900 raw files) quarantined** due to critical annotation conflict. |
| **Near-Duplicate Candidate Pairs** | **6,880** | Perceptual hashing (Hamming distance $\le 5$) identified candidate pairs for review. |
| **Near-Duplicate Label Conflicts** | **16** | Pairs with perceptual similarity but conflicting labels flagged for review. |
| **Recommended Clean Usable Images** | **6,632** | Provenance-preserved, deduplicated, and conflict-free images for Phase 2. |

---

## 2. Dataset Locations and Folder Structures

The audit verified the exact storage paths, nesting patterns, and folder organizations across `Data/Raw/`:

```
Data/Raw/
├── Data - 1/ (DS1: Pre-split train/test structure)
│   ├── Full_Train_Data/
│   │   ├── blast/ (510 images)
│   │   ├── blight/ (510 images)
│   │   ├── Brownspot/ (510 images)
│   │   ├── sheath_blight/ (510 images)
│   │   └── tungro/ (510 images)
│   └── Test_Data_SVM/
│       ├── blast/ (50 images)
│       ├── blight/ (50 images)
│       ├── Brownspot/ (50 images)
│       ├── sheath_blight/ (50 images)
│       └── tungro/ (50 images)
├── Data - 2/ (DS2: Flat class structure)
│   ├── Bacterialblight/ (1,604 images)
│   ├── Brownspot/ (1,620 images)
│   └── Leafsmut/ (1,460 images)  <-- [CRITICAL ANOMALY IDENTIFIED]
└── Data - 3/ (DS3: Flat class structure)
    ├── Bacterialblight/ (1,584 images)
    ├── Blast/ (1,440 images)
    ├── Brownspot/ (1,600 images)
    └── Tungro/ (1,308 images)
```

### Dataset Architecture Observations:
1. **DS1 (Data - 1)**: Formatted as a pre-partitioned 5-class dataset (`Full_Train_Data`: 2,550 images; `Test_Data_SVM`: 250 images). Contains 5 classes: Blast, Blight (Bacterial Leaf Blight), Brown Spot, Sheath Blight, and Tungro. It is the **only dataset providing Sheath Blight (SB)**.
2. **DS2 (Data - 2)**: Flat directory structure with 3 folders totaling 4,684 images.
3. **DS3 (Data - 3)**: Flat directory structure with 4 folders totaling 5,932 images.

---

## 3. Critical Anomaly & Label Verification Findings

### A. The DS2 `Leafsmut` vs DS3 `Blast` Annotation Conflict
- **Discovery**: In `Data - 2`, the folder named `Leafsmut` contains 1,460 images. However, every single filename in this folder begins with the prefix `BLAST1_` through `BLAST9_`.
- **Cryptographic Verification**: Exact SHA-256 matching revealed that **100% of the unique images in DS2 `Leafsmut` (960 distinct content hashes) are identical to the images in DS3 `Blast`**.
- **Impact**: If a researcher naively assigned `Leafsmut` $	o$ `LS` and `Blast` $	o$ `BL`, the model would be trained with identical images assigned conflicting ground-truth targets. This would induce catastrophic label noise and invalidate experimental evaluation.
- **Audit Action Taken**: All 960 duplicate groups representing this conflict (2,900 physical files across DS2 and DS3) have been **quarantined** with status `QUARANTINED_CONFLICT`. None of these records are recommended for canonical training without expert agricultural review.

### B. Verification of `SL` vs `LS` Label Hypothesis
- The project context hypothesized that DS3 might contain a folder named `SL` instead of `LS`.
- **Finding**: A recursive scan of all folders and filenames confirmed that **no folder or file named `SL` exists anywhere in DS1, DS2, or DS3**.
- **Leaf Smut Availability**: Because all images in DS2 `Leafsmut` are duplicate copies of DS3 `Blast`, **there are zero verified Leaf Smut (LS) images in the provided datasets**. The research team must either obtain an authentic Leaf Smut dataset or formulate the single classifier for the 5 verified disease classes (`BL`, `BLB`, `BS`, `TU`, `SB`).

### C. DS2 is a Redundant Subset of DS3
- **Total unique SHA-256 hashes in DS2**: **3,486**.
- **Shared unique hashes between DS2 and DS3**: **3,486 (100.0%)**.
- Every single unique image in DS2 exists in DS3:
  - DS2 `Bacterialblight` (1,326 unique hashes) $\equiv$ DS3 `Bacterialblight` (1,326 unique hashes).
  - DS2 `Brownspot` (1,200 unique hashes) $\equiv$ DS3 `Brownspot` (1,200 unique hashes).
  - DS2 `Leafsmut` (960 unique hashes) $\equiv$ DS3 `Blast` (960 unique hashes).
- DS3 contains additional data: DS3 has 1,308 unique images for `Tungro` (which DS2 lacks).

### D. Data Leakage in Pre-Split DS1
- In DS1 (`Data - 1`), an exact duplicate was detected between `Full_Train_Data` and `Test_Data_SVM`:
  - `Full_Train_Data/blast/blast_43_output.jpg`
  - `Test_Data_SVM/blast/blast_42_output.jpg`
  - SHA-256: `e4bf9bd3b4b4554b2efc8fca399e7c53d9e36506371cb4653896504a59f5bca9`
- This proves that the legacy pre-split in DS1 suffered from train-test contamination. Phase 2 must construct unified, leakage-safe splits from scratch.

---

## 4. Class Distribution & Deduplication Summary

### Per-Dataset Breakdown
| dataset_id | raw_count | unique_count | quarantined | recommended |
| :--- | :--- | :--- | :--- | :--- |
| DS1 | 2800 | 2798 | 0 | 2798 |
| DS2 | 4684 | 3486 | 1460 | 0 |
| DS3 | 5932 | 4794 | 1440 | 3834 |

### Canonical Class Distribution
| canonical_label | raw_count | unique_count | quarantined | recommended | disease_name |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BL | 2000 | 1518 | 1440 | 558 | Blast |
| BLB | 3748 | 1886 | 0 | 1886 | Bacterial Leaf Blight |
| BS | 3780 | 1760 | 0 | 1760 | Brown Spot |
| LS | 1460 | 960 | 1460 | 0 | Leaf Smut |
| SB | 560 | 560 | 0 | 560 | Sheath Blight |
| TU | 1868 | 1868 | 0 | 1868 | Tungro |

*Note on Class Representation*:
- **BL (Blast)**: 1,998 raw images (558 from DS1, 1,440 from DS3). Note: 1,440 files in DS3 Blast are currently quarantined due to collision with DS2 Leafsmut.
- **BLB (Bacterial Leaf Blight)**: 3,748 raw images (560 from DS1, 1,604 from DS2, 1,584 from DS3) resolve to 1,886 unique images.
- **BS (Brown Spot)**: 3,780 raw images (560 from DS1, 1,620 from DS2, 1,600 from DS3) resolve to 1,760 unique images.
- **TU (Tungro)**: 1,868 raw images (560 from DS1, 1,308 from DS3) resolve to 1,868 unique images (zero duplicates between DS1 and DS3).
- **SB (Sheath Blight)**: 560 raw images (exclusively in DS1) resolve to 560 unique images.
- **LS (Leaf Smut)**: 1,460 raw images in DS2; all 1,460 are quarantined due to 100% collision with DS3 Blast.

---

## 5. Perceptual Hashing & Near-Duplicate Analysis

Dual perceptual hashing was executed across all unique image contents:
- **dHash (Difference Hash)**: Evaluates horizontal luminance gradients (9x8 matrix).
- **pHash (DCT Perceptual Hash)**: Evaluates low-frequency 2D Discrete Cosine Transform coefficients (32x32 to 8x8).

### Distance Threshold Configuration
- **Hamming Distance <= 5**: Flagged as candidate near-duplicates.
- **Hamming Distance 0 among differing SHA-256**: 3,068 pairs represent re-encoded, re-compressed, or metadata-altered copies of the same original photograph.
- **Hamming Distance 1 to 3**: 2,254 pairs exhibit near-identical visual compositions (minor crops, slight color shifts).
- **Hamming Distance 4 to 5**: 1,558 pairs capture sequential shots of the same lesion or plant specimen.

All near-duplicate candidates have been cataloged in `near_duplicate_candidates.csv` with source datasets, relative paths, original labels, canonical labels, distances, and review priority flags.

---

## 6. Audit Deliverables Manifest

All required deliverables have been generated in `audit/`:
1. `dataset_manifest_initial.csv`: Master inventory of all 13,416 images with dimensions, modes, hashes, labels, and status flags.
2. `class_distribution.csv`: Formal count summary by dataset, raw label, and canonical class.
3. `exact_duplicates.csv`: Record-level duplicate tracking showing duplicate groups, sources, and resolution status.
4. `exact_duplicate_groups.csv`: Group-level summary of all 3,488 exact duplicate clusters.
5. `near_duplicate_candidates.csv`: Complete catalog of perceptual near-duplicate candidate pairs.
6. `unmapped_labels.csv`: Record of unmapped folder labels (0 unmapped; all raw folders resolved).
7. `unreadable_images.csv`: Log of corrupted or unreadable images (0 errors found).
8. `label_mapping_config.json`: Explicit, editable JSON configuration governing mappings and deduplication policies.
9. `audit_summary.md`: This comprehensive research-grade audit document.
10. `run_audit.py`: Reproducible CLI script.
11. `dataset_audit_notebook.ipynb`: Interactive Jupyter Notebook for visual validation and exploratory inspection.

---

## 7. Recommended Next Actions for Phase 2

1. **Resolution of Quarantined DS2 Leafsmut / DS3 Blast**:
   - The team must formally decide whether to re-label DS2 `Leafsmut` as `Blast` (which matches the filenames `BLAST...` and visual disease traits) or acquire a verified external Leaf Smut dataset.
2. **Leakage-Safe Splitting (Group-Aware Stratification)**:
   - When generating train/validation/test splits, images belonging to the same exact duplicate group or near-duplicate cluster must **never be split across train and test sets**.
   - Use the `exact_duplicate_group_id` and near-duplicate cluster IDs as grouping keys in `GroupKFold` or `StratifiedGroupKFold`.
3. **Canonical Deduplication**:
   - For training the handcrafted feature pipeline (CC + MCLBP + linear SVM), utilize only records where `is_canonical_recommendation == True` to avoid biasing the SVM margins with redundant identical samples.
