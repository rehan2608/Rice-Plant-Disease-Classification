# Dataset Audit & Validation Summary
## Rice Disease Single Classifier (5 Classes: BL, BLB, BS, SB, TU)

**Audit Date**: 2026-10-29  
**Status**: AUDIT COMPLETE — STOPPED BEFORE DATASET PREPARATION / SPLITTING / TRAINING  
**Target Taxonomy**: 5 Disease Classes (`BL`, `BLB`, `BS`, `SB`, `TU`). Class `LS` (Leaf Smut) officially EXCLUDED.

---

### 1. Executive Metrics & Recalculated Totals

| Metric | Recalculated Count | Description & Integrity Notes |
| :--- | :--- | :--- |
| **Total Raw Files Audited** | **13,416** | 100% verified to physically exist on disk (DS1: 2,800; DS2: 4,684; DS3: 5,932). |
| **Unique Cryptographic Images (SHA-256)** | **7,592** | Distinct image contents across all 3 raw datasets. |
| **Class LS Excluded Images** | **1,460** | All 1,460 raw images in `Data - 2/Leafsmut` (960 unique hashes) excluded due to conflicting labels. |
| **Redundant Duplicate Files** | **4,364** | Identical duplicate copies within and across datasets excluded to prevent sample inflation (3,884 among consistent groups + 480 internal DS3 Blast duplicates). |
| **Quarantined Conflict Files (Strict Audit)** | **1,440** | 1,440 physical files in `Data - 3/Blast` (960 unique hashes) quarantined in strict audit due to raw collision with `DS2/Leafsmut`. |
| **Strict Clean Canonical Images** | **6,632** | Zero-conflict verified images (BL: 558; BLB: 1,886; BS: 1,760; SB: 560; TU: 1,868). |
| **Provisional Expanded Images** | **7,592** | If DS3 Blast is accepted under LS exclusion (BL: 1,518; BLB: 1,886; BS: 1,760; SB: 560; TU: 1,868). |
| **Leakage Prevention Groups** | **4,009** | Connected component clusters (exact + near duplicates) to prevent data leakage in train/test splits. |

---

### 2. Recalculated Class Distribution Table

| Canonical Label | Disease Name | Raw Files | Unique Hashes | Redundant Dups | Strict Clean Usable | Provisional Usable | Leakage Groups |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BL** | Blast | 2,000 | 1,518 | 482 | **558** | **1,518** | 1,033 |
| **BLB** | Bacterial Leaf Blight | 3,748 | 1,886 | 1,862 | **1,886** | **1,886** | 1,056 |
| **BS** | Brown Spot | 3,780 | 1,760 | 2,020 | **1,760** | **1,760** | 1,160 |
| **SB** | Sheath Blight | 560 | 560 | 0 | **560** | **560** | 128 |
| **TU** | Tungro | 1,868 | 1,868 | 0 | **1,868** | **1,868** | 632 |
| **LS** | Leaf Smut (Excluded) | 1,460 | 960 | 500 | **0** | **0** | 0 |

---

### 3. Inclusion of Valid Images from All Three Datasets

Both `blight` and `Bacterialblight` folders are fully merged and validated under canonical class **`BLB`**:
- **DS1 `blight`**: 560 images (560 unique hashes).
- **DS2 `Bacterialblight`**: 1,604 images (1,326 unique hashes).
- **DS3 `Bacterialblight`**: 1,584 images (1,326 unique hashes, identical to DS2).
- **Unified BLB Unique Images**: **1,886** distinct cryptographic images across all 3 datasets.

Similarly, for **`BS` (Brown Spot)**:
- **DS1 `Brownspot`**: 560 images (560 unique hashes).
- **DS2 `Brownspot`**: 1,620 images (1,200 unique hashes).
- **DS3 `Brownspot`**: 1,600 images (1,200 unique hashes, identical to DS2).
- **Unified BS Unique Images**: **1,760** distinct cryptographic images across all 3 datasets.

For **`SB` (Sheath Blight)**:
- Exclusively provided by **DS1 `sheath_blight`**: 560 images (560 unique hashes, 128 visual lesion leakage groups).

For **`TU` (Tungro)**:
- **DS1 `tungro`**: 560 images (560 unique hashes).
- **DS3 `Tungro`**: 1,308 images (1,308 unique hashes).
- **Unified TU Unique Images**: **1,868** distinct cryptographic images.

---

### 4. Detailed Conflict & Exclusion Analysis

#### A. The Leaf Smut (`LS`) Exclusion & DS3 Blast Provenance
1. **Misannotation Evidence**: In `Data - 2`, the folder named `Leafsmut` contains 1,460 images. However, every single image filename is prefixed with `BLAST1_` through `BLAST9_`.
2. **Cryptographic Collision**: All 960 unique SHA-256 hashes in `Data - 2/Leafsmut` are **100% identical** to the 960 unique hashes in `Data - 3/Blast`.
3. **Action Taken**:
   - Class `LS` is officially **EXCLUDED** from the classifier per instructions.
   - All 1,460 physical files in `Data - 2/Leafsmut` are marked `EXCLUDED_CLASS_LS`.
   - In strict audit mode, the 1,440 files in `Data - 3/Blast` are quarantined (`QUARANTINED_CONFLICT_DS3_BLAST`) because they share identical hashes with an excluded conflicting class.
   - In provisional expanded mode, 960 unique images from `Data - 3/Blast` are tracked as provisional canonical `BL` representatives (`is_provisional_recommended = True`), providing 1,518 total BL images if agricultural authorities confirm DS3 Blast as genuine Blast.

#### B. Perceptual Near-Duplicate Conflicts
- Dual perceptual hashing (pHash Hamming distance $\le 5$ and dHash cross-validation) identified **16 cross-label candidate pairs**.
- **All 16 pairs** are between `Data - 1/blast` (labeled `BL`) and `Data - 2/Leafsmut` (labeled `LS`).
- There are **ZERO near-duplicate conflicts** between any of the 5 canonical classes (`BL`, `BLB`, `BS`, `SB`, `TU`).

---

### 5. Leakage Prevention Grouping

To prevent data leakage during train/validation/test split generation:
- A total of **4,009 Leakage Prevention Groups (`leakage_group_id`)** were computed via connected components over exact duplicates and same-label near-duplicates.
- **Rule for Downstream Splitting**: All images sharing the same `leakage_group_id` **MUST ALWAYS** be assigned to the same split (train, validation, or test) using `StratifiedGroupKFold` or `GroupShuffleSplit`.

---

### 6. Validation Status & Stopping Condition

- **Integrity Validation**: PASSED. 0 missing files, 0 unmapped labels, 0 corrupted image decodings.
- **Deduplication Validation**: PASSED. 4,364 redundant exact duplicate copies flagged and segregated across datasets.
- **Leakage Prevention Validation**: PASSED. All exact and near-duplicates clustered into group IDs.
- **Label Conflict Safety**: PASSED WITH STOPPING PROTOCOL. No conflicting labels were silently resolved. All 960 exact duplicate conflict groups and 16 near-duplicate conflict pairs are fully documented.
- **STOPPING DIRECTIVE**: In accordance with user instructions, execution is **halted** before model training, image corruptions, augmentations, or split generation.
