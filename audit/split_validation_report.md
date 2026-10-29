# Leakage-Safe Dataset Split & Validation Report
## Rice Disease Single Classifier (5 Classes: BL, BLB, BS, SB, TU)

**Split Date**: 2026-10-29  
**Random Seed**: `42` (Fully Deterministic & Reproducible)  
**Partition Strategy**: Group-Aware Stratified Splitting over Connected-Component `leakage_group_id`  
**Target Ratios**: 70.0% Train / 15.0% Validation / 15.0% Test  
**Source Manifest**: Strict-Clean subset of [`audit/dataset_manifest_final.csv`](file:///c:/Users/suhail/OneDrive/Desktop/Machine%20Learning/Project%20-single%20classifier/audit/dataset_manifest_final.csv) (`is_clean_recommended == True`)

---

### 1. Executive Split Summary

| Split Partition | Image Count | Percentage | Leakage Groups | Hash Count | File Integrity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | **4,642** | **69.99%** | **2,463** | **4,642** | 100% on disk, 0 missing |
| **Validation Set** | **995** | **15.00%** | **531** | **995** | 100% on disk, 0 missing |
| **Test Set** | **995** | **15.00%** | **535** | **995** | 100% on disk, 0 missing |
| **Total Clean Dataset** | **6,632** | **100.00%** | **3,529** | **6,632** | 100% on disk, 0 missing |

---

### 2. Per-Class Stratification Breakdown

| Canonical Class | Disease Name | Total Images | Total Groups | Train (Count / %) | Val (Count / %) | Test (Count / %) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **BL** | Blast | 558 | 553 | 390 (69.89%) | 84 (15.05%) | 84 (15.05%) |
| **BLB** | Bacterial Leaf Blight | 1,886 | 1,056 | 1,320 (69.99%) | 283 (15.01%) | 283 (15.01%) |
| **BS** | Brown Spot | 1,760 | 1,160 | 1,232 (70.00%) | 264 (15.00%) | 264 (15.00%) |
| **SB** | Sheath Blight | 560 | 128 | 392 (70.00%) | 84 (15.00%) | 84 (15.00%) |
| **TU** | Tungro | 1,868 | 632 | 1,308 (70.02%) | 280 (14.99%) | 280 (14.99%) |
| **TOTAL** | ****ALL CLASSES COMBINED**** | **6,632** | **3,529** | **4,642 (69.99%)** | **995 (15.00%)** | **995 (15.00%)** |

---

### 3. Handling of Excluded Records & Mixed Groups

In accordance with project guidelines:
1. **Class Label Purity**:
   - Every single one of the **3,529 leakage groups** present in the strict-clean dataset contains records of **exactly one disease class** (0 mixed-class groups).
2. **Excluded Records Handling**:
   - Exactly **1,122 leakage groups** in the clean dataset also correspond to **3,884 redundant exact duplicate records** in the master manifest (from DS2 and DS3 redundant copies).
   - **Isolation**: These redundant records have `is_clean_recommended == False` and `audit_status == "EXCLUDED_REDUNDANT_DUPLICATE"`. They were explicitly excluded from the split manifest creation.
   - **Provenance Guarantee**: Because those redundant copies share the exact same `leakage_group_id` with their canonical representative, if redundant files are ever inspected, they map strictly to the same partition, preventing any latent leakage.
3. **Quarantined & Excluded Classes**:
   - `EXCLUDED_CLASS_LS` (all 1,460 Leafsmut files) and `QUARANTINED_CONFLICT_DS3_BLAST` (all 1,440 DS3 Blast files) share **zero** leakage groups with the strict-clean records. None of them are present in train, val, or test splits.

---

### 4. Comprehensive Validation Results

All six strict validation criteria have passed:

| Validation Check | Result | Verification Details |
| :--- | :---: | :--- |
| **No Group Leakage** | **PASSED** | 0 `leakage_group_id` values appear in more than one split (Train $\cap$ Val = $\emptyset$, Train $\cap$ Test = $\emptyset$, Val $\cap$ Test = $\emptyset$). |
| **No Cryptographic Hash Overlap** | **PASSED** | 0 SHA-256 hashes are shared across splits. |
| **Zero Excluded Records** | **PASSED** | 0 LS records, 0 redundant copies, and 0 quarantined Blast records in any manifest. |
| **Physical File Existence** | **PASSED** | 100% of 6,632 split files confirmed to exist on local disk. |
| **Five-Class Representation** | **PASSED** | All five classes (`BL`, `BLB`, `BS`, `SB`, `TU`) are represented in Train, Val, and Test. |
| **Count Reconciliation** | **PASSED** | Exactly reconciles with 6,632 strict-clean records: 4,642 + 995 + 995 = 6,632. |

---

### 5. Reproducibility & Output Files

- **Execution Script**: [`audit/create_splits.py`](file:///c:/Users/suhail/OneDrive/Desktop/Machine%20Learning/Project%20-single%20classifier/audit/create_splits.py)
- **Train Split Manifest**: [`audit/train_manifest.csv`](file:///c:/Users/suhail/OneDrive/Desktop/Machine%20Learning/Project%20-single%20classifier/audit/train_manifest.csv) (4,642 images)
- **Validation Split Manifest**: [`audit/val_manifest.csv`](file:///c:/Users/suhail/OneDrive/Desktop/Machine%20Learning/Project%20-single%20classifier/audit/val_manifest.csv) (995 images)
- **Test Split Manifest**: [`audit/test_manifest.csv`](file:///c:/Users/suhail/OneDrive/Desktop/Machine%20Learning/Project%20-single%20classifier/audit/test_manifest.csv) (995 images)
- **Summary Metrics**: [`audit/split_summary.csv`](file:///c:/Users/suhail/OneDrive/Desktop/Machine%20Learning/Project%20-single%20classifier/audit/split_summary.csv)
