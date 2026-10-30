#!/usr/bin/env python3
"""
Rice Plant Disease Classification - Dataset Audit Execution Script
Phase 1: Dataset Audit

Usage:
    python audit/run_audit.py
    python audit/run_audit.py --raw-dir "Data/Raw" --output-dir "audit" --config "audit/label_mapping_config.json"
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit.audit_engine import DatasetAuditEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Phase 1 Dataset Audit for Rice Plant Disease Classification."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="audit/label_mapping_config.json",
        help="Path to label mapping configuration JSON file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="Data/Raw",
        help="Root directory containing raw datasets (Data - 1, Data - 2, Data - 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="audit",
        help="Directory where audit CSVs, summaries, and reports are saved.",
    )
    parser.add_argument(
        "--phash-thresh",
        type=int,
        default=5,
        help="Hamming distance threshold for pHash near-duplicate candidate mining.",
    )
    parser.add_argument(
        "--dhash-thresh",
        type=int,
        default=5,
        help="Hamming distance threshold for dHash cross-validation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 75)
    print("  RICE PLANT DISEASE CLASSIFICATION: PHASE 1 DATASET AUDIT")
    print("=" * 75)
    print(f"Configuration:    {args.config}")
    print(f"Raw Datasets:     {args.raw_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"pHash Threshold:  {args.phash_thresh} bits")
    print(f"dHash Threshold:  {args.dhash_thresh} bits")
    print("-" * 75)

    engine = DatasetAuditEngine(
        config_path=args.config,
        raw_root=args.raw_dir,
        output_dir=args.output_dir,
        phash_threshold=args.phash_thresh,
        dhash_threshold=args.dhash_thresh,
    )

    # 1. Scan and verify datasets
    engine.scan_datasets()

    # 2. Process exact cryptographic duplicates (SHA-256)
    engine.process_exact_duplicates()

    # 3. Detect perceptual near-duplicate candidates (pHash & dHash)
    engine.detect_near_duplicates()

    # 4. Compile distribution summaries
    engine.compile_class_distribution()

    # 5. Export deliverables
    outputs = engine.export_audit_outputs()

    print("\n" + "=" * 75)
    print("  AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 75)
    print(f"Total raw images audited:       {len(engine.manifest_df):,}")
    print(f"Unique cryptographic images:   {engine.manifest_df['sha256'].nunique():,}")
    print(f"Exact duplicate groups:        {len(engine.exact_duplicate_groups_df):,}")
    print(f"Near-duplicate candidate pairs: {len(engine.near_duplicates_df):,}")
    print(f"Corrupted / unreadable images: {len(engine.unreadable_records):,}")
    print("All artifacts stored in:", Path(args.output_dir).resolve())
    print("=" * 75)


if __name__ == "__main__":
    main()
