"""
Phase 8 — Pipeline Orchestration
=================================

Single entry-point script that runs the entire entity resolution pipeline
end-to-end:

  1. Load + normalize all source TSVs (train + test)
  2. Build validation split (stratified by country)
  3. Run multi-key blocking -> generate candidate pairs
  4. Compute features for all candidate pairs
  5. Hard-negative sampling + LightGBM training
  6. Threshold sweep on validation split -> select best threshold
  7. Batch inference on test candidates
  8. ID-existence check + superset assertion
  9. Write both output TSVs
  10. Run validator

Usage:
    python run_pipeline.py \
        --train-dir dataset/train \
        --test-dir dataset/test \
        --output-dir output \
        --val-fraction 0.1
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingest.load_normalize import load_and_normalize
from src.blocking.candidate_gen import InvertedIndexBlocker, merge_s2_s3
from src.features.similarity import FeatureExtractor, build_lookup, FEATURE_COLUMNS
from src.matching.train import build_training_pairs, train_lgbm
from src.matching.predict import batch_predict, apply_threshold, threshold_sweep
from src.evaluate.score_f05 import (
    parse_ground_truth,
    score_f05_macro,
    print_score_summary,
    create_val_split,
    error_gallery,
    print_error_gallery,
)
from src.submit.package import (
    check_id_existence,
    check_superset,
    check_s1_coverage,
    write_matching_results_tsv,
    write_candidate_pairs_tsv,
    candidates_df_to_dict,
    run_validator,
)


def print_stage(n: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  STAGE {n}: {title}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Resolvent — Full Entity Resolution Pipeline"
    )
    parser.add_argument("--train-dir", required=True,
                        help="Directory with train_source1/2/3.tsv + train_ground_truth.tsv")
    parser.add_argument("--test-dir", required=True,
                        help="Directory with test_source1/2/3.tsv")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory for matching_results.tsv + candidate_pairs.tsv")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of train S1 entities for validation (default: 0.1)")
    parser.add_argument("--data-dir", default="data",
                        help="Data directory for intermediate files (default: data)")
    parser.add_argument("--model-dir", "--models-dir", dest="model_dir", default="models",
                        help="Directory to save/load model files")
    parser.add_argument("--max-train-s1", type=int, default=100000,
                        help="Max S1 entities to sample for training (default: 100000, 0 for all)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold (skip sweep). If not set, sweeps on val.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing model)")
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip validation split creation")
    parser.add_argument("--validator", default="utils/validate_submission.py",
                        help="Path to organizer's validator script")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    pipeline_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    val_dir = os.path.join(data_dir, "val_split")

    train_s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    train_s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    train_s3_path = os.path.join(args.train_dir, "train_source3.tsv")
    train_gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")

    test_s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(args.test_dir, "test_source3.tsv")

    model_path = os.path.join(args.model_dir, "lgbm_matcher.txt")

    if not args.skip_train or not os.path.exists(model_path):
        # ── STAGE 1: Load & Normalize (Train) ─────────────────────────
        print_stage(1, "LOAD & NORMALIZE (TRAIN)")

        print("Loading and normalizing train sources...")
        train_s1 = load_and_normalize(train_s1_path)
        train_s2 = load_and_normalize(train_s2_path)
        train_s3 = load_and_normalize(train_s3_path)
        print(f"  Train S1: {len(train_s1):,}")
        print(f"  Train S2: {len(train_s2):,}")
        print(f"  Train S3: {len(train_s3):,}")

        print("\nLoading ground truth...")
        gt_dict = parse_ground_truth(train_gt_path)
        print(f"  Ground truth: {len(gt_dict):,} S1 entities")

        # ── STAGE 2: Validation Split ─────────────────────────────────
        print_stage(2, "VALIDATION SPLIT")
        # Split train_s1 into train & val subsets for threshold sweep
        rng = np.random.RandomState(42)
        n_val = min(10000, max(100, int(len(train_s1) * args.val_fraction)))
        val_indices = rng.choice(train_s1.index.values, size=n_val, replace=False)
        val_s1 = train_s1.loc[val_indices].copy().reset_index(drop=True)
        val_s1_ids = set(val_s1["entity_id"].values)
        val_gt_dict = {eid: gt_dict.get(eid, set()) for eid in val_s1_ids}
        print(f"  In-memory validation set: {len(val_s1):,} S1 entities")

        # Sample train S1 if requested
        if args.max_train_s1 and 0 < args.max_train_s1 < len(train_s1):
            remaining_indices = [idx for idx in train_s1.index.values if idx not in set(val_indices)]
            n_train = min(len(remaining_indices), args.max_train_s1)
            train_sub_indices = rng.choice(remaining_indices, size=n_train, replace=False)
            train_s1_block = train_s1.loc[train_sub_indices].copy().reset_index(drop=True)
            print(f"  Sampled {len(train_s1_block):,} S1 entities for training model")
        else:
            train_s1_block = train_s1

        # ── STAGE 3: Blocking (Train) ─────────────────────────────────
        print_stage(3, "BLOCKING (TRAIN)")

        train_s23 = merge_s2_s3(train_s2, train_s3)
        print(f"  Train S23 (merged): {len(train_s23):,}")

        blocker = InvertedIndexBlocker()
        train_gt_df = pd.read_csv(train_gt_path, sep="\t", encoding="utf-8",
                                  encoding_errors="replace", dtype=str, keep_default_na=False)
        train_candidates = blocker.generate_candidates(train_s1_block, train_s23, ground_truth=train_gt_df)
        print(f"  Train candidate pairs: {len(train_candidates):,}")

        # ── STAGE 4: Feature Engineering (Train) ──────────────────────
        print_stage(4, "FEATURE ENGINEERING (TRAIN)")

        extractor = FeatureExtractor()
        extractor.fit(train_s1_block, train_s23)

        s1_lookup = build_lookup(train_s1_block)
        train_needed_s23 = set(train_candidates["source23_entity_id"].values)
        s23_lookup = build_lookup(train_s23, needed_ids=train_needed_s23)

        train_features = extractor.extract(train_candidates, s1_lookup, s23_lookup)

        # ── STAGE 5: Training ─────────────────────────────────────────
        print_stage(5, "LGBM TRAINING")
        df_train = build_training_pairs(train_features, gt_dict)
        model = train_lgbm(df_train, model_output=model_path)

        # ── STAGE 6: Threshold Sweep on Validation ────────────────────
        print_stage(6, "THRESHOLD SWEEP")
        threshold = args.threshold
        if threshold is None:
            print("  Blocking validation set...")
            val_blocker = InvertedIndexBlocker()
            val_candidates = val_blocker.generate_candidates(val_s1, train_s23)

            print("  Computing validation features...")
            val_s1_lookup = build_lookup(val_s1)
            val_needed_s23 = set(val_candidates["source23_entity_id"].values)
            val_s23_lookup = build_lookup(train_s23, needed_ids=val_needed_s23)
            val_features = extractor.extract(val_candidates, val_s1_lookup, val_s23_lookup)

            print("  Running validation inference...")
            import lightgbm as lgb
            booster = model.booster_ if isinstance(model, lgb.LGBMClassifier) else model
            val_probs = batch_predict(val_features, booster)

            threshold, best_f05, sweep_df = threshold_sweep(
                val_features, val_probs, val_gt_dict
            )
            print(f"  Selected threshold: {threshold:.2f} (Val F0.5: {best_f05:.4f})")
        else:
            print(f"  Using fixed threshold: {threshold:.2f}")

        # Free training data from memory before test stage
        del train_s1, train_s2, train_s3, train_s23, train_candidates, train_features, df_train
        if 'train_s1_block' in locals():
            del train_s1_block
        if 'val_s1' in locals():
            del val_s1, val_candidates, val_features
        import gc
        gc.collect()

    else:
        import lightgbm as lgb
        print_stage(5, "LOAD EXISTING MODEL")
        print(f"  Loading existing model from {model_path}")
        model = lgb.Booster(model_file=model_path)
        threshold = args.threshold if args.threshold is not None else 0.89
        print(f"  Threshold: {threshold:.2f}")

    # ── STAGE 7: Blocking + Inference (Test) ──────────────────────────
    print_stage(7, "BLOCKING + INFERENCE (TEST)")

    print("Loading and normalizing test sources...")
    test_s1 = load_and_normalize(test_s1_path)
    test_s2 = load_and_normalize(test_s2_path)
    test_s3 = load_and_normalize(test_s3_path)
    print(f"  Test S1: {len(test_s1):,}")
    print(f"  Test S2: {len(test_s2):,}")
    print(f"  Test S3: {len(test_s3):,}")

    valid_s23_ids = set(test_s2["entity_id"].values) | set(test_s3["entity_id"].values)
    test_s23 = merge_s2_s3(test_s2, test_s3)
    print(f"  Test S23 (merged): {len(test_s23):,}")
    # Free raw test_s2, test_s3 to reclaim 2.5 GB RAM
    del test_s2, test_s3
    import gc
    gc.collect()

    print("  Fitting test feature extractor...")
    test_extractor = FeatureExtractor()
    test_extractor.fit(test_s1, test_s23)

    import lightgbm as lgb
    booster = model.booster_ if isinstance(model, lgb.LGBMClassifier) else model

    all_test_s1_ids = set(test_s1["entity_id"].values)
    test_matches: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_test_s1_ids}
    test_cand_dict: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_test_s1_ids}

    # Group by country to keep memory lean
    test_countries = sorted(test_s1["country"].dropna().unique())
    print(f"  Processing test set by country partition: {test_countries}")

    test_blocker = InvertedIndexBlocker()

    for country in test_countries:
        c_t0 = time.time()
        print(f"\n  --- Country partition: {country} ---")
        s1_c = test_s1[test_s1["country"] == country]
        s23_c = test_s23[test_s23["country"] == country]
        print(f"    S1: {len(s1_c):,}, S23: {len(s23_c):,}")

        c_pairs = test_blocker._block_within_country(s1_c, s23_c)
        print(f"    Generated candidate pairs: {len(c_pairs):,}")

        if not c_pairs:
            print(f"    No candidate pairs for country {country}")
            continue

        # Add all generated pairs to test_cand_dict
        for s1_eid, s23_eid in c_pairs:
            test_cand_dict[s1_eid].add(s23_eid)

        df_cand_c = pd.DataFrame(c_pairs, columns=["source1_entity_id", "source23_entity_id"])
        del c_pairs
        gc.collect()

        s1_c_lookup = build_lookup(s1_c)
        needed_s23_c = set(df_cand_c["source23_entity_id"].values)
        s23_c_lookup = build_lookup(s23_c, needed_ids=needed_s23_c)

        print(f"    Extracting features for {len(df_cand_c):,} pairs...")
        feats_c = test_extractor.extract(df_cand_c, s1_c_lookup, s23_c_lookup)
        del s1_c_lookup, s23_c_lookup, needed_s23_c
        gc.collect()

        print(f"    Running model inference...")
        probs_c = batch_predict(feats_c, booster)

        s1_c_ids = set(s1_c["entity_id"].values)
        matches_c = apply_threshold(feats_c, probs_c, threshold, s1_c_ids)
        for s1_id, mset in matches_c.items():
            test_matches[s1_id] = mset

        n_c_matched = sum(1 for m in matches_c.values() if m)
        print(f"    Partition {country} complete: {n_c_matched:,} matched out of {len(s1_c):,} ({time.time() - c_t0:.1f}s)")

        del df_cand_c, feats_c, probs_c, matches_c
        gc.collect()

    del test_s1, test_s23
    gc.collect()

    n_matched = sum(1 for m in test_matches.values() if m)
    n_singleton = sum(1 for m in test_matches.values() if not m)
    print(f"\n  Final Test Matches (Threshold={threshold:.2f}): {n_matched:,} matched, {n_singleton:,} singletons")

    # ── STAGE 8: Packaging ────────────────────────────────────────────
    print_stage(8, "SUBMISSION PACKAGING")


    matching_output = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_output = os.path.join(args.output_dir, "candidate_pairs.tsv")

    # ID checks
    check_s1_coverage(test_matches, all_test_s1_ids, "matching_results")
    check_s1_coverage(test_cand_dict, all_test_s1_ids, "candidate_pairs")
    check_id_existence(test_matches, valid_s23_ids, "matching_results")
    check_superset(test_matches, test_cand_dict)

    # Write outputs
    write_matching_results_tsv(test_matches, matching_output)
    write_candidate_pairs_tsv(test_cand_dict, candidate_output)

    # ── STAGE 9: Validator Gate ───────────────────────────────────────
    print_stage(9, "VALIDATOR GATE")
    run_validator(matching_output, candidate_output, args.test_dir, args.validator)

    # ── STAGE 10: Submission Zip Assembly ─────────────────────────────
    print_stage(10, "SUBMISSION ZIP ASSEMBLY")
    import zipfile
    zip_path = "Resolvent_Submission.zip"
    print(f"Creating final submission archive: {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(matching_output, arcname="output/matching_results.tsv")
        zf.write(candidate_output, arcname="output/candidate_pairs.tsv")
        if os.path.exists("Documentation_template.md"):
            zf.write("Documentation_template.md", arcname="Documentation_template.md")
        code_dir = "code/business_entity_resolution"
        for root, _, files in os.walk(code_dir):
            for file in files:
                if not file.endswith((".pyc", ".pyo")) and "__pycache__" not in root:
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, ".")
                    zf.write(full_p, arcname=rel_p)
    print(f"  [OK] Submission zip created: {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")

    # ── Summary ───────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  Output    : {matching_output}")
    print(f"            : {candidate_output}")
    print(f"  Zip       : {zip_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
