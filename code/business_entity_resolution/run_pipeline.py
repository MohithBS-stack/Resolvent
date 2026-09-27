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
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold (skip sweep). If not set, sweeps on val.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing model)")
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip validation split creation")
    parser.add_argument("--validator", default="utils/validate_submission.py",
                        help="Path to organizer's validator script")
    args = parser.parse_args()

    pipeline_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    val_dir = os.path.join(data_dir, "val_split")

    # ── STAGE 1: Load & Normalize ─────────────────────────────────────
    print_stage(1, "LOAD & NORMALIZE")

    train_s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    train_s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    train_s3_path = os.path.join(args.train_dir, "train_source3.tsv")
    train_gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")

    test_s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(args.test_dir, "test_source3.tsv")

    print("Loading and normalizing train sources...")
    train_s1 = load_and_normalize(train_s1_path)
    train_s2 = load_and_normalize(train_s2_path)
    train_s3 = load_and_normalize(train_s3_path)
    print(f"  Train S1: {len(train_s1):,}")
    print(f"  Train S2: {len(train_s2):,}")
    print(f"  Train S3: {len(train_s3):,}")

    print("\nLoading and normalizing test sources...")
    test_s1 = load_and_normalize(test_s1_path)
    test_s2 = load_and_normalize(test_s2_path)
    test_s3 = load_and_normalize(test_s3_path)
    print(f"  Test S1: {len(test_s1):,}")
    print(f"  Test S2: {len(test_s2):,}")
    print(f"  Test S3: {len(test_s3):,}")

    print("\nLoading ground truth...")
    gt_dict = parse_ground_truth(train_gt_path)
    print(f"  Ground truth: {len(gt_dict):,} S1 entities")

    # ── STAGE 2: Validation Split ─────────────────────────────────────
    print_stage(2, "VALIDATION SPLIT")

    if not args.skip_val:
        create_val_split(
            train_s1_path, train_s2_path, train_s3_path, train_gt_path,
            val_dir, args.val_fraction,
        )
    else:
        print("  Skipped (--skip-val)")

    # Load val split for later threshold sweep
    val_gt_path = os.path.join(val_dir, "val_ground_truth.tsv")
    val_s1_path = os.path.join(val_dir, "val_source1.tsv")
    if os.path.exists(val_gt_path):
        val_gt_dict = parse_ground_truth(val_gt_path)
        val_s1 = load_and_normalize(val_s1_path)
        print(f"  Val S1: {len(val_s1):,}, Val GT: {len(val_gt_dict):,}")
    else:
        val_gt_dict = None
        val_s1 = None
        print("  No validation split found")

    # ── STAGE 3: Blocking (Train) ─────────────────────────────────────
    print_stage(3, "BLOCKING (TRAIN)")

    train_s23 = merge_s2_s3(train_s2, train_s3)
    print(f"  Train S23 (merged): {len(train_s23):,}")

    blocker = InvertedIndexBlocker()
    train_gt_df = pd.read_csv(train_gt_path, sep="\t", encoding="utf-8",
                              encoding_errors="replace", dtype=str, keep_default_na=False)
    train_candidates = blocker.generate_candidates(train_s1, train_s23, ground_truth=train_gt_df)
    print(f"  Train candidate pairs: {len(train_candidates):,}")

    # ── STAGE 4: Feature Engineering (Train) ──────────────────────────
    print_stage(4, "FEATURE ENGINEERING (TRAIN)")

    extractor = FeatureExtractor()
    extractor.fit(train_s1, train_s23)

    s1_lookup = build_lookup(train_s1)
    s23_lookup = build_lookup(train_s23)

    train_features = extractor.extract(train_candidates, s1_lookup, s23_lookup)

    # ── STAGE 5: Training ─────────────────────────────────────────────
    print_stage(5, "LGBM TRAINING")

    model_path = os.path.join(args.model_dir, "lgbm_matcher.txt")

    if not args.skip_train:
        df_train = build_training_pairs(train_features, gt_dict)
        model = train_lgbm(df_train, model_output=model_path)
    else:
        import lightgbm as lgb
        print(f"  Loading existing model from {model_path}")
        model = lgb.Booster(model_file=model_path)

    # ── STAGE 6: Threshold Sweep on Validation ────────────────────────
    print_stage(6, "THRESHOLD SWEEP")

    threshold = args.threshold

    if threshold is None and val_gt_dict is not None and val_s1 is not None:
        # Block validation set
        val_s2_path = os.path.join(val_dir, "val_source2.tsv")
        val_s3_path = os.path.join(val_dir, "val_source3.tsv")

        if os.path.exists(val_s2_path) and os.path.exists(val_s3_path):
            val_s2 = load_and_normalize(val_s2_path)
            val_s3 = load_and_normalize(val_s3_path)
            val_s23 = merge_s2_s3(val_s2, val_s3)

            print("  Blocking validation set...")
            val_blocker = InvertedIndexBlocker()
            val_candidates = val_blocker.generate_candidates(val_s1, val_s23)

            print("  Computing validation features...")
            val_extractor = FeatureExtractor()
            val_extractor.fit(val_s1, val_s23)
            val_s1_lookup = build_lookup(val_s1)
            val_s23_lookup = build_lookup(val_s23)
            val_features = val_extractor.extract(val_candidates, val_s1_lookup, val_s23_lookup)

            print("  Running validation inference...")
            import lightgbm as lgb
            if isinstance(model, lgb.LGBMClassifier):
                booster = model.booster_
            else:
                booster = model
            val_probs = batch_predict(val_features, booster)

            threshold, best_f05, sweep_df = threshold_sweep(
                val_features, val_probs, val_gt_dict
            )
        else:
            print("  Val S2/S3 not found, using default threshold=0.5")
            threshold = 0.5
    elif threshold is None:
        threshold = 0.5
        print(f"  No validation split available, using default threshold={threshold}")
    else:
        print(f"  Using fixed threshold={threshold}")

    # ── STAGE 7: Blocking + Inference (Test) ──────────────────────────
    print_stage(7, "BLOCKING + INFERENCE (TEST)")

    test_s23 = merge_s2_s3(test_s2, test_s3)
    print(f"  Test S23 (merged): {len(test_s23):,}")

    print("  Blocking test set...")
    test_blocker = InvertedIndexBlocker()
    test_candidates = test_blocker.generate_candidates(test_s1, test_s23)

    print("  Computing test features...")
    test_extractor = FeatureExtractor()
    test_extractor.fit(test_s1, test_s23)
    test_s1_lookup = build_lookup(test_s1)
    test_s23_lookup = build_lookup(test_s23)
    test_features = test_extractor.extract(test_candidates, test_s1_lookup, test_s23_lookup)

    print("  Running test inference...")
    import lightgbm as lgb
    if isinstance(model, lgb.LGBMClassifier):
        booster = model.booster_
    else:
        booster = model
    test_probs = batch_predict(test_features, booster)

    all_test_s1_ids = set(test_s1["entity_id"].values)
    test_matches = apply_threshold(test_features, test_probs, threshold, all_test_s1_ids)

    n_matched = sum(1 for m in test_matches.values() if m)
    n_singleton = sum(1 for m in test_matches.values() if not m)
    print(f"  Threshold={threshold:.2f}: {n_matched:,} matched, {n_singleton:,} singletons")

    # ── STAGE 8: Packaging ────────────────────────────────────────────
    print_stage(8, "SUBMISSION PACKAGING")

    # Build candidate dict for the output format
    test_cand_dict = candidates_df_to_dict(test_candidates)
    # Ensure all test S1 IDs are in candidates
    for s1_id in all_test_s1_ids:
        if s1_id not in test_cand_dict:
            test_cand_dict[s1_id] = set()

    # Valid S2/S3 IDs for existence check
    valid_s23_ids = set(test_s2["entity_id"].values) | set(test_s3["entity_id"].values)

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

    # ── Summary ───────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  Output    : {matching_output}")
    print(f"            : {candidate_output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
