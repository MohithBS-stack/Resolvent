"""
End-to-end integration test on synthetic data.

Creates small fake source files, runs the full pipeline, and verifies
all stages work and wire together correctly.
"""
import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingest.load_normalize import load_and_normalize
from src.blocking.candidate_gen import InvertedIndexBlocker, merge_s2_s3
from src.features.similarity import FeatureExtractor, build_lookup, FEATURE_COLUMNS
from src.matching.train import build_training_pairs, train_lgbm
from src.matching.predict import batch_predict, apply_threshold, threshold_sweep
from src.evaluate.score_f05 import parse_ground_truth, score_f05_macro, print_score_summary


def create_synthetic_data(tmpdir):
    """Create small synthetic TSV files for testing."""
    # S1: 10 entities
    s1_data = {
        "entity_id": [f"S1-{i:09d}" for i in range(1, 11)],
        "business_name": [
            "Acme Corporation", "Beta Solutions Pvt Ltd", "Gamma Inc",
            "Delta Technologies", "Epsilon Services LLC",
            "Zeta Industries", "Eta Consulting", "Theta Labs",
            "Iota Global", "Kappa Systems",
        ],
        "business_address": [
            "123 Main St, New York", "456 Oak Ave, Mumbai",
            "789 Pine Rd, Delhi", "101 Elm Blvd, LA",
            "202 Cedar Ln, Chicago", "303 Birch Dr, Houston",
            "404 Maple Ct, Phoenix", "505 Walnut Pl, Philly",
            "606 Cherry Hwy, San Antonio", "707 Ash St, San Diego",
        ],
        "country": ["US", "India", "India", "US", "US",
                     "US", "US", "US", "India", "US"],
    }
    df_s1 = pd.DataFrame(s1_data)

    # S2: some matching records + distractors
    s2_data = {
        "entity_id": [f"S2-{i:09d}" for i in range(1, 9)],
        "business_name": [
            "Acme Corp", "Beta Solutions", "Gamma Limited",
            "Random Business", "Delta Tech", "Epsilon Svc",
            "Totally Different Co", "Another Random Inc",
        ],
        "business_address": [
            "123 Main Street, New York", "456 Oak Avenue, Mumbai",
            "789 Pine Road, Delhi", "999 Nowhere",
            "101 Elm Boulevard, LA", "202 Cedar Lane, Chicago",
            "111 Fake St", "222 Other Ave",
        ],
        "country": ["US", "India", "India", "US", "US", "US", "US", "US"],
    }
    df_s2 = pd.DataFrame(s2_data)

    # S3: a few more matches
    s3_data = {
        "entity_id": [f"S3-{i:09d}" for i in range(1, 5)],
        "business_name": [
            "ACME CORPORATION", "Zeta Industries Ltd",
            "Iota Global Services", "Kappa Sys",
        ],
        "business_address": [
            "123 Main St NYC", "303 Birch Drive, Houston",
            "606 Cherry Highway, San Antonio", "707 Ash Street, San Diego",
        ],
        "country": ["US", "US", "India", "US"],
    }
    df_s3 = pd.DataFrame(s3_data)

    # Ground truth
    gt_data = {
        "source1_entity_id": [f"S1-{i:09d}" for i in range(1, 11)],
        "matched_entity_ids": [
            "S2-000000001,S3-000000001",  # Acme
            "S2-000000002",               # Beta
            "S2-000000003",               # Gamma
            "S2-000000005",               # Delta
            "S2-000000006",               # Epsilon
            "S3-000000002",               # Zeta
            "",                           # Eta - singleton
            "",                           # Theta - singleton
            "S3-000000003",               # Iota
            "S3-000000004",               # Kappa
        ],
    }
    df_gt = pd.DataFrame(gt_data)

    # Write files
    s1_path = os.path.join(tmpdir, "train_source1.tsv")
    s2_path = os.path.join(tmpdir, "train_source2.tsv")
    s3_path = os.path.join(tmpdir, "train_source3.tsv")
    gt_path = os.path.join(tmpdir, "train_ground_truth.tsv")

    df_s1.to_csv(s1_path, sep="\t", index=False)
    df_s2.to_csv(s2_path, sep="\t", index=False)
    df_s3.to_csv(s3_path, sep="\t", index=False)
    df_gt.to_csv(gt_path, sep="\t", index=False)

    return s1_path, s2_path, s3_path, gt_path


def test_full_pipeline():
    tmpdir = tempfile.mkdtemp(prefix="resolvent_test_")
    try:
        print("1. Creating synthetic data...")
        s1_path, s2_path, s3_path, gt_path = create_synthetic_data(tmpdir)

        print("\n2. Loading and normalizing...")
        s1 = load_and_normalize(s1_path)
        s2 = load_and_normalize(s2_path)
        s3 = load_and_normalize(s3_path)
        assert len(s1) == 10
        assert "norm_name" in s1.columns
        assert "norm_address" in s1.columns
        print(f"   S1={len(s1)}, S2={len(s2)}, S3={len(s3)}")

        print("\n3. Blocking...")
        s23 = merge_s2_s3(s2, s3)
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        blocker = InvertedIndexBlocker()
        candidates = blocker.generate_candidates(s1, s23, ground_truth=gt_df)
        assert len(candidates) > 0
        assert "source1_entity_id" in candidates.columns
        print(f"   Candidates: {len(candidates)}")

        print("\n4. Feature extraction...")
        extractor = FeatureExtractor()
        extractor.fit(s1, s23)
        s1_lookup = build_lookup(s1)
        s23_lookup = build_lookup(s23)
        features = extractor.extract(candidates, s1_lookup, s23_lookup)
        assert len(features) == len(candidates)
        for col in FEATURE_COLUMNS:
            assert col in features.columns, f"Missing feature: {col}"
        print(f"   Features: {len(features)} rows x {len(FEATURE_COLUMNS)} features")

        print("\n5. Training...")
        gt_dict = parse_ground_truth(gt_path)
        train_df = build_training_pairs(features, gt_dict, neg_ratio=3, max_neg=10)
        assert "label" in train_df.columns
        assert train_df["label"].sum() > 0  # has positives

        model_path = os.path.join(tmpdir, "model.txt")
        model = train_lgbm(train_df, model_output=model_path,
                          params={"n_estimators": 50, "verbose": -1})
        assert os.path.exists(model_path)
        print(f"   Model saved: {model_path}")

        print("\n6. Prediction...")
        import lightgbm as lgb
        booster = model.booster_
        probs = batch_predict(features, booster)
        assert len(probs) == len(features)
        assert probs.min() >= 0 and probs.max() <= 1
        print(f"   Probabilities: min={probs.min():.4f}, max={probs.max():.4f}")

        print("\n7. Threshold sweep...")
        best_thresh, best_f05, sweep_df = threshold_sweep(
            features, probs, gt_dict,
            thresholds=np.arange(0.1, 0.9, 0.1)
        )
        assert 0 < best_thresh < 1
        print(f"   Best threshold: {best_thresh:.2f}, F0.5: {best_f05:.4f}")

        print("\n8. Applying threshold & scoring...")
        all_s1_ids = set(s1["entity_id"].values)
        matches = apply_threshold(features, probs, best_thresh, all_s1_ids)
        results = score_f05_macro(matches, gt_dict)
        print_score_summary(results)

        assert results["n_entities"] == 10
        assert 0 <= results["f05_macro"] <= 1

        print("\n9. Packaging & self-checks...")
        from src.submit.package import (
            candidates_df_to_dict,
            check_s1_coverage,
            check_id_existence,
            check_superset,
            write_matching_results_tsv,
            write_candidate_pairs_tsv,
        )
        cand_dict = candidates_df_to_dict(candidates)
        for s1_id in all_s1_ids:
            if s1_id not in cand_dict:
                cand_dict[s1_id] = set()

        valid_s23_ids = set(s2["entity_id"].values) | set(s3["entity_id"].values)
        assert check_s1_coverage(matches, all_s1_ids, "matching_results")
        assert check_s1_coverage(cand_dict, all_s1_ids, "candidate_pairs")
        assert check_id_existence(matches, valid_s23_ids, "matching_results")
        assert check_superset(matches, cand_dict)

        out_match = os.path.join(tmpdir, "matching_results.tsv")
        out_cand = os.path.join(tmpdir, "candidate_pairs.tsv")
        write_matching_results_tsv(matches, out_match)
        write_candidate_pairs_tsv(cand_dict, out_cand)
        assert os.path.exists(out_match)
        assert os.path.exists(out_cand)

        print("\n[PASS] Full pipeline integration test passed!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_full_pipeline()
