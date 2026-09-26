"""
Phase 5b — Matching Model: Prediction & Threshold Sweep
========================================================

Batch inference on candidate pairs, F0.5-calibrated threshold sweep on
validation set, and final match set generation.

Usage:
    python -m src.matching.predict \
        --features data/features.tsv \
        --model models/lgbm_matcher.txt \
        --threshold 0.5 \
        --output output/matching_results.tsv

    # Threshold sweep (requires ground truth):
    python -m src.matching.predict \
        --features data/val_features.tsv \
        --model models/lgbm_matcher.txt \
        --ground-truth data/val_split/val_ground_truth.tsv \
        --sweep
"""

import argparse
import os
import time
from typing import Dict, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.similarity import FEATURE_COLUMNS


# ──────────────────────────────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────────────────────────────

def batch_predict(
    df_features: pd.DataFrame,
    model: lgb.Booster,
    chunk_size: int = 100000,
) -> np.ndarray:
    """
    Predict match probability for all candidate pairs in chunks.

    Returns array of probabilities (P(match)).
    """
    n = len(df_features)
    probs = np.zeros(n, dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        X = df_features.iloc[start:end][FEATURE_COLUMNS].values.astype(np.float32)
        probs[start:end] = model.predict(X)

    return probs


# ──────────────────────────────────────────────────────────────────────
# Threshold application → match sets
# ──────────────────────────────────────────────────────────────────────

def apply_threshold(
    df_features: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    all_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Apply threshold to probabilities and collect per-S1 match sets.

    Parameters
    ----------
    df_features : DataFrame
        Must have columns [source1_entity_id, source23_entity_id].
    probs : ndarray
        Match probabilities (same length as df_features).
    threshold : float
        Pairs with prob >= threshold are considered matches.
    all_s1_ids : set, optional
        If provided, ensures every S1 entity has an entry (empty set for singletons).

    Returns
    -------
    dict: {source1_entity_id: set(matched source23 entity_ids)}
    """
    matches: Dict[str, Set[str]] = {}

    mask = probs >= threshold
    matched_df = df_features[mask]

    for _, row in matched_df.iterrows():
        s1_id = row["source1_entity_id"]
        s23_id = row["source23_entity_id"]
        if s1_id not in matches:
            matches[s1_id] = set()
        matches[s1_id].add(s23_id)

    # Ensure all S1 entities are present (singletons get empty set)
    if all_s1_ids:
        for s1_id in all_s1_ids:
            if s1_id not in matches:
                matches[s1_id] = set()

    return matches


# ──────────────────────────────────────────────────────────────────────
# F0.5-calibrated threshold sweep
# ──────────────────────────────────────────────────────────────────────

def threshold_sweep(
    df_features: pd.DataFrame,
    probs: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Sweep thresholds on validation set, pick the one maximizing F0.5 macro.

    Returns (best_threshold, best_f05, sweep_results_df).
    """
    from src.evaluate.score_f05 import score_f05_macro

    if thresholds is None:
        thresholds = np.arange(0.10, 0.96, 0.01)

    all_s1_ids = set(ground_truth.keys())
    results = []

    print(f"  Sweeping {len(thresholds)} thresholds...")
    t0 = time.time()

    for t in thresholds:
        matches = apply_threshold(df_features, probs, t, all_s1_ids)
        scores = score_f05_macro(matches, ground_truth)
        results.append({
            "threshold": t,
            "f05": scores["f05_macro"],
            "precision": scores["precision_macro"],
            "recall": scores["recall_macro"],
        })

    elapsed = time.time() - t0
    df_results = pd.DataFrame(results)

    best_idx = df_results["f05"].idxmax()
    best_thresh = df_results.loc[best_idx, "threshold"]
    best_f05 = df_results.loc[best_idx, "f05"]

    print(f"    Sweep complete in {elapsed:.1f}s")
    print(f"\n    Best threshold: {best_thresh:.2f}")
    print(f"    Best F0.5     : {best_f05:.4f}")
    print(f"    Precision     : {df_results.loc[best_idx, 'precision']:.4f}")
    print(f"    Recall        : {df_results.loc[best_idx, 'recall']:.4f}")

    # Show nearby thresholds for context
    print("\n    Threshold sweep (top 10 by F0.5):")
    top10 = df_results.nlargest(10, "f05")
    for _, row in top10.iterrows():
        print(f"      t={row['threshold']:.2f}  F0.5={row['f05']:.4f}  "
              f"P={row['precision']:.4f}  R={row['recall']:.4f}")

    return best_thresh, best_f05, df_results


# ──────────────────────────────────────────────────────────────────────
# Write matching_results.tsv
# ──────────────────────────────────────────────────────────────────────

def write_matching_results(
    matches: Dict[str, Set[str]],
    output_path: str,
) -> None:
    """
    Write matching_results.tsv with columns:
    [source1_entity_id, matched_entity_ids]
    """
    rows = []
    for s1_id in sorted(matches.keys()):
        matched = ",".join(sorted(matches[s1_id]))
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"Wrote {len(df):,} rows to {output_path}")

    n_matched = sum(1 for m in matches.values() if m)
    n_singleton = sum(1 for m in matches.values() if not m)
    print(f"  Matched entities: {n_matched:,}")
    print(f"  Singletons      : {n_singleton:,}")


# ──────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5b: Predict matches & threshold sweep")
    parser.add_argument("--features", required=True, help="Path to features TSV")
    parser.add_argument("--model", required=True, help="Path to LightGBM model file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold (if not sweeping)")
    parser.add_argument("--output", default="output/matching_results.tsv",
                        help="Output matching results TSV")
    parser.add_argument("--ground-truth", default=None,
                        help="Ground truth TSV (for threshold sweep)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run threshold sweep (requires --ground-truth)")
    parser.add_argument("--all-s1-ids", default=None,
                        help="Path to S1 TSV (to ensure all S1 entities in output)")
    args = parser.parse_args()

    print("Loading features...")
    df_features = pd.read_csv(args.features, sep="\t", encoding="utf-8",
                              encoding_errors="replace")
    print(f"  Features: {len(df_features):,} rows")

    print("Loading model...")
    model = lgb.Booster(model_file=args.model)
    print("  Model loaded")

    print("Running inference...")
    probs = batch_predict(df_features, model)
    print(f"  Probability stats: min={probs.min():.4f}, max={probs.max():.4f}, "
          f"mean={probs.mean():.4f}")

    # Threshold selection
    threshold = args.threshold
    if args.sweep and args.ground_truth:
        from src.evaluate.score_f05 import parse_ground_truth
        gt = parse_ground_truth(args.ground_truth)
        threshold, best_f05, _ = threshold_sweep(df_features, probs, gt)
    elif threshold is None:
        threshold = 0.5
        print(f"  Using default threshold: {threshold}")

    # Load all S1 IDs for complete output
    all_s1_ids = None
    if args.all_s1_ids:
        s1_df = pd.read_csv(args.all_s1_ids, sep="\t", encoding="utf-8",
                            encoding_errors="replace")
        all_s1_ids = set(s1_df["entity_id"].values)

    # Apply threshold
    print(f"\nApplying threshold={threshold:.2f}...")
    matches = apply_threshold(df_features, probs, threshold, all_s1_ids)

    # Write output
    write_matching_results(matches, args.output)
