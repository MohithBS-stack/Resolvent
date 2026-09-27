"""
Phase 5a — Matching Model: Training
====================================

Hard-negative sampling + LightGBM gradient-boosted classifier.

Training pair construction:
  For each S1 entity e:
    positives = true matches from ground truth
    negatives = sample K negatives from e's blocking candidates (not true matches)
    K = min(5 * |positives|, 50)  # class balance control

Usage:
    python -m src.matching.train \
        --features data/features.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --model-output models/lgbm_matcher.txt
"""

import argparse
import os
import pickle
import sys
import time
from typing import Dict, Optional, Set, Tuple

# Ensure package root is on sys.path for standalone script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.similarity import FEATURE_COLUMNS


# ──────────────────────────────────────────────────────────────────────
# Hard-negative sampling
# ──────────────────────────────────────────────────────────────────────

def build_training_pairs(
    df_features: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    neg_ratio: int = 5,
    max_neg: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build labeled training DataFrame with hard-negative sampling.

    For each S1 entity:
      - positives: candidate pairs that are true matches
      - negatives: sample K from blocking candidates that are NOT true matches
        (these are hard negatives — similar-but-different businesses)
      - K = min(neg_ratio * |positives|, max_neg)

    Returns DataFrame with FEATURE_COLUMNS + 'label' (1=match, 0=non-match).
    """
    rng = np.random.RandomState(seed)
    print(f"  Building training pairs (neg_ratio={neg_ratio}, max_neg={max_neg})...")

    # Group candidates by S1 entity
    grouped = df_features.groupby("source1_entity_id")

    pos_rows = []
    neg_rows = []

    for s1_id, group in grouped:
        true_matches = ground_truth.get(s1_id, set())
        if not true_matches:
            continue  # singleton — no positives to train on

        # Find positives and negatives within this S1's candidates
        pos_mask = group["source23_entity_id"].isin(true_matches)
        positives = group[pos_mask]
        negatives = group[~pos_mask]

        if len(positives) == 0:
            continue  # true matches not in candidate set (blocking miss)

        pos_rows.append(positives)

        # Hard-negative sampling
        n_neg = min(neg_ratio * len(positives), max_neg, len(negatives))
        if n_neg > 0:
            sampled_neg = negatives.sample(n=n_neg, random_state=rng)
            neg_rows.append(sampled_neg)

    if not pos_rows:
        raise ValueError("No positive training pairs found! Check ground truth and candidates.")

    df_pos = pd.concat(pos_rows, ignore_index=True)
    df_pos["label"] = 1

    if neg_rows:
        df_neg = pd.concat(neg_rows, ignore_index=True)
        df_neg["label"] = 0
    else:
        df_neg = pd.DataFrame(columns=list(df_pos.columns))

    df_train = pd.concat([df_pos, df_neg], ignore_index=True)
    # Shuffle
    df_train = df_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_pos = len(df_pos)
    n_neg = len(df_neg)
    print(f"    Positives: {n_pos:,}")
    print(f"    Negatives: {n_neg:,}")
    print(f"    Total    : {len(df_train):,}")
    print(f"    Ratio    : 1:{n_neg / max(n_pos, 1):.1f}")

    return df_train


# ──────────────────────────────────────────────────────────────────────
# LightGBM training
# ──────────────────────────────────────────────────────────────────────

def train_lgbm(
    df_train: pd.DataFrame,
    feature_cols: list = None,
    model_output: str = "models/lgbm_matcher.txt",
    params: Optional[Dict] = None,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM binary classifier on the training pairs.

    Returns the fitted model.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    X = df_train[feature_cols].values.astype(np.float32)
    y = df_train["label"].values.astype(int)

    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "class_weight": "balanced",
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    print(f"  Training LightGBM ({default_params['n_estimators']} trees)...")
    t0 = time.time()

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X, y)

    elapsed = time.time() - t0
    print(f"    Training time: {elapsed:.1f}s")

    # Feature importance
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    print("\n    Feature importances (top 10):")
    for _, row in imp_df.head(10).iterrows():
        print(f"      {row['feature']:30s}  {row['importance']:6d}")

    # Save model
    os.makedirs(os.path.dirname(model_output) or ".", exist_ok=True)
    model.booster_.save_model(model_output)
    print(f"\n    Model saved to {model_output}")

    return model


# ──────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5a: Train LightGBM matcher")
    parser.add_argument("--features", required=True, help="Path to features TSV")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth TSV")
    parser.add_argument("--model-output", default="models/lgbm_matcher.txt",
                        help="Where to save the model")
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--max-neg", type=int, default=50)
    args = parser.parse_args()

    from src.evaluate.score_f05 import parse_ground_truth

    print("Loading features...")
    df_features = pd.read_csv(args.features, sep="\t", encoding="utf-8",
                              encoding_errors="replace")
    print(f"  Features: {len(df_features):,} rows, {len(FEATURE_COLUMNS)} feature columns")

    print("Loading ground truth...")
    gt = parse_ground_truth(args.ground_truth)
    print(f"  Ground truth: {len(gt):,} S1 entities")

    df_train = build_training_pairs(df_features, gt, args.neg_ratio, args.max_neg)
    model = train_lgbm(df_train, model_output=args.model_output)
