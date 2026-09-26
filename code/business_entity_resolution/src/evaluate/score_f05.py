"""
Phase 3 — F0.5 Scorer & Held-Out Validation Split
==================================================

Provides:
  1. ``score_f05_macro`` — vectorized F0.5 macro-averaged scorer
  2. ``create_val_split`` — stratified held-out validation split generator
  3. ``error_gallery`` — worst false merges and worst misses for debugging

Usage:
    python -m src.evaluate.score_f05 \
        --predictions output/matching_results.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv

    python -m src.evaluate.score_f05 \
        --create-split \
        --s1 dataset/train/train_source1.tsv \
        --s2 dataset/train/train_source2.tsv \
        --s3 dataset/train/train_source3.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --val-fraction 0.1 \
        --output-dir data/val_split
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Ground truth & prediction parsing
# ──────────────────────────────────────────────────────────────────────

def parse_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """
    Parse ground truth TSV into {source1_entity_id: set(matched S2/S3 ids)}.
    Singletons have an empty set.
    """
    df = pd.read_csv(gt_path, sep="\t", encoding="utf-8", encoding_errors="replace",
                     dtype=str, keep_default_na=False)
    result: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_raw = str(row.get("matched_entity_ids", "")).strip()
        if matched_raw == "" or matched_raw == "nan":
            result[s1_id] = set()
        else:
            result[s1_id] = {m.strip() for m in matched_raw.split(",") if m.strip()}
    return result


def parse_predictions(pred_path: str) -> Dict[str, Set[str]]:
    """
    Parse predictions TSV into {source1_entity_id: set(matched S2/S3 ids)}.
    """
    df = pd.read_csv(pred_path, sep="\t", encoding="utf-8", encoding_errors="replace",
                     dtype=str, keep_default_na=False)
    result: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_raw = str(row.get("matched_entity_ids", "")).strip()
        if matched_raw == "" or matched_raw == "nan":
            result[s1_id] = set()
        else:
            result[s1_id] = {m.strip() for m in matched_raw.split(",") if m.strip()}
    return result


# ──────────────────────────────────────────────────────────────────────
# F0.5 computation (vectorized via numpy)
# ──────────────────────────────────────────────────────────────────────

def f05_single(pred: Set[str], truth: Set[str]) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F0.5 for a single S1 entity.

    Special cases (per competition rules):
      - truth={} and pred={} → singleton correctly predicted → (1.0, 1.0, 1.0)
      - truth={} and pred≠{} → singleton wrongly merged → (0.0, 0.0, 0.0)
      - truth≠{} and pred={} → missed all matches → (0.0, 0.0, 0.0)
    """
    # Singleton handling
    if len(truth) == 0 and len(pred) == 0:
        return 1.0, 1.0, 1.0
    if len(truth) == 0 and len(pred) > 0:
        return 0.0, 0.0, 0.0  # false merge of a singleton
    if len(truth) > 0 and len(pred) == 0:
        return 0.0, 0.0, 0.0  # missed all matches

    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    beta_sq = 0.25  # 0.5^2
    f05 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return precision, recall, f05


def score_f05_macro(
    pred: Dict[str, Set[str]],
    truth: Dict[str, Set[str]],
) -> Dict[str, object]:
    """
    Compute macro-averaged F0.5 across all S1 entities.

    Returns dict with:
      - f05_macro: float
      - precision_macro: float
      - recall_macro: float
      - n_entities: int
      - n_singletons: int
      - n_singletons_correct: int
      - per_entity: list of (s1_id, precision, recall, f05) sorted by f05 ascending
    """
    all_s1 = sorted(set(truth.keys()) | set(pred.keys()))

    precisions = np.zeros(len(all_s1))
    recalls = np.zeros(len(all_s1))
    f05s = np.zeros(len(all_s1))
    per_entity: List[Tuple[str, float, float, float]] = []

    n_singletons = 0
    n_singletons_correct = 0

    for i, s1_id in enumerate(all_s1):
        t = truth.get(s1_id, set())
        p = pred.get(s1_id, set())

        prec, rec, f = f05_single(p, t)
        precisions[i] = prec
        recalls[i] = rec
        f05s[i] = f
        per_entity.append((s1_id, prec, rec, f))

        if len(t) == 0:
            n_singletons += 1
            if len(p) == 0:
                n_singletons_correct += 1

    return {
        "f05_macro": float(np.mean(f05s)),
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "n_entities": len(all_s1),
        "n_singletons": n_singletons,
        "n_singletons_correct": n_singletons_correct,
        "per_entity": sorted(per_entity, key=lambda x: x[3]),  # worst first
    }


# ──────────────────────────────────────────────────────────────────────
# Error gallery
# ──────────────────────────────────────────────────────────────────────

def error_gallery(
    pred: Dict[str, Set[str]],
    truth: Dict[str, Set[str]],
    top_k: int = 20,
) -> Dict[str, List]:
    """
    Return worst false merges and worst misses for debugging.
    """
    false_merges = []  # pred has IDs not in truth
    misses = []        # truth has IDs not in pred

    for s1_id in truth:
        t = truth[s1_id]
        p = pred.get(s1_id, set())
        fps = p - t
        fns = t - p
        if fps:
            false_merges.append((s1_id, fps, t, p))
        if fns:
            misses.append((s1_id, fns, t, p))

    # Sort by number of errors descending
    false_merges.sort(key=lambda x: -len(x[1]))
    misses.sort(key=lambda x: -len(x[1]))

    return {
        "false_merges": false_merges[:top_k],
        "misses": misses[:top_k],
    }


def print_error_gallery(gallery: Dict[str, List]) -> None:
    """Pretty-print the error gallery."""
    print("\n--- WORST FALSE MERGES (pred has IDs not in truth) ---")
    for s1_id, fps, truth_set, pred_set in gallery["false_merges"][:10]:
        print(f"  {s1_id}: false_pos={fps}  truth={truth_set}  pred={pred_set}")

    print("\n--- WORST MISSES (truth has IDs not in pred) ---")
    for s1_id, fns, truth_set, pred_set in gallery["misses"][:10]:
        print(f"  {s1_id}: false_neg={fns}  truth={truth_set}  pred={pred_set}")


# ──────────────────────────────────────────────────────────────────────
# Validation split creation
# ──────────────────────────────────────────────────────────────────────

def create_val_split(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    gt_path: str,
    output_dir: str,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Create a stratified held-out validation split.

    Stratified by country so both US and India are represented in val.
    Splits S1 entities, then filters S2/S3/GT to only include relevant records.
    """
    print(f"Creating validation split (fraction={val_fraction}, seed={seed})...")
    os.makedirs(output_dir, exist_ok=True)

    # Load S1
    df_s1 = pd.read_csv(s1_path, sep="\t", encoding="utf-8", encoding_errors="replace")
    print(f"  S1 total: {len(df_s1):,}")

    # Stratified split by country
    rng = np.random.RandomState(seed)
    val_indices = []
    train_indices = []

    for country, group in df_s1.groupby("country"):
        n_val = max(1, int(len(group) * val_fraction))
        shuffled = rng.permutation(group.index.values)
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])
        print(f"  country='{country}': total={len(group):,}, val={n_val:,}, train={len(group) - n_val:,}")

    val_s1 = df_s1.loc[val_indices].copy()
    train_s1 = df_s1.loc[train_indices].copy()
    val_s1_ids = set(val_s1["entity_id"].values)
    train_s1_ids = set(train_s1["entity_id"].values)

    # Load ground truth
    df_gt = pd.read_csv(gt_path, sep="\t", encoding="utf-8", encoding_errors="replace",
                        dtype=str, keep_default_na=False)

    val_gt = df_gt[df_gt["source1_entity_id"].isin(val_s1_ids)].copy()
    train_gt = df_gt[df_gt["source1_entity_id"].isin(train_s1_ids)].copy()

    # Determine which S2/S3 IDs belong to val vs train
    val_s23_ids = set()
    for _, row in val_gt.iterrows():
        matched_raw = str(row.get("matched_entity_ids", "")).strip()
        if matched_raw and matched_raw != "nan":
            for mid in matched_raw.split(","):
                mid = mid.strip()
                if mid:
                    val_s23_ids.add(mid)

    # Load S2, S3
    df_s2 = pd.read_csv(s2_path, sep="\t", encoding="utf-8", encoding_errors="replace")
    df_s3 = pd.read_csv(s3_path, sep="\t", encoding="utf-8", encoding_errors="replace")

    # For val split, include all S2/S3 records from the same countries as val S1
    # (the blocker needs to search through all S23 in that country partition)
    val_countries = set(val_s1["country"].unique())
    val_s2 = df_s2[df_s2["country"].isin(val_countries)].copy()
    val_s3 = df_s3[df_s3["country"].isin(val_countries)].copy()

    # Write outputs
    val_s1.to_csv(os.path.join(output_dir, "val_source1.tsv"), sep="\t", index=False)
    val_s2.to_csv(os.path.join(output_dir, "val_source2.tsv"), sep="\t", index=False)
    val_s3.to_csv(os.path.join(output_dir, "val_source3.tsv"), sep="\t", index=False)
    val_gt.to_csv(os.path.join(output_dir, "val_ground_truth.tsv"), sep="\t", index=False)

    train_s1.to_csv(os.path.join(output_dir, "train_source1.tsv"), sep="\t", index=False)
    train_gt.to_csv(os.path.join(output_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    print(f"\n  Val  : S1={len(val_s1):,}, S2={len(val_s2):,}, S3={len(val_s3):,}, GT={len(val_gt):,}")
    print(f"  Train: S1={len(train_s1):,}, GT={len(train_gt):,}")
    print(f"  Written to {output_dir}/")


# ──────────────────────────────────────────────────────────────────────
# Country stress test split (train US -> validate India)
# ──────────────────────────────────────────────────────────────────────

def create_country_stress_split(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    gt_path: str,
    output_dir: str,
    holdout_country: str = "India",
) -> None:
    """
    Create a stress-test split where all entities from *holdout_country*
    are in validation and everything else is in training.
    Simulates France OOD before test data reveals it.
    """
    stress_dir = os.path.join(output_dir, "stress_test")
    os.makedirs(stress_dir, exist_ok=True)
    print(f"Creating country stress split (holdout='{holdout_country}')...")

    df_s1 = pd.read_csv(s1_path, sep="\t", encoding="utf-8", encoding_errors="replace")
    df_gt = pd.read_csv(gt_path, sep="\t", encoding="utf-8", encoding_errors="replace",
                        dtype=str, keep_default_na=False)
    df_s2 = pd.read_csv(s2_path, sep="\t", encoding="utf-8", encoding_errors="replace")
    df_s3 = pd.read_csv(s3_path, sep="\t", encoding="utf-8", encoding_errors="replace")

    val_s1 = df_s1[df_s1["country"] == holdout_country].copy()
    train_s1 = df_s1[df_s1["country"] != holdout_country].copy()

    val_s1_ids = set(val_s1["entity_id"].values)
    val_gt = df_gt[df_gt["source1_entity_id"].isin(val_s1_ids)].copy()
    train_s1_ids = set(train_s1["entity_id"].values)
    train_gt = df_gt[df_gt["source1_entity_id"].isin(train_s1_ids)].copy()

    val_s2 = df_s2[df_s2["country"] == holdout_country].copy()
    val_s3 = df_s3[df_s3["country"] == holdout_country].copy()

    val_s1.to_csv(os.path.join(stress_dir, "val_source1.tsv"), sep="\t", index=False)
    val_s2.to_csv(os.path.join(stress_dir, "val_source2.tsv"), sep="\t", index=False)
    val_s3.to_csv(os.path.join(stress_dir, "val_source3.tsv"), sep="\t", index=False)
    val_gt.to_csv(os.path.join(stress_dir, "val_ground_truth.tsv"), sep="\t", index=False)
    train_s1.to_csv(os.path.join(stress_dir, "train_source1.tsv"), sep="\t", index=False)
    train_gt.to_csv(os.path.join(stress_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    print(f"  Holdout '{holdout_country}': val S1={len(val_s1):,}, train S1={len(train_s1):,}")
    print(f"  Written to {stress_dir}/")


# ──────────────────────────────────────────────────────────────────────
# Print results
# ──────────────────────────────────────────────────────────────────────

def print_score_summary(results: Dict[str, object]) -> None:
    """Pretty-print scoring results."""
    print("\n" + "=" * 60)
    print("F0.5 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  F0.5 (macro)     : {results['f05_macro']:.4f}")
    print(f"  Precision (macro): {results['precision_macro']:.4f}")
    print(f"  Recall (macro)   : {results['recall_macro']:.4f}")
    print(f"  Total entities   : {results['n_entities']:,}")
    print(f"  Singletons       : {results['n_singletons']:,} "
          f"({results['n_singletons_correct']:,} correct)")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: F0.5 scorer & validation split generator"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Score subcommand
    score_p = subparsers.add_parser("score", help="Score predictions against ground truth")
    score_p.add_argument("--predictions", required=True, help="Path to matching_results.tsv")
    score_p.add_argument("--ground-truth", required=True, help="Path to ground truth TSV")
    score_p.add_argument("--show-errors", action="store_true",
                         help="Show error gallery (worst false merges/misses)")

    # Split subcommand
    split_p = subparsers.add_parser("split", help="Create validation split")
    split_p.add_argument("--s1", required=True, help="Path to train_source1.tsv")
    split_p.add_argument("--s2", required=True, help="Path to train_source2.tsv")
    split_p.add_argument("--s3", required=True, help="Path to train_source3.tsv")
    split_p.add_argument("--ground-truth", required=True, help="Path to train_ground_truth.tsv")
    split_p.add_argument("--val-fraction", type=float, default=0.1)
    split_p.add_argument("--output-dir", default="data/val_split")
    split_p.add_argument("--stress-test", action="store_true",
                         help="Also create country stress-test split (holdout India)")

    args = parser.parse_args()

    if args.command == "score":
        truth = parse_ground_truth(args.ground_truth)
        pred = parse_predictions(args.predictions)
        results = score_f05_macro(pred, truth)
        print_score_summary(results)
        if args.show_errors:
            gallery = error_gallery(pred, truth)
            print_error_gallery(gallery)

    elif args.command == "split":
        create_val_split(
            args.s1, args.s2, args.s3, args.ground_truth,
            args.output_dir, args.val_fraction,
        )
        if args.stress_test:
            create_country_stress_split(
                args.s1, args.s2, args.s3, args.ground_truth,
                args.output_dir, holdout_country="India",
            )
    else:
        parser.print_help()
