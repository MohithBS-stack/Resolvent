"""
Dashboard Data Exporter
========================

Generates JSON files from pipeline outputs that the dashboard reads.

Usage:
    python export_dashboard_data.py \
        --val-predictions output/val_matching_results.tsv \
        --val-ground-truth data/val_split/val_ground_truth.tsv \
        --blocking-stats data/blocking_stats.json \
        --model models/lgbm_matcher.txt \
        --output-dir dashboard/data
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluate.score_f05 import parse_ground_truth, parse_predictions, score_f05_macro
from src.features.similarity import FEATURE_COLUMNS


def export_scorecard(pred_path, gt_path, threshold, output_dir):
    """Export scorecard JSON."""
    gt = parse_ground_truth(gt_path)
    pred = parse_predictions(pred_path)
    results = score_f05_macro(pred, gt)

    scorecard = {
        "f05_macro": results["f05_macro"],
        "precision_macro": results["precision_macro"],
        "recall_macro": results["recall_macro"],
        "n_entities": results["n_entities"],
        "n_singletons": results["n_singletons"],
        "n_singletons_correct": results["n_singletons_correct"],
        "best_threshold": threshold,
    }

    path = os.path.join(output_dir, "scorecard.json")
    with open(path, "w") as f:
        json.dump(scorecard, f, indent=2)
    print(f"  Wrote {path}")


def export_feature_importances(model_path, output_dir):
    """Export feature importances JSON."""
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path)
    importances = model.feature_importance(importance_type="split")
    names = model.feature_name()

    data = {
        "feature_importances": [
            {"name": n, "importance": int(v)}
            for n, v in sorted(zip(names, importances), key=lambda x: -x[1])
        ]
    }

    path = os.path.join(output_dir, "features.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export dashboard JSON data")
    parser.add_argument("--val-predictions", help="Path to val matching_results.tsv")
    parser.add_argument("--val-ground-truth", help="Path to val ground truth")
    parser.add_argument("--model", help="Path to LightGBM model")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="dashboard/data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.val_predictions and args.val_ground_truth:
        export_scorecard(args.val_predictions, args.val_ground_truth,
                        args.threshold, args.output_dir)

    if args.model:
        export_feature_importances(args.model, args.output_dir)

    print("Dashboard data export complete.")
