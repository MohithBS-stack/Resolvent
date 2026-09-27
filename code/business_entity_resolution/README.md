# Business Entity Resolution — Code Package

## Overview

End-to-end pipeline to resolve business entities across three noisy data sources
(Source 1, Source 2, Source 3) for the Amazon ML Challenge 2026.
Produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Repository Structure

```
code/business_entity_resolution/
├── src/
│   ├── ingest/         # Phase 1 — Load & normalize raw TSVs
│   ├── blocking/       # Phase 2 — Multi-key inverted-index candidate generation
│   ├── features/       # Phase 4 — Similarity feature engineering
│   ├── matching/       # Phase 5 — LightGBM pairwise classifier
│   ├── evaluate/       # Phase 3 — F₀.5 evaluation harness + val split
│   └── submit/         # Phase 6 — Output packaging + ID-existence validation
├── README.md           # this file
└── requirements.txt    # pinned dependencies
```

## Setup

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## Quick Start (Single-Command Execution)

To run the complete pipeline end-to-end (normalization, validation split, blocking, feature extraction, LightGBM training, F0.5 threshold sweep, test inference, packaging, and validation check):

```bash
python code/business_entity_resolution/run_pipeline.py \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output \
    --data-dir data \
    --models-dir models
```

---

## Step-by-Step Modular Execution

> All commands run from the **project root** (where `dataset/` lives).

### Step 1 — Normalize all sources

```bash
# Train
python code/business_entity_resolution/src/ingest/load_normalize.py \
    --input dataset/train/train_source1.tsv \
    --output data/norm_train_source1.tsv

python code/business_entity_resolution/src/ingest/load_normalize.py \
    --input dataset/train/train_source2.tsv \
    --output data/norm_train_source2.tsv

python code/business_entity_resolution/src/ingest/load_normalize.py \
    --input dataset/train/train_source3.tsv \
    --output data/norm_train_source3.tsv

# Test
python code/business_entity_resolution/src/ingest/load_normalize.py \
    --input dataset/test/test_source1.tsv \
    --output data/norm_test_source1.tsv

python code/business_entity_resolution/src/ingest/load_normalize.py \
    --input dataset/test/test_source2.tsv \
    --output data/norm_test_source2.tsv

python code/business_entity_resolution/src/ingest/load_normalize.py \
    --input dataset/test/test_source3.tsv \
    --output data/norm_test_source3.tsv
```

### Step 2 — Generate candidate pairs (blocking)
```bash
python code/business_entity_resolution/src/blocking/candidate_gen.py \
    --s1 data/norm_test_source1.tsv \
    --s2 data/norm_test_source2.tsv \
    --s3 data/norm_test_source3.tsv \
    --output output/candidate_pairs.tsv
```

### Step 3 — Build validation split
```bash
python code/business_entity_resolution/src/evaluate/score_f05.py split \
    --s1 dataset/train/train_source1.tsv \
    --s2 dataset/train/train_source2.tsv \
    --s3 dataset/train/train_source3.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --val-fraction 0.1 \
    --output-dir data/val_split
```

### Step 4 — Extract features & train matching model
```bash
# Extract features on training candidates
python code/business_entity_resolution/src/features/similarity.py \
    --s1 data/norm_train_source1.tsv \
    --s23 data/norm_train_s23.tsv \
    --candidates data/train_candidate_pairs.tsv \
    --output data/train_features.tsv

# Train LightGBM matcher with hard-negative sampling
python code/business_entity_resolution/src/matching/train.py \
    --features data/train_features.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --model-output models/lgbm_matcher.txt
```

### Step 5 — Run inference & thresholding
```bash
# Extract test features
python code/business_entity_resolution/src/features/similarity.py \
    --s1 data/norm_test_source1.tsv \
    --s23 data/norm_test_s23.tsv \
    --candidates output/candidate_pairs.tsv \
    --output data/test_features.tsv

# Predict and apply calibrated threshold
python code/business_entity_resolution/src/matching/predict.py \
    --features data/test_features.tsv \
    --model models/lgbm_matcher.txt \
    --all-s1-ids data/norm_test_source1.tsv \
    --output output/matching_results.tsv \
    --threshold 0.70
```

### Step 6 — Self-check & validate before submitting
```bash
# Self-checks (ID existence, superset assertion, S1 coverage) + validator gate
python code/business_entity_resolution/src/submit/package.py \
    --matching-results output/matching_results.tsv \
    --candidate-pairs output/candidate_pairs.tsv \
    --test-s1 dataset/test/test_source1.tsv \
    --test-s2 dataset/test/test_source2.tsv \
    --test-s3 dataset/test/test_source3.tsv \
    --test-dir dataset/test \
    --validator utils/validate_submission.py
# Must print PASS — never upload without this
```

## Constraints Compliance

| Constraint | Status |
|---|---|
| TSV I/O, exact column names | ✅ enforced throughout |
| Every S1 test entity in output | ✅ checked in submit module |
| S2-/S3- IDs only in matches | ✅ ID-existence assertion |
| No external API/database calls | ✅ self-contained, stdlib + pip only |
| Model ≤ 8B params, MIT/Apache-2.0 | ✅ LightGBM (MIT) |
| No external data augmentation | ✅ only provided TSVs used |
