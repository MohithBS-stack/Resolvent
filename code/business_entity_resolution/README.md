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

### Step 3 — Build validation split + F₀.5 harness
```bash
python code/business_entity_resolution/src/evaluate/score_f05.py \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --s1 data/norm_train_source1.tsv \
    --val-fraction 0.1 \
    --output-dir data/val_split
```

### Step 4 — Train matching model
```bash
python code/business_entity_resolution/src/matching/train.py \
    --s1 data/norm_train_source1.tsv \
    --s2 data/norm_train_source2.tsv \
    --s3 data/norm_train_source3.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --val-dir data/val_split \
    --model-out data/model.lgb
```

### Step 5 — Run inference
```bash
python code/business_entity_resolution/src/matching/predict.py \
    --candidates output/candidate_pairs.tsv \
    --model data/model.lgb \
    --s1 data/norm_test_source1.tsv \
    --s2 data/norm_test_source2.tsv \
    --s3 data/norm_test_source3.tsv \
    --output output/matching_results.tsv \
    --threshold 0.5
```

### Step 6 — Validate before submitting
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
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
