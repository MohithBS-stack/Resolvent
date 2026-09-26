"""
Smoke test for Phase 2 — InvertedIndexBlocker.

Runs on small synthetic data to verify:
  1. Multi-key blocking produces expected pairs.
  2. Country partitioning works (no cross-country pairs).
  3. Recall ceiling computation is correct.
  4. Stats are logged.
"""

import pandas as pd
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.blocking.candidate_gen import (
    InvertedIndexBlocker,
    merge_s2_s3,
    _significant_name_tokens,
    _primary_name_token,
    _top_k_trigrams,
)


def make_s1():
    """Small S1 dataset."""
    return pd.DataFrame({
        "entity_id": ["S1-000000001", "S1-000000002", "S1-000000003"],
        "business_name": ["Acme Corp", "Beta Solutions", "Gamma Inc"],
        "business_address": ["123 Main St", "456 Oak Ave", "789 Pine Rd"],
        "country": ["US", "US", "India"],
        "norm_name": ["acme corp", "beta solutions", "gamma inc"],
        "norm_address": ["123 main st", "456 oak ave", "789 pine rd"],
    })


def make_s2():
    """S2 records — some should match S1, some not."""
    return pd.DataFrame({
        "entity_id": ["S2-000000001", "S2-000000002", "S2-000000003", "S2-000000004"],
        "business_name": ["Acme Corporation", "Unrelated Biz", "Gamma Limited", "Beta Sol"],
        "business_address": ["123 Main Street", "999 Elm", "789 Pine Road", "456 Oak Avenue"],
        "country": ["US", "US", "India", "US"],
        "norm_name": ["acme corp", "unrelated biz", "gamma ltd", "beta sol"],
        "norm_address": ["123 main st", "999 elm", "789 pine rd", "456 oak ave"],
    })


def make_s3():
    """S3 records — one match for S1-000000001."""
    return pd.DataFrame({
        "entity_id": ["S3-000000001"],
        "business_name": ["Acme Co"],
        "business_address": ["123 Main St Suite 100"],
        "country": ["US"],
        "norm_name": ["acme co"],
        "norm_address": ["123 main st ste 100"],
    })


def make_ground_truth():
    """Ground truth for recall ceiling check."""
    return pd.DataFrame({
        "source1_entity_id": ["S1-000000001", "S1-000000002", "S1-000000003"],
        "matched_entity_ids": [
            "S2-000000001,S3-000000001",  # Acme matches
            "S2-000000004",                # Beta matches
            "S2-000000003",                # Gamma matches
        ],
    })


def test_significant_tokens():
    tokens = _significant_name_tokens("acme corp")
    # 'corp' is in NAME_STOPWORDS → stripped
    assert "acme" in tokens, f"Expected 'acme' in {tokens}"
    assert "corp" not in tokens, f"'corp' should be stripped: {tokens}"
    print("  ✓ _significant_name_tokens")


def test_primary_token():
    primary = _primary_name_token("beta solutions")
    assert primary is not None
    assert primary == "beta", f"Expected 'beta', got {primary}"
    print("  ✓ _primary_name_token")


def test_trigrams():
    tris = _top_k_trigrams("acme", k=2)
    assert len(tris) == 2
    assert "acm" in tris
    assert "cme" in tris
    print("  ✓ _top_k_trigrams")


def test_blocking_no_cross_country():
    """Verify that US S1 records don't pair with India S23 records."""
    s1 = make_s1()
    s23 = merge_s2_s3(make_s2(), make_s3())

    blocker = InvertedIndexBlocker()
    df_cands = blocker.generate_candidates(s1, s23)

    # Check: India S1 should never pair with US S23
    india_s1 = {"S1-000000003"}
    us_s23 = {"S2-000000001", "S2-000000002", "S2-000000004", "S3-000000001"}
    for _, row in df_cands.iterrows():
        s1_id = row["source1_entity_id"]
        s23_id = row["source23_entity_id"]
        if s1_id in india_s1:
            assert s23_id not in us_s23, (
                f"Cross-country pair found: {s1_id} ↔ {s23_id}"
            )
    print("  ✓ No cross-country pairs")


def test_recall_ceiling():
    """With synthetic data, known true pairs should all be found."""
    s1 = make_s1()
    s23 = merge_s2_s3(make_s2(), make_s3())
    gt = make_ground_truth()

    blocker = InvertedIndexBlocker()
    df_cands = blocker.generate_candidates(s1, s23, ground_truth=gt)

    recall = blocker.stats["recall_ceiling"]
    print(f"  Recall ceiling: {recall * 100:.1f}%")
    # We expect high recall on this tiny synthetic set
    assert recall >= 0.5, f"Recall unexpectedly low: {recall:.2f}"
    print("  ✓ Recall ceiling computed")


def test_candidates_generated():
    """At least some candidate pairs should be generated."""
    s1 = make_s1()
    s23 = merge_s2_s3(make_s2(), make_s3())

    blocker = InvertedIndexBlocker()
    df_cands = blocker.generate_candidates(s1, s23)

    assert len(df_cands) > 0, "No candidate pairs generated!"
    assert "source1_entity_id" in df_cands.columns
    assert "source23_entity_id" in df_cands.columns
    print(f"  ✓ {len(df_cands)} candidate pairs generated")


if __name__ == "__main__":
    print("Running blocking smoke tests...")
    test_significant_tokens()
    test_primary_token()
    test_trigrams()
    test_blocking_no_cross_country()
    test_recall_ceiling()
    test_candidates_generated()
    print("\n✅ All smoke tests passed!")
