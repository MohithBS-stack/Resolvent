"""
Phase 6 — Submission Packaging
===============================

Writes output TSVs (matching_results.tsv, candidate_pairs.tsv) with:
  1. ID-existence self-check (catches phantom IDs that the validator misses)
  2. Superset assertion (every matched ID must appear in candidate_pairs.tsv)
  3. Validator gate (runs the organizer's validate_submission.py)

Usage:
    python -m src.submit.package \
        --matching-results output/matching_results.tsv \
        --candidate-pairs output/candidate_pairs.tsv \
        --test-s1 dataset/test/test_source1.tsv \
        --test-s2 dataset/test/test_source2.tsv \
        --test-s3 dataset/test/test_source3.tsv
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, Optional, Set

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# ID-existence self-check
# ──────────────────────────────────────────────────────────────────────

def check_id_existence(
    matches: Dict[str, Set[str]],
    valid_s23_ids: Set[str],
    label: str = "matching_results",
) -> bool:
    """
    Check that every matched S2/S3 ID actually exists in the test source files.

    The organizer's validator has --check-ids OFF by default. Phantom IDs
    silently cost leaderboard score. We catch them here.

    Returns True if all IDs are valid, False otherwise.
    """
    all_matched_ids: Set[str] = set()
    for match_set in matches.values():
        all_matched_ids |= match_set

    bad_ids = all_matched_ids - valid_s23_ids
    if bad_ids:
        print(f"  [FAIL] {label}: {len(bad_ids):,} phantom ID(s) not in test sources:")
        for bid in sorted(bad_ids)[:20]:
            print(f"    - {bid}")
        if len(bad_ids) > 20:
            print(f"    ... and {len(bad_ids) - 20} more")
        return False
    else:
        print(f"  [OK] {label}: all {len(all_matched_ids):,} matched IDs exist in test sources")
        return True


# ──────────────────────────────────────────────────────────────────────
# Superset assertion (candidates must contain all matches)
# ──────────────────────────────────────────────────────────────────────

def check_superset(
    matches: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
) -> bool:
    """
    Assert that candidate_pairs.tsv is a superset of matching_results.tsv.

    Every matched ID for an S1 entity must appear in that entity's candidate set.
    A violation means a pipeline bug (match generated outside the blocking set).
    """
    violations = []
    for s1_id, matched_ids in matches.items():
        if not matched_ids:
            continue
        cand_ids = candidates.get(s1_id, set())
        missing = matched_ids - cand_ids
        if missing:
            violations.append((s1_id, missing))

    if violations:
        print(f"  [FAIL] Superset check: {len(violations):,} S1 entities have matches "
              "not in candidates:")
        for s1_id, missing in violations[:10]:
            print(f"    {s1_id}: missing from candidates = {missing}")
        if len(violations) > 10:
            print(f"    ... and {len(violations) - 10} more")
        return False
    else:
        print(f"  [OK] Superset check: all matches are in candidates")
        return True


# ──────────────────────────────────────────────────────────────────────
# S1 coverage check
# ──────────────────────────────────────────────────────────────────────

def check_s1_coverage(
    matches: Dict[str, Set[str]],
    required_s1_ids: Set[str],
    label: str = "matching_results",
) -> bool:
    """
    Check that every S1 entity from the test set has a row in the output.
    """
    present = set(matches.keys())
    missing = required_s1_ids - present
    extra = present - required_s1_ids

    ok = True
    if missing:
        print(f"  [FAIL] {label}: {len(missing):,} required S1 entity(ies) missing")
        for mid in sorted(missing)[:10]:
            print(f"    - {mid}")
        ok = False
    else:
        print(f"  [OK] {label}: all {len(required_s1_ids):,} required S1 entities present")

    if extra:
        print(f"  [WARN] {label}: {len(extra):,} extra S1 entity(ies) not in test set")

    return ok


# ──────────────────────────────────────────────────────────────────────
# Write output TSVs
# ──────────────────────────────────────────────────────────────────────

def write_matching_results_tsv(
    matches: Dict[str, Set[str]],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(matches.keys()):
            matched = ",".join(sorted(matches[s1_id]))
            f.write(f"{s1_id}\t{matched}\n")
            count += 1
    print(f"  Wrote matching_results.tsv: {count:,} rows -> {output_path}")


def write_candidate_pairs_tsv(
    candidates: Dict[str, Set[str]],
    output_path: str,
) -> None:
    """Write candidate_pairs.tsv (blocking audit trail)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates.keys()):
            cand = ",".join(sorted(candidates[s1_id]))
            f.write(f"{s1_id}\t{cand}\n")
            count += 1
    print(f"  Wrote candidate_pairs.tsv: {count:,} rows -> {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Convert flat candidate pairs DataFrame to dict
# ──────────────────────────────────────────────────────────────────────

def candidates_df_to_dict(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Convert a flat candidate pairs DataFrame
    (columns: source1_entity_id, source23_entity_id)
    to {s1_id: set(s23_ids)}.
    """
    result: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"])
        s23_id = str(row["source23_entity_id"])
        if s1_id not in result:
            result[s1_id] = set()
        result[s1_id].add(s23_id)
    return result


# ──────────────────────────────────────────────────────────────────────
# Run organizer's validator
# ──────────────────────────────────────────────────────────────────────

def run_validator(
    matching_path: str,
    candidate_path: str,
    test_dir: str,
    validator_script: str = "utils/validate_submission.py",
) -> bool:
    """
    Run the organizer's validate_submission.py.
    Returns True if it exits with code 0.
    """
    if not os.path.isfile(validator_script):
        print(f"  [SKIP] Validator script not found: {validator_script}")
        return True

    cmd = [
        sys.executable, validator_script,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir,
    ]
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print("  [OK] Validator PASS")
        return True
    else:
        print("  [FAIL] Validator FAILED (exit code {})".format(result.returncode))
        return False


# ──────────────────────────────────────────────────────────────────────
# Full packaging pipeline
# ──────────────────────────────────────────────────────────────────────

def package_submission(
    matches: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    required_s1_ids: Set[str],
    valid_s23_ids: Set[str],
    matching_output: str = "output/matching_results.tsv",
    candidate_output: str = "output/candidate_pairs.tsv",
    test_dir: str = "dataset/test",
    validator_script: str = "utils/validate_submission.py",
) -> bool:
    """
    Full packaging: ID checks, write TSVs, run validator.
    Returns True if everything passes.
    """
    print("\n" + "=" * 60)
    print("SUBMISSION PACKAGING")
    print("=" * 60)

    all_ok = True

    # 1. S1 coverage
    if not check_s1_coverage(matches, required_s1_ids, "matching_results"):
        all_ok = False
    if not check_s1_coverage(candidates, required_s1_ids, "candidate_pairs"):
        all_ok = False

    # 2. ID existence
    if not check_id_existence(matches, valid_s23_ids, "matching_results"):
        all_ok = False

    # 3. Superset assertion
    if not check_superset(matches, candidates):
        all_ok = False

    # 4. Write TSVs
    write_matching_results_tsv(matches, matching_output)
    write_candidate_pairs_tsv(candidates, candidate_output)

    # 5. Validator gate
    if not run_validator(matching_output, candidate_output, test_dir, validator_script):
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("PACKAGING COMPLETE - All checks passed")
    else:
        print("PACKAGING COMPLETE - Some checks FAILED (review above)")
    print("=" * 60)

    return all_ok


# ──────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 6: Submission packaging with ID checks + validator gate"
    )
    parser.add_argument("--matching-results", required=True,
                        help="Path to matching_results.tsv to validate")
    parser.add_argument("--candidate-pairs", required=True,
                        help="Path to candidate_pairs.tsv to validate")
    parser.add_argument("--test-s1", required=True,
                        help="Path to test_source1.tsv (for required S1 IDs)")
    parser.add_argument("--test-s2", default=None,
                        help="Path to test_source2.tsv (for ID existence check)")
    parser.add_argument("--test-s3", default=None,
                        help="Path to test_source3.tsv (for ID existence check)")
    parser.add_argument("--test-dir", default="dataset/test",
                        help="Test directory for the organizer's validator")
    parser.add_argument("--validator", default="utils/validate_submission.py",
                        help="Path to organizer's validator script")
    args = parser.parse_args()

    print("Loading test S1 IDs...")
    s1_df = pd.read_csv(args.test_s1, sep="\t", encoding="utf-8", encoding_errors="replace")
    required_s1_ids = set(s1_df["entity_id"].values)
    print(f"  Required S1 entities: {len(required_s1_ids):,}")

    # Load valid S2/S3 IDs
    valid_s23_ids: Set[str] = set()
    if args.test_s2 and os.path.isfile(args.test_s2):
        s2_df = pd.read_csv(args.test_s2, sep="\t", encoding="utf-8",
                            encoding_errors="replace")
        valid_s23_ids |= set(s2_df["entity_id"].values)
    if args.test_s3 and os.path.isfile(args.test_s3):
        s3_df = pd.read_csv(args.test_s3, sep="\t", encoding="utf-8",
                            encoding_errors="replace")
        valid_s23_ids |= set(s3_df["entity_id"].values)
    print(f"  Valid S2/S3 IDs: {len(valid_s23_ids):,}")

    # Parse existing output files
    print("Loading matching_results.tsv...")
    from src.evaluate.score_f05 import parse_predictions
    matches = parse_predictions(args.matching_results)

    print("Loading candidate_pairs.tsv...")
    cand_df = pd.read_csv(args.candidate_pairs, sep="\t", encoding="utf-8",
                          encoding_errors="replace", dtype=str, keep_default_na=False)
    # candidate_pairs uses "candidate_entity_ids" column
    candidates: Dict[str, Set[str]] = {}
    for _, row in cand_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        cand_raw = str(row.get("candidate_entity_ids", "")).strip()
        if cand_raw and cand_raw != "nan":
            candidates[s1_id] = {c.strip() for c in cand_raw.split(",") if c.strip()}
        else:
            candidates[s1_id] = set()

    # Run checks
    all_ok = True
    if not check_s1_coverage(matches, required_s1_ids, "matching_results"):
        all_ok = False
    if not check_id_existence(matches, valid_s23_ids, "matching_results"):
        all_ok = False
    if not check_superset(matches, candidates):
        all_ok = False

    # Run validator
    run_validator(args.matching_results, args.candidate_pairs,
                  args.test_dir, args.validator)

    if all_ok:
        print("\nAll submission checks PASSED")
    else:
        print("\nSome submission checks FAILED")
