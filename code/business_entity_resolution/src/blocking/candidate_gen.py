"""
Phase 2 — Blocking / Candidate Generation
==========================================

Multi-key inverted-index blocking that generates candidate S1–S2/S3 pairs
for downstream matching. All operations are dict-based inverted index lookups
— no O(n²) cross-joins.

Blocking passes (union of all, deduplicated):
  1. Name token overlap (sorted frozenset of significant tokens)
  2. Soundex of primary name token
  3. Metaphone of primary name token
  4. Address token overlap (sorted frozenset of street tokens)
  5. Trigram n-gram backstop (top-3 char trigrams from name+address)

Candidates are generated **within the same country** partition only (open-set,
no hardcoding — France falls into its own bucket automatically).

Usage:
    python -m src.blocking.candidate_gen \
        --s1 data/train_source1_norm.tsv \
        --s2 data/train_source2_norm.tsv \
        --s3 data/train_source3_norm.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --output output/candidate_pairs.tsv
"""

import argparse
import os
import time
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import jellyfish
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Stopwords to strip from name tokens before blocking key generation.
# Kept minimal — these are function words that inflate bucket sizes
# without adding discriminative signal.
# ──────────────────────────────────────────────────────────────────────
NAME_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "for", "to", "at", "by",
    "on", "is", "it", "its", "as", "with", "from", "that", "this",
    "pvt", "ltd", "inc", "corp", "llc", "llp", "co", "company",
    "limited", "incorporated", "corporation", "private",
})

# Suffixes to strip from legal entity names during blocking key construction
LEGAL_SUFFIXES = frozenset({
    "pvt", "ltd", "inc", "corp", "llc", "llp", "co",
    "pvt ltd", "limited", "incorporated", "corporation", "private",
})


# ──────────────────────────────────────────────────────────────────────
# Helper: extract character trigrams from a string
# ──────────────────────────────────────────────────────────────────────
def _char_trigrams(text: str) -> List[str]:
    """Return all character-level trigrams from *text*."""
    if len(text) < 3:
        return [text] if text else []
    return [text[i:i + 3] for i in range(len(text) - 2)]


def _top_k_trigrams(text: str, k: int = 3) -> List[str]:
    """
    Return the *k* most common trigrams by simple frequency count.
    Ties broken alphabetically for determinism.
    """
    trigrams = _char_trigrams(text)
    if not trigrams:
        return []
    freq: Dict[str, int] = defaultdict(int)
    for t in trigrams:
        freq[t] += 1
    # Sort by (-count, alphabetical) for determinism, take top k
    sorted_trigrams = sorted(freq.keys(), key=lambda x: (-freq[x], x))
    return sorted_trigrams[:k]


# ──────────────────────────────────────────────────────────────────────
# Helper: extract significant name tokens (for blocking key)
# ──────────────────────────────────────────────────────────────────────
def _significant_name_tokens(norm_name: str) -> List[str]:
    """
    Split normalised name into tokens, strip stopwords and legal
    suffixes, return remaining tokens sorted alphabetically.
    """
    tokens = norm_name.split()
    significant = [t for t in tokens if t not in NAME_STOPWORDS and len(t) > 1]
    return sorted(significant)


def _primary_name_token(norm_name: str) -> Optional[str]:
    """
    Return the first significant token from the normalised name.
    Used as input to phonetic blocking passes (Soundex/Metaphone).
    """
    tokens = _significant_name_tokens(norm_name)
    return tokens[0] if tokens else None


# ──────────────────────────────────────────────────────────────────────
# InvertedIndexBlocker
# ──────────────────────────────────────────────────────────────────────
class InvertedIndexBlocker:
    """
    Multi-key inverted-index blocker that generates candidate pairs
    between Source 1 and Source 2/3 records within the same country.

    Each blocking pass builds an inverted index mapping
    ``blocking_key → [entity_ids]``, then candidate pairs are the
    union of all pairs sharing at least one key in any pass.
    """

    def __init__(self):
        # Populated after build_index / generate_candidates
        self.stats: Dict[str, object] = {}

    # ──────────────────────────────────────────────────────────────────
    # Core public API
    # ──────────────────────────────────────────────────────────────────

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s23: pd.DataFrame,
        ground_truth: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate candidate (S1, S2/S3) pairs via multi-key blocking,
        partitioned by country.

        Parameters
        ----------
        df_s1 : DataFrame
            Source 1 records with columns
            ``[entity_id, norm_name, norm_address, country]``.
        df_s23 : DataFrame
            Source 2+3 records concatenated, same columns.
        ground_truth : DataFrame, optional
            If provided (columns ``[source1_entity_id, matched_entity_ids]``),
            used to compute blocking recall ceiling.

        Returns
        -------
        DataFrame with columns ``[source1_entity_id, source23_entity_id]``.
        """
        t0 = time.time()
        countries_s1 = df_s1["country"].unique()
        countries_s23 = df_s23["country"].unique()
        all_countries = set(countries_s1) | set(countries_s23)

        all_pairs: List[Tuple[str, str]] = []
        per_country_stats: Dict[str, Dict] = {}

        for country in sorted(all_countries):
            s1_c = df_s1[df_s1["country"] == country]
            s23_c = df_s23[df_s23["country"] == country]
            if s1_c.empty or s23_c.empty:
                print(f"  [skip] country='{country}': S1={len(s1_c)}, S23={len(s23_c)}")
                continue

            print(f"  Blocking country='{country}': S1={len(s1_c):,}, S23={len(s23_c):,}")
            pairs = self._block_within_country(s1_c, s23_c)
            all_pairs.extend(pairs)
            per_country_stats[country] = {
                "s1_count": len(s1_c),
                "s23_count": len(s23_c),
                "candidate_pairs": len(pairs),
                "cross_product": len(s1_c) * len(s23_c),
            }

        # Deduplicate across countries (shouldn't happen, but safety)
        pair_set = set(all_pairs)
        df_candidates = pd.DataFrame(
            list(pair_set), columns=["source1_entity_id", "source23_entity_id"]
        )

        elapsed = time.time() - t0

        # ── Compute stats ────────────────────────────────────────────
        total_cross = sum(s["cross_product"] for s in per_country_stats.values())
        total_candidates = len(df_candidates)
        reduction_ratio = total_cross / total_candidates if total_candidates > 0 else float("inf")

        self.stats = {
            "total_candidates": total_candidates,
            "total_cross_product": total_cross,
            "reduction_ratio": reduction_ratio,
            "elapsed_seconds": elapsed,
            "per_country": per_country_stats,
        }

        # ── Recall ceiling (if ground truth provided) ────────────────
        if ground_truth is not None:
            recall = self._compute_recall_ceiling(df_candidates, ground_truth)
            self.stats["recall_ceiling"] = recall
        else:
            self.stats["recall_ceiling"] = None

        self._log_stats()
        return df_candidates

    # ──────────────────────────────────────────────────────────────────
    # Internal: block within a single country partition
    # ──────────────────────────────────────────────────────────────────

    def _block_within_country(
        self,
        s1: pd.DataFrame,
        s23: pd.DataFrame,
    ) -> List[Tuple[str, str]]:
        """
        Run all 5 blocking passes for a single country partition and
        return the union of candidate pairs (deduplicated).
        """
        # Pre-compute blocking key data for S1 and S23
        s1_keys = self._compute_blocking_keys(s1)
        s23_keys = self._compute_blocking_keys(s23)

        candidate_pairs: Set[Tuple[str, str]] = set()

        # Each pass: build inverted index on S23, probe with S1 keys
        pass_names = [
            "name_token",
            "soundex",
            "metaphone",
            "address_token",
            "trigram",
        ]

        for pass_name in pass_names:
            t_pass = time.time()
            # Build inverted index from S23 side
            s23_index: Dict[str, List[str]] = defaultdict(list)
            for eid, keys_dict in s23_keys.items():
                for key in keys_dict.get(pass_name, []):
                    if key:  # skip empty keys
                        s23_index[key].append(eid)

            # Probe with S1 keys
            pass_pairs = 0
            for s1_eid, keys_dict in s1_keys.items():
                for key in keys_dict.get(pass_name, []):
                    if key and key in s23_index:
                        for s23_eid in s23_index[key]:
                            pair = (s1_eid, s23_eid)
                            if pair not in candidate_pairs:
                                candidate_pairs.add(pair)
                                pass_pairs += 1

            elapsed_pass = time.time() - t_pass
            print(f"    pass={pass_name}: +{pass_pairs:,} new pairs ({elapsed_pass:.1f}s)")

        return list(candidate_pairs)

    # ──────────────────────────────────────────────────────────────────
    # Blocking key computation (all 5 passes at once per entity)
    # ──────────────────────────────────────────────────────────────────

    def _compute_blocking_keys(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        For every row in *df*, compute blocking keys for all 5 passes.

        Returns
        -------
        dict mapping entity_id → {pass_name → [key1, key2, ...]}
        """
        result: Dict[str, Dict[str, List[str]]] = {}

        # Vectorised extraction of columns for speed
        entity_ids = df["entity_id"].values
        norm_names = df["norm_name"].fillna("").values
        norm_addrs = df["norm_address"].fillna("").values

        for i in range(len(entity_ids)):
            eid = entity_ids[i]
            name = str(norm_names[i])
            addr = str(norm_addrs[i])

            keys: Dict[str, List[str]] = {}

            # ── Pass 1: Name token ───────────────────────────────────
            sig_tokens = _significant_name_tokens(name)
            if sig_tokens:
                # Each individual token is a blocking key (share ≥1 token)
                keys["name_token"] = sig_tokens
            else:
                keys["name_token"] = []

            # ── Pass 2: Soundex of primary name token ────────────────
            primary = _primary_name_token(name)
            if primary and primary.isascii() and primary.isalpha():
                try:
                    sx = jellyfish.soundex(primary)
                    keys["soundex"] = [sx]
                except Exception:
                    keys["soundex"] = []
            else:
                keys["soundex"] = []

            # ── Pass 3: Metaphone of primary name token ──────────────
            if primary and primary.isascii() and primary.isalpha():
                try:
                    mp = jellyfish.metaphone(primary)
                    keys["metaphone"] = [mp]
                except Exception:
                    keys["metaphone"] = []
            else:
                keys["metaphone"] = []

            # ── Pass 4: Address token ────────────────────────────────
            addr_tokens = [t for t in addr.split() if len(t) > 1]
            if addr_tokens:
                keys["address_token"] = sorted(set(addr_tokens))
            else:
                keys["address_token"] = []

            # ── Pass 5: Trigram backstop ─────────────────────────────
            combined = (name + " " + addr).strip()
            top_trigrams = _top_k_trigrams(combined, k=3)
            keys["trigram"] = top_trigrams

            result[eid] = keys

        return result

    # ──────────────────────────────────────────────────────────────────
    # Recall ceiling computation
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_recall_ceiling(
        df_candidates: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> float:
        """
        Fraction of true (S1, S2/S3) match pairs that appear in the
        candidate set. This is the upper bound on downstream F₀.5.
        """
        # Parse ground truth into (s1_id, s23_id) pairs
        true_pairs: Set[Tuple[str, str]] = set()
        for _, row in ground_truth.iterrows():
            s1_id = str(row["source1_entity_id"])
            matched_raw = row.get("matched_entity_ids", "")
            if pd.isna(matched_raw) or str(matched_raw).strip() == "":
                continue  # singleton — no true match
            for s23_id in str(matched_raw).split(","):
                s23_id = s23_id.strip()
                if s23_id:
                    true_pairs.add((s1_id, s23_id))

        if not true_pairs:
            return 1.0  # no true matches to miss

        # Build set of candidate pairs for fast lookup
        cand_set: Set[Tuple[str, str]] = set(
            zip(
                df_candidates["source1_entity_id"].values,
                df_candidates["source23_entity_id"].values,
            )
        )

        found = sum(1 for pair in true_pairs if pair in cand_set)
        return found / len(true_pairs)

    # ──────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────

    def _log_stats(self) -> None:
        """Print mandatory blocking stats."""
        s = self.stats
        print("\n" + "=" * 60)
        print("BLOCKING STATS")
        print("=" * 60)
        if s.get("recall_ceiling") is not None:
            print(f"  Blocking recall ceiling : {s['recall_ceiling'] * 100:.2f}%")
        else:
            print("  Blocking recall ceiling : N/A (no ground truth provided)")
        print(f"  Reduction ratio         : {s['reduction_ratio']:,.2f}x")
        print(f"  Candidate pairs         : {s['total_candidates']:,}")
        print(f"  Full cross-product      : {s['total_cross_product']:,}")
        print(f"  Elapsed                 : {s['elapsed_seconds']:.1f}s")
        if s.get("per_country"):
            print("  Per-country breakdown:")
            for c, cs in s["per_country"].items():
                print(
                    f"    {c}: S1={cs['s1_count']:,}  S23={cs['s23_count']:,}  "
                    f"pairs={cs['candidate_pairs']:,}  "
                    f"cross={cs['cross_product']:,}"
                )
        print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────
# Convenience: merge S2 + S3 into a single DataFrame
# ──────────────────────────────────────────────────────────────────────
def merge_s2_s3(df_s2: pd.DataFrame, df_s3: pd.DataFrame) -> pd.DataFrame:
    """Concatenate S2 and S3, reset index, deduplicate by entity_id."""
    df = pd.concat([df_s2, df_s3], ignore_index=True)
    df = df.drop_duplicates(subset=["entity_id"], keep="first")
    return df


# ──────────────────────────────────────────────────────────────────────
# Write candidate_pairs.tsv
# ──────────────────────────────────────────────────────────────────────
def write_candidate_pairs(
    df_candidates: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Write the candidate pairs TSV with correct column names:
    ``source1_entity_id`` and ``source23_entity_id``.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_candidates.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"Wrote {len(df_candidates):,} candidate pairs → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Multi-key inverted-index blocking / candidate generation"
    )
    parser.add_argument("--s1", required=True, help="Path to normalised S1 TSV")
    parser.add_argument("--s2", required=True, help="Path to normalised S2 TSV")
    parser.add_argument("--s3", required=True, help="Path to normalised S3 TSV")
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to ground truth TSV (for recall ceiling; optional)",
    )
    parser.add_argument(
        "--output",
        default="output/candidate_pairs.tsv",
        help="Output path for candidate_pairs.tsv",
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────
    print("Loading normalised sources...")
    df_s1 = pd.read_csv(args.s1, sep="\t", encoding="utf-8", encoding_errors="replace")
    df_s2 = pd.read_csv(args.s2, sep="\t", encoding="utf-8", encoding_errors="replace")
    df_s3 = pd.read_csv(args.s3, sep="\t", encoding="utf-8", encoding_errors="replace")

    print(f"  S1 : {len(df_s1):,} rows")
    print(f"  S2 : {len(df_s2):,} rows")
    print(f"  S3 : {len(df_s3):,} rows")

    df_s23 = merge_s2_s3(df_s2, df_s3)
    print(f"  S23 (merged): {len(df_s23):,} rows")

    gt = None
    if args.ground_truth:
        gt = pd.read_csv(
            args.ground_truth, sep="\t", encoding="utf-8", encoding_errors="replace"
        )
        print(f"  Ground truth: {len(gt):,} rows")

    # ── Block ─────────────────────────────────────────────────────────
    blocker = InvertedIndexBlocker()
    df_candidates = blocker.generate_candidates(df_s1, df_s23, ground_truth=gt)

    # ── Write ─────────────────────────────────────────────────────────
    write_candidate_pairs(df_candidates, args.output)
