"""
Phase 4 — Feature Engineering
==============================

Computes pairwise similarity features between S1 and S2/S3 candidate pairs.
All features are language/country agnostic (no US/India-specific logic).

Feature groups:
  - Name features: Jaro-Winkler, Levenshtein, Jaccard, char-trigram TF-IDF cosine,
    exact match, phonetic match, length ratio
  - Address features: Jaccard, Levenshtein, char-trigram TF-IDF cosine,
    both-empty flag, one-empty flag
  - Structural features: country match, name token count difference

Usage:
    python -m src.features.similarity \
        --s1 data/train_source1_norm.tsv \
        --s23 data/train_s23_norm.tsv \
        --candidates output/candidate_pairs.tsv \
        --output data/features.tsv
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import jellyfish
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


# ──────────────────────────────────────────────────────────────────────
# Feature names (fixed order for reproducibility)
# ──────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    # Name features
    "name_jaro_winkler",
    "name_levenshtein_norm",
    "name_token_jaccard",
    "name_tfidf_cosine",
    "name_exact_norm",
    "name_phonetic_match",
    "name_len_ratio",
    # Address features
    "addr_token_jaccard",
    "addr_levenshtein_norm",
    "addr_tfidf_cosine",
    "addr_empty_both",
    "addr_empty_one",
    # Structural features
    "country_match",
    "name_token_count_diff",
]


# ──────────────────────────────────────────────────────────────────────
# Scalar feature functions (operate on single string pairs)
# ──────────────────────────────────────────────────────────────────────

def _jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    from rapidfuzz.distance import JaroWinkler
    return float(JaroWinkler.similarity(a, b))


def _levenshtein_norm(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return Levenshtein.normalized_similarity(a, b)


def _token_jaccard(a: str, b: str) -> float:
    tokens_a = set(a.split()) if a else set()
    tokens_b = set(b.split()) if b else set()
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _phonetic_match(a: str, b: str) -> float:
    """Check if primary tokens have same Soundex or Metaphone code."""
    if not a or not b:
        return 0.0
    tok_a = a.split()[0] if a.split() else ""
    tok_b = b.split()[0] if b.split() else ""
    if not tok_a or not tok_b:
        return 0.0
    # Only compute phonetics for ASCII tokens
    if not tok_a.isascii() or not tok_b.isascii():
        return 0.0
    if not tok_a.isalpha() or not tok_b.isalpha():
        return 0.0
    try:
        sx_match = jellyfish.soundex(tok_a) == jellyfish.soundex(tok_b)
        mp_match = jellyfish.metaphone(tok_a) == jellyfish.metaphone(tok_b)
        return 1.0 if (sx_match or mp_match) else 0.0
    except Exception:
        return 0.0


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


# ──────────────────────────────────────────────────────────────────────
# TF-IDF cosine (batch computation for efficiency)
# ──────────────────────────────────────────────────────────────────────

class CharTrigramTfidf:
    """
    Char trigram TF-IDF vectorizer. Script-agnostic — works on any Unicode.
    Fitted on a representative sample of corpus strings, then computes
    row-wise cosine similarity for candidate pairs in vectorized batches.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 3),
            max_features=50000,  # cap vocabulary for memory
            dtype=np.float32,
        )
        self._fitted = False

    def fit_corpus(self, texts: List[str]) -> None:
        """Fit vocabulary & IDF weights on a sample of text."""
        self.vectorizer.fit(texts)
        self._fitted = True

    def fit(self, texts) -> None:
        """Backward-compatible fit: accepts dict or list."""
        if isinstance(texts, dict):
            corpus = list(texts.values())
        else:
            corpus = list(texts)
        self.fit_corpus(corpus)

    def compute_pair_cosines(self, list_a: List[str], list_b: List[str]) -> np.ndarray:
        """
        Compute row-wise cosine similarity for aligned pairs of texts (list_a[i], list_b[i]).
        Vectorized with scipy sparse matrix math.
        """
        if not self._fitted or not list_a:
            return np.zeros(len(list_a), dtype=np.float32)

        vecs_a = self.vectorizer.transform(list_a)
        vecs_b = self.vectorizer.transform(list_b)

        dots = np.asarray(vecs_a.multiply(vecs_b).sum(axis=1)).ravel()
        sq_a = np.asarray(vecs_a.multiply(vecs_a).sum(axis=1)).ravel()
        sq_b = np.asarray(vecs_b.multiply(vecs_b).sum(axis=1)).ravel()
        denom = np.sqrt(sq_a) * np.sqrt(sq_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosines = np.where(denom > 0, dots / denom, 0.0).astype(np.float32)
        return cosines

    def batch_cosine(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Fallback for compatibility."""
        return np.zeros(len(pairs), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
# Feature extractor (main class)
# ──────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Computes all pairwise features for (S1, S23) candidate pairs.
    """

    def __init__(self):
        self.name_tfidf = CharTrigramTfidf()
        self.addr_tfidf = CharTrigramTfidf()

    def fit(self, df_s1: pd.DataFrame, df_s23: pd.DataFrame, max_sample: int = 100000) -> None:
        """
        Fit TF-IDF vectorizers on representative sample of names and addresses.
        """
        print("  Fitting name TF-IDF vectorizer...", flush=True)
        s1_names = df_s1["norm_name"].dropna().astype(str)
        s23_names = df_s23["norm_name"].dropna().astype(str)
        
        if len(s1_names) > max_sample:
            s1_names = s1_names.sample(n=max_sample, random_state=42)
        if len(s23_names) > max_sample:
            s23_names = s23_names.sample(n=max_sample, random_state=42)
            
        train_names = list(s1_names) + list(s23_names)
        self.name_tfidf.fit_corpus(train_names)
        print(f"    Vocabulary: {len(self.name_tfidf.vectorizer.vocabulary_):,} trigrams", flush=True)

        print("  Fitting address TF-IDF vectorizer...", flush=True)
        s1_addrs = df_s1["norm_address"].dropna().astype(str)
        s23_addrs = df_s23["norm_address"].dropna().astype(str)
        
        if len(s1_addrs) > max_sample:
            s1_addrs = s1_addrs.sample(n=max_sample, random_state=42)
        if len(s23_addrs) > max_sample:
            s23_addrs = s23_addrs.sample(n=max_sample, random_state=42)
            
        train_addrs = list(s1_addrs) + list(s23_addrs)
        self.addr_tfidf.fit_corpus(train_addrs)
        print(f"    Vocabulary: {len(self.addr_tfidf.vectorizer.vocabulary_):,} trigrams", flush=True)

    def extract(
        self,
        df_candidates: pd.DataFrame,
        s1_lookup: Dict[str, Dict],
        s23_lookup: Dict[str, Dict],
        chunk_size: int = 100000,
    ) -> pd.DataFrame:
        """
        Extract features for all candidate pairs.

        Parameters
        ----------
        df_candidates : DataFrame
            Columns: [source1_entity_id, source23_entity_id]
        s1_lookup : dict
            entity_id -> {norm_name, norm_address, country}
        s23_lookup : dict
            entity_id -> {norm_name, norm_address, country}
        chunk_size : int
            Process in chunks for memory efficiency.

        Returns
        -------
        DataFrame with [source1_entity_id, source23_entity_id] + FEATURE_COLUMNS
        """
        n = len(df_candidates)
        print(f"  Extracting features for {n:,} candidate pairs...", flush=True)

        all_features = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = df_candidates.iloc[start:end]
            t0 = time.time()

            feat_rows = np.zeros((len(chunk), len(FEATURE_COLUMNS)), dtype=np.float32)

            s1_ids = chunk["source1_entity_id"].values
            s23_ids = chunk["source23_entity_id"].values

            # Pre-extract texts for vectorized TF-IDF
            names_a = [s1_lookup.get(eid, {}).get("norm_name", "") for eid in s1_ids]
            names_b = [s23_lookup.get(eid, {}).get("norm_name", "") for eid in s23_ids]
            addrs_a = [s1_lookup.get(eid, {}).get("norm_address", "") for eid in s1_ids]
            addrs_b = [s23_lookup.get(eid, {}).get("norm_address", "") for eid in s23_ids]

            # Vectorized TF-IDF cosine (runs in C++ via scipy sparse)
            name_tfidf_scores = self.name_tfidf.compute_pair_cosines(names_a, names_b)
            addr_tfidf_scores = self.addr_tfidf.compute_pair_cosines(addrs_a, addrs_b)

            for i in range(len(chunk)):
                s1_id = s1_ids[i]
                s23_id = s23_ids[i]

                s1_data = s1_lookup.get(s1_id, {})
                s23_data = s23_lookup.get(s23_id, {})

                name_a = names_a[i]
                name_b = names_b[i]
                addr_a = addrs_a[i]
                addr_b = addrs_b[i]
                country_a = s1_data.get("country", "")
                country_b = s23_data.get("country", "")

                # Name features
                feat_rows[i, 0] = _jaro_winkler(name_a, name_b)
                feat_rows[i, 1] = _levenshtein_norm(name_a, name_b)
                feat_rows[i, 2] = _token_jaccard(name_a, name_b)
                feat_rows[i, 3] = name_tfidf_scores[i]
                feat_rows[i, 4] = 1.0 if (name_a and name_b and name_a == name_b) else 0.0
                feat_rows[i, 5] = _phonetic_match(name_a, name_b)
                feat_rows[i, 6] = _len_ratio(name_a, name_b)

                # Address features
                feat_rows[i, 7] = _token_jaccard(addr_a, addr_b)
                feat_rows[i, 8] = _levenshtein_norm(addr_a, addr_b)
                feat_rows[i, 9] = addr_tfidf_scores[i]
                feat_rows[i, 10] = 1.0 if (not addr_a and not addr_b) else 0.0
                feat_rows[i, 11] = 1.0 if (bool(addr_a) != bool(addr_b)) else 0.0

                # Structural features
                feat_rows[i, 12] = 1.0 if country_a == country_b else 0.0
                tokens_a = len(name_a.split()) if name_a else 0
                tokens_b = len(name_b.split()) if name_b else 0
                feat_rows[i, 13] = abs(tokens_a - tokens_b)

            df_feat = pd.DataFrame(feat_rows, columns=FEATURE_COLUMNS)
            df_feat.insert(0, "source1_entity_id", s1_ids)
            df_feat.insert(1, "source23_entity_id", s23_ids)
            all_features.append(df_feat)

            elapsed = time.time() - t0
            print(f"    Chunk {start:,}-{end:,}: {elapsed:.1f}s", flush=True)

        result = pd.concat(all_features, ignore_index=True)
        print(f"  Done: {len(result):,} feature rows, {len(FEATURE_COLUMNS)} features each", flush=True)
        return result


# ──────────────────────────────────────────────────────────────────────
# Helpers: build lookup dicts from DataFrames
# ──────────────────────────────────────────────────────────────────────

def build_lookup(df: pd.DataFrame, needed_ids: Optional[Set[str]] = None) -> Dict[str, Dict]:
    """
    Build entity_id -> {norm_name, norm_address, country} dict.
    If needed_ids is provided, only extracts those IDs for massive memory savings.
    Uses fast vectorized arrays instead of slow iterrows().
    """
    if needed_ids is not None:
        sub_df = df[df["entity_id"].isin(needed_ids)]
    else:
        sub_df = df

    eids = sub_df["entity_id"].values
    names = sub_df["norm_name"].fillna("").values
    addrs = sub_df["norm_address"].fillna("").values
    countries = sub_df["country"].fillna("").values

    lookup = {}
    for i in range(len(eids)):
        lookup[eids[i]] = {
            "norm_name": str(names[i]),
            "norm_address": str(addrs[i]),
            "country": str(countries[i]),
        }
    return lookup


# ──────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: Feature engineering for candidate pairs"
    )
    parser.add_argument("--s1", required=True, help="Path to normalised S1 TSV")
    parser.add_argument("--s23", required=True, help="Path to normalised S2+S3 TSV (merged)")
    parser.add_argument("--candidates", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--output", default="data/features.tsv", help="Output features TSV")
    args = parser.parse_args()

    print("Loading data...")
    df_s1 = pd.read_csv(args.s1, sep="\t", encoding="utf-8", encoding_errors="replace")
    df_s23 = pd.read_csv(args.s23, sep="\t", encoding="utf-8", encoding_errors="replace")
    df_candidates = pd.read_csv(args.candidates, sep="\t", encoding="utf-8",
                                encoding_errors="replace")
    print(f"  S1: {len(df_s1):,}, S23: {len(df_s23):,}, Candidates: {len(df_candidates):,}")

    extractor = FeatureExtractor()
    extractor.fit(df_s1, df_s23)

    s1_lookup = build_lookup(df_s1)
    s23_lookup = build_lookup(df_s23)

    df_features = extractor.extract(df_candidates, s1_lookup, s23_lookup)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_features.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote features to {args.output}")
