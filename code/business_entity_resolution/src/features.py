"""Pairwise similarity feature engineering for (Source-1, candidate) pairs.

Features are computed on both raw-ish normalized strings and token sets, so
the classifier can learn to rely on address when the business name is
corrupted/gibberish/non-Latin, and vice versa.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from normalize import name_tokens, address_tokens, address_numbers, normalize_name, normalize_address


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _overlap_coef(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def compute_pair_features(name1, addr1, country1, name2, addr2, country2) -> dict:
    n1_norm = normalize_name(name1)
    n2_norm = normalize_name(name2)
    t1 = set(name_tokens(name1))
    t2 = set(name_tokens(name2))

    a1_norm = normalize_address(addr1, country1)
    a2_norm = normalize_address(addr2, country2)
    at1 = set(address_tokens(addr1, country1))
    at2 = set(address_tokens(addr2, country2))

    num1 = set(x for x in address_numbers(addr1) if len(x) >= 2)
    num2 = set(x for x in address_numbers(addr2) if len(x) >= 2)

    feats = {
        "name_jaccard": _jaccard(t1, t2),
        "name_overlap": _overlap_coef(t1, t2),
        "name_ratio": fuzz.ratio(n1_norm, n2_norm) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(n1_norm, n2_norm) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(n1_norm, n2_norm) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(n1_norm, n2_norm) / 100.0,
        "name_len_ratio": (min(len(n1_norm), len(n2_norm)) / max(len(n1_norm), len(n2_norm))) if (n1_norm and n2_norm) else 0.0,
        "name1_empty": float(len(n1_norm) == 0),
        "name2_empty": float(len(n2_norm) == 0),

        "addr_jaccard": _jaccard(at1, at2),
        "addr_overlap": _overlap_coef(at1, at2),
        "addr_ratio": fuzz.ratio(a1_norm, a2_norm) / 100.0,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(a1_norm, a2_norm) / 100.0,
        "addr_token_set_ratio": fuzz.token_set_ratio(a1_norm, a2_norm) / 100.0,
        "addr1_empty": float(len(a1_norm) == 0),
        "addr2_empty": float(len(a2_norm) == 0),

        "num_jaccard": _jaccard(num1, num2),
        "num_overlap": _overlap_coef(num1, num2),
        "num_exact_any_match": float(len(num1 & num2) > 0),
        "num1_count": float(len(num1)),
        "num2_count": float(len(num2)),

        "country_match": float((country1 or "").strip().lower() == (country2 or "").strip().lower()),
    }
    return feats


FEATURE_COLUMNS = [
    "name_jaccard", "name_overlap", "name_ratio", "name_token_sort_ratio",
    "name_token_set_ratio", "name_partial_ratio", "name_len_ratio",
    "name1_empty", "name2_empty",
    "addr_jaccard", "addr_overlap", "addr_ratio", "addr_token_sort_ratio",
    "addr_token_set_ratio", "addr1_empty", "addr2_empty",
    "num_jaccard", "num_overlap", "num_exact_any_match", "num1_count", "num2_count",
    "country_match", "nkeys", "block_score",
]


_PASSTHROUGH_COLUMNS = ["nkeys", "block_score"]
_COMPUTED_COLUMNS = [c for c in FEATURE_COLUMNS if c not in _PASSTHROUGH_COLUMNS]


def compute_features_batch(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """pairs_df must have columns: name1, addr1, country1, name2, addr2, country2.

    Optional `nkeys` / `block_score` columns (already computed for free
    during candidate generation — see blocking.merge_candidates) are passed
    through as extra features when present.
    """
    records = []
    cols = pairs_df[["name1", "addr1", "country1", "name2", "addr2", "country2"]].itertuples(index=False, name=None)
    for name1, addr1, country1, name2, addr2, country2 in cols:
        records.append(compute_pair_features(name1, addr1, country1, name2, addr2, country2))
    feat_df = pd.DataFrame.from_records(records, columns=_COMPUTED_COLUMNS)
    feat_df.index = pairs_df.index
    for col in _PASSTHROUGH_COLUMNS:
        feat_df[col] = pairs_df[col].values if col in pairs_df.columns else 0.0
    # float32 halves memory vs. pandas' float64 default; irrelevant precision
    # loss for bounded similarity scores feeding a tree model.
    return feat_df[FEATURE_COLUMNS].astype("float32")
