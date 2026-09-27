"""Pairwise similarity feature engineering for (Source-1, candidate) pairs.

Features are computed on both raw-ish normalized strings and token sets, so
the classifier can learn to rely on address when the business name is
corrupted/gibberish/non-Latin, and vice versa.
"""
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from normalize import (ADDRESS_STOPWORDS, NAME_STOPWORDS, address_numbers,
                       normalize_address, normalize_name)


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


def _longest_shared_number(a: set, b: set) -> float:
    shared = a & b
    return float(max((len(x) for x in shared), default=0))


@lru_cache(maxsize=400_000)
def _prep_name(name: str):
    """Everything the features need from one name, computed once per record
    (each Source-1 record appears in hundreds of candidate pairs)."""
    norm = normalize_name(name)
    toks = [t for t in norm.split(" ") if t and t not in NAME_STOPWORDS] if norm else []  # == name_tokens(name)
    cat = "".join(toks)
    return norm, frozenset(toks), toks, cat, "".join(sorted(cat))


@lru_cache(maxsize=400_000)
def _prep_addr(addr: str, country: str):
    norm = normalize_address(addr, country)
    # == address_tokens(addr, country), without normalizing twice
    toks = frozenset(t for t in norm.split(" ") if t and t not in ADDRESS_STOPWORDS and not t.isdigit()) \
        if norm else frozenset()
    nums = address_numbers(addr)
    return norm, toks, nums, frozenset(x for x in nums if len(x) >= 2)


def compute_pair_features(name1, addr1, country1, name2, addr2, country2, cand_id="") -> dict:
    n1_norm, t1, t1_list, cat1, srt1 = _prep_name(name1)
    n2_norm, t2, t2_list, cat2, srt2 = _prep_name(name2)
    a1_norm, at1, nums1, num1 = _prep_addr(addr1, country1)
    a2_norm, at2, nums2, num2 = _prep_addr(addr2, country2)
    # Name tokens found in the other side's address: sources sometimes shuffle
    # part of the name into the address field.
    name_in_addr = (len(t1 & at2) / len(t1)) if t1 else 0.0

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

        "name_cat_ratio": fuzz.ratio(cat1, cat2) / 100.0,
        "name_cat_partial": fuzz.partial_ratio(cat1, cat2) / 100.0 if (cat1 and cat2) else 0.0,
        "name_sorted_chars_ratio": fuzz.ratio(srt1, srt2) / 100.0,
        "name_jw": JaroWinkler.similarity(n1_norm, n2_norm),
        "name_first_tok_eq": float(bool(t1_list) and bool(t2_list) and t1_list[0] == t2_list[0]),
        "name_in_addr": name_in_addr,
        "addr_partial_ratio": fuzz.partial_ratio(a1_norm, a2_norm) / 100.0 if (a1_norm and a2_norm) else 0.0,
        "addr_jw": JaroWinkler.similarity(a1_norm, a2_norm),
        "num_longest_shared": _longest_shared_number(num1, num2),
        "num_first_eq": float(bool(nums1) and bool(nums2) and nums1[0] == nums2[0]),
        "num_last_eq": float(bool(nums1) and bool(nums2) and nums1[-1] == nums2[-1]),
        "num_only_in_1": float(len(num1 - num2)),
        "num_only_in_2": float(len(num2 - num1)),
        "addr_tok_only_in_1": float(len(at1 - at2)),
        "addr_tok_only_in_2": float(len(at2 - at1)),
        "is_source3": float(str(cand_id).startswith("S3")),
    }
    return feats


FEATURE_COLUMNS = [
    "name_jaccard", "name_overlap", "name_ratio", "name_token_sort_ratio",
    "name_token_set_ratio", "name_partial_ratio", "name_len_ratio",
    "name1_empty", "name2_empty",
    "addr_jaccard", "addr_overlap", "addr_ratio", "addr_token_sort_ratio",
    "addr_token_set_ratio", "addr1_empty", "addr2_empty",
    "num_jaccard", "num_overlap", "num_exact_any_match", "num1_count", "num2_count",
    "country_match",
    "name_cat_ratio", "name_cat_partial", "name_sorted_chars_ratio", "name_jw",
    "name_first_tok_eq", "name_in_addr", "addr_partial_ratio", "addr_jw",
    "num_longest_shared", "num_first_eq", "num_last_eq", "num_only_in_1", "num_only_in_2",
    "addr_tok_only_in_1", "addr_tok_only_in_2", "is_source3",
    "nkeys", "block_score", "block_max_idf", "block_rank", "block_score_rel", "block_n_cands",
]


# Computed for free during candidate generation (see blocking.merge_candidates_prebuilt).
_PASSTHROUGH_COLUMNS = ["nkeys", "block_score", "block_max_idf", "block_rank", "block_score_rel", "block_n_cands"]
_COMPUTED_COLUMNS = [c for c in FEATURE_COLUMNS if c not in _PASSTHROUGH_COLUMNS]


def compute_features_batch(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """pairs_df must have columns: name1, addr1, country1, name2, addr2, country2, cand_id.

    Optional `nkeys` / `block_score` columns (already computed for free
    during candidate generation — see blocking.merge_candidates) are passed
    through as extra features when present.
    """
    records = []
    cols = pairs_df[["name1", "addr1", "country1", "name2", "addr2", "country2", "cand_id"]].itertuples(index=False, name=None)
    for name1, addr1, country1, name2, addr2, country2, cand_id in cols:
        records.append(compute_pair_features(name1, addr1, country1, name2, addr2, country2, cand_id))
    feat_df = pd.DataFrame.from_records(records, columns=_COMPUTED_COLUMNS) if records \
        else pd.DataFrame(columns=_COMPUTED_COLUMNS)
    feat_df.index = pairs_df.index
    for col in _PASSTHROUGH_COLUMNS:
        feat_df[col] = pairs_df[col].values if col in pairs_df.columns else 0.0
    # float32 halves memory vs. pandas' float64 default; irrelevant precision
    # loss for bounded similarity scores feeding a tree model.
    return feat_df[FEATURE_COLUMNS].astype("float32")
