"""Pairwise similarity feature engineering for (Source-1, candidate) pairs.

Features are computed on both raw-ish normalized strings and token sets, so
the classifier can learn to rely on address when the business name is
corrupted/gibberish/non-Latin, and vice versa.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

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


def _longest_shared_number(a: set, b: set) -> float:
    shared = a & b
    return float(max((len(x) for x in shared), default=0))


def compute_pair_features(name1, addr1, country1, name2, addr2, country2, cand_id="") -> dict:
    n1_norm = normalize_name(name1)
    n2_norm = normalize_name(name2)
    t1 = set(name_tokens(name1))
    t2 = set(name_tokens(name2))

    a1_norm = normalize_address(addr1, country1)
    a2_norm = normalize_address(addr2, country2)
    at1 = set(address_tokens(addr1, country1))
    at2 = set(address_tokens(addr2, country2))

    nums1 = address_numbers(addr1)
    nums2 = address_numbers(addr2)
    num1 = set(x for x in nums1 if len(x) >= 2)
    num2 = set(x for x in nums2 if len(x) >= 2)

    # Space/punctuation-free and letter-sorted forms of the name: robust to
    # "@HENDERSONFORTRESS" vs "Henderson Fortress" and to character scrambles.
    t1_list = name_tokens(name1)
    t2_list = name_tokens(name2)
    cat1 = "".join(t1_list)
    cat2 = "".join(t2_list)
    srt1 = "".join(sorted(cat1))
    srt2 = "".join(sorted(cat2))
    # Name tokens found anywhere in the other side's address (and vice versa):
    # sources sometimes shuffle part of the name into the address field.
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
