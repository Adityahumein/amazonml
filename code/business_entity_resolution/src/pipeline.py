"""Shared data-loading / candidate-assembly helpers used by train.py and predict.py.

The scoring path (build_lookup / attach_fields_prebuilt / iter_scored_batches)
is batched over Source-1 entities end to end: the full candidate set for a
multi-million-entity file doesn't fit in memory as a single DataFrame, even
with float32 features. Processing a bounded batch at a time and discarding
everything but its scored result keeps peak memory roughly constant
regardless of input size.
"""
import gc

import pandas as pd

from blocking import build_other_keys, build_s1_keys, merge_candidates, merge_candidates_prebuilt
from features import FEATURE_COLUMNS
from parallel_features import compute_features_parallel

S1_BATCH_SIZE = 50_000


def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)


def build_candidate_pairs(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """Pruned union of blocking candidates from Source-2 and Source-3 for every Source-1
    record (see blocking.merge_candidates for the pruning rule).

    Only safe to call on a small/moderate s1_df (e.g. a single batch) — see
    iter_scored_batches for the memory-bounded way to process a full file.
    """
    s1_keys = build_s1_keys(s1_df)
    cand2 = merge_candidates(s1_keys, s2_df, "entity_id")
    cand3 = merge_candidates(s1_keys, s3_df, "entity_id")
    pairs = pd.concat([cand2, cand3], ignore_index=True)
    return pairs


def build_lookup(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.drop_duplicates("entity_id").set_index("entity_id")[["business_name", "business_address", "country"]]
    idx.columns = ["name", "addr", "country_"]
    return idx


def attach_fields(pairs_df: pd.DataFrame, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """Join name/address/country for both sides of each (source1_entity_id, cand_id) pair.

    Convenience one-shot version — builds the S2+S3 lookup on every call, so
    prefer build_lookup() once + attach_fields_prebuilt() when calling this
    repeatedly (e.g. once per S1 batch) against the same S2/S3 data.
    """
    other_lookup = build_lookup(pd.concat([s2_df, s3_df], ignore_index=True))
    return attach_fields_prebuilt(pairs_df, build_lookup(s1_df), other_lookup)


def attach_fields_prebuilt(pairs_df: pd.DataFrame, s1_lookup: pd.DataFrame, other_lookup: pd.DataFrame) -> pd.DataFrame:
    out = pairs_df.join(s1_lookup, on="source1_entity_id")
    out = out.rename(columns={"name": "name1", "addr": "addr1", "country_": "country1"})
    out = out.join(other_lookup, on="cand_id")
    out = out.rename(columns={"name": "name2", "addr": "addr2", "country_": "country2"})
    return out


def iter_s1_batches(s1_df: pd.DataFrame, batch_size: int = S1_BATCH_SIZE):
    for i in range(0, len(s1_df), batch_size):
        yield s1_df.iloc[i:i + batch_size].reset_index(drop=True)


def iter_scored_batches(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                         booster, batch_size: int = S1_BATCH_SIZE):
    """Yield (s1_batch, scored_pairs) one Source-1 batch at a time.

    s2_keys/s3_keys and the S2+S3 name/address lookup are built once (the
    expensive part) and reused across every batch; only a batch's own
    candidates, features and scores are ever materialized at once, and are
    discarded before the next batch starts.
    """
    s2_keys = build_other_keys(s2_df, "entity_id")
    s3_keys = build_other_keys(s3_df, "entity_id")
    other_lookup = build_lookup(pd.concat([s2_df, s3_df], ignore_index=True))

    for s1_batch in iter_s1_batches(s1_df, batch_size):
        s1_keys_batch = build_s1_keys(s1_batch)
        cand2 = merge_candidates_prebuilt(s1_keys_batch, s2_keys)
        cand3 = merge_candidates_prebuilt(s1_keys_batch, s3_keys)
        pairs = pd.concat([cand2, cand3], ignore_index=True)
        del cand2, cand3, s1_keys_batch

        s1_lookup_batch = build_lookup(s1_batch)
        feat_input = attach_fields_prebuilt(pairs, s1_lookup_batch, other_lookup)
        X = compute_features_parallel(feat_input)
        del feat_input, s1_lookup_batch

        pairs = pairs.reset_index(drop=True)
        pairs["score"] = booster.predict(X[FEATURE_COLUMNS].to_numpy(dtype="float32")) if len(pairs) else []
        del X
        gc.collect()

        yield s1_batch, pairs
        del pairs
        gc.collect()


def candidate_ids_to_str(ids) -> str:
    return ",".join(ids)
