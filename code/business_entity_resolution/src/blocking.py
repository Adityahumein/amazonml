"""Candidate generation (blocking) via multi-key inverted-index style joins.

For each record we derive cheap "blocking keys": a 2-token name-prefix combo
(or an exact single-token match for one-word names), address house/PIN
numbers, and the longest, most distinctive normalized address tokens. Two
records become a candidate pair if they share any key value, restricted to
the same country (ground truth never crosses country).

Uses pandas hash-joins (vectorized, C-level) instead of manual Python
inverted indices. Overly generic key values are dropped before the join via
a frequency cap on both sides, and joined candidates are further pruned per
Source-1 entity by an IDF-weighted score (see merge_candidates_prebuilt).
Both the join and the pruning are done per-country and batched over Source-1
entities (see _merge_one_partition) to keep peak memory bounded regardless
of country size or candidate-set size.
"""
import gc
import multiprocessing as mp

import numpy as np
import pandas as pd

from normalize import name_tokens, address_tokens, address_numbers

# Leave one core for the OS.
N_JOBS = max(1, mp.cpu_count() - 1)

# S1-side rows processed per merge batch within a country partition (see
# _merge_one_partition) — bounds the raw pre-prune join size independent of
# country size or MAX_POSTINGS.
MERGE_BATCH_ENTITIES = 60_000

NAME_PREFIX_LEN = 4
# Drop a key value shared by more than this many records on either side of
# the join. Single-signal keys (a soundex code, a 3-char prefix) have too few
# distinct buckets and blow up combinatorially even after capping, so every
# key type here is a multi-token/exact-token combination with many distinct
# values. Measured pair-level recall ceiling vs. train_ground_truth.tsv:
# 250->77.4%, 400->81.7%, 600->85.3%, 900->88.2%.
MAX_POSTINGS = 900


def _name_key_parts(name: str):
    toks = sorted(set(name_tokens(name)))
    if not toks:
        return []
    keys = []
    if len(toks) >= 2:
        keys.append("p2:" + toks[0][:NAME_PREFIX_LEN] + "|" + toks[1][:NAME_PREFIX_LEN])
    else:
        keys.append("full1:" + toks[0])
    return keys


MAX_ADDR_TOKEN_KEYS = 3  # only the longest (most discriminative) tokens become blocking keys


def _addr_key_parts(addr: str, country: str):
    nums = [n for n in address_numbers(addr) if len(n) >= 2][:3]
    toks = [t for t in set(address_tokens(addr, country)) if len(t) >= 5]
    toks.sort(key=len, reverse=True)
    keys = ["n:" + n for n in nums]
    keys.extend("t:" + t for t in toks[:MAX_ADDR_TOKEN_KEYS])
    return keys


def _build_key_table(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Explode each record into (entity_id, country, key) rows."""
    ids = df[id_col].to_numpy()
    countries = df["country"].to_numpy()
    names = df["business_name"].to_numpy()
    addrs = df["business_address"].to_numpy()

    rows_id = []
    rows_country = []
    rows_key = []
    for i in range(len(df)):
        country = countries[i]
        keys = _name_key_parts(names[i])
        keys.extend(_addr_key_parts(addrs[i], country))
        if not keys:
            continue
        rows_id.extend([ids[i]] * len(keys))
        rows_country.extend([country] * len(keys))
        rows_key.extend(keys)

    return pd.DataFrame({id_col: rows_id, "country": rows_country, "key": rows_key})


def _build_key_table_chunk(args):
    df_chunk, id_col = args
    return _build_key_table(df_chunk, id_col)


def _build_key_table_parallel(df: pd.DataFrame, id_col: str, chunk_size: int = 150_000) -> pd.DataFrame:
    """Row-chunked multiprocessing version of _build_key_table."""
    if len(df) <= chunk_size or N_JOBS <= 1:
        return _build_key_table(df, id_col)
    cols = [id_col, "country", "business_name", "business_address"]
    chunks = [(df[cols].iloc[i:i + chunk_size], id_col) for i in range(0, len(df), chunk_size)]
    with mp.Pool(N_JOBS) as pool:
        results = pool.map(_build_key_table_chunk, chunks)
    return pd.concat(results, ignore_index=True)


def build_side_keys(df: pd.DataFrame, id_col: str, cap_postings: bool) -> pd.DataFrame:
    keys = _build_key_table_parallel(df, id_col)
    if cap_postings:
        sizes = keys.groupby(["country", "key"])[id_col].transform("size")
        keys = keys[sizes <= MAX_POSTINGS].copy()
        keys["group_size"] = sizes[sizes <= MAX_POSTINGS].values
    return keys


def _combined_key(df: pd.DataFrame) -> pd.Series:
    return df["country"].astype(str) + "\x1f" + df["key"]


def _prepare_other_keyed(other_part: pd.DataFrame):
    """Factorize the (country, key) vocabulary once per partition so repeated
    S1 batches (see _merge_one_partition) reuse the same category codes."""
    cat = pd.Categorical(_combined_key(other_part))
    other_keyed = other_part.assign(_jk=cat.codes.astype("int32"))
    return other_keyed, cat.categories


def build_s1_keys(s1_df: pd.DataFrame) -> pd.DataFrame:
    # Cap the S1 side too — a generic key shared by many S1 records
    # multiplies against every match on the other side.
    keys = build_side_keys(s1_df, "entity_id", cap_postings=True)
    return keys.rename(columns={"entity_id": "source1_entity_id"})


MAX_CANDIDATES_PER_S1_PER_SOURCE = 250


def _merge_one_batch(s1_batch: pd.DataFrame, other_keyed: pd.DataFrame, other_categories, max_per_s1: int) -> pd.DataFrame:
    codes = pd.Categorical(_combined_key(s1_batch), categories=other_categories).codes
    s1_batch = s1_batch.assign(_jk=codes.astype("int32"))
    s1_batch = s1_batch[s1_batch["_jk"] >= 0]  # -1 = key has no possible match on the other side

    merged = s1_batch[["source1_entity_id", "_jk"]].merge(
        other_keyed[["cand_id", "_jk", "group_size"]], on="_jk", how="inner")
    merged["idf"] = 1.0 / merged["group_size"]

    grouped = merged.groupby(["source1_entity_id", "cand_id"], as_index=False).agg(
        nkeys=("idf", "size"), block_score=("idf", "sum"))
    del merged
    grouped["rank"] = grouped.groupby("source1_entity_id")["block_score"].rank(method="first", ascending=False)
    pruned = grouped[grouped["rank"] <= max_per_s1]
    return pruned[["source1_entity_id", "cand_id", "nkeys", "block_score"]].copy()


def _merge_one_partition(args):
    """Join + IDF-score + prune for one (already country-scoped) partition.

    Blocking keys never match across countries, so per-country partitions
    are exact, not approximate — safe to run in separate processes. Within a
    partition, the S1 side is further batched (MERGE_BATCH_ENTITIES entities
    at a time, each entity's keys staying in one batch so ranking/pruning is
    still exact) so the raw pre-prune join for a large country is never
    materialized in full at once.
    """
    s1_part, other_part, max_per_s1 = args
    if len(s1_part) == 0 or len(other_part) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "cand_id", "nkeys", "block_score"])

    other_keyed, other_categories = _prepare_other_keyed(other_part)

    entity_ids = s1_part["source1_entity_id"].unique()
    if len(entity_ids) <= MERGE_BATCH_ENTITIES:
        return _merge_one_batch(s1_part, other_keyed, other_categories, max_per_s1)

    results = []
    for i in range(0, len(entity_ids), MERGE_BATCH_ENTITIES):
        batch_ids = entity_ids[i:i + MERGE_BATCH_ENTITIES]
        s1_batch = s1_part[s1_part["source1_entity_id"].isin(batch_ids)]
        results.append(_merge_one_batch(s1_batch, other_keyed, other_categories, max_per_s1))
        del s1_batch
        gc.collect()
    return pd.concat(results, ignore_index=True)


def build_other_keys(other_df: pd.DataFrame, other_id_col: str) -> pd.DataFrame:
    """Build the capped S2/S3-side key table once, for reuse across many S1
    batches (see pipeline.iter_scored_batches)."""
    other_keys = build_side_keys(other_df, other_id_col, cap_postings=True)
    return other_keys.rename(columns={other_id_col: "cand_id"})


def merge_candidates_prebuilt(s1_keys: pd.DataFrame, other_keys: pd.DataFrame,
                               max_per_s1: int = MAX_CANDIDATES_PER_S1_PER_SOURCE) -> pd.DataFrame:
    """Same as merge_candidates, but takes an already-built other_keys table
    (from build_other_keys) instead of the raw dataframe + id column.

    Ranking signal is an IDF-style score: each shared key contributes
    1/group_size (a key shared by a handful of records is stronger evidence
    than one shared by hundreds), summed over every key the pair shares
    (`nkeys`, the raw shared-key count, is kept too as a classifier feature).
    This out-performed a plain shared-key count in held-out recall-retention
    testing. Both terms fall out of the join itself — no extra similarity
    computation needed.

    Runs one process per distinct country (US / India / France / ...).
    """
    countries = sorted(set(s1_keys["country"].unique()) | set(other_keys["country"].unique()))
    tasks = [
        (s1_keys[s1_keys["country"] == c].reset_index(drop=True),
         other_keys[other_keys["country"] == c].reset_index(drop=True),
         max_per_s1)
        for c in countries
    ]

    if len(tasks) <= 1 or N_JOBS <= 1:
        results = [_merge_one_partition(t) for t in tasks]
    else:
        with mp.Pool(min(N_JOBS, len(tasks))) as pool:
            results = pool.map(_merge_one_partition, tasks)
    return pd.concat(results, ignore_index=True)


def merge_candidates(s1_keys: pd.DataFrame, other_df: pd.DataFrame, other_id_col: str,
                      max_per_s1: int = MAX_CANDIDATES_PER_S1_PER_SOURCE) -> pd.DataFrame:
    """One-shot version: builds other_keys and merges in one call. Prefer
    build_other_keys() + merge_candidates_prebuilt() when calling repeatedly
    (e.g. once per S1 batch) against the same S2/S3 data."""
    other_keys = build_other_keys(other_df, other_id_col)
    return merge_candidates_prebuilt(s1_keys, other_keys, max_per_s1)


def generate_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame, other_id_col: str) -> pd.DataFrame:
    """Return DataFrame[source1_entity_id, cand_id, nkeys, block_score]."""
    s1_keys = build_s1_keys(s1_df)
    return merge_candidates(s1_keys, other_df, other_id_col)
