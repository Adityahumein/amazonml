"""Candidate generation (blocking) via multi-key inverted-index style joins.

For each record we derive cheap "blocking keys" from its name and address.
Two records become a candidate pair if they share any key value, restricted
to the same country (ground truth never crosses country).

Key families (all scoped to country):

- name:    `p2` two alphabetically-first token prefixes (or `full1` for
           one-token names), `cat` all tokens concatenated (catches
           "@HENDERSONFORTRESS" vs "Henderson Fortress"), `ana` the sorted
           letters of that concatenation (catches character scrambles), and
           `nt` each long name token on its own.
- address: `n` numeric runs, `t` longest tokens, plus combination keys
           `hn` (number + token) and `tt` (token + token). A single common
           token or number ("main", "400001") exceeds the posting cap and is
           dropped, but a combination of two of them is specific, so the
           combos recover exactly the matches the cap used to throw away.
- cross:   `xn` name prefix + address number, for records whose address
           tokens are too corrupted but the house/PIN number survived.

Keys are hashed to int64 (blake2b, stable across processes) and records are
referenced by integer row position, so the key tables and the join are
purely numeric — several times smaller and faster than string columns.
Overly generic key values are dropped before the join via a frequency cap,
and joined candidates are pruned per Source-1 entity by an IDF-weighted
score (see merge_candidates_prebuilt). The join runs per country and is
batched over Source-1 rows to keep peak memory bounded.
"""
import gc
import hashlib
import multiprocessing as mp
from itertools import combinations

import numpy as np
import pandas as pd

from normalize import name_tokens, address_tokens, address_numbers

# Leave one core for the OS.
N_JOBS = max(1, mp.cpu_count() - 1)

# S1-side rows processed per merge batch within a country partition.
MERGE_BATCH_ENTITIES = 60_000

NAME_PREFIX_LEN = 4
# Drop a key value shared by more than this many records on either side.
MAX_POSTINGS = 900

MAX_ADDR_TOKEN_KEYS = 3   # longest address tokens used as single keys
MAX_COMBO_TOKENS = 4      # longest address tokens used in tt/hn combos
MAX_NUMBERS = 3

MAX_CANDIDATES_PER_S1_PER_SOURCE = 250

PAIR_COLUMNS = ["source1_entity_id", "cand_id", "nkeys", "block_score", "block_max_idf",
                "block_rank", "block_score_rel", "block_n_cands"]


def _name_key_parts(name: str):
    toks = name_tokens(name)
    if not toks:
        return []
    uniq = sorted(set(toks))
    keys = []
    if len(uniq) >= 2:
        keys.append("p2:" + uniq[0][:NAME_PREFIX_LEN] + "|" + uniq[1][:NAME_PREFIX_LEN])
    else:
        keys.append("full1:" + uniq[0])
    cat = "".join(toks)
    if len(cat) >= 6:
        keys.append("cat:" + cat)
        letters = "".join(sorted(c for c in cat if c.isalpha()))
        if len(letters) >= 8:
            keys.append("ana:" + letters)
    long_toks = sorted((t for t in uniq if len(t) >= 5 and not t.isdigit()), key=len, reverse=True)
    keys.extend("nt:" + t for t in long_toks[:2])
    return keys


def _addr_key_parts(addr: str, country: str):
    nums = []
    for n in address_numbers(addr):
        if len(n) >= 2 and n not in nums:
            nums.append(n)
    nums = nums[:MAX_NUMBERS]
    toks = sorted(set(t for t in address_tokens(addr, country) if len(t) >= 4), key=lambda t: (-len(t), t))
    keys = ["n:" + n for n in nums]
    keys.extend("t:" + t for t in toks[:MAX_ADDR_TOKEN_KEYS] if len(t) >= 5)
    combo = toks[:MAX_COMBO_TOKENS]
    for a, b in combinations(sorted(combo), 2):
        keys.append("tt:" + a + "|" + b)
    for n in nums[:2]:
        for t in combo[:2]:
            keys.append("hn:" + n + "|" + t)
    return keys, nums


def record_keys(name: str, addr: str, country: str):
    keys = _name_key_parts(name)
    addr_keys, nums = _addr_key_parts(addr, country)
    keys.extend(addr_keys)
    toks = sorted(set(name_tokens(name)))
    if toks and nums:
        pref = toks[0][:NAME_PREFIX_LEN]
        for n in nums[:2]:
            keys.append("xn:" + pref + "|" + n)
    return keys


def _hash_key(country: str, key: str) -> int:
    h = hashlib.blake2b((country + "\x1f" + key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=True)


def _build_key_arrays(args):
    """Explode records into (row_position, key_hash) arrays."""
    names, addrs, countries, offset = args
    rows, keys = [], []
    for i in range(len(names)):
        country = countries[i]
        ks = set(record_keys(names[i], addrs[i], country))
        if not ks:
            continue
        rows.extend([offset + i] * len(ks))
        keys.extend(_hash_key(country, k) for k in ks)
    return np.asarray(rows, dtype="int64"), np.asarray(keys, dtype="int64")


def _build_key_table(df: pd.DataFrame, chunk_size: int = 100_000) -> pd.DataFrame:
    names = df["business_name"].to_numpy()
    addrs = df["business_address"].to_numpy()
    countries = df["country"].to_numpy()
    chunks = [(names[i:i + chunk_size], addrs[i:i + chunk_size], countries[i:i + chunk_size], i)
              for i in range(0, len(df), chunk_size)]
    if len(chunks) <= 1 or N_JOBS <= 1:
        results = [_build_key_arrays(c) for c in chunks]
    else:
        with mp.Pool(N_JOBS) as pool:
            results = pool.map(_build_key_arrays, chunks)
    if results:
        rows = np.concatenate([r[0] for r in results])
        keys = np.concatenate([r[1] for r in results])
    else:
        rows = np.zeros(0, dtype="int64")
        keys = np.zeros(0, dtype="int64")
    return pd.DataFrame({"row": rows.astype("int32"), "key": keys,
                         "country": pd.Categorical(countries[rows] if len(rows) else [])})


def build_side_keys(df: pd.DataFrame, cap_postings: bool = True) -> pd.DataFrame:
    """Key table [row, key, country, group_size]; `row` indexes into df."""
    keys = _build_key_table(df)
    sizes = keys.groupby("key")["row"].transform("size").to_numpy(dtype="int32")
    keys["group_size"] = sizes
    if cap_postings:
        keys = keys[sizes <= MAX_POSTINGS].reset_index(drop=True)
    return keys


class SideKeys:
    """A capped key table plus the entity ids its `row` column refers to."""

    def __init__(self, df: pd.DataFrame, id_col: str = "entity_id"):
        self.ids = df[id_col].to_numpy()
        self.keys = build_side_keys(df)
        self._parts = None

    def partitions(self) -> dict:
        """Key table split by country (cached — the other side is reused
        across every S1 batch)."""
        if self._parts is None:
            self._parts = {c: g.reset_index(drop=True)
                           for c, g in self.keys.groupby("country", observed=True)}
        return self._parts


def build_s1_keys(s1_df: pd.DataFrame) -> SideKeys:
    # Cap the S1 side too — a generic key shared by many S1 records
    # multiplies against every match on the other side.
    return SideKeys(s1_df)


def build_other_keys(other_df: pd.DataFrame, other_id_col: str = "entity_id") -> SideKeys:
    """Build the capped S2/S3-side key table once, for reuse across many S1 batches."""
    return SideKeys(other_df, other_id_col)


def _merge_one_batch(s1_batch: pd.DataFrame, other_part: pd.DataFrame, n_other: int, max_per_s1: int):
    merged = s1_batch[["row", "key"]].merge(other_part[["row", "key", "group_size"]], on="key",
                                            how="inner", suffixes=("_s1", "_o"))
    if len(merged) == 0:
        return None
    pair = merged["row_s1"].to_numpy(dtype="int64") * n_other + merged["row_o"].to_numpy(dtype="int64")
    idf = 1.0 / merged["group_size"].to_numpy(dtype="float64")
    del merged
    g = pd.DataFrame({"pair": pair, "idf": idf}).groupby("pair")["idf"].agg(["size", "sum", "max"])
    pair = g.index.to_numpy(dtype="int64")
    s1_row = pair // n_other
    o_row = pair % n_other
    score = g["sum"].to_numpy()
    nkeys = g["size"].to_numpy()
    max_idf = g["max"].to_numpy()
    del g

    order = np.lexsort((-score, s1_row))
    s1_row, o_row, score, nkeys, max_idf = s1_row[order], o_row[order], score[order], nkeys[order], max_idf[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s1_row)) + 1]
    counts = np.diff(np.r_[starts, len(s1_row)])
    rank = np.arange(len(s1_row)) - np.repeat(starts, counts)
    best = np.repeat(score[starts], counts)
    n_c = np.repeat(np.minimum(counts, max_per_s1), counts)
    keep = rank < max_per_s1
    return {
        "s1_row": s1_row[keep].astype("int32"), "o_row": o_row[keep].astype("int32"),
        "nkeys": nkeys[keep].astype("float32"), "block_score": score[keep].astype("float32"),
        "block_max_idf": max_idf[keep].astype("float32"), "block_rank": rank[keep].astype("float32"),
        "block_score_rel": (score[keep] / best[keep]).astype("float32"),
        "block_n_cands": n_c[keep].astype("float32"),
    }


# Per-country other-side key tables for the current merge call. Set in the
# parent right before forking the worker pool, so workers inherit them
# copy-on-write instead of having a multi-GB table pickled to them on every
# S1 batch.
_OTHER_PARTS = {}


def _merge_one_partition(args):
    """Join + IDF-score + prune for one (already country-scoped) partition,
    batched over S1 rows so the raw pre-prune join is never materialized in
    full. Each S1 row's keys stay within one batch, so pruning is exact."""
    s1_part, country, n_other, max_per_s1 = args
    other_part = _OTHER_PARTS[country]
    if len(s1_part) == 0 or len(other_part) == 0:
        return []
    r = s1_part["row"].to_numpy()
    rows = np.unique(r)
    out = []
    for i in range(0, len(rows), MERGE_BATCH_ENTITIES):
        lo, hi = rows[i], rows[min(i + MERGE_BATCH_ENTITIES, len(rows)) - 1]
        batch = s1_part[(r >= lo) & (r <= hi)]
        res = _merge_one_batch(batch, other_part, n_other, max_per_s1)
        if res is not None:
            out.append(res)
        del batch
        gc.collect()
    return out


def merge_candidates_prebuilt(s1_keys: SideKeys, other_keys: SideKeys,
                              max_per_s1: int = MAX_CANDIDATES_PER_S1_PER_SOURCE) -> pd.DataFrame:
    """Candidates for every S1 row against one other source.

    Ranking signal is an IDF-style score: each shared key contributes
    1/group_size, summed over every key the pair shares. The raw shared-key
    count, the single most specific shared key (`block_max_idf`), the pair's
    rank within its S1 entity, its score relative to that entity's best
    candidate and the entity's candidate count are all returned as free
    classifier features.

    Runs one process per distinct country.
    """
    s1k, ok = s1_keys.keys, other_keys.keys
    n_other = max(1, len(other_keys.ids))
    countries = sorted(set(s1k["country"].unique()) & set(ok["country"].unique()))
    other_parts = other_keys.partitions()
    _OTHER_PARTS.clear()
    _OTHER_PARTS.update({c: other_parts[c] for c in countries})
    tasks = [(s1k[s1k["country"] == c], c, n_other, max_per_s1) for c in countries]

    if len(tasks) <= 1 or N_JOBS <= 1 or "fork" not in mp.get_all_start_methods():
        results = [_merge_one_partition(t) for t in tasks]
    else:
        with mp.get_context("fork").Pool(min(N_JOBS, len(tasks))) as pool:
            results = pool.map(_merge_one_partition, tasks)
    _OTHER_PARTS.clear()
    parts = [p for r in results for p in r]
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype="object" if c in ("source1_entity_id", "cand_id") else "float32")
                             for c in PAIR_COLUMNS})
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out = pd.DataFrame({
        "source1_entity_id": s1_keys.ids[cols.pop("s1_row")],
        "cand_id": other_keys.ids[cols.pop("o_row")],
    })
    for k, v in cols.items():
        out[k] = v
    return out[PAIR_COLUMNS]


def merge_candidates(s1_keys: SideKeys, other_df: pd.DataFrame, other_id_col: str,
                     max_per_s1: int = MAX_CANDIDATES_PER_S1_PER_SOURCE) -> pd.DataFrame:
    """One-shot version: builds other_keys and merges in one call."""
    return merge_candidates_prebuilt(s1_keys, build_other_keys(other_df, other_id_col), max_per_s1)


def generate_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame, other_id_col: str) -> pd.DataFrame:
    return merge_candidates(build_s1_keys(s1_df), other_df, other_id_col)
