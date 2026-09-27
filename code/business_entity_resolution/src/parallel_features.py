"""Multiprocessing wrapper around features.compute_features_batch for scale."""
import multiprocessing as mp

import pandas as pd

from features import compute_features_batch, FEATURE_COLUMNS

# Same policy as blocking.N_JOBS: per-worker memory is bounded by the caller's
# batching, so it's safe to use most cores — just leave one for the OS.
N_JOBS = max(1, mp.cpu_count() - 1)


def _worker(chunk: pd.DataFrame) -> pd.DataFrame:
    return compute_features_batch(chunk)


def compute_features_parallel(pairs_df: pd.DataFrame, n_jobs: int = None, chunk_size: int = 200_000) -> pd.DataFrame:
    if len(pairs_df) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    n_jobs = n_jobs or N_JOBS
    if len(pairs_df) <= chunk_size or n_jobs <= 1:
        return compute_features_batch(pairs_df)

    chunks = [pairs_df.iloc[i:i + chunk_size] for i in range(0, len(pairs_df), chunk_size)]
    with mp.Pool(n_jobs) as pool:
        results = pool.map(_worker, chunks)
    out = pd.concat(results)
    out = out.loc[pairs_df.index]
    del results, chunks
    return out
