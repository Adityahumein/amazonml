"""Train the pairwise matcher: blocking -> features -> LightGBM -> threshold tuning.

Usage (from the code/business_entity_resolution directory):
    python3 src/train.py --train-dir ../../dataset/train --model-dir artifacts

Produces artifacts/model.txt (LightGBM booster), artifacts/threshold.json and
artifacts/feature_columns.json, plus a printed validation report (recall
ceiling of blocking, candidate-set size, and the final macro F_0.5 achieved
on a held-out split of the training Source-1 entities).

Processed in bounded batches of Source-1 entities (see pipeline.iter_s1_batches)
rather than materializing the full candidate table at once — at this
pipeline's candidate-set sizes that table is hundreds of millions of rows.
Training features accumulate as compact float32 numpy arrays; validation
pairs are cached to parquet per batch and streamed back through a second
pass (features recomputed on read) for threshold tuning, rather than held in
memory alongside the training data.
"""
import argparse
import gc
import glob
import hashlib
import json
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import build_other_keys, build_s1_keys, merge_candidates_prebuilt
from features import FEATURE_COLUMNS
from parallel_features import compute_features_parallel
from pipeline import attach_fields_prebuilt, build_lookup, iter_s1_batches, load_ground_truth, load_source

NEG_PER_S1_CAP = 30
VAL_FRACTION = 0.15
RANDOM_SEED = 13
S1_BATCH_SIZE = 50_000


def _split_hash(entity_id: str, frac: float) -> bool:
    """Deterministic pseudo-random split so re-runs are reproducible without storing indices."""
    h = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
    return (h % 10_000) / 10_000.0 < frac


def _label_batch(pairs: pd.DataFrame, pos_key_set: set) -> pd.DataFrame:
    """pos_key_set: set of "source1_entity_id\\x1fcand_id" strings (built once
    in main() from the full ground truth). A vectorized `.isin()` against a
    Python set — not a per-row Python function call — since this runs on
    every batch and a per-row loop would dominate runtime at this scale."""
    combined = pairs["source1_entity_id"] + "\x1f" + pairs["cand_id"]
    pairs = pairs.copy()
    pairs["label"] = combined.isin(pos_key_set).to_numpy(dtype="int8")
    return pairs


def _cap_negatives(neg: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(neg) == 0:
        return neg
    r = rng.random(len(neg))
    order = neg.assign(_r=r)
    order["_rank"] = order.groupby("source1_entity_id")["_r"].rank(method="first")
    return order[order["_rank"] <= cap].drop(columns=["_r", "_rank"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="../../dataset/train")
    ap.add_argument("--model-dir", default="artifacts")
    ap.add_argument("--batch-size", type=int, default=S1_BATCH_SIZE,
                     help="Source-1 entities processed per batch (bounds peak memory).")
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    val_cache_dir = os.path.join(args.model_dir, "_val_cache")
    if os.path.exists(val_cache_dir):
        shutil.rmtree(val_cache_dir)
    os.makedirs(val_cache_dir)

    t0 = time.time()
    s1 = load_source(os.path.join(args.train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(args.train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(args.train_dir, "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))
    print(f"[{time.time()-t0:.1f}s] loaded train data: s1={len(s1)} s2={len(s2)} s3={len(s3)} gt={len(gt)}", flush=True)

    pos_key_set = set()
    n_pos = 0
    for s1_id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if ids:
            for cid in ids.split(","):
                pos_key_set.add(s1_id + "\x1f" + cid)
                n_pos += 1
    print(f"[{time.time()-t0:.1f}s] ground truth positives: {n_pos}", flush=True)

    # Decided per S1 entity up front, not per candidate row: an entity with
    # zero blocking candidates still needs scoring (correctly predicting
    # nothing for a true singleton is worth 1.0), so it must be in this set
    # even though it never appears in a candidate batch below.
    val_id_set = set(eid for eid in s1["entity_id"] if _split_hash(eid, VAL_FRACTION))
    print(f"[{time.time()-t0:.1f}s] val S1 entities: {len(val_id_set)} / {len(s1)}", flush=True)

    s2_keys = build_other_keys(s2, "entity_id")
    s3_keys = build_other_keys(s3, "entity_id")
    other_lookup = build_lookup(pd.concat([s2, s3], ignore_index=True))
    print(f"[{time.time()-t0:.1f}s] built S2/S3 blocking keys + lookup", flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    train_X_parts, train_y_parts = [], []
    n_candidates_total = 0
    n_pos_covered_total = 0
    n_s1_with_candidates = 0

    for batch_idx, s1_batch in enumerate(iter_s1_batches(s1, args.batch_size)):
        s1_keys_batch = build_s1_keys(s1_batch)
        cand2 = merge_candidates_prebuilt(s1_keys_batch, s2_keys)
        cand3 = merge_candidates_prebuilt(s1_keys_batch, s3_keys)
        pairs = pd.concat([cand2, cand3], ignore_index=True)
        del cand2, cand3, s1_keys_batch

        pairs = _label_batch(pairs, pos_key_set)

        n_candidates_total += len(pairs)
        n_pos_covered_total += int(pairs["label"].sum())
        n_s1_with_candidates += pairs["source1_entity_id"].nunique()

        is_val = pairs["source1_entity_id"].isin(val_id_set)
        val_part = pairs[is_val]
        train_part = pairs[~is_val]
        del pairs, is_val

        if len(val_part) > 0:
            val_part.to_parquet(os.path.join(val_cache_dir, f"val_{batch_idx:05d}.parquet"))
        del val_part

        train_pos = train_part[train_part["label"] == 1]
        train_neg = _cap_negatives(train_part[train_part["label"] == 0], NEG_PER_S1_CAP, rng)
        train_subset = pd.concat([train_pos, train_neg], ignore_index=True)
        del train_part, train_pos, train_neg

        if len(train_subset) > 0:
            s1_lookup_batch = build_lookup(s1_batch)
            feat_input = attach_fields_prebuilt(train_subset, s1_lookup_batch, other_lookup)
            X_batch = compute_features_parallel(feat_input).to_numpy(dtype="float32", copy=False)
            y_batch = train_subset["label"].to_numpy(dtype="int8")
            train_X_parts.append(X_batch)
            train_y_parts.append(y_batch)
            del feat_input, s1_lookup_batch, train_subset

        gc.collect()
        if batch_idx % 5 == 0:
            print(f"[{time.time()-t0:.1f}s] batch {batch_idx}: "
                  f"{n_s1_with_candidates}/{(batch_idx+1)*args.batch_size} S1 processed, "
                  f"{n_candidates_total} candidates so far", flush=True)

    print(f"[{time.time()-t0:.1f}s] phase 1 done. blocking recall ceiling (pair-level): "
          f"{n_pos_covered_total}/{n_pos} = {n_pos_covered_total/n_pos:.4f}", flush=True)
    print(f"avg candidates/S1: {n_candidates_total/len(s1):.2f}  "
          f"entities-with-candidates: {n_s1_with_candidates}/{len(s1)}", flush=True)

    train_X = np.concatenate(train_X_parts, axis=0)
    train_y = np.concatenate(train_y_parts, axis=0)
    del train_X_parts, train_y_parts
    gc.collect()
    print(f"[{time.time()-t0:.1f}s] training matrix: {train_X.shape} "
          f"positives={int(train_y.sum())}", flush=True)

    n = len(train_X)
    perm = rng.permutation(n)
    cut = int(n * 0.9)
    fit_idx, es_idx = perm[:cut], perm[cut:]

    train_set = lgb.Dataset(train_X[fit_idx], label=train_y[fit_idx], feature_name=FEATURE_COLUMNS)
    es_set = lgb.Dataset(train_X[es_idx], label=train_y[es_idx], reference=train_set)
    del train_X, train_y, perm, fit_idx, es_idx
    gc.collect()

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": RANDOM_SEED,
    }
    booster = lgb.train(
        params, train_set, num_boost_round=1000,
        valid_sets=[es_set], valid_names=["es"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    del train_set, es_set
    gc.collect()
    print(f"[{time.time()-t0:.1f}s] trained LightGBM, best_iteration={booster.best_iteration}", flush=True)

    # Stream the cached validation pairs back through for threshold tuning,
    # all thresholds at once (numpy broadcast + pandas groupby-sum) per file
    # rather than a per-row Python loop, which would dominate runtime here.
    thresholds = np.round(np.arange(0.30, 0.96, 0.02), 2)
    thr_cols = [f"t{k:.2f}" for k in thresholds]

    n_true_acc = pd.Series(dtype="int64")
    pred_acc = pd.DataFrame()
    tp_acc = pd.DataFrame()

    val_files = sorted(glob.glob(os.path.join(val_cache_dir, "*.parquet")))
    for vf in val_files:
        val_part = pd.read_parquet(vf)
        if len(val_part) == 0:
            continue
        s1_lookup_batch = build_lookup(s1[s1["entity_id"].isin(val_part["source1_entity_id"].unique())])
        feat_input = attach_fields_prebuilt(val_part, s1_lookup_batch, other_lookup)
        X = compute_features_parallel(feat_input)
        scores = booster.predict(X)
        del feat_input, X, s1_lookup_batch

        labels = val_part["label"].to_numpy()
        idx = pd.Index(val_part["source1_entity_id"].to_numpy(), name="s1")

        is_pred_mat = (scores[:, None] >= thresholds[None, :]).astype("int32")
        tp_mat = is_pred_mat * labels[:, None]

        grp_true = pd.Series(labels, index=idx).groupby(level=0).sum()
        grp_pred = pd.DataFrame(is_pred_mat, index=idx, columns=thr_cols).groupby(level=0).sum()
        grp_tp = pd.DataFrame(tp_mat, index=idx, columns=thr_cols).groupby(level=0).sum()

        n_true_acc = grp_true if n_true_acc.empty else n_true_acc.add(grp_true, fill_value=0)
        pred_acc = grp_pred if pred_acc.empty else pred_acc.add(grp_pred, fill_value=0)
        tp_acc = grp_tp if tp_acc.empty else tp_acc.add(grp_tp, fill_value=0)

        del val_part, scores, labels, idx, is_pred_mat, tp_mat, grp_true, grp_pred, grp_tp
        gc.collect()

    all_val_ids = list(val_id_set)
    n_true_arr = n_true_acc.reindex(all_val_ids, fill_value=0).to_numpy()
    pred_acc = pred_acc.reindex(all_val_ids, fill_value=0)
    tp_acc = tp_acc.reindex(all_val_ids, fill_value=0)

    best_thr, best_f, best_col = None, -1.0, None
    for thr, col in zip(thresholds, thr_cols):
        n_pred_arr = pred_acc[col].to_numpy()
        tp_arr = tp_acc[col].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(n_pred_arr > 0, tp_arr / n_pred_arr, 0.0)
            recall = np.where(n_true_arr > 0, tp_arr / n_true_arr, 0.0)
            denom = 0.25 * precision + recall
            f = np.where(denom > 0, 1.25 * precision * recall / denom, 0.0)
        f = np.where((n_true_arr == 0) & (n_pred_arr == 0), 1.0, f)
        macro_f = float(f.mean())
        if macro_f > best_f:
            best_f, best_thr, best_col = macro_f, float(thr), col
    print(f"[{time.time()-t0:.1f}s] best threshold={best_thr:.2f}  macro F0.5={best_f:.4f} "
          f"(val S1 entities scored: {len(all_val_ids)})", flush=True)

    tp_sum = int(tp_acc[best_col].sum())
    n_pred_sum = int(pred_acc[best_col].sum())
    n_true_sum = int(n_true_arr.sum())
    precision = tp_sum / n_pred_sum if n_pred_sum else 0.0
    recall = tp_sum / n_true_sum if n_true_sum else 0.0
    print(f"validation micro precision={precision:.4f} recall={recall:.4f} "
          f"(tp={tp_sum} pred={n_pred_sum} true={n_true_sum})", flush=True)

    booster.save_model(os.path.join(args.model_dir, "model.txt"), num_iteration=booster.best_iteration)
    with open(os.path.join(args.model_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": best_thr, "val_macro_f0.5": best_f}, f, indent=2)
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    shutil.rmtree(val_cache_dir, ignore_errors=True)
    print(f"[{time.time()-t0:.1f}s] saved artifacts to {args.model_dir}", flush=True)


if __name__ == "__main__":
    main()
