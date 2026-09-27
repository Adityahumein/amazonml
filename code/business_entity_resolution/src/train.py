"""Train the pairwise matcher: blocking -> features -> LightGBM -> decision-rule tuning.

Usage (from the code/business_entity_resolution directory):
    python3 src/train.py --train-dir ../../dataset/train --model-dir artifacts

Produces artifacts/model.txt (LightGBM booster), artifacts/threshold.json
(the full tuned decision rule, see decision.py) and
artifacts/feature_columns.json, plus a printed validation report.

Validation macro F_0.5 is computed against the *ground-truth* match count of
every held-out Source-1 entity, so true matches that blocking never produced
count as misses — exactly as the leaderboard counts them. (An earlier version
used only the positives present in the candidate set as the denominator,
which silently ignored the blocking recall ceiling and overstated the score:
0.941 on validation vs 0.842 on the leaderboard.)

Processed in bounded batches of Source-1 entities (see pipeline.iter_s1_batches).
Training features accumulate as compact float32 numpy arrays; validation
pairs are cached to parquet per batch and streamed back through a second
pass for tuning.
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
from decision import MIN_SCORE_KEEP, macro_f05_from_counts, pre_threshold_mask
from error_analysis import report as error_report
from features import FEATURE_COLUMNS
from parallel_features import compute_features_parallel
from pipeline import (Progress, predict_scores, attach_fields_prebuilt, build_lookup, fmt_duration, iter_s1_batches,
                      load_ground_truth, load_source)

# Negatives kept per S1 entity for training: the hardest ones by blocking
# score (the look-alikes the classifier actually has to separate) plus a
# random sample of the rest (so easy negatives stay represented too).
HARD_NEG_PER_S1 = 12
RANDOM_NEG_PER_S1 = 8
VAL_FRACTION = 0.15
ES_EVERY_N_BATCHES = 10  # every Nth training batch is held out for early stopping
RANDOM_SEED = 13
# Large model is affordable at prediction time thanks to cascade scoring
# (pipeline.predict_scores): only the ~1% of pairs that aren't obvious
# non-matches go through all trees.
MAX_ROUNDS = 2500
S1_BATCH_SIZE = 50_000
# Exclusivity is only allowed when S2/S3 records (almost) never belong to
# more than one S1 entity in the ground truth.
EXCLUSIVE_MAX_SHARED_RATE = 0.01


def _split_hash(entity_id: str, frac: float) -> bool:
    """Deterministic pseudo-random split so re-runs are reproducible without storing indices."""
    h = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
    return (h % 10_000) / 10_000.0 < frac


def _label_batch(pairs: pd.DataFrame, pos_key_set: set) -> pd.DataFrame:
    combined = pairs["source1_entity_id"] + "\x1f" + pairs["cand_id"]
    pairs = pairs.copy()
    pairs["label"] = combined.isin(pos_key_set).to_numpy(dtype="int8")
    return pairs


def _sample_negatives(neg: pd.DataFrame, n_hard: int, n_random: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(neg) == 0:
        return neg
    hard_rank = neg.groupby("source1_entity_id")["block_score"].rank(method="first", ascending=False)
    is_hard = hard_rank.to_numpy() <= n_hard
    rest = neg[~is_hard]
    rest = rest.assign(_r=rng.random(len(rest)))
    rnd_rank = rest.groupby("source1_entity_id")["_r"].rank(method="first")
    rest = rest[rnd_rank.to_numpy() <= n_random].drop(columns=["_r"])
    return pd.concat([neg[is_hard], rest], ignore_index=True)


def _ground_truth_stats(gt: pd.DataFrame):
    """Match-count per S1, max per source, and how often an S2/S3 record is
    shared by more than one S1 entity."""
    n_true = {}
    pos_key_set = set()
    cand_owner = {}
    n_links = 0
    max_per_src = {"S2": 0, "S3": 0}
    for s1_id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        lst = [c for c in ids.split(",") if c] if ids else []
        n_true[s1_id] = len(lst)
        per_src = {"S2": 0, "S3": 0}
        for cid in lst:
            pos_key_set.add(s1_id + "\x1f" + cid)
            cand_owner.setdefault(cid, s1_id)
            n_links += 1
            src = cid[:2]
            if src in per_src:
                per_src[src] += 1
        for k in per_src:
            max_per_src[k] = max(max_per_src[k], per_src[k])
    # Extra links beyond one owner per record, relative to distinct records.
    shared_rate = (n_links - len(cand_owner)) / len(cand_owner) if cand_owner else 0.0
    return n_true, pos_key_set, max_per_src, shared_rate, cand_owner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="../../dataset/train")
    ap.add_argument("--model-dir", default="artifacts")
    ap.add_argument("--batch-size", type=int, default=S1_BATCH_SIZE,
                    help="Source-1 entities processed per batch (bounds peak memory).")
    ap.add_argument("--hard-neg", type=int, default=HARD_NEG_PER_S1)
    ap.add_argument("--random-neg", type=int, default=RANDOM_NEG_PER_S1)
    ap.add_argument("--s1-sample", type=float, default=1.0,
                    help="Fast-experiment mode: use only this fraction of Source-1 entities (deterministic). "
                         "Source-2/3 stay complete, so blocking statistics stay realistic.")
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

    n_true_map, pos_key_set, max_per_src, shared_rate, cand_owner = _ground_truth_stats(gt)
    if args.s1_sample < 1.0:
        s1 = s1[[_split_hash("sample" + e, args.s1_sample) for e in s1["entity_id"]]].reset_index(drop=True)
        keep_ids = set(s1["entity_id"])
        pos_key_set = {k for k in pos_key_set if k.split("\x1f", 1)[0] in keep_ids}
        gt = gt[gt["source1_entity_id"].isin(keep_ids)]
        print(f"[{time.time()-t0:.1f}s] --s1-sample {args.s1_sample}: using {len(s1)} Source-1 entities", flush=True)
    n_pos = len(pos_key_set)
    gt_max_per_source = max(max_per_src.values()) or None
    allow_exclusive = shared_rate <= EXCLUSIVE_MAX_SHARED_RATE
    print(f"[{time.time()-t0:.1f}s] ground truth positives: {n_pos}; max matches per source: {max_per_src}; "
          f"S2/S3 records shared by >1 S1: {shared_rate:.4%} -> exclusivity "
          f"{'allowed' if allow_exclusive else 'disabled'}", flush=True)

    val_ids = [eid for eid in s1["entity_id"] if _split_hash(eid, VAL_FRACTION)]
    val_id_set = set(val_ids)
    print(f"[{time.time()-t0:.1f}s] val S1 entities: {len(val_id_set)} / {len(s1)}", flush=True)

    s2_keys = build_other_keys(s2, "entity_id")
    s3_keys = build_other_keys(s3, "entity_id")
    other_lookup = build_lookup(pd.concat([s2, s3], ignore_index=True))
    print(f"[{time.time()-t0:.1f}s] built S2/S3 blocking keys ({len(s2_keys.keys)} + {len(s3_keys.keys)} rows) + lookup",
          flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    fit_X, fit_y, es_X, es_y = [], [], [], []
    n_candidates_total = 0
    n_pos_covered_total = 0
    n_s1_with_candidates = 0

    n_batches = (len(s1) + args.batch_size - 1) // args.batch_size
    prog = Progress("phase 1/3 blocking + training features", n_batches, t0)
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

        is_val = pairs["source1_entity_id"].isin(val_id_set).to_numpy()
        val_part = pairs[is_val]
        train_part = pairs[~is_val]
        del pairs, is_val

        if len(val_part) > 0:
            val_part.to_parquet(os.path.join(val_cache_dir, f"val_{batch_idx:05d}.parquet"))
        del val_part

        train_pos = train_part[train_part["label"] == 1]
        train_neg = _sample_negatives(train_part[train_part["label"] == 0], args.hard_neg, args.random_neg, rng)
        train_subset = pd.concat([train_pos, train_neg], ignore_index=True)
        del train_part, train_pos, train_neg

        if len(train_subset) > 0:
            s1_lookup_batch = build_lookup(s1_batch)
            feat_input = attach_fields_prebuilt(train_subset, s1_lookup_batch, other_lookup)
            X_batch = compute_features_parallel(feat_input)[FEATURE_COLUMNS].to_numpy(dtype="float32")
            y_batch = train_subset["label"].to_numpy(dtype="int8")
            if batch_idx % ES_EVERY_N_BATCHES == ES_EVERY_N_BATCHES - 1:
                es_X.append(X_batch)
                es_y.append(y_batch)
            else:
                fit_X.append(X_batch)
                fit_y.append(y_batch)
            del feat_input, s1_lookup_batch, train_subset

        gc.collect()
        seen = min((batch_idx + 1) * args.batch_size, len(s1))
        prog.update(batch_idx + 1, f"{n_candidates_total / seen:.0f} cand/S1")

    prog.done()
    print(f"[{time.time()-t0:.1f}s] phase 1 done. blocking recall ceiling (pair-level): "
          f"{n_pos_covered_total}/{n_pos} = {n_pos_covered_total/max(n_pos,1):.4f}", flush=True)
    print(f"avg candidates/S1: {n_candidates_total/len(s1):.2f}  "
          f"entities-with-candidates: {n_s1_with_candidates}/{len(s1)}", flush=True)

    if not es_X:  # tiny inputs: fall back to the last fit batch
        es_X.append(fit_X[-1])
        es_y.append(fit_y[-1])
    X_fit = np.concatenate(fit_X, axis=0)
    y_fit = np.concatenate(fit_y, axis=0)
    del fit_X, fit_y
    X_es = np.concatenate(es_X, axis=0)
    y_es = np.concatenate(es_y, axis=0)
    del es_X, es_y
    gc.collect()
    print(f"[{time.time()-t0:.1f}s] training matrix: {X_fit.shape} positives={int(y_fit.sum())}; "
          f"early-stop matrix: {X_es.shape}", flush=True)

    train_set = lgb.Dataset(X_fit, label=y_fit, feature_name=FEATURE_COLUMNS, free_raw_data=True)
    es_set = lgb.Dataset(X_es, label=y_es, reference=train_set, free_raw_data=True)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "verbose": -1,
        "seed": RANDOM_SEED,
        "num_threads": os.cpu_count(),
    }
    print(f"[total {fmt_duration(time.time()-t0)}] phase 2/3 LightGBM training started "
          f"(logs every 100 rounds, up to {MAX_ROUNDS} rounds)", flush=True)
    t_fit = time.time()

    def _eta_cb(env):
        it = env.iteration + 1
        if it % 100 == 0:
            el = time.time() - t_fit
            print(f"[total {fmt_duration(time.time()-t0)}] phase 2/3: round {it}, {fmt_duration(el)} elapsed, "
                  f"at most {fmt_duration(el / it * (env.end_iteration - it))} left", flush=True)

    booster = lgb.train(
        params, train_set, num_boost_round=MAX_ROUNDS,
        valid_sets=[es_set], valid_names=["es"],
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False), lgb.log_evaluation(100), _eta_cb],
    )
    # Keep exactly the trees that get saved, so validation scores == test scores.
    booster = lgb.Booster(model_str=booster.model_to_string(num_iteration=booster.best_iteration))
    booster.best_iteration = 0
    del train_set, es_set, X_fit, y_fit, X_es, y_es
    gc.collect()
    print(f"[total {fmt_duration(time.time()-t0)}] trained LightGBM: {booster.num_trees()} trees kept", flush=True)

    imp = sorted(zip(FEATURE_COLUMNS, booster.feature_importance("gain")),
                 key=lambda x: -x[1])
    print("top features (gain): " + ", ".join(f"{n}={v:.3g}" for n, v in imp[:15]), flush=True)

    # ---- Validation: score every cached val pair, keep those above the
    # decision floor, and tune the decision rule against the real metric.
    val_code = {eid: i for i, eid in enumerate(val_ids)}
    n_true = np.array([n_true_map.get(eid, 0) for eid in val_ids], dtype=np.int64)
    s1_parts, cand_parts, s3_parts, score_parts, label_parts = [], [], [], [], []
    cand_vocab = {}
    blocked_pos = np.zeros(len(val_ids), dtype=np.int64)
    pos_parts = []
    val_files = sorted(glob.glob(os.path.join(val_cache_dir, "*.parquet")))
    prog = Progress("phase 3/3 scoring validation", len(val_files), t0)
    for fi, vf in enumerate(val_files):
        val_part = pd.read_parquet(vf)
        if len(val_part) == 0:
            continue
        s1_lookup_batch = build_lookup(s1[s1["entity_id"].isin(val_part["source1_entity_id"].unique())])
        feat_input = attach_fields_prebuilt(val_part, s1_lookup_batch, other_lookup)
        X = compute_features_parallel(feat_input)[FEATURE_COLUMNS].to_numpy(dtype="float32")
        scores = predict_scores(booster, X)
        del feat_input, X, s1_lookup_batch
        is_pos = val_part["label"].to_numpy() == 1
        pos = val_part.loc[is_pos, ["source1_entity_id", "cand_id"]].copy()
        pos["score"] = scores[is_pos]
        pos_parts.append(pos)
        np.add.at(blocked_pos, pos["source1_entity_id"].map(val_code).to_numpy(dtype=np.int64), 1)
        keep = scores >= MIN_SCORE_KEEP
        vp = val_part[keep]
        s1_parts.append(vp["source1_entity_id"].map(val_code).to_numpy(dtype=np.int64))
        cand_parts.append(np.array([cand_vocab.setdefault(c, len(cand_vocab)) for c in vp["cand_id"]], dtype=np.int64))
        s3_parts.append(vp["cand_id"].str.startswith("S3").to_numpy())
        score_parts.append(scores[keep].astype(np.float32))
        label_parts.append(vp["label"].to_numpy(dtype=np.int8))
        del val_part, vp, scores, keep
        gc.collect()
        prog.update(fi + 1)
    prog.done()

    v_s1 = np.concatenate(s1_parts) if s1_parts else np.zeros(0, np.int64)
    v_cand = np.concatenate(cand_parts) if cand_parts else np.zeros(0, np.int64)
    v_s3 = np.concatenate(s3_parts) if s3_parts else np.zeros(0, bool)
    v_score = np.concatenate(score_parts) if score_parts else np.zeros(0, np.float32)
    v_label = np.concatenate(label_parts) if label_parts else np.zeros(0, np.int8)
    n_val = len(val_ids)

    # Evenly spaced in log-odds: fine resolution near 1, where a strong
    # model's optimum sits (0.99 / 0.999 / 0.9999 are equally far apart here).
    thresholds = np.unique(np.round(1.0 / (1.0 + np.exp(-np.linspace(-3.0, 11.0, 141))), 6))
    caps = [None] + ([gt_max_per_source] if gt_max_per_source else [])
    exclusives = [False, True] if allow_exclusive else [False]

    results = []
    for excl in exclusives:
        for cap in caps:
            base = pre_threshold_mask(v_s1, v_cand, v_s3, v_score, excl, cap)
            for thr in thresholds:
                sel = base & (v_score >= thr)
                n_pred = np.bincount(v_s1[sel], minlength=n_val)
                tp = np.bincount(v_s1[sel], weights=v_label[sel], minlength=n_val)
                f = macro_f05_from_counts(n_pred, tp, n_true)
                results.append((f, float(thr), excl, cap, int(sel.sum()), int(tp.sum())))

    results.sort(key=lambda r: -r[0])
    for excl in exclusives:
        for cap in caps:
            best = max(r for r in results if r[2] == excl and r[3] == cap)
            print(f"  exclusive={excl!s:5} cap={cap!s:4}: best thr={best[1]:.6f} macro F0.5={best[0]:.4f}", flush=True)
    best_f, best_thr, best_excl, best_cap, n_pred_sum, tp_sum = results[0]
    n_true_sum = int(n_true.sum())
    print(f"[{time.time()-t0:.1f}s] BEST: threshold={best_thr:.6f} exclusive={best_excl} "
          f"max_per_source={best_cap}  macro F0.5={best_f:.4f} (val S1 entities: {n_val})", flush=True)
    print(f"validation micro precision={tp_sum/max(n_pred_sum,1):.4f} recall={tp_sum/max(n_true_sum,1):.4f} "
          f"(tp={tp_sum} pred={n_pred_sum} true={n_true_sum})", flush=True)

    try:
        s1_val = s1[s1["entity_id"].isin(val_id_set)]
        val_country = s1_val.set_index("entity_id").loc[val_ids, "country"].to_numpy()
        gt_val = gt[gt["source1_entity_id"].isin(val_id_set)]
        gt_pairs = [(a, c) for a, ids in zip(gt_val["source1_entity_id"], gt_val["matched_entity_ids"])
                    if ids for c in ids.split(",") if c]
        gt_val_pairs = pd.DataFrame(gt_pairs, columns=["source1_entity_id", "cand_id"])
        pos_rows = pd.concat(pos_parts, ignore_index=True) if pos_parts else \
            pd.DataFrame(columns=["source1_entity_id", "cand_id", "score"])
        cand_ids = np.empty(len(cand_vocab), dtype=object)
        for c, i in cand_vocab.items():
            cand_ids[i] = c
        error_report(os.path.join(args.model_dir, "error_analysis"), val_ids, val_country, n_true, blocked_pos,
                     v_s1, cand_ids[v_cand], v_s3, v_score, v_label,
                     pre_threshold_mask(v_s1, v_cand, v_s3, v_score, best_excl, best_cap), best_thr,
                     pos_rows, gt_val_pairs, cand_owner, build_lookup(s1_val), other_lookup)
    except Exception as e:  # analysis is diagnostics only — never lose a trained model over it
        print(f"error analysis failed: {e!r}", flush=True)

    booster.save_model(os.path.join(args.model_dir, "model.txt"))
    with open(os.path.join(args.model_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": best_thr, "exclusive": bool(best_excl), "max_per_source": best_cap,
                   "min_score_keep": MIN_SCORE_KEEP, "val_macro_f0.5": best_f}, f, indent=2)
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    shutil.rmtree(val_cache_dir, ignore_errors=True)
    print(f"[total {fmt_duration(time.time()-t0)}] saved artifacts to {args.model_dir}", flush=True)


if __name__ == "__main__":
    main()
