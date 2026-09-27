"""Macro F_0.5 scoring utilities, matching the challenge's evaluation definition."""
import numpy as np
import pandas as pd


def macro_f_beta(pred_sets: dict, true_sets: dict, beta: float = 0.5) -> float:
    """pred_sets / true_sets: {source1_entity_id: set(matched_ids)}.

    Every key in true_sets must be scored (missing keys in pred_sets are
    treated as an empty prediction), matching how the leaderboard scores a
    submission that must cover every Source-1 entity.
    """
    beta2 = beta * beta
    total = 0.0
    n = 0
    for s1_id, true_ids in true_sets.items():
        pred_ids = pred_sets.get(s1_id, set())
        n += 1
        if not true_ids and not pred_ids:
            total += 1.0
            continue
        if not pred_ids:
            continue  # precision undefined -> contributes 0
        tp = len(true_ids & pred_ids)
        if tp == 0:
            continue
        precision = tp / len(pred_ids)
        recall = tp / len(true_ids)
        denom = beta2 * precision + recall
        if denom == 0:
            continue
        f = (1 + beta2) * precision * recall / denom
        total += f
    return total / n if n else 0.0


def macro_f_beta_vectorized(source1_ids, labels, is_pred, all_s1_ids, beta: float = 0.5) -> float:
    """Same metric as macro_f_beta, computed with numeric groupby-sums instead
    of Python sets — needed once the candidate table is large (100M+ rows).
    `source1_ids`/`labels`/`is_pred` are equal-length arrays over candidate
    pairs; `all_s1_ids` is every S1 entity that must be scored, including
    ones with zero candidate rows (still a correctly/incorrectly predicted
    singleton).
    """
    beta2 = beta * beta
    label_arr = np.asarray(labels, dtype="int8")
    pred_arr = np.asarray(is_pred, dtype="int8")
    tp_arr = label_arr & pred_arr
    grp = pd.DataFrame({"s1": np.asarray(source1_ids), "n_pred": pred_arr, "n_true": label_arr, "tp": tp_arr})
    agg = grp.groupby("s1").sum()
    agg = agg.reindex(all_s1_ids, fill_value=0)

    n_pred = agg["n_pred"].to_numpy(dtype="float64")
    n_true = agg["n_true"].to_numpy(dtype="float64")
    tp = agg["tp"].to_numpy(dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(n_pred > 0, tp / n_pred, 0.0)
        recall = np.where(n_true > 0, tp / n_true, 0.0)
        denom = beta2 * precision + recall
        f = np.where(denom > 0, (1 + beta2) * precision * recall / denom, 0.0)

    # singleton correctly predicted empty -> 1.0; a false merge on a true
    # singleton already scores 0.0 above (tp==0 forces f==0).
    is_singleton_correct = (n_true == 0) & (n_pred == 0)
    f = np.where(is_singleton_correct, 1.0, f)
    return float(f.mean())


def sets_from_gt_df(gt_df: pd.DataFrame) -> dict:
    out = {}
    for s1_id, ids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        out[s1_id] = set(ids.split(",")) if ids else set()
    return out
