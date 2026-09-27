"""Turn scored (Source-1, candidate) pairs into final match sets.

Shared by train.py (to tune the rule on validation) and predict.py (to apply
it), so the tuned rule is exactly the one that gets submitted.

Rule, in order:
1. Exclusivity (optional, enabled only when the training ground truth shows
   Source-2/3 records essentially never belong to more than one Source-1
   entity): a candidate record is kept only for the Source-1 entity that
   scores it highest. This removes the long tail of "one popular template
   name matched to dozens of unrelated Source-1 entities" false merges.
2. Per-source cap (optional): keep at most `max_per_source` candidates per
   (Source-1 entity, source), best score first — ground truth never exceeds
   a small number per source.
3. Probability threshold.

All pair arrays are integer-coded so this runs on tens of millions of rows
in seconds.
"""
import numpy as np

# Pairs scoring below this are never kept around for the decision step
# (they could not pass any threshold in the tuning grid). Exclusivity only
# competes among pairs above it, identically in validation and prediction.
MIN_SCORE_KEEP = 0.02


def exclusive_mask(cand_codes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """True for rows holding the (first) maximum score of their candidate."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-scores, cand_codes))
    first = np.r_[True, cand_codes[order][1:] != cand_codes[order][:-1]]
    mask = np.zeros(n, dtype=bool)
    mask[order[first]] = True
    return mask


def rank_within_group(group_codes: np.ndarray, scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """0-based rank by descending score within each group, counting only
    `valid` rows (invalid rows get a large rank)."""
    n = len(scores)
    rank = np.full(n, np.iinfo(np.int32).max, dtype=np.int64)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return rank
    g = group_codes[idx]
    order = np.lexsort((-scores[idx], g))
    gs = g[order]
    starts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(gs)])
    r = np.arange(len(gs)) - np.repeat(starts, counts)
    rank[idx[order]] = r
    return rank


def pre_threshold_mask(s1_codes, cand_codes, is_s3, scores, exclusive: bool, max_per_source):
    """Everything in the rule except the threshold itself (so a threshold
    sweep can reuse it)."""
    keep = np.ones(len(scores), dtype=bool)
    if exclusive:
        keep &= exclusive_mask(cand_codes, scores)
    if max_per_source:
        group = s1_codes.astype(np.int64) * 2 + is_s3.astype(np.int64)
        keep &= rank_within_group(group, scores, keep) < int(max_per_source)
    return keep


def select_matches(s1_codes, cand_codes, is_s3, scores, cfg: dict) -> np.ndarray:
    keep = pre_threshold_mask(s1_codes, cand_codes, is_s3, scores,
                              cfg.get("exclusive", False), cfg.get("max_per_source"))
    return keep & (scores >= cfg["threshold"])


def macro_f05_from_counts(n_pred: np.ndarray, tp: np.ndarray, n_true: np.ndarray, beta: float = 0.5) -> float:
    """Challenge macro F_beta from per-entity counts. `n_true` must be the
    ground-truth match count (NOT just the positives blocking found)."""
    b2 = beta * beta
    n_pred = n_pred.astype("float64")
    tp = tp.astype("float64")
    n_true = n_true.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tp / n_pred, 0.0)
        r = np.where(n_true > 0, tp / n_true, 0.0)
        d = b2 * p + r
        f = np.where(d > 0, (1 + b2) * p * r / d, 0.0)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)
    return float(f.mean()) if len(f) else 0.0
