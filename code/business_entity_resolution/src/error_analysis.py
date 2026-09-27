"""Validation error analysis: where exactly does the pipeline lose macro F0.5?

Called by train.py after the decision rule is tuned. Prints a breakdown and
writes example rows to <model-dir>/error_analysis/ so failures can be read
side by side (Source-1 name/address vs candidate name/address):

- points lost per failure type: how much of the macro F0.5 gap to 1.0 comes
  from false merges on true singletons, from entities where blocking found
  none of the true matches, from entities where every true match was
  rejected, and from partially right entities. This says which stage to
  work on next.
- pair-level misses split into blocking misses (never a candidate),
  classifier misses (candidate, but scored below the threshold) and rule
  misses (above the threshold, removed by exclusivity / per-source cap).
- false merges split by whether the wrongly matched record really belongs
  to another Source-1 entity (a look-alike) or to none.
- macro F0.5 per country and precision / recall per source.
"""
import os

import numpy as np
import pandas as pd

from decision import f05_per_entity

N_EXAMPLES = 3000


def _sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return df.sample(n=n, random_state=seed) if len(df) > n else df


def _with_text(df: pd.DataFrame, s1_lookup: pd.DataFrame, other_lookup: pd.DataFrame) -> pd.DataFrame:
    out = df.join(s1_lookup[["name", "addr", "country_"]].rename(
        columns={"name": "s1_name", "addr": "s1_addr", "country_": "country"}), on="source1_entity_id")
    out = out.join(other_lookup[["name", "addr"]].rename(
        columns={"name": "cand_name", "addr": "cand_addr"}), on="cand_id")
    return out


def report(out_dir, val_ids, val_country, n_true, blocked_pos,
           v_s1, v_cand_ids, v_s3, v_score, v_label, keep_rule, threshold,
           pos_rows: pd.DataFrame, gt_val_pairs: pd.DataFrame, cand_owner: dict,
           s1_lookup: pd.DataFrame, other_lookup: pd.DataFrame):
    """
    val_ids/val_country/n_true/blocked_pos: per validation S1 entity (aligned).
    v_*: scored validation pairs above the decision floor; keep_rule is the
      pre-threshold mask of the chosen rule (exclusivity / cap).
    pos_rows: every blocked true pair [source1_entity_id, cand_id, score].
    gt_val_pairs: every true pair of the validation entities [source1_entity_id, cand_id].
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(val_ids)
    sel = keep_rule & (v_score >= threshold)
    n_pred = np.bincount(v_s1[sel], minlength=n)
    tp = np.bincount(v_s1[sel], weights=v_label[sel], minlength=n).astype(np.int64)
    f = f05_per_entity(n_pred, tp, n_true)
    macro = f.mean() if n else 0.0

    print("=" * 78, flush=True)
    print(f"ERROR ANALYSIS (validation, {n} Source-1 entities, macro F0.5={macro:.4f})", flush=True)

    loss = 1.0 - f
    cats = {
        "false merge on a true singleton (scores 0)": (n_true == 0) & (n_pred > 0),
        "blocking found none of its true matches": (n_true > 0) & (blocked_pos == 0),
        "blocked, but every true match rejected": (n_true > 0) & (blocked_pos > 0) & (tp == 0),
        "partially right (some tp, some fp/fn)": (tp > 0) & (f < 1.0),
    }
    print("Macro-F0.5 points lost by failure type (sum = 1 - score):", flush=True)
    for name, m in cats.items():
        print(f"  {name:45s} entities={int(m.sum()):8d}  points lost={loss[m].sum() / max(n, 1):.4f}", flush=True)

    # ---- pair-level misses
    n_true_pairs = int(n_true.sum())
    n_blocked = int(blocked_pos.sum())
    tp_pairs = int(tp.sum())
    above = v_label.astype(bool) & (v_score >= threshold)
    rule_miss = int((above & ~keep_rule).sum())
    clf_miss = n_blocked - tp_pairs - rule_miss
    print(f"True pairs: {n_true_pairs}. Found: {tp_pairs} ({tp_pairs / max(n_true_pairs, 1):.2%}). Missed:", flush=True)
    print(f"  blocking never produced it : {n_true_pairs - n_blocked:8d} ({(n_true_pairs - n_blocked) / max(n_true_pairs, 1):.2%})",
          flush=True)
    print(f"  classifier scored < thr    : {clf_miss:8d} ({clf_miss / max(n_true_pairs, 1):.2%})", flush=True)
    print(f"  removed by excl./cap rule  : {rule_miss:8d} ({rule_miss / max(n_true_pairs, 1):.2%})", flush=True)

    # ---- false merges
    fp_mask = sel & (v_label == 0)
    fp_cands = v_cand_ids[fp_mask]
    fp_s1 = np.asarray(val_ids, dtype=object)[v_s1[fp_mask]]
    owner = np.array([cand_owner.get(c, "") for c in fp_cands], dtype=object)
    on_singleton = n_true[v_s1[fp_mask]] == 0
    has_owner = owner != ""
    n_fp = int(fp_mask.sum())
    print(f"False merges: {n_fp} (precision {tp_pairs / max(tp_pairs + n_fp, 1):.4f})", flush=True)
    print(f"  on true-singleton entities          : {int(on_singleton.sum()):8d}", flush=True)
    print(f"  record truly belongs to another S1  : {int(has_owner.sum()):8d}  (look-alike / template name)", flush=True)
    print(f"  record belongs to no S1 at all      : {int((~has_owner).sum()):8d}", flush=True)

    # ---- per country / per source
    print("Per country:", flush=True)
    for c in sorted(set(val_country)):
        m = val_country == c
        print(f"  {c:10s} entities={int(m.sum()):8d}  macro F0.5={f[m].mean():.4f}", flush=True)
    true_s3 = gt_val_pairs["cand_id"].str.startswith("S3").to_numpy()
    print("Per source:", flush=True)
    for name, is3 in (("Source 2", False), ("Source 3", True)):
        m = sel & (v_s3 == is3)
        tps = int(v_label[m].sum())
        nt = int((true_s3 == is3).sum())
        print(f"  {name}: precision={tps / max(int(m.sum()), 1):.4f} recall={tps / max(nt, 1):.4f} "
              f"(tp={tps} pred={int(m.sum())} true={nt})", flush=True)

    # ---- example files
    blocked_keys = set(pos_rows["source1_entity_id"] + "\x1f" + pos_rows["cand_id"])
    gk = gt_val_pairs["source1_entity_id"] + "\x1f" + gt_val_pairs["cand_id"]
    blk = gt_val_pairs[~gk.isin(blocked_keys).to_numpy()]
    _with_text(_sample(blk, N_EXAMPLES), s1_lookup, other_lookup).to_csv(
        os.path.join(out_dir, "blocking_misses.tsv"), sep="\t", index=False)

    tp_keys = set(pd.Series(np.asarray(val_ids, dtype=object)[v_s1[sel & (v_label == 1)]]) + "\x1f"
                  + pd.Series(v_cand_ids[sel & (v_label == 1)]))
    pk = pos_rows["source1_entity_id"] + "\x1f" + pos_rows["cand_id"]
    clf = pos_rows[~pk.isin(tp_keys).to_numpy()].sort_values("score", ascending=False)
    _with_text(_sample(clf, N_EXAMPLES), s1_lookup, other_lookup).to_csv(
        os.path.join(out_dir, "classifier_misses.tsv"), sep="\t", index=False)

    fp_df = pd.DataFrame({"source1_entity_id": fp_s1, "cand_id": fp_cands, "score": v_score[fp_mask],
                          "s1_is_singleton": on_singleton, "cand_true_owner": owner})
    fp_df = _with_text(_sample(fp_df.sort_values("score", ascending=False), N_EXAMPLES), s1_lookup, other_lookup)
    fp_df.to_csv(os.path.join(out_dir, "false_positives.tsv"), sep="\t", index=False)
    print(f"Example rows written to {out_dir}/ "
          f"(blocking_misses.tsv, classifier_misses.tsv, false_positives.tsv)", flush=True)
    print("=" * 78, flush=True)
