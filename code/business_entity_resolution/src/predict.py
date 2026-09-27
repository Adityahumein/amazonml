"""Run the full pipeline (blocking -> features -> scoring -> decision) on a
dataset split and write output/candidate_pairs.tsv and output/matching_results.tsv.

Pass 1 processes Source-1 entities in bounded batches (see
pipeline.iter_scored_batches), writes each batch's candidate list straight
to candidate_pairs.tsv, and keeps only the compact, integer-coded pairs that
score above decision.MIN_SCORE_KEEP. Pass 2 applies the decision rule tuned
in train.py (threshold + optional exclusivity + per-source cap) over the
whole file at once — exclusivity must see every Source-1 entity competing
for the same Source-2/3 record, not just the ones in one batch.

Usage:
    python3 src/predict.py --data-dir ../../dataset/test --model-dir artifacts --out-dir ../../output
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from decision import MIN_SCORE_KEEP, select_matches
from pipeline import S1_BATCH_SIZE, Progress, fmt_duration, iter_scored_batches, load_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../dataset/test")
    ap.add_argument("--model-dir", default="artifacts")
    ap.add_argument("--out-dir", default="../../output")
    ap.add_argument("--source1-file", default="test_source1.tsv")
    ap.add_argument("--source2-file", default="test_source2.tsv")
    ap.add_argument("--source3-file", default="test_source3.tsv")
    ap.add_argument("--batch-size", type=int, default=S1_BATCH_SIZE,
                    help="Source-1 entities processed per batch (bounds peak memory).")
    ap.add_argument("--max-candidates-per-s1", type=int, default=200,
                    help="Cap (across both sources, best score first) on each entity's candidate_pairs.tsv list.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    s1 = load_source(os.path.join(args.data_dir, args.source1_file))
    s2 = load_source(os.path.join(args.data_dir, args.source2_file))
    s3 = load_source(os.path.join(args.data_dir, args.source3_file))
    print(f"[{time.time()-t0:.1f}s] loaded: s1={len(s1)} s2={len(s2)} s3={len(s3)}", flush=True)

    booster = lgb.Booster(model_file=os.path.join(args.model_dir, "model.txt"))
    with open(os.path.join(args.model_dir, "threshold.json")) as f:
        cfg = json.load(f)
    cfg.setdefault("exclusive", False)
    cfg.setdefault("max_per_source", None)
    floor = min(cfg.get("min_score_keep", MIN_SCORE_KEEP), cfg["threshold"])
    print(f"decision rule: {cfg}", flush=True)

    s1_ids = s1["entity_id"].to_numpy()
    s1_code = pd.Index(s1_ids)
    other_code = pd.Index(np.concatenate([s2["entity_id"].to_numpy(), s3["entity_id"].to_numpy()]))
    other_ids = other_code.to_numpy()

    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    match_path = os.path.join(args.out_dir, "matching_results.tsv")

    k_s1, k_cand, k_s3, k_score = [], [], [], []
    n_done = 0
    total_cand_rows = 0
    prog = Progress("scoring test", len(s1), t0)
    with open(cand_path, "w") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_batch, pairs in iter_scored_batches(s1, s2, s3, booster, batch_size=args.batch_size):
            pairs = pairs.sort_values(["source1_entity_id", "score"], ascending=[True, False])
            pairs["rank"] = pairs.groupby("source1_entity_id").cumcount()
            capped = pairs[pairs["rank"] < args.max_candidates_per_s1]
            cand_map = capped.groupby("source1_entity_id")["cand_id"].apply(list).to_dict()
            for s1_id in s1_batch["entity_id"]:
                fc.write(f"{s1_id}\t{','.join(cand_map.get(s1_id, []))}\n")

            kept = capped[capped["score"] >= floor]
            k_s1.append(s1_code.get_indexer(kept["source1_entity_id"]).astype(np.int64))
            k_cand.append(other_code.get_indexer(kept["cand_id"]).astype(np.int64))
            k_s3.append(kept["cand_id"].str.startswith("S3").to_numpy())
            k_score.append(kept["score"].to_numpy(dtype=np.float32))

            n_done += len(s1_batch)
            total_cand_rows += len(capped)
            prog.update(n_done, f"{total_cand_rows / n_done:.0f} cand/S1")

    prog.done()
    v_s1 = np.concatenate(k_s1) if k_s1 else np.zeros(0, np.int64)
    v_cand = np.concatenate(k_cand) if k_cand else np.zeros(0, np.int64)
    v_s3 = np.concatenate(k_s3) if k_s3 else np.zeros(0, bool)
    v_score = np.concatenate(k_score) if k_score else np.zeros(0, np.float32)
    sel = select_matches(v_s1, v_cand, v_s3, v_score, cfg)

    order = np.lexsort((-v_score[sel], v_s1[sel]))
    m_s1 = v_s1[sel][order]
    m_cand = other_ids[v_cand[sel][order]]
    starts = np.r_[0, np.flatnonzero(np.diff(m_s1)) + 1] if len(m_s1) else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], len(m_s1)] if len(m_s1) else np.zeros(0, np.int64)
    match_map = {int(m_s1[a]): ",".join(m_cand[a:b]) for a, b in zip(starts, ends)}

    with open(match_path, "w") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for i, s1_id in enumerate(s1_ids):
            fm.write(f"{s1_id}\t{match_map.get(i, '')}\n")

    counts = np.bincount(m_s1, minlength=len(s1_ids)) if len(s1_ids) else np.zeros(0)
    print(f"[total {fmt_duration(time.time()-t0)}] wrote output files to {args.out_dir}: {int(sel.sum())} matches, "
          f"{int((counts == 0).sum())} entities with no match, max matches/entity {int(counts.max()) if len(counts) else 0}",
          flush=True)


if __name__ == "__main__":
    main()
