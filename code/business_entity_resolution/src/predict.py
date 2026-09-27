"""Run the full pipeline (blocking -> features -> scoring) on a dataset split
and write output/candidate_pairs.tsv and output/matching_results.tsv.

Processes Source-1 entities in bounded batches (see pipeline.iter_scored_batches)
and writes each batch's rows to the output files immediately, so memory use
stays roughly constant regardless of how large test_source1.tsv is.

Usage:
    python3 src/predict.py --data-dir ../../dataset/test --model-dir artifacts --out-dir ../../output
"""
import argparse
import json
import os
import time

import lightgbm as lgb

from pipeline import S1_BATCH_SIZE, iter_scored_batches, load_source


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
    ap.add_argument("--max-candidates-per-s1", type=int, default=500,
                     help="Safety cap (across both sources) on the final candidate_pairs.tsv size, "
                          "applied after model scoring within each batch.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    s1 = load_source(os.path.join(args.data_dir, args.source1_file))
    s2 = load_source(os.path.join(args.data_dir, args.source2_file))
    s3 = load_source(os.path.join(args.data_dir, args.source3_file))
    print(f"[{time.time()-t0:.1f}s] loaded: s1={len(s1)} s2={len(s2)} s3={len(s3)}", flush=True)

    booster = lgb.Booster(model_file=os.path.join(args.model_dir, "model.txt"))
    with open(os.path.join(args.model_dir, "threshold.json")) as f:
        threshold = json.load(f)["threshold"]

    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    match_path = os.path.join(args.out_dir, "matching_results.tsv")

    n_done = 0
    total_cand_rows = 0
    with open(cand_path, "w") as fc, open(match_path, "w") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")

        for s1_batch, pairs in iter_scored_batches(s1, s2, s3, booster, batch_size=args.batch_size):
            pairs = pairs.sort_values(["source1_entity_id", "score"], ascending=[True, False])
            pairs["rank"] = pairs.groupby("source1_entity_id").cumcount()
            capped = pairs[pairs["rank"] < args.max_candidates_per_s1]

            cand_map = capped.groupby("source1_entity_id")["cand_id"].apply(list).to_dict()
            match_map = (capped[capped["score"] >= threshold]
                         .groupby("source1_entity_id")["cand_id"].apply(list).to_dict())

            for s1_id in s1_batch["entity_id"]:
                fc.write(f"{s1_id}\t{','.join(cand_map.get(s1_id, []))}\n")
                fm.write(f"{s1_id}\t{','.join(match_map.get(s1_id, []))}\n")

            n_done += len(s1_batch)
            total_cand_rows += len(capped)
            print(f"[{time.time()-t0:.1f}s] {n_done}/{len(s1)} S1 entities "
                  f"({total_cand_rows} candidate rows so far)", flush=True)

    print(f"[{time.time()-t0:.1f}s] wrote output files to {args.out_dir}")


if __name__ == "__main__":
    main()
