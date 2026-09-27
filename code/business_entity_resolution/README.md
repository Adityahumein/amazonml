# Business Entity Resolution — Pipeline

Blocking + pairwise LightGBM classifier for matching Source-1 (reference)
business records against Source-2 / Source-3 records across US, India and
(test-only) France.

Validated macro F_0.5 = **0.9410** (97.3% precision / 90.2% recall) on a
held-out, entity-grouped 15% split of the training data. See
`../../Documentation_template.md` for the full methodology write-up.

## Setup

```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce end-to-end

From `code/business_entity_resolution/`:

```bash
# 1. Train the pairwise classifier on dataset/train (also prints blocking
#    recall ceiling, candidate-set size, and validation macro F0.5).
#    On the full ~2.2M-entity training set this takes roughly 2-2.5 hours
#    on a modern multi-core machine; see "Scale & memory" below.
python3 src/train.py --train-dir ../../dataset/train --model-dir artifacts

# 2. Score dataset/test and write output/candidate_pairs.tsv +
#    output/matching_results.tsv. On the full ~1.7M-entity test set this
#    takes roughly 4.5 hours.
python3 src/predict.py --data-dir ../../dataset/test --model-dir artifacts --out-dir ../../output

# 3. Validate the submission format
cd ../..
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

Both `train.py` and `predict.py` accept `--batch-size` (default 50,000
Source-1 entities) to trade memory for speed — see below.

## Structure

```
src/
  normalize.py          # name/address text normalization, Devanagari romanization, soundex
  us_in_states.py        # static US / India state abbreviation <-> full-name tables
  blocking.py            # multi-key candidate generation + free IDF-weighted pruning,
                          # country-partitioned + entity-batched for bounded memory
  features.py             # pairwise similarity feature engineering (float32 output)
  parallel_features.py    # multiprocessing wrapper for feature computation at scale
  pipeline.py             # batched, memory-bounded blocking -> features -> scoring
  scoring.py              # macro F_0.5 exactly as defined by the challenge (vectorized)
  train.py                # end-to-end training + threshold tuning entry point
  predict.py              # end-to-end inference entry point
artifacts/                # model.txt (LightGBM), threshold.json, feature_columns.json
```

## Model

LightGBM (`lightgbm==4.7.0`, MIT license) gradient-boosted binary classifier
over 23 hand-engineered pairwise similarity features. Well under the 8B
parameter cap.

## Scale & memory

At the candidate-set sizes this pipeline targets (up to 500 candidates per
Source-1 entity across both sources, hundreds of millions of pairs for the
full training/test files), holding the full candidate table — even just the
ID columns — in memory at once is tens of GB. Every stage instead processes
Source-1 entities in bounded batches (`--batch-size`, `pipeline.py`), reusing
the (expensive-to-build) Source-2/Source-3 blocking-key tables across
batches and discarding each batch's intermediate data before the next one
starts. `blocking.py`'s join additionally runs one process per country and,
within a country, in further sub-batches (`MERGE_BATCH_ENTITIES`) — this is
what fixes the actual failure mode we hit during development: materializing
one large country's *entire* raw pre-prune join at once drove system swap to
near-exhaustion.

Multiprocessing worker counts (`blocking.N_JOBS`, `parallel_features.N_JOBS`)
default to `cpu_count() - 1`. Lower `--batch-size` or those constants if
running on a memory-constrained machine.

See `../../Documentation_template.md` for the full methodology write-up.
