# Business Entity Resolution — Pipeline

Blocking + pairwise LightGBM classifier for matching Source-1 (reference)
business records against Source-2 / Source-3 records across US, India and
(test-only) France.

Previous version: leaderboard macro F_0.5 = 0.842. (Its reported validation
score of 0.941 was inflated: validation recall only counted true matches that
blocking had found. `train.py` now scores validation against the full ground
truth, so its number should track the leaderboard.) See
`../../Documentation_template.md` for the full methodology write-up.

### What changed in v2

1. **Validation metric fixed.** Recall now uses each entity's ground-truth
   match count, so matches missed by blocking count as misses. Thresholds
   are tuned against the real objective.
2. **Higher-recall blocking.** Keys are hashed to int64 and rows are integer
   positions, which makes the tables several times smaller. New keys:
   concatenated-name (`@HENDERSONFORTRESS`), sorted-letter anagram
   (character scrambles), per-token name keys, and address combo keys
   (number+token, token+token, name-prefix+number). The combo keys recover
   the matches whose single keys were too common and got dropped by the
   posting cap.
3. **More features (44 vs 24).** Concatenated/sorted-char name similarity,
   Jaro-Winkler, partial address ratio, house-number/postal-code agreement,
   unmatched-token counts, source flag, and free blocking-context features
   (best shared-key specificity, rank within the entity, score relative to
   the entity's best candidate).
4. **Hard-negative mining.** Training keeps the top negatives by blocking
   score plus random ones.
5. **Tuned decision rule** (`src/decision.py`). The threshold search is finer
   near 1. Two optional rules are chosen on validation: *exclusivity* (a
   Source-2/3 record goes only to the Source-1 entity that scores it
   highest; allowed only if ground truth shows records are almost never
   shared) and a *per-source cap* (the ground-truth max). `predict.py`
   applies the rule over the whole test set in a second pass.
6. **French and Indian normalization.** French legal forms (SARL, SAS, ...),
   French street types (rue, bd, chemin, ...) and Indian address terms
   (marg, nagar, sector, ...). France appears only in the test set.

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
#    v1 took ~2-2.5 h on the full ~2.2M-entity training set. v2 features are
#    ~2x faster per pair (each record normalized once, cached) and validation
#    uses cascade scoring, so expect similar or less; the live ETA printed per
#    phase gives the real number for your machine. Add --s1-sample 0.1 for a
#    ~10x faster experiment run.
python3 src/train.py --train-dir ../../dataset/train --model-dir artifacts

# 2. Score dataset/test and write output/candidate_pairs.tsv +
#    output/matching_results.tsv. v1 took ~4.5 h on the full ~1.7M-entity
#    test set; v2's cascade scoring (~25x faster model evaluation) and cached
#    normalization (~2x faster features) should cut this substantially.
#    Progress and ETA are printed per batch.
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

## Improving the score: the error-analysis loop

Every `train.py` run ends with an **error analysis** of the validation split
(which has labels) and writes example rows to `artifacts/error_analysis/`:

```
Macro-F0.5 points lost by failure type (sum = 1 - score):
  false merge on a true singleton (scores 0)     entities=...  points lost=0.0xx
  blocking found none of its true matches        entities=...  points lost=0.0xx
  blocked, but every true match rejected         entities=...  points lost=0.0xx
  partially right (some tp, some fp/fn)          entities=...  points lost=0.0xx
True pairs: ... Missed: blocking / classifier / exclusivity-cap rule
False merges: on singletons / record belongs to another S1 / belongs to none
Per country: macro F0.5 ...     Per source: precision / recall ...
```

Work on whichever line loses the most points:

| Biggest loss | Look at | Typical fix |
|---|---|---|
| blocking found none / blocking misses | `blocking_misses.tsv` | new key type in `blocking.record_keys` for the pattern you see |
| every true match rejected / classifier misses | `classifier_misses.tsv` | a feature in `features.py` capturing the pattern |
| false merges (look-alike) | `false_positives.tsv` (`cand_true_owner` set) | a feature that separates the two look-alikes; exclusivity already helps |
| false merges on singletons | `false_positives.tsv` (`s1_is_singleton`) | stricter features on house number / postcode |

For fast iterations, train on a sample of Source-1 (Source-2/3 stay full,
so blocking behaves as at full scale):

```bash
python3 src/train.py --train-dir ../../dataset/train --model-dir artifacts_exp --s1-sample 0.1
```

Compare the printed validation macro F0.5 between runs, and keep a change
only if it goes up. Then do one full run (`--s1-sample 1.0`) for the
submission model.

Both scripts print progress with an ETA for every phase, e.g.
`[total 0h41m10s] phase 1/3 blocking + training features: 12/45 (26.7%) | phase 0h38m02s | ETA 1h44m31s`.

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
  error_analysis.py       # validation failure breakdown + example rows (run by train.py)
  decision.py             # threshold + exclusivity + per-source cap (tuned in train, applied in predict)
  scoring.py              # macro F_0.5 exactly as defined by the challenge (vectorized)
  train.py                # end-to-end training + threshold tuning entry point
  predict.py              # end-to-end inference entry point
artifacts/                # model.txt (LightGBM), threshold.json (full decision rule), feature_columns.json
                          # NOTE: the committed artifacts are from v1 and must be regenerated with train.py
```

## Model

LightGBM (`lightgbm==4.7.0`, MIT license) gradient-boosted binary classifier
over 44 pairwise similarity / blocking-context features. Well under the 8B
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
