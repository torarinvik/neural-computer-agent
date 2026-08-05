# Static pre-authored annotation table v3 — 2026-08-04

## Decision

Promoted. Three independently trained byte frontends align a static,
pre-authored caption table into the same frozen amodal neural-IR basis used by
the earlier pixel-only and phrase-corpus bridges.

This is a stronger data boundary than v2: the corpus stores complete sentences
and contains no runtime format slots. The source receives rendered pixels only
to join each image to its annotation row. The controller remains frozen and
receives only the resulting opaque event tensor. This is controlled synthetic
visible-scene caption transport, not open-world language understanding,
speech, or semantic reasoning.

## Population replay

Each row below is the worst/best value across bars, diamonds, and dot pairs at
held-out styles 3 and 4, with 1,024 lives per cell:

| seed | min fused | max shuffled | max contradictory | min flip | min vision |
|---|---:|---:|---:|---:|---:|
| 1003001 | 92.83% | 54.51% | 15.57% | 77.77% | 97.60% |
| 1003002 | 91.39% | 54.63% | 16.27% | 75.43% | 97.54% |
| 1003003 | 92.15% | 53.89% | 16.46% | 75.68% | 98.05% |

The strict gate is fused ≥90%, shuffled ≤60%, contradictory ≤25%,
contradiction flip ≥75%, full vision ≥95%, unchanged controller parameters,
and optimizer-free saved-frontend replay. All three seeds pass.

## What was rejected

The four-update probe showed no causal signal. The first full-sentence table
used avoidable synonym variation and failed one seed on hard diamond cells.
Diamond replay, paired text views, and a 32-update horizon did not repair that
failure. Keeping the core and frontend fixed, the final table stabilized the
content vocabulary and used a compact held-out paraphrase; the resulting
25-update configuration passed both training and population replay.

## Accounting

- 0 unique verifier bits
- 0 unique logical lifetimes during adapter training
- 25 optimizer updates per seed
- 57,600 paired unlabeled frames per seed
- 0 replayed training examples
- wall time: 119.19s, 164.71s, and 118.87s
- promotion replay: 1,024 lives per appearance/style cell
- latency and fresh-learner transfer ratio: not measured in this frozen-core
  representation qualification

## Machine-checkable evidence

Verify the one-use holdout claim and promotion decision with:

```sh
uv run python scripts/verify_promotion_record.py \
  session_records/natural_text_grounding_external_annotation_table_v3_2026-08-04/promotion.json \
  --holdout-ledger session_records/natural_text_grounding_external_annotation_table_v3_2026-08-04/promotion_holdout_ledger.jsonl
```

The exact annotation-table hash, source hashes, training reports, replay
reports, curated frontend hashes, and rejected-variant history are recorded in
the neighboring JSON files.
