# Independent external-caption corpus v2 — 2026-08-04

## Decision

This is a promoted three-seed extension of the pixel-grounded UTF-8 bridge.
The source is now a separately versioned, independently authored phrase
corpus. v1 exposed a real variance bottleneck: one seed failed the 1,024-life
replay on the hard diamond/style-4 cell. v2 adds two training paraphrase
variants per style, including clause-order variation, without changing the
controller, input bus, decoder, or byte frontend. All three v2 frontends pass
the 1,024-life saved-artifact gate.

This promotes controlled corpus-backed text transport, not open-world natural-
language understanding, speech, or semantic reasoning.

## Boundary and mechanism

The corpus is stored in
`experiments/natural_text_grounding/external_caption_corpus_v2.json`. A
separate caption source derives only visible colour, brightness, position, and
coarse geometry from pixels, then instantiates the authored phrases. The
learner receives padded UTF-8 bytes and no source descriptors, IDs, outcomes,
actions, task labels, or context metadata. The corpus SHA-256 is recorded in
every training report and saved frontend.

The only successful repair was data-side: two training variants per source
style. Paired-view consistency alone and extra optimizer updates were rejected
as weaker or overfitting interventions. The frozen neural-IR controller and
vision/input-bus checkpoints are identical to the promoted pixel-only bridge.

## Promotion replay

| seed | min fused | max shuffled | max contradictory | min flip | min vision |
|---|---:|---:|---:|---:|---:|
| 1002001 | 93.65% | 53.54% | 14.79% | 78.87% | 97.95% |
| 1002002 | 92.30% | 54.59% | 15.18% | 77.13% | 97.52% |
| 1002003 | 93.57% | 54.20% | 13.71% | 81.31% | 97.75% |

Each result is the minimum/maximum across bars, diamonds, and dot pairs at
held-out styles 3 and 4, with 1,024 lives per cell. The strict gate is fused
≥90%, shuffled ≤60%, contradictory ≤25%, contradiction flip ≥75%, full vision
≥95%, and unchanged controller parameters.

## Promotion record

The machine-checkable evidence is [promotion.json](promotion.json), with the
one-use local holdout lease in
[promotion_holdout_ledger.jsonl](promotion_holdout_ledger.jsonl). Verify it:

```sh
uv run python scripts/verify_promotion_record.py \
  session_records/natural_text_grounding_external_corpus_v2_2026-08-04/promotion.json \
  --holdout-ledger session_records/natural_text_grounding_external_corpus_v2_2026-08-04/promotion_holdout_ledger.jsonl
```

## Accounting

Each seed used 16 optimizer updates, 36,864 paired unlabeled frames, zero
verifier bits, zero logical lifetimes, and no replayed training examples.
Wall times were 135.97s, 133.23s, and 119.58s. Saved replay loaded no
optimizer state and left the controller unchanged.

Reports and manifests:

- [training summary](training_summary.json)
- [saved replay summary](saved_frontend_replay_summary.json)
- [sample-efficiency ledger](sample_efficiency_ledger.json)
- [development manifest](development_manifest.json)
- [promotion manifest](promotion_manifest.json)
- [promotion configuration](promotion_configuration.json)
- [seed 1002001 training report](training_seed1002001_u16.json)
- [seed 1002002 training report](training_seed1002002_u16.json)
- [seed 1002003 training report](training_seed1002003_u16.json)
- [seed 1002001 replay report](saved_replay_seed1002001_n1024.json)
- [seed 1002002 replay report](saved_replay_seed1002002_n1024.json)
- [seed 1002003 replay report](saved_replay_seed1002003_n1024.json)

## Next boundary

Keep v2 frozen and test a real independently produced caption stream with the
same corpus hash, metadata audit, causal controls, and 1,024-life replay. Do
not weaken the gate or add a language-specific reasoning branch.
