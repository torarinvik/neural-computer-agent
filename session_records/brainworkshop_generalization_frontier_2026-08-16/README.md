# Generalization frontier (2026-08-16)

Status: **development transfer diagnostic; not promoted**.

This run reuses the verified world-independent operator bundle on two fresh
target ring dynamics, with matched fresh, irrelevant, corrupted, and raw
source-world successor controls. The source and target transition tables are
different in both replicates; no curated artifact or reserved holdout was
used.

## Result

| arm | stable verifier bits (replicate 0 / 1) | target behavior |
| --- | --- | --- |
| reusable operator | 128 / 128 | normalized return 0.984--1.000 |
| fresh learner | 128 / 256 | normalized return 0.938--0.967 |
| irrelevant operator | 128 / 256 | same as fresh control |
| corrupted operator | never / never | normalized return 0.0 |
| raw source successor | never / never | normalized return 0.0--0.25 |

The reusable control-flow contract transferred across changed dynamics and
reached the stable threshold in 128 bits in both replicates. The raw successor
artifact did not transfer, which separates a reusable learning/operator
procedure from a stale world-specific model. The measured transfer ratio
against a fresh learner was 0.75.

## Boundary

This is evidence for world/dynamics transfer only. It does not yet establish
cross-frontend transfer, symbol remapping, larger action spaces, occlusion,
or belief-state integration. Those axes need fresh pixel rerenders and
separate controls; this record must not be read as broad generalization across
all modalities.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.operator_world_transfer \
  --replicates 2 --training-episodes 12 --evaluation-episodes 4 \
  --output session_records/brainworkshop_generalization_frontier_2026-08-16
```

The complete per-replicate report is in `operator_world_transfer.json`.
