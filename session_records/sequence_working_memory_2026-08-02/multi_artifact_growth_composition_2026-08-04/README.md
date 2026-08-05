# Multi-artifact growth composition — 2026-08-04

This is the first end-to-end execution audit after top-k artifact promotion.
Two independently acquired, same-schema working-memory growth artifacts were
promoted from one opaque candidate query, remapped into separate generic
controller growth slots, and executed by one frozen controller.

The result replicated across seeds:

- `complement`: parent `34.6%`, isolated factor `70.2%`, composed `70.2%`
  (seed 69001); replica `35.0% -> 71.0% -> 70.8%`.
- `complement_reverse`: parent `34.7%`, isolated factor `59.3%`, composed
  `59.3%`; replica `34.6% -> 58.9% -> 58.9%`.
- Top-k promotion returned both verified rows with scores above the memory
  read threshold.
- Loading both artifacts changed 14 growth entries in disjoint namespaces;
  the frozen controller core digest was unchanged.
- A blank two-slot successor was bit-exact with the parent, and zeroing either
  factor removed only that factor's causal behavior.

This promotes a narrow mechanism claim: multiple verified opaque artifacts can
be composed as independently replaceable growth state and retain both learned
procedures in one frozen controller. It does **not** claim arbitrary program
synthesis, sequential factor algebra, unrestricted composition of unrelated
artifacts, or general cognition.

## Reproduction

```bash
uv run python -m experiments.working_memory_continuous.compose_acquired_growth \
  --parent artifacts/checkpoints/span8_addressed_parent_scale1_seed32001.pt \
  --artifacts \
    session_records/sequence_working_memory_2026-08-02/frozen_growth_span10_complement_512_2026-08-04/memory \
    session_records/sequence_working_memory_2026-08-02/frozen_growth_span10_complement_reverse_512_2026-08-04/memory \
  --bank session_records/sequence_working_memory_2026-08-02/multi_artifact_growth_composition_2026-08-04/bank \
  --report session_records/sequence_working_memory_2026-08-02/multi_artifact_growth_composition_2026-08-04/report.json
```

The independent replication is in
`multi_artifact_growth_composition_replication_2026-08-04/`.
