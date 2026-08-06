# Multi-step external view growth

This pressure test follows the promoted one-failure fifth-view result. It
acquires two new executable views in sequence while keeping the controller and
the original four-view router frozen. The first new view is tried after the
old route fails; the second is tried only after the first new view also fails.

Each new route extension trains only from fresh scalar outcomes for its own
procedure. The prior extension is frozen and its examples are not replayed.
Both additions are compacted into the same physical artifact row, then
reloaded and tested with permutation, causal wrong-view, and corruption
controls.

Passing this audit qualifies a bounded two-step external fallback chain. It
does not establish arbitrary open-ended task discovery or unrestricted
continual learning.

## Retention-safe v2

The promoted v2 audit adds the retention ledger to both transactions. Four old
views are probed and protected before the first addition. The `rotate`
candidate is built and probed in a disposable store, then protected before the
`complement_rotate` candidate is attempted. Each candidate must clear a `0.70`
stable prefix floor, and an independent behavior verifier must preserve every
previous capability within `0.05` of its baseline.

The matched promoted command is:

```bash
python -m experiments.multistep_view_growth_amodal.train \
  --report-out /tmp/multistep-retention-512-69316.json \
  --seed 69316 --updates 512 --extension-artifact-updates 512 \
  --route-updates 2048 --extension-updates 512 --batch-size 16 \
  --route-batch-size 16 --audit-count 64 --retention-probes 8 \
  --retention-threshold 0.70 --behavior-tolerance 0.05
```

Seed `69317` uses the same budgets. Both seeds retained six opaque views in
one physical row, kept the controller and old router frozen, passed route,
permutation, causal, reward-shuffle, reload, corruption, and zero-replay
controls, and protected the intermediate state before the second extension.
Evidence is archived under
`session_records/sequence_working_memory_2026-08-02/multistep_view_growth_retention_v2_2026-08-06/`.
This is still bounded two-step growth, not open-ended skill discovery,
unrestricted memory growth, or general continual learning.
