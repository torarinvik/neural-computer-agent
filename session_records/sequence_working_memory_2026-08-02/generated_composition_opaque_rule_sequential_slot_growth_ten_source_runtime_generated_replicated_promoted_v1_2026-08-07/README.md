# Strict isolated growth to ten opaque procedures (2026-08-07)

Status: promoted replicated bounded replay-free growth result.

The frozen controller acquires nine verifier-private, runtime-generated
eight-step opaque procedures one at a time. Each new procedure is trained in a
fresh external neural slot from fresh rendered outcomes, appended under an
opaque alias, and admitted only after every retained alias passes fresh
retention verification. A tenth generated procedure is then learned from the
first retained slot and admitted into a separate grown row.

| gate | seed 69316 | seed 69317 |
| --- | :---: | :---: |
| nine sequential fresh-slot additions | pass | pass |
| minimum source behavior after reload | 0.9141 | 0.8320 |
| target behavior after reload | 1.0000 | 1.0000 |
| all aliases share one physical row | pass | pass |
| reversal/recovery, reload, corruption | pass | pass |
| frozen controller unchanged | pass | pass |
| replayed examples | 0 | 0 |

The source family is sampled at audit time from the 256 verifier-private
three-cell local rules; rule tokens, truth tables, program IDs, and correct
actions remain outside the deployed learner. The source payload is
`3,019,104` bytes for nine isolated slots, so this result demonstrates
retention-safe capacity growth rather than neural compression. The target was
not a positive-transfer win: inherited and fresh target stable prefixes tied
at `2,048` verifier bits for seed 69316, while the inherited arm required
`4,096` versus `2,048` fresh bits for seed 69317.

The full accounting is identical in both replicas: `274,432` unique verifier
bits, `92,160` logical lifetimes, `2,944` optimizer updates, `124` retention
observations, and zero replayed examples. Wall time was `536.64s` and
`279.25s`; the audit explicitly used one PyTorch thread for this small model.
Retention probes are batched per alias but preserve independent fresh seeds;
the equivalence test is covered in the runtime-program test suite.

The short 16-update control is retained beside the promoted reports. It
rejected the first append at behavior `0.6250`, proving the hard mastery gate
still blocks under-trained growth.

This promotes ten-procedure bounded isolated external growth over an opaque
runtime-generated rule family. It does not establish arbitrary Turing-complete
program induction, learned compression, unrestricted memory growth, positive
transfer against a fresh learner, or general continual learning. The next
bottleneck is shared computation/capacity selection that can compress or
reuse old capability structure without replay or interference.

The promoted command was:

```text
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_sequential_slot_isolated_consolidation \
  --program-seed 4242 --primitive-family opaque_rule \
  --program-count 10 --program-depth 8 \
  --source-ids 0 1 2 3 4 5 6 7 8 --target-id 9 \
  --parent-updates 128 --source-updates 256 --target-updates 256 \
  --batch-size 16 --audit-count 64 --retention-probes 4 \
  --eval-every 32 --torch-threads 1
```
