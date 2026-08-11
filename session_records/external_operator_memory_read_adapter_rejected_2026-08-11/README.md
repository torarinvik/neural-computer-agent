# External operator file-read adapter rung — rejected

Date: 2026-08-11

## Question

Does a frozen, independently persisted external operator file become causally
necessary when its file values are read once from a bound route and supplied
through a replaceable memory-side intention adapter?

## Configuration

This was the matched deeper rendered-event diagnostic with seed `69316`:

- parent updates `32`, source updates `128`, operator calibration updates `256`;
- target warmup/focus/target updates `16/16/64`;
- span `3`, batch `16`, two operator slots, rank `4`;
- protected bounded-meta operator, counterfactual route credit, and bind-once
  routing;
- independent v2 `slot_values` file tokens and a trainable then frozen
  `EpisodicIntentAdapter`;
- zero replay, paired zero-read controls, and exact reload checks.

## Result

Calibration was accepted, the v2 external file round-tripped with exact
reload, and the adapter was present and frozen for target acquisition. The
file was not causally necessary:

| target | normal | zero-read | drop | gate |
| --- | ---: | ---: | ---: | --- |
| `reverse → adjacent_xor → complement` | 0.78125 | 0.78125 | 0.00 pp | fail |
| `reverse → complement → adjacent_xor` | 0.75000 | 0.75000 | 0.00 pp | fail |

No target was accepted and no transfer promotion was claimed. Memory reload
was exact, while shuffled-outcome and missing-evidence controls stayed at or
near chance. The external file ABI and read path are therefore retained as
infrastructure, not as a learned capability result.

Accounting: `239,872` unique verifier bits, `2,144` optimizer updates,
`36,992` logical lifetimes, and `0` replayed examples. The controller remained
frozen; the read adapter was trained during calibration and frozen before the
target rung.

## Interpretation

This rejects the current target-use design, not independent persistence or
bind-once routing. The target learner can still solve the pressure test through
the source/interpreter path without using the file token. The next experiment
must make the held-out computation unavailable except through the external
file-read path, then repeat the read-ablation and fresh-learner transfer gates.
Increasing operator-bank capacity before that test would not address the
bypass.

## Reproduction

```bash
.venv/bin/python -m experiments.external_register_composition_amodal.audit_interleaved_basis_acquisition \
  --report-out /tmp/operator-read-rung3-v2.json \
  --seed 69316 --parent-updates 32 --source-updates 128 \
  --sequence-calibration-updates 256 --warmup-updates 16 --focus-updates 16 \
  --target-updates 64 --batch-size 16 --span 3 --warmup-span 2 \
  --composition-program-count 2 --target-composition-start 0 \
  --operator-mode factorized_protected_bounded_meta --operator-rank 4 \
  --use-operator-sequence-memory --use-operator-sequence-router \
  --use-route-outcome-credit --bind-operator-route \
  --use-operator-read-adapter
```
