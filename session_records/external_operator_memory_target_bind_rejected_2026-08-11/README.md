# External operator memory target-use rung — rejected

Date: 2026-08-11

## Question

Does a frozen, independently persisted external operator bank become causally
necessary for fresh target acquisition when its contextual route is bound once
per rollout and then reused through fixed recurrent execution?

## Configuration

This was a one-seed deeper rendered-event diagnostic (`69316`) using the
outcome-only interleaved acquisition harness:

- source acquisition: 32 parent updates and 128 sequential source updates;
- external operator calibration: 256 updates, rank 4, two append-only slots;
- target acquisition: 16 warmup, 16 focus, and 64 target updates;
- span 3, batch 16, protected bounded-meta operator, counterfactual scalar
  outcome routing, and bind-once routing enabled;
- no replayed examples; fresh transfer received a matched 256-update external
  calibration budget.

## Result

The calibration boundary was accepted and the external bank round-tripped with
an exact checksum and exact paired reload accuracy. The target path still did
not depend strongly enough on the external file to pass the causal gate:

| target | normal | paired zeroed bank | drop | causal gate |
| --- | ---: | ---: | ---: | --- |
| `reverse → adjacent_xor → complement` | 0.78125 | 0.765625 | 1.56 pp | fail |
| `reverse → complement → adjacent_xor` | 0.72917 | 0.72396 | 0.52 pp | fail |

Both memory reload checks passed, and shuffled-outcome and missing-evidence
controls remained at or below chance. No target was accepted and no transfer
promotion was claimed. Accounting was 239,488 unique verifier bits, 2,144
optimizer updates, 36,928 logical lifetimes, and zero replayed examples.

## Interpretation

This rejects the current target-use boundary as a learned memory-use result,
not the bind-once ABI or persistence mechanism. The file is durable and the
route is executed once, but the fresh target learner can mostly solve the
pressure test without relying on the calibrated external operator. The next
high-ROI intervention is to make external-file consumption an independently
measurable learned pathway—such as an explicit file-read/trace interface and
matched read-ablated training control—before scaling the operator bank or
claiming continual-learning gains.

## Reproduction

```bash
.venv/bin/python -m experiments.external_register_composition_amodal.audit_interleaved_basis_acquisition \
  --report-out /tmp/operator-memory-bound-rung3-paired-v2.json \
  --seed 69316 --parent-updates 32 --source-updates 128 \
  --sequence-calibration-updates 256 --warmup-updates 16 --focus-updates 16 \
  --target-updates 64 --batch-size 16 --span 3 --warmup-span 2 \
  --composition-program-count 2 --target-composition-start 0 \
  --operator-mode factorized_protected_bounded_meta --operator-rank 4 \
  --use-operator-sequence-memory --use-operator-sequence-router \
  --use-route-outcome-credit --bind-operator-route
```
