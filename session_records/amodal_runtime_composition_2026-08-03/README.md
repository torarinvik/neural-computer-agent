# Complete amodal runtime composition audit (2026-08-03)

## Question

Does the new `AmodalControllerRuntime` preserve the already-qualified
complementary-input capability while actually running through the target
boundary:

```text
two independently registered encoders -> one frozen controller -> decoder bus
```

The controller and the previously trained generic input bus were loaded from
real promoted artifacts. Two independent copies of the external vision
frontend were registered under different process-local handles. The handles
select adapters only; no stream name or task metadata is added to the event
payload.

## Results

The 4,096-lifetime MPS audit and an independent 512-lifetime MPS replica passed
all three appearance gates:

| audit | bars fused | diamonds fused | dot-pairs fused |
|---|---:|---:|---:|
| 4,096 | 96.57% | 91.13% | 95.56% |
| replica 512 | 97.03% | 90.27% | 95.66% |

The individual streams stayed near chance (roughly 44–56%), shuffled partners
stayed near chance (49.5–52.9%), and contradictory partners flipped the
prediction 73.2–92.1% of the time. N=1 and duplicate-event controls passed.

Most importantly, every wrapper result was exactly equal to the old explicit
`bus -> ExtractedAmodalRuntime.step_intention_event -> decoder` path:

- wrapper/legacy action-logit difference: **0.0** for every appearance and
  control;
- wrapper/legacy intention and recurrent-state parity: exact in the paired
  execution;
- controller parameters remained unchanged;
- two decoder-output registration and stream permutation remained runtime
  properties, not controller branches.

This is the first behavioral audit of the complete N-to-M runtime wrapper, not
merely a unit test of its plumbing. It establishes that the architecture
boundary can carry an existing causal two-stream skill without changing its
behavior.

## Loader repair discovered by the audit

The promoted amodal checkpoint predated a zero-initialized optional
`skill_adapter_critic_scales.*` parameter. The strict loader therefore rejected
it before the new wrapper could run. The compatibility loader now permits only
those known absent zero-initialized keys and continues to reject every other
missing or unexpected checkpoint key. A regression test covers this narrow
compatibility rule.

## Honest boundary

This does not yet prove that arbitrary audio/language encoders can be plugged
in and learn transfer, or that a cold-start controller can discover cross-modal
relevance. The next experiment is a second synthetic encoder with a different
latent basis, trained or calibrated through the wrapper while the controller
is frozen, followed by modality dropout, temporal shuffle, and sample-efficiency
comparisons against a fresh controller.

## Artifacts

- `runtime_wrapper_audit_4096_mps.json`
- `runtime_wrapper_audit_512_replica_mps.json`
- controller:
  `artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt`
- input bus:
  `artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt`
