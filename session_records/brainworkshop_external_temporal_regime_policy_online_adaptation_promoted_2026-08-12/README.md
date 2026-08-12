# Replay-free online partial-overlap adaptation — promoted boundary

Status: `PROMOTED` as a bounded anti-forgetting result.

The detector is first trained on stable versus fully shifted opaque banks. A
zero-initialized `GatedResidualRegimeChangePolicy` then learns online from 144
fresh scalar verifier utilities across non-periodic stable intervals,
partially overlapping shifts, and disjoint shifts. The original detector is
frozen. Deterministic inference falls back to the frozen detector unless the
new residual has positive, stronger evidence, so the new capability is
isolated in external trainable state.

| seed | before stable / partial / disjoint | after stable / partial / disjoint | exact stable / shift after | fresh partial |
| ---: | --- | --- | --- | --- |
| 17 | `1.0000/0.0156/1.0000` | `1.0000/0.8203/1.0000` | `1.0000/1.0000` | `0.0000` |
| 18 | `1.0000/0.0156/1.0000` | `1.0000/0.8906/1.0000` | `1.0000/1.0000` | `0.0000` |

The stream achieved `0.7083` and `0.7153` mean utility with zero replayed
examples. Both seeds passed stable and disjoint retention, partial-overlap
acquisition, exact old-boundary retention, frozen controller/encoder, and
zero-replay gates.

The rejected `rejected_naive_single_policy_seed-17.json` is an important
negative control: directly updating one shared detector reached perfect
partial replacement but collapsed stable keep from `1.0000` to `0.0000`.
This supports parameter-isolated, gated external growth over unconstrained
in-place online updates.

This promotes one bounded residual-growth mechanism. It does not establish
unrestricted adapter growth, autonomous slot routing for arbitrary skills,
arbitrary new computation, or general continual learning.

Reports and accounting:

- `seed-17.json`
- `seed-18.json`
- `rejected_naive_single_policy_seed-17.json`
- `sample_efficiency_ledger.json`
