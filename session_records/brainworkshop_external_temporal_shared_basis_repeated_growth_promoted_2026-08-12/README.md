# Repeated raw-value shared-structure growth — promoted narrow boundary

Status: `PROMOTED` as a repeated external structure-policy transfer result.

The v2 `OpaqueSharedBasisStructurePolicy` consumes opaque value rows and an
occupancy mask. Internally it uses a fixed-width singular-spectrum summary plus
row-permutation-invariant normalized pairwise statistics. It learns from one
scalar verifier utility per fresh bank and never receives precomputed
candidate reconstruction error, semantic labels, task IDs, or correct
actions. Unoccupied padding is ignored by the feature contract.

The canonical live stream has four cohorts of six records:

`rank 2 → rank 4 → rank 4 → rank 4`

Each proposal is copy-on-write and must pass the independent route/value
retention verifier and expected-version check before persistence. The stream
is run in forward and reversed physical insertion order. The controller and
event encoder remain frozen and no live record is replayed.

| seed | held-out rank 1/2/4 | fresh rank 1/2/4 | live choices | final physical/dense value scalars |
| ---: | --- | --- | --- | ---: |
| 17 | `0.8594/0.9844/1.0000` | `0.6094/0.0000/0.0000` | `2 → 4 → 4 → 4` | `160/384` |
| 18 | `0.9219/1.0000/1.0000` | `0.5938/0.0000/0.0000` | `2 → 4 → 4 → 4` | `160/384` |

Both seeds passed every held-out, fresh-control, per-stage acceptance, prefix
retention, reversed-order, reload, stale-version, corruption, frozen-core,
and zero-replay gate. All 24 routes remained readable after all three
successor transitions.

The v2 representation reduces the previous 50,000-update requirement to
20,000 updates per seed on this rung. The 3,000-update calibration passed all
live memory safety gates but was rejected for insufficient held-out transfer;
that evidence is archived rather than hidden.

This promotes repeated bounded structure-policy transfer, not semantic
structure discovery, arbitrary new computation, unrestricted memory growth,
regime reversal, or general continual learning. The next pressure is changing
or competing subspaces, genuine reversal/regime-shift controls, and lower
scalar-feedback cost.

Reports and accounting:

- `seed-17.json`
- `seed-18.json`
- `rejected_calibration.json`
- `sample_efficiency_ledger.json`
