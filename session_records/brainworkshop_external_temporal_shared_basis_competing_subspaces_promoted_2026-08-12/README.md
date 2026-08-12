# Competing-subspace shared-structure growth — promoted narrow boundary

Status: `PROMOTED` as a dynamic-rank external structure-policy transfer
result.

This rung trains the v2 `OpaqueSharedBasisStructurePolicy` on runtime
candidate ranks `(2, 4, 8)`. The policy sees only opaque value rows,
occupancy, and candidate ranks; it computes the spectral plus normalized
pairwise structure summary internally. It receives one scalar verifier utility
per fresh bank and never receives precomputed candidate reconstruction error,
semantic labels, task IDs, or correct actions.

The frozen canonical memory receives four rank-two cohorts in four distinct
orthogonal subspaces. Their union grows from rank 2 to rank 4 to rank 6 to rank
8, so the safe choices are:

`2 → 4 → 8 → 8`

The stream is evaluated in both subspace-arrival orders and both physical row
orders. Every candidate is independently route/value verified, version
checked, and committed copy-on-write. The controller and event encoder remain
frozen; replay is zero.

| seed | held-out rank 2/4/8 | fresh rank 2/4/8 | all stream choices | final physical/dense value scalars |
| ---: | --- | --- | --- | ---: |
| 17 | `0.9688/1.0000/1.0000` | `0.9688/0.0000/0.0000` | `2 → 4 → 8 → 8` | `512/768` |
| 18 | `0.9688/1.0000/1.0000` | `0.9688/0.0000/0.0000` | `2 → 4 → 8 → 8` | `512/768` |

Both seeds passed all sixteen verifier-gated transactions, complete-prefix
retention, subspace-order reversal, physical-order reversal, exact reload,
stale-version rejection, checksum corruption rejection, frozen-core, and zero
replay. The 100-update calibration is retained as rejected evidence: live
safety passed, but held-out rank-2/4 transfer and dynamic choices had not yet
stabilized.

This promotes bounded dynamic-rank structure selection under competing
subspaces. It does not establish semantic structure discovery, arbitrary new
computation, unrestricted memory growth, or general continual learning. The
next pressure is genuinely changing subspaces with removal/replacement or
reversal, capacity pressure, and lower scalar-feedback cost.

Reports and accounting:

- `seed-17.json`
- `seed-18.json`
- `rejected_calibration.json`
- `sample_efficiency_ledger.json`
