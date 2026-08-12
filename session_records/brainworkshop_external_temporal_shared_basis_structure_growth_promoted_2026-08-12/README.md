# Raw-value shared structure policy growth — promoted narrow boundary

Status: `PROMOTED` as an external structure-policy transfer result.

This rung trains `OpaqueSharedBasisStructurePolicy` from one scalar verifier
utility per generic opaque value bank. The policy receives the value rows and
occupancy mask, computes a fixed-width row-permutation-invariant singular
spectrum summary, and proposes a runtime-sized rank. It does **not** receive
the memory evaluator's precomputed candidate reconstruction error, task labels,
semantic family IDs, correct actions, or verifier-private targets.

The memory backend remains authoritative: each proposal is copy-on-write,
retention-verified, version-checked, and persistently checksummed.

| seed | held-out rank 1/2/4 | fresh rank 1/2/4 | live choices | final physical/dense value scalars |
| ---: | --- | --- | --- | ---: |
| 17 | `0.9375/1.0000/1.0000` | `0.0469/0.0156/1.0000` | `2 → 4` | `112/192` |
| 18 | `0.9219/1.0000/1.0000` | `0.5938/0.0000/0.0000` | `2 → 4` | `112/192` |

Both seeds passed held-out transfer, fresh comparison, old/new retention after
growth, forward and reversed physical order, exact reload, stale-version
rejection, checksum-corruption rejection, frozen controller/encoder, and zero
replay. Six old routes remained readable while six rank-four successor routes
were admitted. The basis was reduced from six to two rows for the first cohort
and from eight to four after successor growth.

The calibration ledger records that 10,000 and 20,000 updates were not enough
for seed 17 to clear the rank-one `0.80` floor. Promotion required 50,000
unique scalar-utility updates. This is a sample-efficiency bottleneck, not a
result to hide.

This promotes outcome-trained external structure selection without a
precomputed reconstruction-error feature across one controlled nonstationary
growth event. It does not establish semantic structure discovery, arbitrary
new computation, unrestricted memory growth, repeated alternation, or general
continual learning. The next pressure is a longer stream with repeated growth,
reversal, and capacity pressure, while reducing the scalar-feedback budget.

Reports and accounting:

- `seed-17.json`
- `seed-18.json`
- `rejected_calibration.json`
- `sample_efficiency_ledger.json`
