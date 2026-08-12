# Outcome-trained shared-basis policy growth — promoted narrow boundary

Status: `PROMOTED` as an external policy-transfer result.

This rung trains `OpaqueSharedBasisCompressionPolicy` only from one scalar
utility per generic candidate bank. The policy sees opaque candidate
statistics—rank, reconstruction error, physical-size ratio, record/width
fractions—and emits a candidate index. It does not see task labels, semantic
families, correct actions, or verifier-private targets.

The frozen canonical stream then has two stages:

1. six old opaque event-key values with rank-two shared structure;
2. six new event-key values with rank-four successor structure.

The policy chooses rank `2` and then rank `4`. Each proposal is independently
checked by the memory route/value verifier before copy-on-write commit. The old
cohort remains readable after the new cohort grows the basis; forward and
reversed physical row orders are both tested.

| seed | held-out rank 1/2/4 | fresh rank 1/2/4 | live choices | candidate errors |
| ---: | --- | --- | --- | --- |
| 17 | `0.875/1.000/1.000` | `0.047/0.016/1.000` | `2 → 4` | `0.01124/0.00521` |
| 18 | `0.875/1.000/1.000` | `0.656/0.969/1.000` | `2 → 4` | `0.00557/0.00459` |

Both seeds passed held-out transfer, fresh comparison, old/new retention after
growth, reversed-order routing, exact reload, stale-version rejection,
checksum-corruption rejection, frozen controller/encoder, and zero replay.

This promotes an outcome-trained external compression preference and its safe
transfer across one controlled nonstationary memory growth event. It does not
establish semantic structure discovery, arbitrary new computation, unrestricted
memory growth, or general continual learning. The policy still receives
candidate reconstruction statistics supplied by the memory-side evaluator, and
the stream has only one successor transition. The next pressure is online
structure discovery from raw evolving capability values, longer repeated
growth/reversal streams, and a learned verifier-independent proposal cost.

Reports:

- `seed-17.json`
- `seed-18.json`
