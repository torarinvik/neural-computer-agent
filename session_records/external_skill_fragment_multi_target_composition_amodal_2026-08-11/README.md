# Multi-target frozen-bank closure — 2026-08-11

This audit extends the promoted four-fragment acquisition result. The four
opaque fragments are acquired once, then reused across three independently
held-out orders. Every target gets a fresh external trace combiner and output
decoder; the acquired register interpreter and fragment bank remain frozen.

| seed | target | inherited | fresh | inherited stable bits | fresh stable bits |
| ---: | --- | ---: | ---: | ---: | ---: |
| 69316 | prefix_parity → complement → reverse → rotate | 0.9583 | 0.9974 | 6,144 | 12,288 |
| 69316 | rotate → reverse → prefix_parity → complement | 0.9167 | 0.9740 | 6,144 | 12,288 |
| 69316 | complement → prefix_parity → rotate → reverse | 0.9974 | 0.9583 | 6,144 | 12,288 |
| 69317 | prefix_parity → complement → reverse → rotate | 1.0000 | 0.9036 | 6,144 | 12,288 |
| 69317 | rotate → reverse → prefix_parity → complement | 1.0000 | 1.0000 | 6,144 | 12,288 |
| 69317 | complement → prefix_parity → rotate → reverse | 0.9583 | 0.8750 | 6,144 | 12,288 |

All promotion gates passed on both seeds: primitive mastery and retention,
all-target mastery and stable prefixes, positive fresh-over-inherited transfer,
wrong-order/zero-code/missing-evidence/reward-shuffled rejection, frozen parent,
frozen acquired bank checksum, route resolution, persistence and corruption
rejection, and zero replayed examples.

The first 64-update replication is retained as a rejected diagnostic in
`seed-69317-64-rejected.json`. Its inherited targets passed, but the matched
fresh learner for the third target ended at 0.75 without a stable prefix.
Doubling only composition exposure to 128 updates resolved the issue. This is
a training-length/variance bottleneck in the fresh control, not evidence that
the acquired bank must be changed.

Claim boundary: this is replicated bounded continual-memory/composition
transfer. It is not arbitrary program induction, unrestricted memory growth,
compression, or general continual learning.
