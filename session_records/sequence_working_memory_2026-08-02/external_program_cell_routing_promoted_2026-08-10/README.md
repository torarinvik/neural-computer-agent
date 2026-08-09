# Overlapping external program-cell routing — promoted

This audit extends the external program memory from one retained source cell
to an append-only bank of independently persisted cells. Two cells receive
identical opaque event features but encode different hidden program relations.
The bank selects a cell only when a copy-on-write verifier probe predicts the
current stream accurately; no context label or relation ID is given to the
router.

| seed | source retention | target mastery | maximum wrong-cell accuracy | shuffled accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 2501 | 90.33% | 90.67% | 0.33% | 9.33% |
| 2502 | 84.00% | 80.67% | 0.67% | 3.67% |
| 2503 | 82.67% | 84.67% | 0.00% | 2.67% |

All three seeds admitted a second cell, selected `[0, 1, 0, 1, 0, 1]` on
alternating identical-feature streams, retained the source cell, rejected the
wrong-cell and shuffled-outcome controls, and restored the bank with an exact
content digest. The frozen controller, interpreter, and executable program
memory were unchanged. Old-cell replay during target adaptation was zero.

This promotes bounded outcome-based routing among overlapping external
program cells. It does not establish learned context formation from raw
modalities, arbitrary program induction, unbounded capacity, or general
continual learning. The next frontier is learning the address representation
and handling gradual relation drift without relying on a complete verifier
bundle.

Reports are protected by `SHA256SUMS`.
