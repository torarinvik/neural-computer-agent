# Adaptive factual-surprise admission screen

The online router now supports an opt-in factual-surprise admission policy.
It retains the complete pending prefix, scores each row against the best
committed factual model, and processes the prefix early only when both a mean
surprise threshold and a minimum surprising-row fraction are met. The policy
is serialized and the conservative six-row default is unchanged.

The first screen used `early_admission_observations=3`,
`early_admission_surprise_threshold=0.02`, and
`early_admission_surprise_fraction=2/3`. It was rejected as a learning gain:

| arm | same-cue n-back 3/4/5 | same-cue total | different-cue n-back-5 |
| --- | --- | --- | --- |
| fixed six-row active | 16 / 15 / 14 | 45/72 | 13/24 |
| adaptive active | 14 / 13 / 8 | 35/72 | 10/24 |
| fixed six-row passive | 12 / 13 / 13 | 38/72 | 11/24 |
| adaptive passive | 10 / 13 / 12 | 35/72 | 7/24 |

The adaptive same-cue active arm consumed `2,130` transition rows once versus
`2,148` for the fixed arm; it did not replay examples. All 96 runs in each
arm preserved the frozen controller and source slot. The result rejects this
naive threshold policy, not adaptive admission as a research direction.

The next attempt must use calibrated opaque context stability and full
held-out promotion risk, with the fixed six-row route retained as the control.
