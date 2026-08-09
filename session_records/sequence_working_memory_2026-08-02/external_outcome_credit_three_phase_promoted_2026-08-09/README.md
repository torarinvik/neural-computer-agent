# External outcome-credit three-phase scaling

Date: 2026-08-09; seeds: `69316`, `69317`

This is the replicated sequence-length follow-up to the two-phase
eligibility-trace promotion. The learner receives one standardized event,
makes three opaque choices, logs exact propensities, and receives only one
terminal scalar verifier outcome. The hidden event-to-sequence relation never
enters the learner.

## Result

| seed | source mastery | target accuracy | stable target episodes | no-trace control | shuffled control | source retention |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 69316 | 0.9400 | 0.9567 | 1,500 | 0.1133 | 0.1267 | 0.9400 |
| 69317 | 0.9267 | 0.9733 | 1,000 | 0.3800 | 0.1533 | 0.9267 |

Each seed used `3,000` source episodes and `4,000` target episodes, accounted
for `7,000` unique verifier bits and logical lifetimes, made zero optimizer
updates, and replayed zero examples. Missing-feedback no-write, persistence,
frozen-rule, no-trace, reward-shuffle, source mastery, and source-retention
gates all passed.

## Claim boundary

This promotes replicated three-phase delayed scalar credit. It does not prove
general continual learning, arbitrary program induction, unrestricted memory,
or efficient long-horizon credit. Four-phase exploratory rungs remain
unpromoted because sparse terminal feedback increases variance and leaves the
stable-prefix/source gates unresolved. The next test is an external value
baseline or another variance-reduced credit mechanism.

The generalized reproducer is
`experiments/external_outcome_credit/train.py`. Per-seed summaries are in
`report_seed69316.json` and `report_seed69317.json`.
