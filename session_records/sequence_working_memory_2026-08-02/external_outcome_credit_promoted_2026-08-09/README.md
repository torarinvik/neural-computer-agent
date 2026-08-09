# External outcome-credit eligibility state

Date: 2026-08-09; seeds: `69316`, `69317`

This audit tests delayed scalar credit in external state. Each episode presents
one learned event tensor, requires two opaque sampled choices, and returns only
one terminal verifier outcome. The hidden event-to-choice relation is private
to the verifier. The main state uses an eligibility trace; matched controls
disable the trace or shuffle the scalar outcomes across episodes.

## Result

| seed | trace target | no-trace control | shuffled control | stable trace gate | source retention |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 69316 | 0.980 | 0.518 | 0.368 | 500 episodes | 0.958 |
| 69317 | 0.972 | 0.470 | 0.194 | 500 episodes | 0.958 |

Both seeds passed source mastery, target mastery, delayed-credit stability,
no-trace rejection, reward-shuffle rejection, missing-feedback no-write,
exact persistence, frozen-rule, and zero-replay gates. Each seed used `2,000`
source episodes and `5,000` target episodes, accounted for `7,000` unique
verifier bits and `7,000` unique logical lifetimes, and made `0` optimizer
updates. The paired controls reuse the target stream only as independent
diagnostics; the promoted learner replays no examples.

## Claim boundary

This promotes a bounded external delayed-credit primitive. It demonstrates
that external state can learn a two-step capability from terminal scalar
feedback while the processor remains frozen. It does not establish general
continual learning, arbitrary program induction, unrestricted capacity,
variable-length credit assignment, or learned consolidation. The next test
must connect the eligibility state to an executable external program and vary
the relation and sequence length.

The reproducer is
`experiments/external_outcome_credit/train.py`. Per-seed summaries are in
`report_seed69316.json` and `report_seed69317.json`.
