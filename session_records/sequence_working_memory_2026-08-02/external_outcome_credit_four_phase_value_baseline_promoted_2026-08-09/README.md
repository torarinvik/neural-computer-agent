# External four-phase outcome credit with value baseline

Date: 2026-08-09; seeds: `69316`, `69317`

This audit tests whether a frozen external feature-conditioned value baseline
reduces the variance of the already-promoted eligibility-trace policy on four
delayed choices. The baseline receives the same learned features and terminal
scalar outcome as the policy, but no correct choices, task identifiers, or
privileged verifier state. Its state is external and tensor-serializable; the
controller and both rule modules remain frozen.

## Result

| seed | source mastery | target accuracy | stable target episodes | no-trace control | shuffled control | source retention |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 69316 | 0.9233 | 0.9067 | 2,500 | 0.2067 | 0.0833 | 0.9233 |
| 69317 | 0.9067 | 0.9033 | 3,000 | 0.0167 | 0.0233 | 0.9067 |

Both matched seeds passed every promotion gate: source mastery and
retention, stable-prefix target mastery, no-trace and reward-shuffle causal
controls, missing-feedback no-write, exact persistence, frozen policy and
baseline rules, and zero replay. Each seed used `3,000` source and `3,000`
target episodes, `6,000` unique verifier bits and logical lifetimes, and zero
optimizer updates.

## Interpretation

This promotes four-phase delayed scalar credit with an external value baseline
as a bounded variance-reduction primitive. It is a meaningful extension of the
three-phase result: the previously unpromoted four-phase budget now passes on
both seeds. It still does not establish general continual learning, arbitrary
program induction, unrestricted memory growth, or transfer across changing
relations. The next bottleneck is integrating this credit path with executable
external programs and testing many-capability interference without replay.

The reproducer is
`experiments/external_outcome_credit/train.py`, using `--phases 4
--value-baseline`. Per-seed summaries are in `report_seed69316.json` and
`report_seed69317.json`.
