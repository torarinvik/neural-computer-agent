# v20 cue-guided retention rung

This is the first outcome-only curriculum test of learned utility-based
retention after the promoted fixed-write v18 binding result.

The one-row memory receives two opaque slot outcomes. An ordinary event cue
identifies which slot will be queried later. The agent receives only opaque
actions and scalar verifier outcomes. The curriculum first trains single-slot
scalar recall, then adds the distractor and randomizes presentation order.

## Result

The ordinary seed-17 sub-minute rung was rejected:

| condition | recall |
|---|---:|
| intact | 0.4922 |
| clear memory | 0.5146 |
| corrupt values | 0.4971 |
| reversed order | 0.4902 |
| random action | 0.5127 |

The reward-shuffled control was also rejected at `0.4863` intact recall. The
ordinary write gate committed on `24.09%` of logged writes versus `10.71%` in
the shuffled control, but that difference did not produce a causal recall gain.
The controller/runtime v20 pair-context path therefore has no promoted
learned-retention result yet.

## Bridged parent control

Because the first parent phase was not mastered at 64 updates, a promoted
bridge was run with 1,024 single-event updates before 128 retention updates.
The single-event history reached a stable `1.0` prefix, but the retention
conditions still failed:

| condition | recall |
|---|---:|
| intact | 0.5020 |
| clear memory | 0.5049 |
| corrupt values | 0.5195 |
| reversed order | 0.4805 |
| random action | 0.4971 |

The ordinary bridge committed `94.69%` of writes, while the reward-shuffled
control committed `1.78%`; neither produced a causal retention gain. This
rules out insufficient parent mastery and absent writes as the immediate
blockers. The next experiment should reduce phase-transition credit-assignment
variance or explicitly protect the mastered parent while testing retention.

A parent-protected diagnostic then froze the mastered controller and trained
only the v20 write policy for 512 retention updates. It produced `0.5344`
intact, `0.5015` clear, `0.5110` corrupt, and `0.5186` reversed-order recall.
That small `+0.033` intact/clear gap is below the promotion gate and is not a
learned capability claim.

Reports:

- `subminute_seed17.json`
- `reward_randomized_seed17.json`
- `bridged_seed17.json`
- `bridged_reward_randomized_seed17.json`
- `sample_efficiency_ledger.json`

## Accounting

Each ordinary run charged 8,192 unique verifier outcome bits, 3,072 unique
logical lifetimes, 192 optimizer updates, and no replayed examples. The run
took 2.38 seconds on CPU. No checkpoint was curated or promoted.
