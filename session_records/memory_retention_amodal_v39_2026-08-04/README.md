# Counterfactual write-utility qualification

This rung tests outcome-only cue-guided retention with a training-only paired
intervention. For one randomly selected event position, one arm is forced to
write and its common-random partner is forced to skip. All other positions use
shared sampled write uniforms. The controller receives only the ordinary cue
and event tokens, opaque actions, and scalar verifier outcomes; the branch
position and verifier state never cross the runtime boundary.

The intervention trains the generic write logit from the observed recall
difference between the write and skip arms. The retained runtime remains the
single v23 controller/memory; the paired verifier rows only provide common
random-number variance reduction during training. The write-policy output
layer is reset to a neutral prior at the parent-to-retention transition, while
the rest of the parent is retained.

The three valid unprotected seeds pass the existing single-run gate:

| seed | parent updates requested/effective | intact | clear | corrupt | reverse | target first | target last | missing-write cue |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1024 / 768 | 0.930 | 0.492 | 0.479 | 0.938 | 0.935 | 0.919 | 0.746 |
| 18 | 512 / 416 | 0.990 | 0.524 | 0.506 | 0.989 | 0.978 | 0.982 | 0.746 |
| 19 | 512 / 320 | 0.949 | 0.516 | 0.479 | 0.937 | 0.951 | 0.920 | 0.537 |

The population means are `0.956` intact, `0.511` clear, `0.488` corrupt,
`0.954` reversed, `0.954` target-first, and `0.940` target-last. Random query
actions remain at `0.505`. The reward-shuffled control stays at chance
(`0.497` intact, `0.503` target-first, `0.473` target-last), so the gain is
not explained by reward noise alone. Stable validation thresholds are reached
at 30,720, 19,456, and 16,384 unique verifier bits for seeds 17, 18, and 19.

The initial seed-17 512-update rung is retained as a rejected parent-gate
attempt (`seed_17.json`); its retention phase was correctly blocked. The
bridged 1,024-update run (`seed_17_bridged.json`) is the valid population
member. Full per-run reports and the aggregate sample-efficiency ledger are
in this directory.

This promotes the counterfactual write-utility protocol and the narrow
outcome-only retention behavior at the sub-minute qualification rung. No
checkpoint is promoted from this record: longer-duration replication,
retention on mastered primitives, transfer, and persistent-memory promotion
remain required before claiming reusable capability beyond this verifier.
