# Policy-free model compounding — recursive rollout verification promoted

This archive adds the missing deployed-capability gate to the policy-free
external transition-model compounding audit. Each target model must now pass
both the existing planner mastery checks and a held-out recursive rollout
probe. The probe feeds each predicted state into the next prediction, so it
measures compounding error rather than only one-step transition fit.

| seed | warm target updates | fresh target updates | warm rollout max by target | promoted |
| ---: | ---: | ---: | ---: | :---: |
| 70311 | 24/24/17 | 38/38/34 | 0.0275/0.0155/0.0269 | yes |
| 70312 | 25/22/17 | 35/38/41 | 0.0214/0.0183/0.0391 | yes |
| 70313 | 25/28/21 | 42/40/31 | 0.0211/0.0240/0.0204 | yes |

All three seeds passed every gate: deployed mastery, warm-vs-fresh update
advantage, prior retention and byte stability, frozen controller, zero old
regime replay during target adaptation, inference-only planning, and recursive
rollout error below `0.05` after each target adaptation.

Claim boundary: this promotes a measurement and retention gate for one small
nested dynamics family. It is not general continual learning, unrestricted
memory growth, or evidence that one-step model loss is sufficient elsewhere.
The zero-shot rollout errors remain reported separately to expose the cost of
new knowledge before adaptation.
