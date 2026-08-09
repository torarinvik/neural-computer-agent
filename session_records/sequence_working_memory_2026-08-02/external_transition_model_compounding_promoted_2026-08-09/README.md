# Accounted policy-free model compounding — promoted

This two-seed fast rung imports the strongest lesson from the exported games
session: a transition model is factual and can be extended, while a policy is
preferential and can become wrong on a new task. Behavior is therefore derived
by opaque model-based search rather than stored as a task policy.

The audit trained one source regime and then added three successive regimes.
Each target model was initialized from the preceding external model, adapted
in isolation, and evaluated by the deployed planner. Fresh target models were
trained under matched budgets. The stopping rule required both the transition
loss threshold and deployed planner mastery, preventing an internal loss from
being mistaken for useful behavior.

| metric | seed 70311 | seed 70312 |
| --- | ---: | ---: |
| target-1 warm/fresh updates | 24 / 38 | 25 / 35 |
| target-2 warm/fresh updates | 23 / 38 | 22 / 38 |
| target-3 warm/fresh updates | 17 / 34 | 17 / 41 |
| cumulative warm updates after target-3 | 1,264 | 1,264 |
| cumulative fresh updates after target-3 | 1,310 | 1,314 |
| target mastery at every rung | 1.0 | 1.0 |
| earlier-regime retention | 1.0 | 1.0 |

All gates passed: the controller stayed frozen, earlier model slots were
byte-stable, old-regime replay was zero, planner expansion/latency was
accounted separately, and persistence was exact. Zero-shot capability is
reported separately from adaptation speed; it ranged from `0.0` to `0.667`
before target adaptation in this small fixture.

Claim boundary: this promotes a replicated downward acquisition-cost signal
for one nested dynamics family. It is not general continual learning,
unrestricted growth, cross-family transfer, or evidence that every future
task will become cheaper. The next test must use genuinely disjoint dynamics,
wider seeds, and a learned online context/address path.

Reports are protected by `SHA256SUMS`.
