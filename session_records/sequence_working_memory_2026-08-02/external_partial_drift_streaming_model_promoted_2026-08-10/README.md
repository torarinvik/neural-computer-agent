# Replay-free partial-evidence and gradual-drift model routing — promoted

This three-seed rung addresses the next model-first bottleneck. Each regime
had `12` available transition rows, but only `8` were presented, in two
four-row admission windows. An affine sufficient-statistics model consumed
each presented row once; no raw provisional rows or old-regime examples were
replayed.

The stream progressed through slopes `1.0 -> 1.5 -> 2.0`. Each drift version
was staged outside the committed bank, verified on held-out rows, and promoted
as a new stable slot. The original regime was then revisited and routed back
to slot `0`, retaining planner mastery. A corrupted stream was rejected by
the retention gate without changing the committed bank.

| seed | held-out errors by version | planner mastery | source return | corruption |
| ---: | --- | --- | :---: | --- |
| 81001 | 0 / 7.1e-15 / 0 | 1.0 / 1.0 / 1.0 | pass | rejected |
| 81002 | 0 / 7.1e-15 / 0 | 1.0 / 1.0 / 1.0 | pass | rejected |
| 81003 | 0 / 7.1e-15 / 0 | 1.0 / 1.0 / 1.0 | pass | rejected |

All gates passed: the controller and prior slots were byte-stable, stable IDs
were `[0, 1, 2]`, persistence was exact, and replayed examples were zero.

Claim boundary: bounded replay-free partial-evidence factual drift versions
with planner verification. This is not learned multimodal context formation,
unrestricted memory growth, or general continual learning. The next pressure
test must introduce richer nonlinear drift and a learned evidence/address
policy while preserving the no-replay and retention gates.
