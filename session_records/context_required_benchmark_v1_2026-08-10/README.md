# A context-required multi-step benchmark, and what it measures (F105)

F104 showed the battery had no multi-step game where context is REQUIRED, and
gave the rule for building one. Built from existing configuration: `forage`
supplies opposing item types, `inverted` swaps which is food, `recentre_every`
+ `spawn_radius` force the avatar to cross ground to each trial.

## Validated BEFORE running the mechanism

| policy | normal | inverted | pair mean |
| --- | ---: | ---: | ---: |
| idle | -0.0501 | -0.0497 | -0.0499 |
| random | -0.0496 | -0.0453 | -0.0474 |
| eat-anything | -0.0368 | -0.0267 | -0.0318 |
| always eat plane 1 | +0.1919 | -0.2640 | -0.0360 |
| always eat plane 2 | -0.2709 | +0.1989 | -0.0360 |
| ORACLE (hidden bit) | +0.1919 | +0.1989 | +0.1954 |

Every inversion-invariant policy loses; a fixed preference nets -0.036.
Headroom obtainable only by reading context: **+0.2272**.

## The mechanism on it

54 worlds, held out by whole twin pair, 40000 updates, all three sparsity fixes.

| arm | pair-mean reward |
| --- | ---: |
| trained worlds | -0.0463 |
| held-out worlds | -0.0466 |
| entry withheld | -0.0472 |
| stranger entry | -0.0471 |
| inverted TWIN entry | -0.0471 |
| untrained control | -0.0483 |
| best invariant policy (ref) | -0.0318 |
| oracle (ref) | +0.1954 |

Entry effect +0.0005 against +0.2272 available — 0.2% of the headroom. And the
score on TRAINED worlds is -0.0463, worse than the best context-free policy: the
stack does not learn these worlds at all, where a ten-line hand-written policy
earns +0.1954. The defect is upstream of the bank.

## The durable part

The benchmark has a measured floor (-0.0318), a measured ceiling (+0.1954), and
a validated guarantee that the gap is reachable only by reading context. Any
future attempt can be scored against those numbers — which is exactly what
F100-F104 lacked and spent four findings discovering.
