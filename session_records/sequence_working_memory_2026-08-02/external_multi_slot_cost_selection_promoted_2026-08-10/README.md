# Multi-slot cost-aware factual retrieval — promoted

This three-seed audit extends the learned-cost planning boundary across three
persistent factual slots. Each slot has a different opaque transition scale,
and the planner must select among all slots from the current opaque goal. No
task/context label is supplied to the planner.

A separately learned scalar-cost model supplies opaque intention costs. Across
the three held-out goal queries, cost-aware search reaches every goal, selects
stable slot IDs [0, 1, 2], and reduces realized route cost from 18 to 9
versus terminal-only search. The controller, factual bank, and cost model
remain frozen during inference; all observations are consumed once and no
replay is used.

| metric | seed 83321 | seed 83322 | seed 83323 |
| --- | ---: | ---: | ---: |
| terminal-only goal mastery | 1.000 | 1.000 | 1.000 |
| cost-aware goal mastery | 1.000 | 1.000 | 1.000 |
| terminal-only total route cost | 18 | 18 | 18 |
| cost-aware total route cost | 9 | 9 | 9 |
| cost saving | 9 | 9 | 9 |
| expected stable slots | true | true | true |
| replayed examples | 0 | 0 | 0 |
| controller updates | 0 | 0 | 0 |
| exact persistence | true | true | true |

Claim boundary: this promotes bounded multi-slot factual retrieval and
cost-aware planning under a three-step synthetic fixture. It does not
establish learned address formation, nonlinear model growth, compression,
unrestricted memory growth, or general continual learning.

Reports are protected by SHA256SUMS.
