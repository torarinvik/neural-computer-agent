# Replay-free cost-aware planning with an irreversible trap — promoted

This three-seed audit extends cost-aware planning to a four-action fixture
with an absorbing trap state. A random-feature factual model and a separate
affine scalar-cost model each consume sixteen opaque rows once. Terminal-only
search and learned-cost search are compared under the same planner budget.

| metric | seed 83311 | seed 83312 | seed 83313 |
| --- | ---: | ---: | ---: |
| terminal-only goal mastery | 1.000 | 1.000 | 1.000 |
| cost-aware goal mastery | 1.000 | 1.000 | 1.000 |
| terminal-only route cost | 5 | 10 | 6 |
| cost-aware route cost | 2 | 2 | 2 |
| cost-aware trap visits | no | no | no |
| cost saving | 3 | 8 | 4 |
| shuffled-cost goal mastery | 0.000 | 0.000 | 0.000 |
| replayed examples | 0 | 0 | 0 |
| controller updates | 0 | 0 | 0 |
| exact persistence | true | true | true |

All gates pass. The planner avoids the irreversible trap using the learned
factual model and chooses the lower-cost direct route using the separately
learned scalar cost model. The controller and both external models remain
unchanged during search.

Claim boundary: this qualifies a bounded replay-free cost-aware planner under
one absorbing-trap fixture. It does not establish broad irreversible-world
generalization, learned long-horizon utility, or general continual learning.

Reports are protected by `SHA256SUMS`.
