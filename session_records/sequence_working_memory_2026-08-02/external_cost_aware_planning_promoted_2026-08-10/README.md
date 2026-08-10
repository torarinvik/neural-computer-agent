# Cost-aware goal-conditioned planning — promoted

This three-seed audit implements the exported session's lifetime-cost lesson.
An opaque affine transition model is learned once from fifteen transition rows.
Terminal-only search and cost-aware search both reach the same goal, but the
cost-aware planner selects an equivalent route with lower verifier cost.

| metric | seed 83301 | seed 83302 | seed 83303 |
| --- | ---: | ---: | ---: |
| terminal-only goal mastery | 1.000 | 1.000 | 1.000 |
| cost-aware goal mastery | 1.000 | 1.000 | 1.000 |
| terminal-only route cost | 10 | 10 | 10 |
| cost-aware route cost | 1 | 1 | 1 |
| cost saving | 9 | 9 | 9 |
| shuffled-cost goal mastery | 0.000 | 0.000 | 0.000 |
| controller updates | 0 | 0 | 0 |
| replayed examples | 0 | 0 | 0 |
| exact persistence | true | true | true |

All seeds pass. The planner remains inference-only; the factual model and
controller are unchanged during search. The cost vector is supplied as an
opaque nonnegative verifier scalar and is never interpreted as a protocol
field.

Claim boundary: this qualifies cost-aware inference over a tiny reversible
fixture. It does not establish learned cost prediction, irreversible-world
planning, positive transfer, or general continual learning.

Reports are protected by `SHA256SUMS`.
