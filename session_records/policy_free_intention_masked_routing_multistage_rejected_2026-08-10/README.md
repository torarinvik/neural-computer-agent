# Multi-stage mask-drift external routing — 2026-08-10

This pressure test replaces the rejected halfway switch with seven evidence
stages: four dimensions are added one at a time, then the two source-only
dimensions are removed one at a time. Each stage is exercised for 34 updates;
the final overlapping mask receives 36 updates. The controller and adapter are
frozen and no examples are replayed.

The result is decisively rejected. Both seeds require the full 240 successor
updates, fail the successor mastery and retention-verifier gates, and are no
faster than matched fresh learners. Static overlapping-mask transfer still
passes in 6–11 updates under the same architecture, isolating the failure to
sequentially mutating one cell across mask distributions rather than to the
mask ABI or route address itself.

| seed | successor score | fresh score | successor updates | fresh updates |
| ---: | ---: | ---: | ---: | ---: |
| 85301 | 0.9258 | 0.9440 | 240 | 240 |
| 85302 | 0.8419 | 0.8607 | 240 | 240 |

This rejection defines the next architectural boundary: reusable computation
must be versioned or factored across evidence distributions. A single
cell-local nonlinear generator is not yet a stable continual learner under
sequential mask drift.
