# Robust noisy partial nonlinear memory — promoted

This pressure test extends the partial/ambiguous nonlinear stream with an
explicit robust inlier resolver. Routing ignores a bounded sparse outlier only
when the configured inlier fraction remains satisfied; a contradiction large
enough to exceed the inlier threshold cannot be averaged into a match.

| seed | recursive max error | held-out max error | sparse noisy revisit | replay |
| ---: | ---: | ---: | :---: | ---: |
| 82501 | 0.000845 | 0.001095 | reused | 0 |
| 82502 | 0.001634 | 0.001075 | reused | 0 |
| 82503 | 0.000526 | 0.002870 | reused | 0 |

All three seeds retained the prior gates: partial evidence, contradiction
quarantine and one-time resolution, recursive candidate promotion, alternating
revisits, corruption rejection, frozen controller, zero replay, and exact
persistence. The injected noisy window reused an existing slot without
consuming capacity or staging a new candidate.

Configuration: minimum inlier fraction `0.75`, outlier tolerance `0.5`, and
recursive rollout tolerance `0.003`.

Claim boundary: bounded robust routing for synthetic nonlinear factual memory,
not general continual learning or unrestricted memory growth. The next test
should combine several noisy streams with learned reliability rather than a
fixed robust threshold.
