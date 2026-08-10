# Partial and ambiguous nonlinear open-world memory — recursive gate promoted

This reruns the promoted partial/ambiguous nonlinear factual-memory pressure
test with recursive rollout verification inside every candidate promotion.
Four regimes arrive through partial windows; two candidates are isolated, an
ambiguous bundle is quarantined and later resolved, and all candidates must
pass one-step held-out fit, recursive rollout, and retention before promotion.

| seed | max held-out MSE | max recursive rollout MSE | quarantine rows | replay |
| ---: | ---: | ---: | ---: | ---: |
| 82501 | 0.001095 | 0.000845 | 8 | 0 |
| 82502 | 0.001075 | 0.001634 | 8 | 0 |
| 82503 | 0.002870 | 0.000526 | 8 | 0 |

All three seeds passed every gate: partial evidence, concurrent candidate
isolation, contradiction quarantine and one-time resolution, recursive rollout
verification, retention, alternating revisits, corruption rejection without a
bank write, copy-on-write address isolation, frozen controller, zero replay,
and exact persistence.

Claim boundary: bounded replay-free nonlinear factual-memory identity under
partial and explicitly ambiguous evidence. It is not general continual
learning, unrestricted memory growth, or arbitrary new computation. The next
pressure test should add noisy evidence to the same recursive promotion path.
