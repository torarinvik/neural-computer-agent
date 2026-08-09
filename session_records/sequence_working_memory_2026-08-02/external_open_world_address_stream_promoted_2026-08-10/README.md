# Open-world-style address formation without encoder pretraining — promoted

This three-seed pressure test removes the source-pretraining assumption from
the learned address boundary. The context encoder starts untrained and gets
zero optimizer updates. Eight nonlinear transition regimes arrive one at a
time; capacity grows `1 -> 8` through retention-verified transactions, and
each new address is formed by isolated copy-on-write adaptation from current
evidence only.

| seed | regimes | final address version | max held-out MSE | replay |
| ---: | ---: | ---: | ---: | ---: |
| 82401 | 8 | 40 | 0.00392 | 0 |
| 82402 | 8 | 40 | 0.00114 | 0 |
| 82403 | 8 | 40 | 0.00253 | 0 |

All gates passed: no encoder pretraining, one verified growth per new regime,
all held-out factual checks, reverse and interleaved revisits, no duplicate
slots, full prior retention, corruption rejection without a bank write,
copy-on-write isolation, frozen controller, no raw candidate rows, and exact
persistence.

Claim boundary: bounded open-world-style online address formation. The stream
is finite, the capacity is finite, and the factual basis is a fixed
random-feature family; this does not establish unrestricted general continual
learning.
