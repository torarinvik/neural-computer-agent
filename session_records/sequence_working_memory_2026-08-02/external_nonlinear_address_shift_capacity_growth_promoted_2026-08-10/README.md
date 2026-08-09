# Nonlinear address adaptation with verified capacity growth — promoted

This is the capacity-growth arm of the long alternating nonlinear address
stream. Each seed starts with capacity `4`, acquires two source and two target
slots, then runs a held-out retention probe before growing the bank to capacity
`6` for the remaining two target regimes. Address adaptation remains
copy-on-write, and historical keys remain immutable.

| seed | growth | final capacity | target held-out MSEs C/D/E/F |
| ---: | --- | ---: | --- |
| 82101 | 4 -> 6 | 6 | 6.23e-5 / 3.09e-4 / 3.54e-4 / 1.41e-3 |
| 82102 | 4 -> 6 | 6 | 3.71e-4 / 6.97e-4 / 1.06e-3 / 2.65e-4 |
| 82103 | 4 -> 6 | 6 | 3.44e-4 / 6.78e-4 / 3.96e-4 / 3.94e-3 |

All gates passed: growth was retention-verified, all target promotions and
revisits passed, prior slots and historical keys were retained, corrupted
evidence was rejected without a bank write, the controller remained frozen,
and persistence was exact. Replay and old-regime replay were zero.

Claim boundary: bounded verifier-gated nonlinear memory growth with learned
copy-on-write addresses. This does not establish unrestricted growth,
autonomous consolidation/compression, or general continual learning.
