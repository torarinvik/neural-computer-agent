# Recursive rollout-gated candidate promotion — promoted

This pressure test moves recursive verification into the copy-on-write factual
memory transaction. A candidate must pass one-step held-out fit, a recursive
held-out rollout, and retention. A failed candidate must leave committed model
content and capacity unchanged.

| seed | source rollout error | target rollout error | corrupted candidate | live state unchanged |
| ---: | ---: | ---: | :---: | :---: |
| 84001 | 0.0 | < 5e-13 | rejected | yes |
| 84002 | 0.0 | < 5e-13 | rejected | yes |
| 84003 | 0.0 | < 5e-13 | rejected | yes |

All three seeds passed every gate. The target regime grew external capacity
from `1` to `2` only after recursive verification; a corrupted candidate that
passed one-step held-out fit but failed its recursive rollout was rejected
without changing the live bank. The controller stayed frozen and old-regime
replay was zero.

Claim boundary: this promotes a reusable copy-on-write verification mechanism
on a tiny affine dynamics family. It is not general continual learning,
unrestricted memory growth, or arbitrary computation. The next pressure test
must apply the same transaction to noisy, partially observed nonlinear streams.
