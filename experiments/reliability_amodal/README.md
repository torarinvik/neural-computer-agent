# Outcome-only redundant reliability

This rung tests whether the canonical controller can retain separately bound
event tokens and learn source reliability from outcomes. The verifier renders
one high-bit stream and three redundant low-bit streams. The low-bit sources
have different hidden flip rates; a flip or omission does not change confidence
or expose a condition label. The learner receives only opaque events,
sampled-action propensity, and scalar verifier reward.

The decisive audit forces the historically reliable source to agree against
two noisy sources, then reverses that conflict by flipping the reliable source.
Promotion requires the controller to resolve the first conflict and fail the
reversal, alongside high reward on clean, a low-reliability-source corruption,
and stream-order-shuffled inputs. Arbitrary corruption of the historically
reliable source remains a deliberately hard reversal control.

The promoted protocol freezes all four raw frontends before outcome training,
so the reliability shift is learned inside the canonical controller. Across
seeds 17, 18, and 19, reliable-source conflict reward is `0.9976`, `0.9912`,
and `0.9966`; the reversal reward is `0.0015`, `0.0103`, and `0.0005`.
One-missing performance remains a diagnostic rather than a promotion claim.

Run one short rung with:

```bash
PYTHONPATH=src:. .venv/bin/python -m experiments.reliability_amodal.train --steps 256 --batch-size 256 --seed 17
```
