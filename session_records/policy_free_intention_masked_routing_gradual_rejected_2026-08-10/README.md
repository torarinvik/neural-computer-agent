# Gradual-mask external routing — 2026-08-10

This is the rejected follow-up curriculum after the promoted overlapping-mask
rung. The successor regime starts with the mastered source mask for 119
updates, then switches to the overlapping mask for 121 updates. The controller
and adapter remain frozen; no examples are replayed.

The current one-switch schedule is still too abrupt. Both seeds require all
240 successor updates, fail to reach the mastery threshold, and do not beat the
matched fresh control. The retention verifier correctly refuses to promote the
successor, while source retention and the causal controls remain informative.

| seed | successor score | fresh score | successor updates | fresh updates |
| ---: | ---: | ---: | ---: | ---: |
| 85301 | 0.9231 | 0.9419 | 240 | 240 |
| 85302 | 0.8329 | 0.8548 | 240 | 240 |

This rejects the schedule, not the mask ABI or the promoted overlapping-mask
boundary. The next curriculum should use multiple intermediate evidence
patterns or an adaptive transition based on verified support rather than one
fixed halfway switch.
