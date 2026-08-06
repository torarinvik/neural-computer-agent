# Nonstationary length-six to length-seven growth — rejected (2026-08-06)

This diagnostic freezes two capabilities learned on length-six episodes, then
acquires eight new external routes and isolated credit heads from fresh
length-seven episodes. The old stream is not replayed after the shift. The
audit uses a larger 64-sample evaluation batch than the training batch so the
reward-shuffled control is not decided by a tiny sample.

## Result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| old route accuracy | 1.0000 | 1.0000 |
| candidate permutation | 1.0000 | 1.0000 |
| minimum new-route selection | 0.9688 | 0.9375 |
| shifted credit accuracy | 1.0000 | 1.0000 |
| replayed examples | 0 | 0 |
| reward-shuffled family 7 | 0.6406 | 0.0000 |
| promoted | no | yes |

The cross-seed rung is rejected because one shifted family on seed 69316
activates under reward-shuffled outcomes. Storage, old-route retention,
permutation, new-route causality, shifted credit, reversal/recovery, and
zero-replay gates otherwise pass. This is retained as a decisive negative
control rather than hidden by lowering the shuffle threshold.

## Claim boundary

The mechanism can represent and retain the temporal distribution shift, but
the reward-to-route calibration is not yet robust across seeds. This is not
general continual learning or open-ended growth; the next repair must isolate
shifted-distribution negative credit without damaging the old stream.
