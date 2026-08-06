# Nonstationary full-bank length-six to length-ten growth — rejected (2026-08-06)

This stress rung freezes two capabilities learned on length-six episodes, then
acquires 18 fresh capabilities from length-ten episodes. It fills the 20-family
bank and uses the same antithetic shuffled null as the promoted 6→7 and 6→8
audits.

## Result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| minimum new-route selection | 0.9219 | 0.8125 |
| old-route/permutation | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| shifted credit old/new/combined | 1/0.9444/0.9500 | 1/0.8333/0.8500 |
| initial full-bank protection | pass | fail |
| promoted | yes | no |

Seed 69317 leaves family 12 below the mastery threshold before reversal, so
the full bank can evict an unmastered row and the retention gate correctly
rejects the cross-seed rung. Positive routing, causality, shuffled-null, and
zero-replay controls otherwise pass.

## Claim boundary

This rejects a large temporal-shift promotion at the retention boundary. It
shows that the current mechanism tolerates 6→8 but not yet 6→10 reliably
across seeds. The next repair is confidence-aware acquisition under large
distribution shifts, not a weaker retention threshold.
