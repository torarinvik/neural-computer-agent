# Nonstationary full-bank length-six to length-ten growth v2 (2026-08-06)

This is the repaired large-shift rung. Two capabilities are acquired on
length-six episodes, then 18 fresh capabilities are acquired from length-ten
episodes, filling the 20-family bank. New-extension acquisition is doubled to
256 updates per family after the 128-update rung exposed a cross-seed mastery
failure. The controller, old routes, and old credit state remain frozen; no
old examples are replayed.

The antithetic shuffled control is explicitly zero-centered in trainer-only
state (`score²`), preventing optimizer roundoff from turning exactly null
paired outcomes into a positive route signal.

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| minimum new-route selection | 0.8438 | 0.8750 |
| old-route/permutation | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| shifted credit old/new/combined | 1/0.9444/0.9500 | 1/0.8333/0.8500 |
| full-bank protection/reversal/recovery | passed | passed |
| reward-shuffled false selections | 0 | 0 |
| replayed examples | 0 | 0 |

Both seeds pass all gates. The earlier 128-update rung remains archived as a
rejection control: it showed that the large shift needed more acquisition
depth, not a lower protection threshold.

## Claim boundary

This promotes a 6→10 temporal shift under a fixed 20-family bank. It does not
establish repeated arbitrary shifts, unbounded expansion, arbitrary program
induction, or general continual learning.
