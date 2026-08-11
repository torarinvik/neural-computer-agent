# Active discovery with tight route matching

The online discovery route was tested with the opaque route-match tolerance
tightened from `0.02` to `0.01`. The masked context summary and the same active
model-disagreement probe/passive matched-exposure control were used on target
n-back-3, n-back-4, and n-back-5 regimes.

Across `72` runs per arm, active discovery passed `48/72` complete gates
versus `44/72` for passive. Source retention passed in `72/72` runs for both
arms; all controllers remained unchanged, replay and optimizer updates were
zero, and both arms consumed `2,160` transition rows once. The per-regime
active/passive results were `16/24` vs `16/24` for n-back-3, `15/24` vs
`14/24` for n-back-4, and `17/24` vs `14/24` for n-back-5.

This promotes tighter route matching as a cross-regime boundary improvement.
It does not promote arbitrary computation, unrestricted memory growth, or
general continual learning. The harder-regime failure is still mostly
pre-admission model-family verification and candidate staging.
