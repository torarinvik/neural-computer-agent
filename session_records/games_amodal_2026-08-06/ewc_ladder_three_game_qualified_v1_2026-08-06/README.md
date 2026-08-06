# Qualified: three-game EWC ladder through one plastic core (2026-08-06)

Snake, Pong, and Breakout acquired sequentially through one fully plastic
controller with summed Fisher consolidation penalties (lambda 100), no
frozen weights, no per-game core state, zero replay, and no environment
re-access after each acquisition.

| final mastery | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| Snake post-acq -> after two later games (protected) | 0.9395 -> 0.9395 | 0.9355 -> 0.9375 |
| Pong post-acq -> after Breakout (protected) | 0.9082 -> 0.9082 | 0.7656 -> 0.7656 |
| Breakout (protected) | 0.5723 | 0.5293 |
| unprotected baseline (snake/pong/breakout) | 0.002/0.063/0.000 | 0.002/0.068/0.000 |
| permuted-Fisher ladder (snake/pong) | 0.041/0.469 | 0.004/0.277 |
| reward-shuffled Breakout twin | 0.0176 | 0.0000 |

Seed 69316 passes all nine gates. Seed 69317 passes eight: its single
miss is Pong acquisition under the Snake penalty (0.7656 vs the 0.8
gate) - a rigidity tax on new learning, not a retention failure.

Findings against the classic EWC weaknesses: (a) two consolidation steps
compose - retention through a second later game is essentially exact on
both seeds; (b) anchor staleness did not bite at this depth; (c) the cost
surfaces as reduced acquisition headroom for later games on one seed,
which is where an enhanced consolidation rule should aim (adaptive
per-parameter release, not stronger anchoring). Breakout mastery
(0.53-0.57 vs its 0.5 gate) is limited by the compound game itself,
consistent with the routing-rung slot plateau.

Stopping rule: archived without escalation; the recorded enhancement
target is acquisition headroom under accumulated penalties.
