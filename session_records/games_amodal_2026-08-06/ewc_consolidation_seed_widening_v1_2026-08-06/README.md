# Promoted: EWC consolidation seed-pool widening (2026-08-06)

Three fresh seeds of the flagship rung
(ewc_consolidation_plastic_core_v1_2026-08-06, identical command), all
fully promoted with every gate passing.

| metric | 69318 | 69319 | 69320 |
| --- | ---: | ---: | ---: |
| Snake before Pong | 0.9531 | 0.9258 | 0.9160 |
| Snake after (protected plastic core) | 0.9336 | 0.9043 | 0.9141 |
| Pong (protected) | 0.9395 | 0.8965 | 0.8691 |
| Snake after (unprotected baseline) | 0.0176 | 0.0020 | 0.0156 |

Combined with the original two seeds, the replay-free two-game
continual-learning result now holds on 5/5 seeds: retention within
epsilon, full new-game acquisition, catastrophic forgetting in every
baseline, permuted-Fisher and reward-shuffled nulls behaving, zero
replay.
