# The slot state made it worse — and the real cause is outcome sparsity (F102)

F101 blamed state representation and prescribed a factored multi-object state.
Built it: six slots off the screen (avatar, nearest positive object, nearest
negative object), dynamics model, outcome model, beam search.

Same seed, same held-out variants, floors bit-identical 12/12:

| arm | mean lift | beats floor |
| --- | ---: | ---: |
| trained | -0.0013 | 5/12 |
| untrained (random plant) | +0.0071 | 10/12 |

Training is worse than not training. F101's diagnosis was wrong: the state was
not the binding constraint.

## The measured cause

20 variants x 1280 random steps:

| outcome class | share |
| --- | ---: |
| nothing | 98.16% |
| cost | 1.53% |
| food | 0.31% |

1. "Always nothing" scores 98.16%, so cross-entropy on random-play data has
   almost no gradient toward the 1.8% that matters.
2. The residual is 5:1 biased toward COST.
3. Beam search maximises P(food)-P(cost) over that near-flat, cost-biased
   landscape, so the agent avoids everything including food — CONSISTENTLY
   wrong where random play is only randomly wrong. That is why it lands below
   the floor.

## This explains F99

`dual` worked (0.667 trained; stranger entry -0.100 reward) because every dual
step resolves a trial — outcomes ~100% non-zero. The mechanism works where the
outcome signal is DENSE and fails at 1.8%. One account covers F99-F102, and it
is about signal density, not the bank, the search, or the state.

## Corrected next step

Class-balanced or importance-weighted outcome loss; outcome-seeking rather than
uniform-random data collection; a VALUE target rather than immediate outcome.
Ordinary RL practice that this line skipped by collecting with a uniform random
policy.

Three formulations and two wrong diagnoses; what identified the cause was
counting the labels, which cost one command and should have come first.
