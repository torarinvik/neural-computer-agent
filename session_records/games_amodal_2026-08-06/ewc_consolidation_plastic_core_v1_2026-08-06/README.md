# Promoted: continual core learning via Fisher consolidation (2026-08-06)

First promoted replay-free continual-learning result on a fully plastic
shared amodal core with real games. Nothing is frozen and nothing
game-specific is added to the controller. Before leaving Snake, the trainer
estimates a diagonal Fisher sensitivity map from 32 fresh on-policy lifetime
batches and snapshots the anchor weights; afterwards only those two
parameter-shaped maps are carried — no experiences, no later Snake
environment access. During Pong the plastic core trains under a quadratic
consolidation penalty (`lambda = 100` on a unit-mean Fisher).

Command (per seed):

```bash
uv run python -m experiments.games_amodal.ewc_plasticity \
  --seed <seed> --updates 600 --batch-size 64 --steps 64 \
  --fisher-batches 32 --ewc-lambda 100 --eval-seeds 8
```

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| Snake before Pong | 0.8555 | 0.9297 |
| Snake after, EWC-protected plastic core | 0.8438 | 0.9395 |
| Snake after, unprotected baseline | 0.0391 | 0.0137 |
| Snake after, permuted-Fisher null | 0.1973 | 0.0352 |
| Pong, EWC-protected | 0.8125 | 0.9688 |
| Pong, reward-shuffled twin | 0.0762 | 0.0391 |
| replayed examples | 0 | 0 |

All gates pass on both seeds: genuine catastrophic forgetting in the
baseline; retention within epsilon under protection (seed 69317 retains
slightly above its pre-Pong score); Pong acquired above the mastery gate
through the same plastic core; the permuted-Fisher null — identical penalty
distribution, wrong parameter assignment — fails to rescue retention,
proving the *assignment* of sensitivity to parameters is causally
load-bearing; shuffled null at chance; zero replay.

Contrast with `protected_plasticity_stale_reference_rejected_v1_2026-08-06/`:
a first-order direction from late-phase policy gradients protected nothing
(engagement at chance), while second-order sensitivity captured before
leaving the game protects almost perfectly at the first tested strength.

## Claim boundary

Promoted: two-game replay-free continual learning in one plastic amodal
core in the strict setting (old environment unreachable after acquisition),
with retention within epsilon and full new-game acquisition. Not promoted:
more than one consolidation step (Fisher maps from successive games must
compose), longer game ladders, positive transfer claims, or robustness of
the single tested `lambda`. The known EWC weaknesses (anchor staleness over
many tasks, diagonal approximation) have not yet been stressed — the
three-game ladder is the designed stressor.
