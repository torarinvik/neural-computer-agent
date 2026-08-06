# Rejected: stale-reference protected plasticity (2026-08-06)

First attempt at replay-free continual core learning on real games: the core
stays fully plastic during Pong, and the trainer projects each update against
a reference direction accumulated from the controller's own last 200 Snake
policy-gradient updates. No per-game parameters, no stored experiences.

Command (per seed):

```bash
uv run python -m experiments.games_amodal.protected_plasticity \
  --seed <seed> --updates 600 --batch-size 64 --steps 64 \
  --reference-window 200 --projection-strength 1.0 --eval-seeds 8
```

## Result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| Snake before Pong | 0.8555 | 0.9297 |
| Snake after, unprotected baseline | 0.0000 | 0.1113 |
| Snake after, protected | 0.0098 | 0.0137 |
| Snake after, random-direction null | 0.0293 | 0.2695 |
| Pong, protected | 0.6211 | 0.9590 |
| gradient-projection engagement | 0.505 | 0.528 |

The forgetting baseline is genuine and catastrophic. Protection rescued
nothing: protected retention is statistically indistinguishable from the
unprotected baseline and the random-direction null. The projection engaged
on ~50% of updates — chance level — showing Pong's gradients were no more
anti-aligned with the reference than with a random direction.

## Diagnosis

The reference direction is noise, not Snake. Late-phase REINFORCE gradients
have centered advantages: near convergence they average toward zero mean and
their sum is dominated by variance. Protecting against a noise direction
protects nothing. The parent repository's promoted use of
`project_gradient_against_reference` computed references from fresh verified
rehearsal streams, not stale end-of-training gradient sums.

## What survives

The harness (four-condition design, plastic core, engagement accounting,
zero replay) is sound and reusable. The open repair is a reference computed
from fresh on-distribution Snake gradients during the Pong phase — fresh
self-played lifetimes, which uses Snake environment access but stores no
data. That weakens the continual-learning setting from "old environment
gone" to "old environment reachable," and must be claimed as such if used.
