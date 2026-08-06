# Snake acquisition from scalar outcomes only (2026-08-06)

First game-playing rung for this repository. The `SnakeVerifier` keeps all
game state, physics, and reward rules private; the learner is a caller-owned
`GridEventEncoder` frontend, an intent adapter, and a `KeypressDecoder` with
exact propensity accounting. Training uses fresh verifier lifetimes each
update with discounted outcome-only policy gradients; no trajectory or
verifier state is ever replayed into a later loss.

Command (per seed):

```bash
uv run python -m experiments.games_amodal.snake_acquisition \
  --seed <seed> --updates 400 --batch-size 64 --steps 64 \
  --gamma 0.95 --learning-rate 1e-3 \
  --event-width 64 --intent-width 32 --hidden 64 --eval-seeds 8
```

A lifetime row scores mastery `1` when its total private reward is positive
(more food than deaths), so chance sits at zero.

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| greedy mastery (8 fresh eval seeds) | 0.9453 | 0.9414 |
| minimum per-seed mastery | 0.9219 | 0.9219 |
| mean foods per lifetime (64 steps) | 4.01 | 3.90 |
| mean survival steps (of 64) | 62.4 | 63.0 |
| random-clone mastery | 0.0000 | 0.0000 |
| reward-shuffled-clone mastery | 0.0000 | 0.0000 |
| capability protected in retention ledger | yes | yes |
| replayed examples | 0 | 0 |

Both seeds pass all gates: acquisition above the 0.8 mastery threshold,
random and reward-shuffled nulls at exactly chance, ledger protection after
the deliberate post-acquisition audit, and zero replay.

## Claim boundary

This promotes single-game acquisition through the amodal event boundary with
causal nulls. The full policy path (frontend, adapter, decoder) is trainable
at this rung; freezing and protected growth begin at the next rung, where
Pong must be acquired with all Snake state frozen and Snake retention
re-audited afterward. No continual-learning claim is made here.
