# Pong growth with all Snake state frozen (2026-08-06)

First cross-game continual-learning audit. The rung reproduces the promoted
Snake acquisition, hashes every Snake parameter, freezes the entire Snake
path, and then acquires Pong as an isolated external slot (its own frontend,
adapter, and keypress decoder) from scalar outcomes only. After Pong
acquisition, Snake is re-audited on the identical evaluation seeds and its
parameter hash is compared bit-for-bit.

Command (per seed):

```bash
uv run python -m experiments.games_amodal.pong_growth \
  --seed <seed> --updates 400 --batch-size 64 --steps 64 \
  --gamma 0.95 --learning-rate 1e-3 \
  --event-width 64 --intent-width 32 --hidden 64 --eval-seeds 8
```

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| Snake mastery before Pong | 0.9688 | 0.9414 |
| Snake mastery after Pong | 0.9688 | 0.9414 |
| Snake parameter hash | unchanged | unchanged |
| Pong mastery (8 fresh eval seeds) | 0.9180 | 0.8340 |
| Pong mean paddle returns per lifetime | 4.59 | 4.17 |
| Pong random-clone mastery | 0.0762 | 0.0391 |
| Pong reward-shuffled-clone mastery | 0.0762 | 0.0293 |
| both capabilities protected in one ledger | yes | yes |
| replayed examples | 0 | 0 |

Both seeds pass all gates: Snake acquisition, exact Snake retention (score
and hash), Pong acquisition above the 0.8 mastery gate, random and
reward-shuffled Pong nulls near chance, dual ledger protection, and zero
replay.

## Claim boundary

This promotes isolated-slot growth: a second game acquired with the first
game's state bit-for-bit frozen and no replay, with catastrophic forgetting
structurally impossible for the frozen slot. It does not promote learned
routing between games from opaque events alone — the caller still selects
which slot plays which game. Learned game routing (via the promoted shared
candidate growth router) is the next rung.
