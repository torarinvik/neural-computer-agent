# Amodal game-playing rung

This experiment package brings real game dynamics behind the same boundary the
Brain Workshop rung uses:

```text
game verifier (private state) -> rendered observation grid
    -> caller-owned frontend encoder -> opaque amodal event
    -> one controller/memory -> integer action -> scalar outcome only
```

The verifiers in `environments.py` (`SnakeVerifier`, `PongVerifier`) own all
game state, physics, and reward rules. The learner receives only learned event
tensors and deterministic scalar rewards; it never sees coordinates, entity
labels, or the collision rule.

The long-term goal for this repository is continual acquisition of multiple
games — Snake, then Pong, then further games — through the promoted
shared-growth-router mechanism, with the standard hard gates: full retention
of earlier games, causal routing, permutation invariance, null controls, and
zero replay.

Run the wiring smoke (random policy, no learning claim):

```bash
uv run python -m experiments.games_amodal.train --seed 0 --report-out /tmp/games-amodal-smoke.json
```

Status:

- `snake_acquisition.py` — promoted on seeds 69316/69317: greedy mastery
  0.9453/0.9414 with random and reward-shuffled nulls at chance and zero
  replay. Evidence:
  `session_records/games_amodal_2026-08-06/snake_acquisition_v1_2026-08-06/`.
- `pong_growth.py` — promoted on both seeds: Pong acquired at 0.9180/0.8340
  mastery as an isolated slot while every Snake parameter stayed bit-for-bit
  frozen and Snake re-audited unchanged. Evidence:
  `session_records/games_amodal_2026-08-06/pong_growth_snake_frozen_v1_2026-08-06/`.

Open next rung: learned routing between games from opaque events alone,
through the promoted shared candidate growth router.
