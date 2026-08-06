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

Status: environments and event-bus wiring only. No acquisition, retention, or
transfer result exists yet for this rung.
