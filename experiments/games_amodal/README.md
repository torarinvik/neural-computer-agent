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

- `game_routing.py` — promoted on both seeds: the shared
  `OpaqueCandidateGrowthRouter` routes opaque event queries to the correct
  frozen slot at 1.0000 accuracy on both games, with routed end-to-end
  mastery 0.9688/0.8516 and 0.9414/0.8047, permutation invariance, an
  outcome-shuffled null at exactly chance, and zero replay. The common
  four-key action space was rejected as an acquisition control. Evidence:
  `session_records/games_amodal_2026-08-06/game_routing_native_actions_v1_2026-08-06/`.

- `shared_controller.py` — qualified: both games run through one
  `AmodalCognitiveController` behind the production N-to-M runtime with exact
  retention, a bit-for-bit frozen core, and zero replay on both seeds. Seed
  69317 fully promoted (Pong 0.9082 through the frozen Snake-trained core,
  half mastery in 119 vs the random core's 215 updates); seed 69316 showed
  genuine negative transfer (0.5195 vs 0.9336 through a random core).
  Reliable positive core transfer is NOT promoted — it is seed-sensitive,
  replicating the parent repository's calibration bottleneck. Evidence:
  `session_records/games_amodal_2026-08-06/shared_controller_two_game_qualified_v1_2026-08-06/`.

- `protected_plasticity.py` — rejected: projecting Pong updates against a
  reference direction accumulated from late-phase Snake policy gradients
  rescues nothing (engagement at chance; the direction is noise). Preserved
  as the first-order control for the promoted consolidation rung.
- `ewc_plasticity.py` — **promoted on both seeds**: replay-free two-game
  continual learning in one fully plastic amodal core. Snake retained at
  0.8438/0.9395 (before: 0.8555/0.9297) while Pong reached 0.8125/0.9688
  through the same moving core; the unprotected baseline forgets to
  0.0391/0.0137 and the permuted-Fisher null fails, proving the sensitivity
  assignment is causal. Strict setting: no stored data, no frozen weights,
  no per-game core state, no old-environment access after acquisition.
  Evidence:
  `session_records/games_amodal_2026-08-06/ewc_consolidation_plastic_core_v1_2026-08-06/`.

Open next rungs: the three-game ladder (composing successive Fisher maps —
the designed stressor for anchor staleness and diagonal blindness, where an
enhanced consolidation rule would earn novelty), lambda robustness,
positive-transfer measurement through the continually trained core, and
routing over more than two games.
