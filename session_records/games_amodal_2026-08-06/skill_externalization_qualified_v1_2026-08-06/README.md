# Qualified: skills stored in bank artifacts, not weights (2026-08-06)

Phase 1 of the dynamic-brain storage rule
(`docs/DYNAMIC_BRAIN_ARCHITECTURE.md`): each game's skill lives in a
512-parameter bank artifact (8 opaque tokens x 64), loaded into the
controller's event window as skill-as-context. Both skills train through one
fixed-size core under a dissociation objective: policy gradients with the
correct artifact fetched; KL-to-uniform ignorance under a withheld bank,
under fresh same-norm decoys, and under the other game's (detached)
artifact. Configuration: 1,600 alternating updates, ignorance weight 2.0.

Command (per seed):

```bash
uv run python -m experiments.games_amodal.skill_externalization \
  --seed <seed> --updates 1600 --ignorance-weight 2.0 --eval-seeds 8
```

## Result

| condition (mastery) | seed 69316 snake/pong | seed 69317 snake/pong |
| --- | ---: | ---: |
| own artifact fetched | 0.8203 / 0.8535 | 0.9336 / 0.9863 |
| bank withheld | 0.0000 / 0.0703 | 0.0000 / 0.0684 |
| random same-norm artifact | 0.0020 / 0.1035 | 0.0000 / 0.0684 |
| cross artifact (other game's) | 0.0020 / 0.0645 | **0.4141** / 0.0098 |
| reward-shuffled twin | 0.0000 / 0.0625 | 0.0000 / 0.0684 |

Seed 69316 passes every gate: the skill is necessary (withheld collapses),
content-bearing (noise collapses), identity-bearing (wrong program
collapses), computing-parameter counts fixed, zero replay. Seed 69317
passes six of seven; its single leak is the Pong artifact partially playing
Snake (0.4141). The promotion is therefore qualified: one full seed, one
seed with a directional identity leak.

## The shortcut ladder (all preserved as archives)

1. `skill_externalization_onswitch_rejected_v1` — one skill, no decoys:
   any same-norm noise switched the skill on (presence cue).
2. `skill_externalization_master_key_rejected_v2` — decoys only: noise
   rejected, but any authentic artifact ran whatever game was on screen.
3. v3/v4 budget controls (in this directory) — cross-artifact ignorance
   closed the master key; remaining failures were margins that responded
   to budget and ignorance-weight calibration.

## Claim boundary

Promoted (qualified): a game skill can be stored as a fetchable external
artifact with the core provably ignorant alone, content and identity
causally verified on one seed and content on both. Not promoted:
seed-robust identity checking (the 69317 cross leak), consolidation of
*already-acquired* weight-stored skills into artifacts (skills here were
acquired externalized from the start), content-addressed fetch (the caller
still selects the artifact), and compositional transfer. Stopping rule:
no further global-dial escalation; the recorded next designs are per-game
null gates scaled to game guessability and richer decoy sets.
