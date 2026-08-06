# Learned game routing over frozen slots (2026-08-06)

The promoted `OpaqueCandidateGrowthRouter` selects which frozen game slot
plays an incoming stream, from opaque events only. Both slots (Snake, Pong)
are trained from scalar outcomes on a common padded observation space with
their native action counts, then frozen and hashed. Each slot contributes one
opaque candidate key derived from its own fresh greedy events. A caller-owned
route encoder maps the first observation of a lifetime to an opaque query;
router training is outcome-only paired counterfactual ranking over both slots
attempted on identical fresh lifetimes. No game label, slot label, or
correct-row target ever reaches the router loss.

Command (per seed):

```bash
uv run python -m experiments.games_amodal.game_routing \
  --seed <seed> --updates 400 --route-updates 256 --batch-size 64 --steps 64 \
  --gamma 0.95 --learning-rate 1e-3 --route-learning-rate 3e-3 \
  --event-width 64 --intent-width 32 --hidden 64 --router-hidden 64 \
  --eval-seeds 8
```

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| routing accuracy snake / pong | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| routed end-to-end mastery snake / pong | 0.9688 / 0.8516 | 0.9414 / 0.8047 |
| candidate permutation accuracy | 1.0000 | 1.0000 |
| outcome-shuffled router routing (mean) | 0.5000 | 0.5000 |
| frozen slot hashes after router training | unchanged | unchanged |
| replayed examples | 0 | 0 |

## Rejected control: common action space

`game_routing_common_actions_rejected_v1_2026-08-06/` preserves the first
configuration, which forced both slots onto one four-key action space with
clamping during slot training. Routing was already near-perfect there
(&geq; 0.977), but the Pong slot plateaued at 0.752/0.760 routed mastery on
seed 69316 even after a 700-update matched budget escalation, while the same
seed reached 0.918 with native actions in the pong-growth rung. The clamped
fourth key biased exploration toward one paddle direction. Native per-slot
action counts repaired acquisition without touching the routing mechanism.

## Claim boundary

This promotes learned two-game routing from opaque events with frozen slots
and zero replay: the system itself decides which capability plays. It does
not yet promote routing over more than two candidates, growth of new slots
through the router's failure gate, or routing from mid-lifetime events (the
query uses the first observation). Those are the next rungs.
