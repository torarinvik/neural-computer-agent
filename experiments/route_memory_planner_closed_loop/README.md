# Closed-loop route-memory capacity-policy learning

This experiment connects the learned `OpaqueCapacityPlanner` to actual
`ExternalTransitionRouteMemory` transactions. The planner proposes admission,
eviction, consolidation, or growth; an independent verifier-gated transaction
then either commits the proposal or leaves memory unchanged.

Every state contains distractor route prototypes. Two alternating evidence
patterns reverse which pair is redundant and which row is the least-supported
eviction target. The planner is first exposed to one pattern, then learns a
balanced mixed stream containing both patterns without replaying the first
phase. Held-out transfer is measured for both patterns and every action.

The controller remains frozen, planner updates never mutate memory, and each
transaction uses a copy-on-write retention probe. This promotes closed-loop
bounded capacity-policy learning; it does not establish unbounded memory,
universal policy composition, autonomous verifier design, or general
continual learning.

Run one seed with:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/route_memory_planner_closed_loop/train.py \
  --seed 85701 \
  --report-out /tmp/route-memory-85701.json
```
