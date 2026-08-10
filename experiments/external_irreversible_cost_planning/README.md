# Replay-free cost-aware planning with an irreversible trap

This pressure test extends cost-aware search to a small irreversible
environment. The external factual model learns four opaque intention
transitions, including an absorbing trap state. A separate external scalar
model learns the verifier's opaque per-intention costs from the same one-pass
observations. The planner must reach the goal, avoid the trap, and choose a
lower-cost route than terminal-only search.

The controller is frozen, both external models are unchanged during search,
and no transition rows are replayed. The cost model is a narrow scalar
prediction fixture; it does not establish general learned utility or
open-world planning.

```text
uv run python experiments/external_irreversible_cost_planning/train.py \
  --seed 83311 \
  --report-out /tmp/external-irreversible-cost-planning.json
```
