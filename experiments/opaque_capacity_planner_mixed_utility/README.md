# Replay-free mixed-action capacity-policy learning

This experiment pressure-tests the external `OpaqueCapacityPlanner` as a
continually trainable policy. It first teaches one consolidation regime, then
switches to a balanced stream of admission, eviction, growth, and
consolidation states without replaying the first phase. A fresh opaque bank is
created for every episode.

The planner receives only learned keys/values and generic occupancy,
protection, and availability facts. The verifier retains the hidden target
action and selector and returns one scalar utility. The controller is frozen;
the policy update mutates neither controller nor memory candidates.

The promotion gates require stable mixed-stream utility, held-out transfer for
all four actions, retention of the earlier consolidation skill, improvement
over a fresh planner, exact one-update-per-utility accounting, and zero
replay. This remains a bounded policy-learning result, not universal
capacity planning or general continual learning.

Run one seed with:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/opaque_capacity_planner_mixed_utility/train.py \
  --seed 85601 \
  --report-out /tmp/mixed-capacity-85601.json
```
