# Online verifier-utility learning for opaque capacity planning

This experiment trains the external `OpaqueCapacityPlanner` online from one
scalar verifier utility per proposed maintenance action. Exploratory action
and pair proposals are sampled; accepted proposals are reinforced and rejected
proposals are suppressed with a centered policy-gradient update.

Each episode contains a fresh opaque memory bank in which one pair is
redundant and therefore the only accepted consolidation pair. The controller
is frozen, memory candidates are never mutated by policy learning, and no old
episode is replayed. The trained planner is evaluated deterministically on a
held-out stream against a fresh planner.

Run one seed with:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/opaque_capacity_planner_online_utility/train.py \
  --seed 85501 \
  --report-out /tmp/capacity-utility-85501.json
```

This promotes online learned maintenance selection and transfer for one
bounded consolidation regime. It does not establish a universal capacity
policy, autonomous verifier design, unbounded memory, or general continual
learning.
