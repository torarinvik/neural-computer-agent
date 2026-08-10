# Masked identity capacity pressure

This pressure test fills a slot's bounded masked-prototype budget, then tests
verifier-gated replacement. A rejected retention probe leaves the live bank
unchanged. An accepted probe replaces only the least-supported redundant
variant while retaining the core identity, the other slot, and the new partial
pattern.

All runtime identity updates use opaque anchor selection; the experiment does
not pass a frontend or slot ID to the bank's update API. The retention probe is
an external verifier over opaque route outcomes.

Run one seed with:

```text
PYTHONPATH=.:src .venv/bin/python experiments/external_goal_alignment_masked_capacity/train.py \
  --seed 85101 --report-out /tmp/external-goal-alignment-masked-capacity.json
```

The claim is bounded: verifier-gated masked-prototype replacement under a
fixed per-slot capacity. It does not establish autonomous retention policy,
unbounded memory growth, semantic open-world identity, or general continual
learning.
