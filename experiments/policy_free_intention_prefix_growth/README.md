# Policy-free intention prefix growth

This experiment is a promoted bounded-retention pressure test for six sequential opaque
regimes. It keeps the controller frozen, appends fresh external cells to an
accumulated routed bank, and compares each successor with a matched fresh
one-cell learner without replaying earlier examples.

Run one seed with:

```bash
.venv/bin/python -m experiments.policy_free_intention_prefix_growth.train \
  --seed 85401 \
  --report-out /tmp/policy-free-prefix-growth-85401.json
```

The result is not a claim of positive transfer or general continual learning.
Promotion requires all six regimes to master, every prior content and route
prefix to remain above its floor, held-out retention qualification, causal
controls, exact persistence, and zero replay. The current bounded-retention
promotion is archived in
`session_records/policy_free_intention_prefix_growth_promoted_2026-08-10/`;
the earlier failed design remains in
`session_records/policy_free_intention_prefix_growth_rejected_2026-08-10/`.
