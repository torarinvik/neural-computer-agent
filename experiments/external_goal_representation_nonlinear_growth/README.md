# Nonlinear goal alignment growth

This rung tests whether a nonlinear alignment memory can grow without replay.
An initial 16-feature adapter consumes one sparse alignment slice and fails
held-out verification. A copy-on-write growth transaction expands it to 80
frozen random features while retaining its prior predictions; the adapter then
consumes only new alignment pairs and must recover the held-out relation.

Promotion requires the growth retention receipt, post-growth held-out
verification, planning mastery, exact persistence, frozen controller/model,
and zero replay of the initial alignment rows.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_goal_representation_nonlinear_growth/train.py \
  --seed 84601 \
  --report-out /tmp/external-goal-representation-nonlinear-growth-84601.json
```
