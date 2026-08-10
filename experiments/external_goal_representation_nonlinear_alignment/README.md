# Nonlinear goal representation alignment

This rung follows the rejected nonlinear-drift candidate with a bounded
nonlinear external adapter. A frozen random-feature basis learns the
replacement frontend-to-old-memory map from one pass of paired tensors; the
old goal verifier outcomes remain untouched and unreplayed.

Promotion requires held-out alignment verification, held-out planning, exact
persistence, a frozen controller/model, and a shuffled-alignment control. The
linear adapter is retained as a rejected baseline on the same nonlinear
representation.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_goal_representation_nonlinear_alignment/train.py \
  --seed 84501 \
  --report-out /tmp/external-goal-representation-nonlinear-alignment-84501.json
```
