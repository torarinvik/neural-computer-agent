# Goal-memory representation migration

This pressure test keeps a one-pass learned goal evaluator frozen in its
original one-dimensional representation space while replacing the frontend
with a two-dimensional affine representation. A separate
`ExternalGoalRepresentationAlignmentStatistics` component learns the new to
old mapping from paired tensors once; old verifier outcomes are not replayed.

The experiment retains held-out noisy-goal planning, shuffled alignment,
missing-alignment, reward-shuffled evaluator, corrupted-goal, frozen
controller, immutability, and exact-persistence controls. Promotion is a
narrow representation-migration boundary, not general continual learning.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_goal_representation_migration/train.py \
  --seed 84301 \
  --report-out /tmp/external-goal-representation-migration-84301.json
```
