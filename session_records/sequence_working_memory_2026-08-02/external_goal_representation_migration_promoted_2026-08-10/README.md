# Goal-memory representation migration — promoted

This four-seed audit keeps a one-pass learned goal evaluator frozen in its
original one-dimensional representation space while replacing the frontend
with a two-dimensional affine representation. A separate
`ExternalGoalRepresentationAlignmentStatistics` component learns the new to
old mapping from `96` paired representation tensors in one pass. The old
`648` verifier outcomes are not replayed.

All four seeds reached `120/120` held-out trials after migration. Held-out
positive probabilities were at least `0.999`, negatives were at most `0.032`,
shuffled-alignment mastery was `0.025`, `0.058`, `0.025`, and `0.025`, missing-
alignment mastery was `0.017` on every seed, reward-shuffled evaluator
mastery was `0.0`, `0.0`, `0.0`, and `0.15`, and corrupted-goal mastery was
`0.0` on every seed.

The controller and factual transition model stayed frozen. The old verifier
memory and alignment statistics were unchanged during search, both persisted
exactly, and each was updated once with zero replay. The replacement frontend
was deliberately wider than the old one, so the result tests an actual
interface-space change rather than a renamed tensor.

This promotes a bounded learned-alignment migration path for an external goal
memory. It does not establish arbitrary nonlinear representation migration,
unsupervised cross-modal grounding, unrestricted memory growth, or general
continual learning. The next pressure test must use nonlinear or partially
observed frontend drift and decide when alignment evidence is insufficient for
promotion.

Reproduce one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_goal_representation_migration/train.py \
  --seed 84301 \
  --report-out /tmp/external-goal-representation-migration-84301.json
```
