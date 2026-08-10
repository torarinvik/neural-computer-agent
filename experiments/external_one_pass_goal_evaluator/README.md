# One-pass learned goal evaluator

This rung replaces the repeated MLP verifier batch with
`ExternalGoalEvaluatorStatistics`, a replaceable external component that
stores only normal-equation sufficient statistics. It consumes graded opaque
state/goal verifier outcomes once and emits planner-compatible goal logits.

The same held-out noisy-goal, goal-shuffled, reward-shuffled, corrupted-goal,
random-floor, frozen-controller, persistence, and model-immutability controls
are retained. Promotion requires no replayed goal rows and a one-pass
statistics update.

This is still a bounded sufficient-statistics relation, not arbitrary
cross-modal goal abstraction or general continual learning. The next pressure
test must vary the representation basis, change the frontend while retaining
the goal contract, and test whether the learned relation can migrate without
replaying old verifier outcomes.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_one_pass_goal_evaluator/train.py \
  --seed 84201 \
  --report-out /tmp/external-one-pass-goal-evaluator-84201.json
```
