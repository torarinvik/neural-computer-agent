# Learned opaque goal evaluator

This pressure test extends the universal-goal rung with a learned external
goal verifier. It trains `ExternalGoalEvaluator` from deterministic graded
scalar outcomes on a finite set of noisy goal pairs, then evaluates held-out goal
values with additional representation noise. The frozen controller never
receives the verifier's training state.

The factual transition model is still consumed once through sufficient
statistics. The goal evaluator is deliberately accounted separately: this
first rung uses a repeated offline verifier batch, so it does not claim
replay-free evaluator learning. Goal-shuffled, reward-shuffled, corrupted
goal, random-floor, evaluator-persistence, model-immutability, and frozen
controller controls are required.

This is the next boundary toward cross-modal goal abstraction, not general
continual learning. The following rung should replace the repeated verifier
batch with a one-pass or sufficient-statistics goal representation and test
representation migration without replay.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_learned_goal_evaluator/train.py \
  --seed 84101 \
  --report-out /tmp/external-learned-goal-evaluator-84101.json
```
