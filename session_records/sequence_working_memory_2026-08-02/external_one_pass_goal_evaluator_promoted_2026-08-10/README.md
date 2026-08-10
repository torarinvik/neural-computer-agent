# One-pass learned goal evaluator — promoted

This four-seed audit replaces the repeated MLP verifier batch with
`ExternalGoalEvaluatorStatistics`, an independently versioned external memory
that stores only normal-equation sufficient statistics. It consumes graded
opaque state/goal verifier outcomes once, while a frozen factual transition
model and frozen controller derive behavior by search on held-out noisy goals.

All four seeds reached `120/120` held-out trials. Held-out positive
probabilities were at least `0.958`, held-out negative probabilities were at
most `0.107`, goal-shuffled mastery was `0.0`, corrupted-goal mastery was
`0.0`, and reward-shuffled evaluator mastery was `0.0`, `0.225`, `0.0`, and
`0.0`. Random floors were `0.033`, `0.033`, `0.017`, and `0.042`.

The evaluator consumed `648` unique graded verifier outcomes once through a
single statistics update. It stored no raw rows, replayed no goal examples,
made zero controller optimizer updates, and remained byte-stable during
planning. The factual transition model consumed `123` transition rows once;
the controller and model were frozen throughout. Exact evaluator persistence
was verified for every seed.

This promotes a bounded replay-free sufficient-statistics goal relation and
its use in held-out planning. It does not establish arbitrary nonlinear goal
abstraction, representation migration, cross-modal grounding, unrestricted
memory growth, or general continual learning. The next pressure test must
change the representation basis or frontend while retaining the goal contract,
then verify migration without replaying the old verifier outcomes.

Reports and checksums are in this directory. Reproduce one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_one_pass_goal_evaluator/train.py \
  --seed 84201 \
  --report-out /tmp/external-one-pass-goal-evaluator-84201.json
```
