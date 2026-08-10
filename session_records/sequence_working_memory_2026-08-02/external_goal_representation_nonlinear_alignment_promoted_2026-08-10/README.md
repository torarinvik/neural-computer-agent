# Nonlinear goal representation alignment — promoted

This four-seed audit follows the rejected linear drift candidate with a
bounded nonlinear external adapter. A frozen random-feature basis learns the
replacement frontend-to-old-memory map from `48` paired tensors in one pass;
the old `648` goal verifier outcomes remain frozen and unreplayed.

All four seeds passed held-out alignment and planning gates. Migrated mastery
was `1.000`, `0.992`, `0.975`, and `0.992`. Nonlinear alignment held-out MSE
was `0.00033`, `0.00446`, `0.00070`, and `0.00114`, all below the `0.005`
tolerance. The linear candidate on the same nonlinear representation was
rejected on every seed with held-out MSE near `0.30`. Shuffled nonlinear
alignment mastery stayed between `0.025` and `0.067`.

The adapter uses frozen random features plus one-pass normal-equation
statistics. The controller, factual transition model, and old goal verifier
memory stayed frozen; the adapter was unchanged during search and restored
exactly. This promotes a bounded nonlinear alignment basis, not arbitrary
nonlinear computation, unrestricted frontend growth, or general continual
learning. The next pressure test is basis growth or quarantine when a new
nonlinear frontend exceeds the current random-feature capacity.

Reproduce one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_goal_representation_nonlinear_alignment/train.py \
  --seed 84501 \
  --report-out /tmp/external-goal-representation-nonlinear-alignment-84501.json
```
