# Universal opaque-goal reacher

This pressure test implements the strongest remaining lesson from the
exported games session. A finite set of target-specific policies can succeed
by memorizing a small habit and ignoring the runtime instruction. The
canonical path instead learns only factual state transitions once and derives
behavior by searching for each current opaque goal.

The factual model receives no goal labels. Evaluation goals are held out from
the finite-goal habit control, and every successful trajectory is also a
potential hindsight goal. The planner uses the caller-opt-in goal-progress
heuristic so long-horizon terminal-only beam search does not prune useful
prefixes. Goal-shuffled, finite-habit, random-floor, frozen-controller,
model-immutability, persistence, and zero-replay gates are recorded.

This is a foundational goal-space/generalization test, not evidence of a
learned cross-modal goal evaluator, unrestricted planning, or general
continual learning. The line fixture uses a replaceable opaque tensor
boundary and a one-pass affine sufficient-statistics model; later rungs must
repeat the test with learned goal representations and richer dynamics.

Run a seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_universal_goal_reacher/train.py \
  --seed 84001 \
  --report-out /tmp/external-universal-goal-reacher-84001.json
```
