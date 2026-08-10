# Universal opaque-goal reacher — promoted

This four-seed audit tests the strongest remaining lesson from the exported
games session: a finite target-specific policy can succeed by ignoring a
small goal vocabulary, while a factual transition model can derive behavior
for the current goal at inference time.

The external model consumed `123` opaque state/intention/next-state rows once
through affine sufficient statistics. It received no goal labels and no
optimizer updates occurred. The planner then searched `24` held-out goals
from `5` starting states each, using the caller-opt-in goal-progress heuristic
to preserve useful long-horizon prefixes. The finite-goal habit saw only nine
training targets.

All seeds reached `120/120` held-out trials. Goal-shuffled evaluation reached
`0/120`, the finite-goal habit reached `0/120`, and random floors were
`0.000`, `0.017`, `0.017`, and `0.033`. The controller stayed frozen, the
factual model remained unchanged during search, persistence was exact, and
replay was zero. The planner expanded `44,640` nodes per seed with roughly
`5 ms` mean search latency.

This promotes held-out goal-space generalization of behavior derived from a
replay-free factual model, plus the planner's opt-in progress heuristic. It
does not establish a learned cross-modal goal evaluator, arbitrary nonlinear
goal abstraction, unrestricted planning, or general continual learning. The
line fixture and affine model are intentionally foundational; the next rung
must repeat the pressure test with learned goal representations and richer or
partially observed dynamics.

Reports and checksums are in this directory. The experiment is reproducible
with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_universal_goal_reacher/train.py \
  --seed 84001 \
  --report-out /tmp/external-universal-goal-reacher-84001.json
```
