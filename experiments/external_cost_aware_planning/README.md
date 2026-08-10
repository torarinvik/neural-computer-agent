# Cost-aware goal-conditioned planning

This pressure test verifies the lifetime-cost lesson imported from the
exported learning session. The external factual model is learned once from
opaque state/intention/next-state observations. Two plans reach the same
opaque goal, but one uses a high-cost intention while the other uses a cheap
two-step route. Terminal-only search is compared with the optional opaque
step-cost objective.

The controller is frozen and the factual model is not mutated during search.
The cost vector is a caller-supplied nonnegative verifier scalar; it is not a
protocol ID or a semantic action label. This qualifies cost-aware inference,
not learned cost prediction or general continual learning.

```text
uv run python experiments/external_cost_aware_planning/train.py \
  --seed 83301 \
  --report-out /tmp/external-cost-aware-planning.json
```
