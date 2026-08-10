# Multi-slot cost-aware factual-model selection

This pressure test exercises the reusable-memory seam over three persistent
external factual slots. Each slot has a different opaque transition scale and
there are three held-out opaque goals. The planner selects a stable slot by
goal-conditioned rollout, with no context/task label supplied to it. A
separately learned scalar-cost model supplies the same opaque intention costs
to every slot.

The cost-aware selector is compared with terminal-only model selection across
a three-step horizon. The controller, bank, and cost model remain frozen
during inference; all transition and cost observations are consumed once.

```text
uv run python experiments/external_multi_slot_cost_selection/train.py \
  --seed 83321 \
  --report-out /tmp/external-multi-slot-cost-selection.json
```
