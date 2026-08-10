# Factored factual transition memory

This pressure test validates the reusable component introduced after the
exported session's policy-free finding. A shared transition base is trained
once and then frozen. New opaque dynamics are represented only by
context-addressed residual facts; behavior is derived by model-based search.

Four genuinely disjoint transition tables are presented in sequence. Source
regimes receive complete evidence; target regimes receive only a verifier-
private target-covering subset. The planner receives learned opaque context
tensors, never regime labels. Earlier contexts are re-evaluated after every
append, the base digest is checked, and the complete component is restored
from its independent payload.

This promotes a factored factual-memory boundary, not arbitrary routing or
general continual learning. The context encoder is pretrained, the residual
store is exact-match and finite, and the target mask is selected so the
measured planner goals remain solvable.

```text
PYTHONPATH=src:.:experiments uv run python experiments/external_factored_transition_residual/train.py \
  --seed 82701 \
  --report-out /tmp/external-factored-transition-residual.json
```
