# External transition-model consolidation

This pressure test checks the safe consolidation boundary for the external
transition bank. Two slots with identical factual behavior are verified on
held-out opaque transitions and then made to share one parameter object while
both context keys and indices remain valid. A source slot and a disjoint target
slot are presented to the same consolidation operation and must be rejected
without mutation.

The controller is frozen, consolidation performs zero optimizer updates, and
the payload round-trip must preserve aliasing and behavior. This is parameter
sharing, not semantic merging: distinct transition functions are never merged
just to reduce the slot count.

```text
.venv/bin/python experiments/external_transition_model_consolidation/train.py \
  --seed 70111 \
  --report-out /tmp/transition-model-consolidation.json
```
