# Verifier-gated multi-mask identity-memory consolidation

This experiment extends masked external identity memory beyond two partial
patterns. Memory grows from one to four prototypes per slot, learns three
distinct partial observations under different masks, then merges two rows on a
copy through a verifier-gated consolidation transaction.

The retention probe must preserve the original full route and all three
partial routes. The controller, transition model, verifier statistics, and
alignment adapters remain frozen; no old examples are replayed.

Run one seed with:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/external_goal_alignment_masked_consolidation/train.py \
  --seed 85301 \
  --report-out /tmp/masked-consolidation-85301.json
```

This is a bounded verifier-gated growth and consolidation result. It does not
establish autonomous compression policy, unbounded memory, semantic
open-world identity, or general continual learning.
