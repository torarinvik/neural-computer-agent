# Outcome-only online alignment growth

This pressure test first masters and routes three external alignment cells.
It then introduces an unregistered fourth event transform, grows a new bridge
cell, and trains only a new router output head from scalar outcomes. Earlier
cells and router heads remain frozen; old transforms are not replayed during
the growth phase.

```text
PYTHONPATH=. .venv/bin/python -m experiments.outcome_only_online_alignment_growth.train \
  --seed 69316 \
  --report-out /tmp/online-alignment-growth.json
```

Passing this rung means bounded online admission and old-route retention. It
does not establish unrestricted growth, eviction/consolidation, or general
continual learning.
