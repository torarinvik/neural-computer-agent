# Bounded masked-memory maintenance stream

This experiment pressure-tests repeated external-memory maintenance rather
than a single transaction. It grows one identity slot from two to five
prototype rows, learns four differently masked patterns, exercises forward and
reverse online order, performs verifier-gated replacement, then verifier-gated
consolidation, and re-admits a reversed pattern after compression.

An opaque capacity planner receives a side-effect-free candidate view at the
maintenance points. Its proposal is advisory; only the memory's verified
growth, replacement, and consolidation transactions can commit state.

Run one seed with:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/external_goal_alignment_masked_maintenance_stream/train.py \
  --seed 85401 \
  --report-out /tmp/masked-maintenance-85401.json
```

This is a bounded online maintenance result. It does not yet establish a
trained capacity policy, autonomous retention/compression, unbounded memory,
semantic open-world identity, or general continual learning.
