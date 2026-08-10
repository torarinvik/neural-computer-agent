# Concurrent external goal-alignment bank

This pressure test keeps the controller and one-pass verifier memory frozen
while four opaque frontend spaces compete for two external alignment slots.
Two valid alignments coexist and remain usable; a shuffled candidate is
rejected and quarantined; a valid third frontend is refused at active capacity,
then promoted from quarantine after a stable-ID eviction passes a retention
gate. The bank persists exact adapter state and keeps the failed candidate
isolated.

Run one seed with:

```bash
PYTHONPATH=. .venv/bin/python experiments/external_goal_alignment_bank/train.py \
  --seed 84701 \
  --report-out /tmp/external-goal-alignment-bank-84701.json
```

This is bounded concurrent external-memory evidence. It does not establish
unrestricted growth, automatic semantic frontend identification, arbitrary new
computation, or general continual learning.
