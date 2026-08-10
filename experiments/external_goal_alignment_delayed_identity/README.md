# Delayed identity resolution for overlapping goal frontends

This pressure test deliberately gives two active frontend alignments the same
identity signature. The bank refuses the overlap, retains two bounded
signature records, refuses a third at quarantine capacity, blocks eviction of
referenced slots, and preserves the records across persistence. A later
disambiguating anchor is written only through an explicit verifier-approved
update; verifier rejection leaves the deferred records byte-stable, while
acceptance resolves them and permits safe eviction.

The controller, factual model, and verifier memory remain frozen. This is
bounded delayed identity evidence, not semantic open-world identity discovery.

Run with:

```bash
PYTHONPATH=. .venv/bin/python experiments/external_goal_alignment_delayed_identity/train.py \
  --seed 84901 \
  --report-out /tmp/external-goal-alignment-delayed-identity-84901.json
```
