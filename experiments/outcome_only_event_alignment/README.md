# Outcome-only event-space alignment

This pressure test trains a small external register and decoder, freezes
them, then cyclically permutes the learned event tensor before a replaceable
event bridge. The bridge alone learns from sampled scalar verifier outcomes.

The required causal pattern is:

- source capability is mastered;
- the changed event space is below mastery before bridge adaptation;
- bridge adaptation recovers mastery from scalar outcomes;
- shuffled outcomes do not recover it;
- source external computation and decoder remain unchanged;
- the parent controller remains byte-stable.

This is a bounded representation-alignment result, not a claim of general
amodal alignment or continual learning. Run it with:

```text
PYTHONPATH=. .venv/bin/python -m experiments.outcome_only_event_alignment.train \
  --seed 69316 \
  --report-out /tmp/outcome-only-event-alignment.json
```
