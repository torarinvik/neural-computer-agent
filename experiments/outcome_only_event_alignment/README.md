# Outcome-only event-space alignment

This pressure test trains a small external register and decoder, freezes
them, then transforms the learned event tensor before a replaceable event
bridge. The bridge alone learns from sampled scalar verifier outcomes. The
default is a cyclic permutation; `composed_orthogonal` applies a fixed opaque
dense orthogonal mix and is the stronger representation-drift rung.

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

Run the composed transform with:

```text
PYTHONPATH=. .venv/bin/python -m experiments.outcome_only_event_alignment.train \
  --seed 69316 \
  --bridge-event-mode composed_orthogonal \
  --report-out /tmp/outcome-only-composed-event-alignment.json
```
