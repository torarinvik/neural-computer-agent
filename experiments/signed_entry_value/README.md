# Signed external-entry value model

This pressure test implements the polarity lesson from the exported games
session. A reusable external value model learns a positive state-only salience
and an odd scalar polarity supplied by an opaque external entry. The source
stream contains only positive entries; the held-out target uses negative
entries and receives no target updates. A matched unfactorized value model is
trained on the same source stream as a control.

The learner sees learned state tensors, opaque entry tensors, and deterministic
scalar verifier outcomes. It receives no polarity label or target action. The
audit requires the signed model to retain source behavior, transfer to the
reversed entry, produce a neutral prediction for a zero entry, preserve exact
oddness under entry negation, and round-trip through its checksum payload.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/signed_entry_value/train.py \
  --seed 9101 \
  --report-out /tmp/signed-entry-value-9101.json
```

This promotes a reusable signed-delta value boundary, not arbitrary value
learning, general continual learning, or unrestricted memory growth.
