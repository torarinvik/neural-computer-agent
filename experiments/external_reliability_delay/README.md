# Replay-free learned reliability and delay

This pressure test separates three kinds of persistent state:

1. the factual transition model and its opaque slot identity;
2. a scalar reliability table learned once from verifier outcomes;
3. a wait/absence table learned from transport utility outcomes.

The learned reliability gate may veto a committed route, but it cannot mutate
the historical model, context key, or slot ID.  A low-error corrupted revisit
is therefore rejected while a fresh gate-disabled control still routes it.
The wait statistics learn that delayed incomplete evidence is worth waiting
for while an equally incomplete fast-absence window should be released.

Run one seed with:

```text
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_reliability_delay/train.py \
  --seed 85101 \
  --report-out /tmp/external-reliability-delay.json
```

This promotes a bounded replay-free reliability/delay boundary.  It does not
establish learned multimodal grounding, unrestricted memory growth, arbitrary
new computation, or general continual learning.
