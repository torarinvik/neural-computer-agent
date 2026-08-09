# Long alternating nonlinear address shift

This pressure test extends the learned-context nonlinear-drift rung to four
target regimes and a long alternating sequence. A source-trained context
encoder is frozen; novel address versions adapt copy-on-write from current
evidence, while committed historical keys and factual model slots remain
immutable unless a held-out retention gate authorizes promotion.

The capacity-growth arm starts with four slots and performs retention-verified
copy-on-write growth to six slots before the fifth regime arrives.

Run one seed with:

```text
PYTHONPATH=. .venv/bin/python experiments/external_nonlinear_address_shift_stream/train.py \
  --seed 82101 \
  --report-out /tmp/external-nonlinear-address-shift.json
```

This remains a bounded pressure test. It measures immutable-address routing
under distribution shift, not unrestricted memory growth or general
continual learning.
