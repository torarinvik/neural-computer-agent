# Open-world-style address stream

This pressure test starts the context encoder untrained and gives it zero
pretraining updates. Each new nonlinear transition regime must create an
isolated copy-on-write address, pass held-out factual verification, and grow
the external bank by one slot. The stream then revisits every regime in reverse
and interleaved order.

The controller is frozen and source examples are not replayed. Run from the
repository root:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/external_open_world_address_stream/train.py \
  --seed 82401 --regimes 8 \
  --report-out /tmp/external-open-world-address-82401.json
```

This is an open-world-style bounded pressure test: it removes encoder
pretraining but still uses finite capacity, a finite stream, and a fixed
random-feature factual basis. It does not establish unrestricted general
continual learning.
