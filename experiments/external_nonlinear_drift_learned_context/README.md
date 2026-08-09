# Learned-context nonlinear drift

This pressure test trains an opaque transition-context encoder on two source
bundles, freezes it, and then sends two nonlinear drift regimes through the
online router one row at a time. Target models use replay-free random-feature
sufficient statistics. Promotion requires held-out prediction and retention of
the earlier factual slots; the controller remains frozen.

Run one seed with:

```text
.venv/bin/python experiments/external_nonlinear_drift_learned_context/train.py \
  --seed 82001 \
  --report-out /tmp/external-nonlinear-drift-learned-context.json
```

The copy-on-write address-adaptation arm adds `--adapt-address`; it keeps the
source encoder and committed keys immutable while adapting isolated candidate
versions from current evidence.

The claim is intentionally bounded: this tests nonlinear drift and learned
address formation, not unbounded memory growth or general continual learning.
