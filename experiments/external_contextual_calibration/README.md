# Context-isolated external calibration

This pressure test follows the global-calibration failure: a single scalar
calibration state adapted to a shifted verifier regime but damaged the source
regime. It keeps the evidence evaluator and controller frozen, then updates
only the scalar calibration state addressed by an opaque context vector.

The target phase receives one deterministic verifier outcome at a time. It
does not replay target examples, update the controller, or expose a regime
label to the deployed boundary. The source slot is measured before and after
target adaptation, and the contextual state is round-tripped through its
external payload.

```text
.venv/bin/python experiments/external_contextual_calibration/train.py \
  --seed 69801 \
  --report-out /tmp/external-contextual-calibration.json
```

This is a bounded continual-memory result, not general continual learning:
the calibrated state grows by context slots and the base evaluator remains
fixed. The next pressure test is online context discovery plus compaction
under a larger stream of verifier regimes.
