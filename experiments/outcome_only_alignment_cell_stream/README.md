# Outcome-only persistent alignment-cell stream

This pressure test treats learned event alignment as external memory. A new
bridge cell is allocated for each opaque frontend transform, trained from
scalar verifier outcomes while the controller, external computation, and
decoder are frozen, then frozen and retained while later cells are learned.

The stream returns to every earlier cell after each admission and includes a
matched shuffled-outcome arm plus single-cell corruption. It tests bounded
continual alignment without replay, not arbitrary modality identification or
general continual learning.

```text
PYTHONPATH=. .venv/bin/python -m experiments.outcome_only_alignment_cell_stream.train \
  --seed 69316 \
  --report-out /tmp/alignment-cell-stream.json
```
