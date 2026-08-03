# Delayed-feedback amodal promotion

This rung sends the first partial event at time `0`, the complementary event
at time `1`, samples an opaque action at time `2`, and returns scalar outcome
feedback at time `3`. The verifier target and correct action remain private.

The audit covers missing second arrivals, contradictory evidence, shuffled
partners, and random actions. This isolates temporal credit assignment and
asynchronous event accumulation. The optional `--with-memory` arm exercises
controller-owned memory reads, writes, and corruption reset, but is not a
persistent-memory capability claim because this task is solvable without it.

Run the short screen with:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.async_memory_amodal.train \
  --steps 256 --batch-size 256 --seed 31
```
