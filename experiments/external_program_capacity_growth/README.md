# Verifier-gated external program capacity growth

This pressure test checks the missing CPU-like operation in the external
program memory: growing the executable address space without changing the
frozen controller or erasing mastered routes. Two opaque programs are learned
from scalar verifier outcomes, capacity is expanded transactionally, and a
third slot is activated and learned afterward.

The growth receipt requires a retention probe to pass on both the source and
zero-initialized expanded state. A rejected probe must not mutate capacity or
external state. The experiment also checks persistence, a reward-shuffled
control, frozen controller/rule weights, zero optimizer updates, and zero raw
example retention. It demonstrates safe bounded external capacity growth; it
does not claim unrestricted program induction or general continual learning.

Run a replicated seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_capacity_growth/train.py \
  --seed 2303 --report-out /tmp/external-program-capacity-growth-2303.json
```
