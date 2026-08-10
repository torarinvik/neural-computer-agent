# Policy-free factual residual stream

This pressure test extends the promoted one-pass factual residual experiment
from one novel regime to six regimes plus a reversal. A shared transition
model is trained once and frozen. Each new regime is written to an opaque
external residual slot without residual optimizer replay, and admission
requires held-out one-step accuracy, recursive rollout accuracy, and complete
prefix retention of every earlier regime.

The run also checks that shuffled reversal evidence is not promoted, empty and
corrupted evidence do not mutate committed memory, persistence is exact, and a
storage codec is selected only when held-out residual behavior survives
decompression. Fresh nonlinear controls are accounted for with their optimizer
updates and replayed examples.

This demonstrates bounded factual-memory scaling. It does not demonstrate
general continual learning, arbitrary new computation, unlimited memory
growth, or policy learning.

Run with:

```bash
PYTHONPATH=src python -m experiments.policy_free_factual_residual_stream.train \
  --seed 101 \
  --report-out /tmp/policy-free-factual-residual-stream-101.json
```
