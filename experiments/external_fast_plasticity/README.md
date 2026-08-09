# External fast-weight plasticity pressure test

This pressure test isolates the proposed frozen-controller learning seam. A
memory-side delta rule receives only opaque query/value tensors and a scalar
outcome. Its fast-weight matrix is external state; the plasticity rule and the
controller are not updated during the acquisition stream.

The test acquires one opaque association, then acquires a second association
without replaying the first. It checks source retention, failed and missing
evidence no-write behavior, state persistence, and exact frozen-rule
parameters. The expected result is a bounded associative-memory primitive,
not general continual learning, arbitrary new computation, or a positive
transfer claim.

Run:

```text
.venv/bin/python -m experiments.external_fast_plasticity.train \
  --report-out /tmp/external-fast-plasticity.json --seed 69316
```
