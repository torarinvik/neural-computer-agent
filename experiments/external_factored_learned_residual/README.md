# External learned factored residual pressure test

This experiment pressure-tests the next boundary after exact residual-memory
retention. A shared transition base is trained on one opaque source regime and
then frozen. Two online regimes use the same state/intention interface, but
the second adds a different affine residual relation. The router receives only
partial current evidence for each online regime, trains an isolated external
residual learner, and promotes it only after an independent held-out factual
check and prior-slot retention probe.

The controller and context encoder are frozen. Source and target evidence are
disjoint from base pretraining; target adaptation receives no source rows.
Revisits alternate after promotion, and a shuffled-outcome candidate must be
rejected. The affine residual learner is selected because the fixture's
residual family is affine; the MLP backend remains available for nonlinear
fixtures and is tested separately in `tests/test_world_model.py`.

Run one seed with:

```bash
PYTHONPATH=src uv run python experiments/external_factored_learned_residual/train.py \
  --seed 81011 --report-out /tmp/external-factored-learned-residual.json
```

This is a bounded factual-learning pressure test. It does not claim arbitrary
new computation, unrestricted memory growth, or general continual learning.
