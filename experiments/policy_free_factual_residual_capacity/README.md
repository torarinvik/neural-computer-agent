# Policy-free factual residual capacity

This pressure test extends the promoted factual residual stream to nine
distinct regimes plus a reversal. The shared transition model is trained once
and frozen. Ten opaque residual slots are admitted without residual optimizer
replay; after four slots, the external bank grows atomically from capacity four
to capacity eight under a retention/integrity probe.

The run then calibrates replay-free scalar reliability statistics from clean
and corrupted verifier outcomes. The learned external gate must allow a clean
known read while rejecting corrupted/out-of-distribution evidence without
mutating the residual bank or the reliability state. Shuffled reversal,
rejected capacity growth, route round-trip, exact persistence, and verified
float16/int4 compression controls are included.

This demonstrates bounded capacity-scaled factual memory and learned evidence
reliability. It does not demonstrate general continual learning, arbitrary new
computation, or unrestricted memory growth.

Run with:

```bash
PYTHONPATH=src python -m experiments.policy_free_factual_residual_capacity.train \
  --seed 101 \
  --report-out /tmp/policy-free-factual-residual-capacity-101.json
```
