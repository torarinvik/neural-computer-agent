# Factorized masked external routing — 2026-08-10

This audit promotes the next reusable-computation increment after the
overlapping-mask routing result. The controller and state adapter remain
frozen. The external intention learner separates a mask-stable nonlinear
content path from a learned value-only context residual. Observation masks
remain available to routing and retention, but cannot mutate the reusable
hidden content representation.

The residual is external state: it is learned from delayed scalar verifier
outcomes, copied on write, protected by held-out verification, persisted with
the generator schema, and never added to the controller. A fresh-cell control
uses the same factorized architecture, so the transfer comparison measures
reuse rather than an architecture mismatch.

Both seeds pass the full promotion gate: delayed/noisy scalar credit,
caller-free routing to a new cell, overlapping-mask transfer faster than a
matched fresh learner, reversal, shuffled-outcome/action controls,
missing-evidence no-op, corruption detection, exact reload, protected
retention, frozen controller/adapter, and zero replay.

| seed | successor updates | matched fresh | transfer ratio | unique verifier bits |
| ---: | ---: | ---: | ---: | ---: |
| 85301 | 9 | 26 | 2.8889 | 393 |
| 85302 | 11 | 20 | 1.8182 | 396 |

Reproduce from the repository root:

```bash
.venv/bin/python -m experiments.policy_free_intention_routing.train \
  --seed 85301 \
  --masked-context \
  --mask-curriculum overlapping \
  --factorized-context-residual \
  --report-out /tmp/policy-free-intention-factorized.json
```

This promotes factorized external content/residual reuse for the bounded
overlapping-mask regime. It does not promote gradual or multi-stage evidence
growth, unrestricted memory growth, learned compression, arbitrary new
computation, or general continual learning. The strict versioned multi-stage
pressure test remains rejected because the warm learner still ties the fresh
learner at the forced final curriculum boundary.
