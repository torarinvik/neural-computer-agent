# Outcome-only context-dependent relevance

The verifier renders one context event and two candidate events. On every
episode, either candidate is relevant; the relevant candidate is the one whose
opaque tag agrees with the context. The controller must use content-dependent
cross-token binding, because candidate identity is randomized and both
frontends are frozen before outcome training.

Promotion requires both candidate assignments, stream-order permutation, and
candidate swapping to remain accurate, while independently shuffled
cross-episode candidates and intention/action interventions fail.

The promoted three-seed rung reaches `0.9995/0.9985/0.9995` on clean,
`0.9985/0.9995/1.0000` on the two forced relevance assignments for seed 17,
and all seeds pass the `0.80` gate. Cross-episode candidate shuffling remains
near `0.51`; reward-shuffling collapses to `0.2534` on seed 17. Frontends are
frozen, so this is controller learning from opaque event relationships rather
than frontend memorization.

Run one short rung with:

```bash
PYTHONPATH=src:. .venv/bin/python -m experiments.relevance_amodal.train --steps 512 --batch-size 256 --seed 17
```
