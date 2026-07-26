# Persistent versus fresh sample efficiency — 2026-07-26

## Accepted result

Persistent physical experience improved reward-guided learning efficiency over
freshly regenerated banks at an equal candidate-verifier-bit budget.

Two paired replicas used:

- seeds 7032 and 7034;
- 16 banks of capacity 6;
- nine updates across old-equal, reliability-dominant, and old-return phases;
- 2,592 candidate verifier bits per arm;
- identical initial controller weights, task seeds, perturbation sequence,
  perturbation magnitude, and step size.

Absolute reward was not compared directly because fresh banks are easier. Each
arm was normalized against a frozen controller evaluated on the same physical
states.

## Aggregate evidence

- persistent normalized verified-reward AUC: `0.1041667`
- fresh normalized verified-reward AUC: `0.0312500`
- persistent/fresh reward-gain ratio: `3.33`
- persistent target-selection advantage AUC: `+1.03125`
- fresh target-selection advantage AUC: `-0.28125`

Persistence beat fresh memory on both metrics in both replicas. Binary and
four-rule retention passed in every arm. The aligned-reward experiment passed;
the existing shuffled-reward control remained rejected.

Canonical comparison:
`experiments/unified_cognitive_controller/reports/persistent_vs_fresh_efficiency_seeds7032_7034.json`

## Interpretation

This is the first short-horizon compounding sample-efficiency result: preserved
physical experience made subsequent updates more useful per verifier bit.

It is not yet cross-primitive transfer. Both arms adapted the same generic
utility residual across related utility switches. The next experiment must
introduce a gradual held-out relation and compare intact persistent memory
against empty, shuffled, and fresh memory. A valid compounding claim requires
fewer verifier bits to reach a fixed held-out accuracy while old skills remain
inside retention gates.
