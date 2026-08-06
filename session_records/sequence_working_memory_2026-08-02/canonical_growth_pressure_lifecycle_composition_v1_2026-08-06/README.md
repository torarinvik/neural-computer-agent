# Lifecycle-backed producer-to-consumer composition (2026-08-06)

This record upgrades the canonical working-memory growth pressure test so
that two independently acquired controller capabilities are composed through
the isolated `ExternalCapabilityLifecycle`. The producer artifact and the
prior-only consumer artifact are replaced transactionally by one namespaced
artifact row. Admission requires a fresh runtime to reload that row and pass
the declared held-out producer-global-parity audit.

## Promoted result

| metric | seed 69204 | seed 69205 |
| --- | ---: | ---: |
| fresh composition verifier | 0.7682 | 0.6406 |
| final composed accuracy | 0.7682 | 0.6406 |
| blank-sequence accuracy | 0.5000 | 0.5052 |
| reward-shuffled accuracy | 0.5938 | 0.4453 |
| physical rows before/after | 2 / 1 | 2 / 1 |
| replayed examples | 0 | 0 |
| controller optimizer updates | 0 during composition | 0 during composition |

Both seeds pass the narrow composition, reload, frozen-core, producer
ablation, prior-read ablation, missing-evidence, and reward-shuffled controls.
The lifecycle rejects admission below the 0.60 above-chance verifier floor;
the final report must also pass every causal gate.

## Claim boundary

This is evidence that isolated learned executable state can be composed and
verified as one external capability. It is not arbitrary program induction,
unrestricted memory growth, positive transfer across unrelated tasks, or
general continual learning. The controller still needs a broader learned
execution substrate and nonstationary transfer tests before those claims are
justified.
