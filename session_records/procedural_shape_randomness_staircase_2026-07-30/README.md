# Ultra-gradual procedural-shape randomness staircase

## Breakthrough

The same controller extended its audited two-item procedural-shape memory
skill from randomness 0.090 to 0.135 through increments of 0.005. This is a
50% increase in the curriculum scalar. The scalar jointly controls bounded
position, size, rotation, colour, background, and contour-deformation
variation; its starting point is already nonzero.

Every candidate is independently rerendered. Identity sequences, answer
patterns, and query permutations remain fully crossed, so neither pixel
matching nor query time reveals the answer. The learner sees only RGB, its own
opaque action, and the scalar outcome of that attempted action.

The policy was deliberately conservative:

1. Evaluate every 0.005 rung zero-shot.
2. Skip training when held-out accuracy is already at least 90%.
3. Train only the first sub-mastery rung.
4. Interleave it with the immediately preceding mastered rung.
5. Preserve the original floor and all earlier cognitive primitives.
6. If acquisition or retention becomes unstable, halve the next increment to
   0.0025 before changing architecture or increasing duration.

## Sample efficiency

| target randomness | zero-shot status | target bits to stable 90% | final |
|---:|---|---:|---:|
| 0.095 | below gate | 512 | 98.68% |
| 0.120 | below gate | 512 | 96.44% |
| 0.135 | below gate | 1,536 | 96.26% |
| 0.135 replica | below gate | 1,024 | 93.63% |
| 0.135 fresh | chance | not reached by 8,192 | 50.00% |
| 0.135 shuffled outcomes | invalid feedback | not reached by 4,096 target bits | 50.00% |

The primary inherited-to-fresh transfer ratio is therefore greater than
8,192 / 1,536 = 5.33x. The replica gives a lower bound greater than 8x.
Fresh learning did not cross the gate, so both are conservative lower bounds.

Acquisition cost did not monotonically fall: 512, 512, then 1,536 target
bits. The result establishes replicated high sample efficiency and repeated
reuse, not yet accelerating acquisition at every successive rung.

## Full retention curve

The final primary controller was rerun on 2,048 held-out lifetimes at every
representative earlier rung:

| randomness | accuracy | blank evidence | full reset |
|---:|---:|---:|---:|
| 0.000 | 100.00% | 50.00% | 50.56% |
| 0.050 | 100.00% | 50.00% | 50.42% |
| 0.090 | 99.93% | 50.05% | 50.29% |
| 0.095 | 99.93% | 50.05% | 50.49% |
| 0.115 | 99.27% | 50.15% | 50.29% |
| 0.120 | 98.73% | 50.20% | 50.24% |
| 0.130 | 97.41% | 50.27% | 49.83% |
| 0.135 | 95.70% | 50.15% | 49.98% |

Visible-identity retention and span-1 recognition are both 100%. There is no
detected catastrophic forgetting across the measured staircase.

At the trained 0.135 rung, the primary model also scores 95.92% under a valid
reversed-presentation rerender and 95.85% under a candidate-identity
counterfactual. Prediction flip rates on changed answers are 95.31% and
94.80%. The independent replica returns to chance with missing evidence and
full reset as well.

## What changed in the implementation

`train_procedural_shape_span.py` now supports:

- explicit per-axis nuisance overrides for localization;
- arbitrary comma-separated mastered scalar rehearsal rungs;
- separate accounting of target and rehearsal verifier bits;
- target, floor, and rehearsal-rung causal audits;
- stable threshold accounting based on target experience rather than total
  rehearsal experience.

## Next frontier

Continue from 0.135 with 0.005 probes. At the first sign of a larger
experience requirement, a missed retention gate, or unstable causal audit,
switch immediately to 0.0025 increments.

After robust 2D nuisance invariance, a deterministic procedural 3D generator
is the natural extension. It should independently control mesh identity,
material, lighting, scale, camera position, and X/Y/Z rotation, inspired by
interactive 3D-shape tools but implemented locally so scenes, counterfactuals,
and verifier answers remain reproducible.

Curated checkpoint:

`artifacts/checkpoints/unified_procedural_shape_randomness135_seed27441.pt`

SHA-256:

`365ff744d9b66d60e29ab895381dcdd1dfa4cec2ad317106e192e1e420e95ec5`
