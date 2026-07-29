# Fixed-parameter experience consolidation

## Result

A 388,191-parameter controller extended its learned relative-magnitude
representation from a 15.625% bars-to-diamonds contour morph to 20.3125%
without adding parameters. The successful schedule used 128 new lifetimes
(768 new verifier bits), one equally sized balanced rehearsal packet, and 16
optimizer passes over that fixed experience packet. It passed acquisition,
causality, and full-repertoire retention on three of three seeds.

This is evidence that, after acquiring useful prior structure, the controller
can trade private computation for verifier experience: consolidating a diverse
packet repeatedly was more effective than continuously replacing it with fresh
events. It is not yet a general claim across task families or a stable
bits-to-threshold curve.

## Learner-visible information

The learner received only:

- rendered RGB events;
- its own opaque attempted actions;
- scalar verifier outcomes for those attempts;
- opaque frozen-controller rehearsal.

It received no semantic task or relation identity, correct unattempted action,
coordinates, generator state, or symbolic solution. Only the latest existing
64-unit magnitude adapter was plastic; every other controller tensor remained
bit-identical. The parameter count stayed fixed at 388,191.

## Localization before repair

The promoted 15.625% parent first failed near 20% morph. Simply doubling
fixed-size refinement to 512 new lifetimes did not solve it:

| seed | new lifetimes | target | bars | accepted |
|---:|---:|---:|---:|:---:|
| 21621 | 512 | 89.29% | 89.87% | no |
| 21622 | 512 | 89.65% | 90.42% | no |

A four-step stationary plateau also damaged the unchanged parent: on identical
events the parent scored 91.51% bars, 91.29% at 15.625%, and 90.09% at the
boundary; refinement reduced the first two to 89.88% and 89.27%.

The gradient probe explained why. At initialization, the new-task gradient
norm was 0.36–1.73, while the frozen-teacher preservation gradient norm was
only `2.9e-8`–`5.2e-8`. Student and teacher initially agree exactly, so
distillation supplies essentially no proactive constraint on the first
destructive update. Larger rehearsal weight helped but only reactively.

## Selected schedule

The trainer now separates unique generated batches from optimizer reuse with
`--epochs-per-batch`. The selected recipe was:

```text
new packet:       128 target lifetimes
rehearsal packet: 128 lifetimes across 8 inherited streams
optimizer passes: 16 over the same packet
learning rate:    0.001
retention weight: 8.0
plasticity:       refine the latest 64-unit slot in place
```

Accounting per seed:

- 128 new logical lifetimes / 768 new verifier bits;
- 128 unique rehearsal lifetimes / 768 rehearsal verifier bits;
- 256 total unique lifetimes / 1,536 total unique verifier bits;
- 16 optimizer updates / 4,096 lifetime exposures;
- 28.71–36.28 seconds for training plus the internal audit.

The larger single packet mattered. Four 32-lifetime packets reused four times
were seed-sensitive, while one balanced 128-lifetime packet reused 16 times
passed all three standardized seeds:

| seed | target | normal | reversal | counterfactual | bars | accepted |
|---:|---:|---:|---:|---:|---:|:---:|
| 21651 | 90.02% | 90.16% | 90.43% | 90.23% | 91.71% | yes |
| 21652 | 90.15% | 90.29% | 90.33% | 90.13% | 91.48% | yes |
| 21653 | 90.47% | 90.65% | 90.37% | 90.28% | 91.65% | yes |

This is a replicated endpoint budget, not a claimed minimum stable
bits-to-threshold.

## Matched controls

| arm | new lifetimes | optimizer updates | target | result |
|---|---:|---:|---:|:---:|
| experienced + consolidation | 128 | 16 | 90.47% | pass |
| reset inherited magnitude slot | 128 | 16 | 89.08% | fail |
| experienced, one pass only | 128 | 1 | 90.08% | causal gate fail |
| experienced, 16 fresh 32-row batches | 512 | 16 | 89.90% | fail |
| shuffled new verifier outcomes | 128 | 16 | 89.48% | fail |

The one-pass arm's headline average crossed 90%, but reversal (89.87%) and the
valid counterfactual (89.81%) did not; it was correctly rejected. The reset
arm establishes that the inherited skill matters. The outcome-shuffled arm
preserved the pixel and outcome marginals but broke their pairing, establishing
that aligned verified experience causes the update.

## Independent causal audit

The selected seed-21653 checkpoint passed a fresh 32,768-lifetime audit:

- trained 20.3125% contour: 90.22%;
- bars retention: 91.74%;
- missing second object: 60.61%;
- inherited-read ablation: 78.50%, an 11.71-point loss;
- mastered 15.625% contour, all three same/different appearances, binary
  mapping, visible context, and visible-context XOR all retained their full
  gates.

The valid counterfactual rerenders pixels, reverses which object is larger,
and recomputes the correct opaque action. No hidden-state swap is used.

## Forward frontier

Parent and child were evaluated on identical fresh 32,768-lifetime streams.
The child was trained only at 20.3125%.

| morph | parent | child | parent mastered | child mastered |
|---:|---:|---:|:---:|:---:|
| 20.3125% | 89.82% | 90.36% | no | yes |
| 20.5078% | 89.66% | 90.17% | no | yes |
| 20.7031% | 89.59% | 90.12% | no | yes |
| 20.8984% | 89.41% | 89.91% | no | no |
| 21.0938% | 89.42% | 89.93% | no | no |

The pre-registered next unseen rung required at least +0.2 percentage points.
The measured gain was +0.51 points, and the child mastered two unseen rungs.
The next exact frontier is therefore 20.8984375%.

## Compute after acquisition

With the learned checkpoint and evidence fixed:

| optional thoughts/event | normal | counterfactual | mastered |
|---:|---:|---:|:---:|
| 0 | 90.25% | 90.40% | yes |
| 1 | 89.18% | 89.29% | no |
| 2 | 87.08% | 87.31% | no |
| 4 | 85.47% | 85.64% | no |
| 8 | 84.30% | 84.25% | no |

The skill is already compiled to the physical minimum of one controller pass
per event. Acquisition should therefore optimize verifier experience first;
optional recurrent compute cannot improve this deployed skill.

## Curated artifact

- Checkpoint:
  `artifacts/checkpoints/unified_pair_magnitude_experience_consolidation_seed21653.pt`
- SHA-256:
  `ffb09143b452f5b9e94b74bc382cf82e83b0e80d7edb1638631d60d4b8d3d6ce`
- Parent:
  `artifacts/checkpoints/unified_pair_magnitude_gradual_bridge_seed21515.pt`
- Raw reports: `reports/`

## Boundary and next experiment

The result establishes fixed-parameter extension of one learned visual concept
and a causal advantage from consolidating a sufficiently diverse experience
packet. It does not establish that 768 bits is the minimum stable threshold,
that the same schedule transfers to unrelated primitives, or that full
diamonds are solved.

The next experiment should advance from 20.7031% to the exact 20.8984375%
failure point. It should first reuse the selected packet-consolidation recipe,
then compare adaptive stopping against the fixed 16-pass budget. Compute may
be reduced only if the full accuracy, causality, retention, and next-rung
transfer gates remain intact.
