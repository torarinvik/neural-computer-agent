# Repeated compounding magnitude acquisition — 2026-07-29

## Breakthrough

The unchanged 388,191-parameter controller stabilized its next genuinely
unmastered magnitude contour at `22.65625%` from only 44 new lifetimes / 264
new verifier bits. Its immediately preceding acquisition required 96 new
lifetimes / 576 bits. Accumulated skill therefore reduced new external
experience by a further **54.2%** while solving a harder visual frontier.

This is the second consecutive verified experience reduction in the magnitude
lineage:

| acquisition | new lifetimes | change |
|---|---:|---:|
| preceding bridge rung | 128 | — |
| first compounding rung, 21.484375% | 96 | -25.0% |
| **second compounding rung, 22.65625%** | **44** | **-54.2%** |

Accuracy, causality, and retention remained hard gates. The learner used twelve
private consolidation passes; no parameter or symbolic interface was added.

## Learner boundary

The learner received only rendered RGB streams, its own opaque sampled
actions, scalar verifier outcomes for attempted actions, its own latent state,
and opaque rehearsal of retained behavior. It did not receive object
coordinates, contour blend, task or relation names, correct unattempted
actions, generator state, or symbolic solutions.

## Genuine frontier

The parent was trained through contour `55/256` (`21.484375%`). It generalized
without new experience through `56/256` and `57/256`, then failed the first
scan at `58/256` (`22.65625%`). Three fresh 16,384-lifetime preflights at
`58/256` all failed:

| seed | normal | counterfactual | mastery |
|---:|---:|---:|:---:|
| 73101 | 89.81% | 90.08% | no |
| 73102 | 90.02% | 89.94% | no |
| 73103 | 89.92% | 89.75% | no |

This prevents already-mastered ability from being credited as acquisition.

## Stable experience threshold

The first 48-lifetime pilot passed and replicated 3/3. The experience floor was
then bracketed with progressively smaller rungs:

| new lifetimes | complete fresh-seed gates | interpretation |
|---:|---:|---|
| 32 | 0/1 | target failed |
| 40 | 0/1 | target failed |
| 42 | 1/3 | unstable; one target and one retention failure |
| **44** | **3/3** | **first stable threshold** |
| 48 | 3/3 | confirmed but dominated |

The selected 44-lifetime target scores were `90.35%`, `90.11%`, and `90.43%`.
Every magnitude, relation, unrelated-skill, missing-evidence, and inherited-read
gate passed.

## Experience and compute accounting

| quantity | first compounding rung | new rung | change |
|---|---:|---:|---:|
| new lifetimes | 96 | **44** | **-54.2%** |
| new verifier bits | 576 | **264** | **-54.2%** |
| replay lifetimes | 132 | 144 | +9.1% |
| total unique lifetimes | 228 | **188** | **-17.5%** |
| total unique verifier bits | 1,368 | **1,128** | **-17.5%** |
| consolidation passes | 12 | 12 | unchanged |
| optimizer-lifetime exposures | 2,736 | **2,256** | **-17.5%** |

The additional replay stream protects the preceding 21.484375% contour. Even
with that stricter retention burden, both external evidence and total optimizer
exposure fell.

## Causal controls

At the identical 44-lifetime / twelve-pass budget:

| arm | target | result |
|---|---:|:---:|
| inherited knowledge + aligned outcomes | 90.11–90.43% | 3/3 pass |
| reset inherited magnitude knowledge | 87.62% | fail |
| shuffled new verifier outcomes | 88.66% | fail |

The acquisition therefore requires both accumulated skill and correctly
aligned new experience.

## Independent audit

The selected seed-23105 checkpoint passed a 32,768-lifetime audit:

- target overall accuracy: `90.259%`;
- target pixel-counterfactual accuracy: `90.277%`;
- counterfactual prediction flips: `80.971%`;
- original bars accuracy: `91.250%`;
- missing-second-object accuracy: `61.138%`;
- inherited-read ablation cost: `11.809` percentage points;
- every trained magnitude contour and every relation/unrelated skill retained.

## Matched population audit

Eight additional 16,384-lifetime streams compared the selected child with its
untouched parent:

| metric | untouched parent | 44-lifetime child |
|---|---:|---:|
| mastery streams | 2/8 | **8/8** |
| mean normal/counterfactual accuracy | 90.010% | **90.278%** |
| mean paired gain | — | **+0.2683 pp** |
| minimum paired gain | — | **+0.2060 pp** |

Every paired stream improved. The result is robust frontier stabilization, not
an isolated threshold crossing.

## Conclusion

This closes a second consecutive compounding step. The controller first spent
128, then 96, and now only 44 new lifetimes to stabilize progressively harder
contours. The evidence does not yet prove unbounded compounding across unrelated
task families, but it does prove that persistent acquired structure can more
than halve the next frontier's verified experience requirement without
forgetting.

## Artifact

- Checkpoint:
  `artifacts/checkpoints/unified_pair_magnitude_repeated_compounding_seed23105.pt`
- SHA-256:
  `c136841d60a5220bd09cd12029b6d59d903dc73d5deddb39248e7327ae48f2a2`
- Parent:
  `artifacts/checkpoints/unified_pair_magnitude_compounding_seed22022.pt`
- Raw curated evidence: [`reports/`](reports/)

## Next frontier

Test whether the experience decline survives a third acquisition and then an
adjacent cognitive primitive. Continue to optimize verified new experience
first; optimize private consolidation and inference latency only after the
capability and retention gates pass.
