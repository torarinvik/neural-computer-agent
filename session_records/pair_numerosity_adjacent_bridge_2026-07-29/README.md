# Adjacent primitive: magnitude to numerosity — 2026-07-29

## Breakthrough

The persistent controller reused its learned greater-than skill across a new
visual representation: from comparing continuous object extent to comparing
the number of disconnected objects. Only **16 new lifetimes / 96 new verifier
bits** were required to stabilize the new bridge frontier.

The selected 406,456-parameter controller kept the inherited controller frozen
and trained one 18,265-parameter zero-output successor slot. It received RGB
frames, its own opaque actions, scalar outcomes for attempted actions, latent
state, and opaque behavioral rehearsal. It never received count values,
coordinates, relation names, correct unattempted actions, or symbolic labels.

## The gradual bridge

`numerosity_appearance_blend` creates a deterministic pixel curriculum:

- `0.0`: the already-mastered continuous magnitude bars;
- `1.0`: spatially separated dot groups whose component count determines the
  answer;
- intermediate values: a continuous visual morph between the two while the
  verifier-private answer remains unchanged.

The frozen parent establishes that this is a real adjacent frontier:

| dot blend | frozen-parent accuracy | gate |
|---:|---:|:---:|
| 0.00 | 93.36% | pass |
| 0.05 | 93.54% | pass |
| 0.10 | 93.23% | pass |
| 0.20 | 89.67% | fail |
| 0.40 | 75.15% | fail |
| 0.60 | 63.37% | fail |
| 1.00 | 48.24% | fail |

Thus prior knowledge solves the first 10% with zero new experience, but not the
later representation shift.

## Acquisition and adversarial controls

Training used 16 unique 22.5%-dot lifetimes. The second replication used 32
private consolidation passes and is the selected checkpoint.

| arm | seed | held-out target | result |
|---|---:|---:|:---:|
| aligned scalar outcomes, 16 passes | 23601 | 90.18% | pass |
| shuffled scalar outcomes, 16 passes | 23601 | 85.39% | fail |
| aligned scalar outcomes, 32 passes | 23602 | 90.29% | pass |
| shuffled scalar outcomes, 32 passes | 23602 | 87.78% | fail |

Correctly aligned experience is therefore causal. The learner does not pass
from architectural perturbation, rehearsal, or compute alone.

## Independent frontier audit

One 32,768-lifetime 22.5% audit missed counterfactual accuracy by 0.015
percentage points, so 22.5% is recorded as threshold-fragile rather than
promoted. The conservative promoted frontier is **22.4% dots**.

The selected checkpoint passed three independent 32,768-lifetime streams at
22.4%:

| audit stream | child normal | child counterfactual | flip rate | parent normal |
|---:|---:|---:|---:|---:|
| 99163602 | 90.15% | 90.20% | 80.44% | 87.80% |
| 99223602 | 90.16% | 90.14% | 80.41% | 87.84% |
| 99323602 | 90.18% | 90.14% | 80.45% | 87.85% |

Every child gate passed; every frozen-parent gate failed. Blank vision remained
at chance, reversing the pixel-level count order reversed predictions, and
removing the second count field sharply reduced accuracy.

## Retention and accounting

All magnitude, relation, binary-mapping, context, and context-XOR families
remained within two percentage points of the frozen parent on exactly matched
held-out lifetimes. The selected seed's worst inherited-skill delta was
`-0.649` percentage points.

| quantity | selected run |
|---|---:|
| new unique lifetimes | **16** |
| new verifier bits | **96** |
| rehearsal lifetimes | 416 |
| total unique lifetimes | 432 |
| optimizer updates | 32 |
| trainable parameters | 18,265 |
| total controller parameters | 406,456 |

## What failed

Direct training on 100% dots plateaued at 62.69% after 128 unique lifetimes.
Inverse-propensity correction, more uniform exploration, competition-aware
losses, and staged inherited reads all performed worse. Stronger retention
penalties protected old behavior but prevented acquisition. These bounded
negatives are recorded in [`negative_results.json`](negative_results.json).

The decisive change was not a larger model or a longer run. It was making the
difficulty step small enough that the inherited magnitude representation
remained useful while new outcomes extended it.

## Conclusion

This is the first verified transfer from the magnitude lineage into an adjacent
cognitive primitive. It demonstrates the intended compounding pattern:

1. previously learned abstract comparison solves the beginning of a new
   representation with zero examples;
2. only 16 new verified experiences extend the stable frontier;
3. shuffled outcomes fail;
4. the old repertoire remains intact.

It does **not** yet demonstrate mastery of pure dot numerosity. The next
frontier is to continue the same verified bridge in very small increments,
promoting only rungs that replicate and retain prior skills.

## Artifacts

- Selected checkpoint:
  `artifacts/checkpoints/unified_pair_numerosity_adjacent_bridge_seed23602.pt`
- SHA-256:
  `da81a71c40b5a2acface987d260cf1cdccbf7f87589f40c4a07c7a46c1cde8d6`
- Parent:
  `artifacts/checkpoints/unified_pair_magnitude_repeated_compounding_seed23105.pt`
- Raw curated evidence: [`reports/`](reports/)
