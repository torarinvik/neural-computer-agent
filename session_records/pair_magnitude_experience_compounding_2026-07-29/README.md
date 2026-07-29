# Compounding magnitude acquisition — 2026-07-29

## Breakthrough

The fixed 388,191-parameter controller used accumulated magnitude knowledge to
stabilize the first genuinely unmastered contour at `21.484375%` from only 96
new lifetimes / 576 new verifier bits. The preceding acquisition rung used 128
new lifetimes / 768 bits. This is a replicated 25% reduction in new external
experience.

The saving is purchased honestly with private computation. Eight consolidation
passes were insufficient at the new frontier; twelve passed three of three
fresh seeds. Accuracy and retention remain hard gates, unique verifier
experience is optimized next, and internal compute breaks ties only after those
conditions are satisfied.

## Learner boundary

The learner received:

- rendered RGB event streams;
- its own opaque sampled actions;
- scalar outcomes for those attempted actions;
- opaque frozen-controller rehearsal of retained behaviors.

It did not receive a task name, relation label, contour blend, object
coordinates, correct unattempted action, generator state, or symbolic solution.
The verifier used private metadata only to construct fresh balanced streams,
valid pixel counterfactuals, ablations, and held-out scores.

## Why this is the real frontier

An attractive first candidate at `21.09375%` was discarded. Although 96
lifetimes passed three training seeds, the untouched parent mastered all three
matched verifier streams. That was capability confirmation, not acquisition.

The earlier 32,768-lifetime frontier audit identified `21.484375%` as the first
unmastered contour. On three new preflight streams at that exact contour, the
untouched parent failed all three:

| seed | parent normal | parent counterfactual | mastery |
|---:|---:|---:|:---:|
| 22021 | 89.85% | 89.98% | no |
| 22022 | 89.76% | 89.99% | no |
| 22023 | 89.85% | 89.98% | no |

## Gradual evidence/compute ladder

The experience count and consolidation budget were increased separately:

| new lifetimes | passes | fresh seeds passed | interpretation |
|---:|---:|---:|---|
| 64 | 8 | 0/3 | task leap too large at this evidence budget |
| 96 | 8 | 1/3 | useful signal, not reliable mastery |
| **96** | **12** | **3/3** | selected operating point |

The selected three target accuracies were `90.34%`, `90.26%`, and `90.11%`.
Every complete magnitude, causality, relation-retention, and unrelated-skill
gate passed.

## Experience and compute accounting

| quantity | prior rung | new rung | change |
|---|---:|---:|---:|
| new lifetimes | 128 | **96** | **-25.0%** |
| new verifier bits | 768 | **576** | **-25.0%** |
| replay lifetimes | 160 | **132** | -17.5% |
| total unique lifetimes | 288 | **228** | **-20.8%** |
| total unique verifier bits | 1,728 | **1,368** | **-20.8%** |
| consolidation passes | 8 | 12 | +50.0% |
| optimizer-lifetime exposures | 2,304 | 2,736 | +18.8% |

This is exactly the intended hierarchy: spend inexpensive private processing
to extract more reusable capability from less scarce verified experience.
Inference remains a separate resource axis.

## Causal controls

At the identical 96-lifetime / 12-pass budget:

| arm | target | result |
|---|---:|:---:|
| inherited knowledge + aligned outcomes | 90.11–90.34% | 3/3 pass |
| reset inherited magnitude slot | 87.95% | fail |
| shuffled new verifier outcomes | 89.39% | fail |

The result therefore needs both accumulated prior skill and correctly paired
new scalar experience.

## Independent audit

The selected seed-22022 checkpoint passed a 32,768-lifetime audit:

- target normal accuracy: `90.45%`;
- target pixel-counterfactual accuracy: `90.43%`;
- counterfactual prediction flips: `81.31%`;
- bars magnitude retention: `91.15%`;
- retained intermediate contours: `90.69–91.29%`;
- missing-second-object accuracy: `61.06%`;
- inherited-read ablation cost: `12.33` percentage points;
- all three same/different appearances, binary mapping, visible context, and
  visible-context XOR retained.

## Matched population audit

One isolated parent stream narrowly crossed the threshold during the
independent audit, so the acquisition claim was tested on eight additional
matched streams rather than inferred from one point.

| metric | untouched parent | trained child |
|---|---:|---:|
| mastery streams | **0/8** | **8/8** |
| mean normal/counterfactual accuracy | 89.873% | 90.341% |
| mean paired gain | — | **+0.4677 pp** |
| minimum paired gain | — | **+0.4506 pp** |

Every child stream improved. The result is therefore robust frontier
stabilization, not a lucky threshold crossing.

## Adaptive-stopping result

A task-agnostic learned stopping rule was investigated before this acquisition
run. Low-resolution 4,096-lifetime gates produced false pass/fail labels near
90%. Rechecking disputed prefixes on 32,768 lifetimes reversed the apparent
failures. Two passive probes then used:

1. loss, retention, locality, and optimization progress; and
2. the controller's sensory latent plus attempted-outcome variance, entropy,
   per-trial accuracy, and confidence margins.

Both were calibrated to forbid training-set failures. The richer probe still
made seven unsafe held-out one-pass stops across 18 streams. It is rejected.
Stable generalization readiness at this razor-thin boundary is not predictable
from one observed packet with the current features. Fixed twelve-pass
consolidation remains the promoted policy for this rung.

This negative is useful: local loss, sensory difficulty, and internal
consistency must not be mistaken for long-horizon causal learning value.

## Artifact

- Checkpoint:
  `artifacts/checkpoints/unified_pair_magnitude_compounding_seed22022.pt`
- SHA-256:
  `5aa030f0fb11d0765752f05cf6c6ecb6334ee31fa1b12a41eeef2603212fe1d4`
- Parent:
  `artifacts/checkpoints/unified_pair_magnitude_half_compute_seed21702.pt`
- Raw curated evidence: [`reports/`](reports/)

## Next frontier

Continue the contour curriculum from this checkpoint. Optimize unique
experience first on the next parent-unmastered rung; then search consolidation
prefixes. Learned stopping should be revisited only after a broader collection
of tasks provides a stable and causally predictable compute-allocation signal.
