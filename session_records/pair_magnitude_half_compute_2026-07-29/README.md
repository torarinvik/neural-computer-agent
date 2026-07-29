# Half-compute magnitude acquisition

## Result

The fixed 388,191-parameter controller acquired the exact next
bars-to-diamonds magnitude contour at 20.8984375% from the same 128 new
lifetimes used by the previous rung. Eight internal consolidation passes were
the first robust measured prefix. Four, six, and seven passes failed; eight,
twelve, and sixteen passed. The selected budget therefore cuts acquisition
optimizer work by 50% without consuming another verifier bit.

This is a replicated consolidation-compute milestone. A separately
pre-registered forward-transfer gain gate missed by 0.0123 percentage points,
so this record does not promote a new compounding-transfer claim.

## Boundary

The previous seed-21653 controller was checked on two new 32,768-lifetime
streams:

- 20.703125% passed both;
- 20.8984375% passed one and failed one;
- 21.09375% failed both.

The experiment therefore treated 20.8984375% as the first non-robust contour.
Training changed only the latest existing 64-unit magnitude adapter; every
other tensor remained bit-identical and no parameter was added.

## Learner-visible information

The learner received rendered RGB events, its own opaque attempted actions,
scalar outcomes for those attempts, and opaque frozen-controller rehearsal.
It received no task name, relation label, correct unattempted action,
coordinates, generator state, or symbolic solution.

## Experience and compute

Every arm used one fixed packet:

- 128 new target lifetimes / 768 new verifier bits;
- 160 unique rehearsal lifetimes across ten inherited streams;
- 288 total unique lifetimes / 1,728 total unique verifier bits.

The gate-leak annealing schedule was fixed at 16 optimizer updates, independent
of run length. Consequently all compared prefixes share the same optimizer
trajectory up to their stopping point. A regression test enforces that
invariance.

| passes | target | old repertoire | complete gate |
|---:|---:|:---:|:---:|
| 4 | 89.70% | fail | fail |
| 6 | 89.92% | pass | fail |
| 7 | 90.04% | pass | fail |
| **8** | **90.06%** | **pass** | **pass** |
| 12 | 90.05% | pass | pass |
| 16 | 90.29% | pass | pass |

Seven passes crossed 90% in the headline average but failed the full causal
mastery gate. Eight is the first measured prefix whose gate remained satisfied
at every later measurement.

At eight passes the learner used 2,304 optimizer-lifetime exposures, versus
4,608 at sixteen. No additional unique training evidence was consumed.

## Replication

The fixed eight-pass recipe passed three of three fresh seeds:

| seed | target | bars | missing object two | inherited-read loss | accepted |
|---:|---:|---:|---:|---:|:---:|
| 21701 | 90.06% | 91.41% | 60.95% | 11.95 pp | yes |
| 21702 | 90.24% | 91.34% | 60.69% | 12.17 pp | yes |
| 21703 | 90.01% | 91.43% | 60.61% | 11.79 pp | yes |

Training plus the internal 16,384-lifetime audit took 39.85–42.39 seconds per
seed on one RTX PRO 6000.

## Matched causal controls

At the identical eight-pass, 128-new-lifetime budget:

| arm | target | result |
|---|---:|:---:|
| inherited, aligned outcomes | 90.01% | pass |
| reset inherited magnitude slot | 88.10% | fail |
| shuffled new verifier outcomes | 89.62% | fail |

Both accumulated prior structure and correctly paired scalar outcomes are
therefore necessary.

## Independent audit

The pre-selected seed-21702 checkpoint passed every gate on 32,768 fresh
lifetimes:

- target contour: 90.21%;
- bars: 91.45%;
- 15.625%: 91.32%;
- 20.3125%: 90.43%;
- 20.7031%: 90.38%;
- missing second object: 60.64%;
- inherited-read ablation cost: 12.09 percentage points;
- all three same/different appearances, binary mapping, visible context, and
  visible-context XOR retained.

The valid counterfactual rerenders the pixels and recomputes the correct opaque
action. It does not swap hidden states.

## Bounded forward-transfer result

The child was trained only at 20.8984375%. Parent and child were evaluated on
identical 32,768-lifetime streams:

| contour | parent | child | gain | parent gate | child gate |
|---:|---:|---:|---:|:---:|:---:|
| 20.8984% | 90.10% | 90.28% | +0.177 pp | fail | pass |
| 21.0938% | 89.90% | 90.09% | +0.188 pp | fail | pass |
| 21.2891% | 89.91% | 90.05% | +0.138 pp | fail | pass |
| 21.4844% | 89.82% | 90.08% | +0.263 pp | fail | fail |
| 21.6797% | 89.64% | 89.90% | +0.254 pp | fail | fail |

The pre-registered next-rung gate required at least +0.200 percentage points.
The measured +0.188 did not pass. The child did master two unseen contours,
but that observation is not promoted as a verified compounding gain.

## Deployed compute

The acquired skill is already compiled to one controller step per event:

| optional thoughts | normal | counterfactual | mastery |
|---:|---:|---:|:---:|
| 0 | 90.23% | 90.11% | yes |
| 1 | 89.15% | 89.24% | no |
| 2 | 87.17% | 86.95% | no |
| 4 | 85.33% | 85.10% | no |
| 8 | 83.61% | 83.80% | no |

The compute saving is entirely in acquisition consolidation, not inference.

## Artifact

- Checkpoint:
  `artifacts/checkpoints/unified_pair_magnitude_half_compute_seed21702.pt`
- SHA-256:
  `e3ae0cd90ec0dc6f2e98c829c2c064d7a6a6008b36fb982213b3b50c795e8ba9`
- Parent:
  `artifacts/checkpoints/unified_pair_magnitude_experience_consolidation_seed21653.pt`
- Raw reports: `reports/`

## Next experiment

Eight is still a fixed budget discovered by an offline prefix ladder. The next
frontier is a task-agnostic stopping policy that reads only learner-visible
optimization statistics and chooses how many consolidation passes to spend.
It must be tested across easier and harder rungs, reduce mean optimizer work
against fixed eight, and preserve every accuracy, causality, retention, and
outcome-shuffle gate.
