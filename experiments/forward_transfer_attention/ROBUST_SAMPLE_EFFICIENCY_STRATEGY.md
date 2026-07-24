# Robust Sample-Efficiency Strategy

## Evidence that changes the plan

The original 64-outcome identify-then-act result remains a valid audited
capability demonstration for seed 211.  It is not yet a robust learning
procedure.

The exact-baseline ignition map used one shared core and independently fitted
readouts at 32, 40, 48, 56, and 64 unique outcomes.  Full pixel-rerender,
missing-evidence, shuffled-action, shuffled-reward, and fresh-core controls ran
at every point.

| Seed | 32-bit normal | 48-bit normal | 64-bit normal | First full pass |
|---:|---:|---:|---:|---:|
| 151 | 49.61% | 52.73% | 55.47% | none |
| 211 | 66.80% | 84.77% | 99.61% | 64 |
| 307 | 55.86% | 72.27% | 81.64% | none |
| Mean | 57.42% | 69.92% | 78.91% | 1/3 seeds at 64 |

Mean target-reversal prediction flips rose from 30.86% at 32 bits to 66.67%
at 64 bits, still below the 75% causal gate.  Therefore no robust ignition
threshold has been demonstrated at or below 64 outcomes.

A second set of curves changed only answer-path initialization and optimizer
seed offsets.  Seed 211 then passed at 48 and 64, while seeds 151 and 307 still
failed.  Initialization matters, but changing the readout seed alone did not
rescue weak runs.  The dominant variance source has not yet been localized.

## Recommendation

Use a population, but not a classic genetic algorithm or unrestricted
population-based training yet.

The next method should be **replicated successive-halving with shared
experience**:

1. Every clone receives the same sensory transitions and verifier outcomes.
2. Clones vary one bounded factor at a time.
3. Fitness is causal learning AULC across multiple held-out streams, not the
   best final accuracy on one seed.
4. Weak clones stop cheaply; exact parents must reproduce before mutation.
5. Search compute and winner experience are reported separately.
6. Inherited weights survive only if they improve the next unseen learning
   curve.

This uses extra compute to discover a robust learner without pretending the
population consumed fewer interactions.  Near-term logged experience can be
shared by all clones because attempted actions and scalar outcomes are known.
Later on-policy populations must count each clone's distinct interactions.

## Why not a genetic algorithm yet?

A genetic algorithm would currently select lucky initializations.  Mutation
would then elaborate noise because we do not know whether variance originates
in the predictive core, the answer head, minibatch order, or the 64-lifetime
experience subset.  It also discards useful gradient information.

Evolution becomes appropriate only after:

- the variance source is localized;
- a parent passes at least two independent seeds;
- a child is judged on new seeds rather than its parent's selection set;
- mutations target a measured bottleneck.

Gradient training should remain the inner learner.  Population search should
choose initializations, objective weights, curricula, and resource allocation.

## Next sub-minute experiment: variance decomposition

Freeze three of four randomness sources and vary the fourth:

1. **Predictive-core initialization/pretraining sampling**
2. **Which unique lifetimes enter each prefix**
3. **Readout initialization**
4. **Readout minibatch sampling**

Use seeds 151, 211, and 307, with 48 and 64 outcomes, one cached sensory pass,
and the full causal audit.  This is a small factorial diagnostic, not a model
search.

Interpretation is pre-registered:

- Core variance dominates → race predictive cores and improve reward-free
  representation stability.
- Lifetime-subset variance dominates → build an outcome/action-balanced,
  task-agnostic experience curriculum.
- Readout initialization dominates → use a more stable head/optimizer or
  multiple-start consolidation.
- Readout sampling dominates → change replay sampling and optimization, not
  architecture.
- Interactions dominate → use a small factorial population, then mutate only
  the measured interaction.

## Variance race result

The nine-horse 64-bit diagnostic completed in 46.36 seconds.  Every horse saw
the same 64 unique outcomes, all behavioral readouts trained against frozen
features, and all predictive cores remained bit-identical with exactly
unchanged held-out predictive metrics.

| Factor varied | Normal-accuracy range | Causal-floor range |
|---|---:|---:|
| Predictive-core initialization | 49.61 points | 74.22 points |
| Predictive pretraining sampling | 20.70 points | 46.09 points |
| Readout initialization | 3.13 points | 7.03 points |
| Readout replay sampling | 2.73 points | 5.86 points |

Core initialization is decisively the dominant variance source.  Three horses
passed every capability, anti-fluke, and retention gate: the anchor,
pretraining-sampling seed 307, and readout-initialization seed 307.  The last
had a 100% causal floor, but it is not promoted from a single race.

The next race should therefore vary predictive-core initialization while
holding pretraining sampling, experience, and the downstream optimizer fixed.
Use successive halving at 32, 48, and 64 outcomes, then replay the exact
winning core on a disjoint lifetime stream and a second downstream seed.

## Core race and replication result

Six core initializations raced under shared experience.  Four survived at 32
outcomes and three at 48.  Seeds 43, 211, and 263 passed every blind
capability, anti-fluke, and frozen-core retention gate at 64 outcomes.

Seed 263 was the early-ignition challenger: it reached a 100% selection causal
floor at 48 bits.  The two scientific parents, seeds 211 and 263, were then
retrained on a disjoint policy stream with readout initialization/replay seed
307:

| Parent | 48-bit replicated floor | 64-bit replicated floor | Stable pass |
|---:|---:|---:|---:|
| 211 | 51.56% | 83.98% | none |
| 263 | 98.05% | 97.27% | 48 bits |

Seed 263 is therefore admitted to the old-primitive retention/compatibility
suite.  This is a population-selected, replicated sample-efficiency result;
the full search cost remains part of the accounting.  No general-agent
checkpoint is promoted until older capabilities are shown to survive
integration.

## Immutable-parent graduation

CUDA adaptive-pooling backward proved nondeterministic even with deterministic
cuDNN settings.  Therefore a seed number is not an adequate parent identity.
The selected seed-263 core was materialized once, stored as immutable weights,
and pinned by SHA-256:

`d027b80a631f61c3a9769b60a079494e0a669e1211d3324a13e5ad7b65a1006d`

Two reloads of that exact core produced metric-for-metric identical curves.
After replacing noisy random-permutation controls with exact binary
complements, the immutable parent passed every behavioral and anti-fluke gate
at 48 and 64 outcomes.  At 40 outcomes it reached 95.31% normal accuracy but
correctly failed because missing-evidence entropy was lower than normal
entropy.  The stable, reproducible threshold is therefore 48—not 40.

The compatibility suite also passed:

- fixed-probe stable mastery at 16 outcomes;
- fixed-target stable mastery at 48 outcomes, improving the prior 64-outcome
  threshold;
- predictive-core parameters bit-identical throughout behavioral learning.

This is a 25% reduction from the previous 64-outcome frontier with stronger
reproducibility and retention evidence.

## Population fitness after localization

For clone \(i\), evaluate:

`fitness_i = robust causal AULC - retention penalty - latency penalty`

where robust causal AULC is the median across seeds minus a lower-tail penalty.
The exact scalarization should be fixed before running the population.

Required reporting:

- unique verifier bits seen by the winner;
- total unique verifier bits generated for the whole search;
- replayed examples and optimizer updates per clone;
- aggregate GPU seconds;
- median, worst-seed, and pass rate;
- causal reversal, missing-evidence, shuffled-outcome, and fresh controls;
- forward transfer to the next primitive.

## What not to do next

- Do not breed the seed-211 48-bit solution.
- Do not add more readout capacity or latent adapters.
- Do not add more auxiliary predictive losses to the same core.
- Do not ensemble clones and call the ensemble sample-efficient.
- Do not increase run length before the sub-minute variance diagnostic points
  to the responsible component.
- Do not treat the best seed as the system's capability frontier.

## Evidence-driven curriculum after the immutable core

The first cross-renderer screen was intentionally capped below one minute per
primitive. Core-263, its vision-only and recurrent-only component transfers,
and its exact fresh initialization all stayed at chance on the older spatial
and same/different tasks through 120 verified outcomes. Scaling was rejected.

The corrected ladder changed one axis at a time inside the already learned
identify world:

| New task | Core-263 stable bits | Fresh stable bits | Replicated |
|---|---:|---:|---:|
| target side | 8 | none through 64 | not required before composition |
| observed effect side | 8 | none through 64 | not required before composition |
| effect matches target | 24 | none through 64 | yes, disjoint stream |

Every passing point requires valid rerender causality, missing-evidence
degradation where applicable, exact-complement controls, and frozen-core
retention. This is the first evidence that previously learned perception can
make a genuinely new composition faster to learn.

The next curriculum bridges toward cross-renderer generalization in
single-axis steps:

1. preserve left/right logic but vary cursor and target appearance;
2. preserve event structure but replace position with color identity;
3. only then retry simultaneous spatial and delayed same/different;
4. after each promotion, rerun the 8/8/24-bit retained ladder.

No rung receives a three-minute budget until a sub-minute screen shows a
causal advantage over matched fresh. No rung receives ten minutes until the
advantage replicates on a disjoint logical/render stream.
