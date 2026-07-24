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
