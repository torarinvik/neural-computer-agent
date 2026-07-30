# Procedural shape `next item`: gradual curriculum and stability frontier

## Verified progress

The same controller that mastered direct lookup and `previous item` learned the
first `next item` anchor using only RGB streams, its own opaque actions, and
scalar attempted-action outcomes.

- span-2 atom: 96.45% overall, 98.83% causal conflicts;
- span-3 first anchor, one query: 95.90%;
- first anchor at the second query position: 95.62%;
- first anchor across all three query positions: 98.33% overall, 97.77%
  `next`, 97.46% causal conflicts;
- previous-item retention at that point: 99.22%.

The span-2 atom required 11,520 target verifier outcomes, versus 14,592 for
the earlier `previous item` atom. This is promising transfer, not yet an
independent replicated sample-efficiency result.

The research checkpoint `seed41151` retains the mastered first anchor at
98.89% and previous-item behavior at 99.29%, while carrying the partially
learned second-anchor frontier. It is intentionally not promoted as mastery.

## Localization of the second-anchor failure

The raw second anchor initially remained near chance. A deterministic control
made item three redundant with item one while preserving the second-anchor
query. Accuracy immediately became 98.75% (`next`) and 98.31% on causal
conflicts without training. Therefore the controller had learned the new
relative anchor; the failure was binding it to a genuinely independent third
memory item.

A 2.5% mixture of fully independent third items was the wrong gradualization.
It produced superficially high 99% aggregate accuracy while the independent
subgroup stayed near chance and its conflicting subgroup fell as low as 0%.
Loss weights of 4, 8, and 16 did not repair this. The strict gate now includes
both independent and conflicting-independent subgroup accuracy, so the easy
majority cannot hide this failure again.

## Target-aligned bridge

A new verifier-balanced bridge asks for the same stored third item through
either direct lookup or `next from item two`. It removes cue disambiguation
temporarily while leaving the learner-visible protocol unchanged.

Zero-shot `next` accuracy was 58.12%. After only 3,072 target outcomes it
reached 84.33% (87.11% on independent conflicting slots), proving that the
bridge supplies a strong learning signal. However, previous-item behavior
fell to 89.10%. A gentler run reached 71.94% from 1,536 target outcomes but
again reduced previous-item causal conflicts to 81.46%.

A subsequent old-skill consolidation phase restored previous-item behavior to
97.80%, but the new bridge fell back to 54.77%. The abilities currently
compete in shared parameters; alternating acquisition and consolidation does
not retain both.

## Protected-plasticity experiments

Three bounded controls separated safety from learnability:

| method | new aligned-next | previous overall | previous conflict |
|---|---:|---:|---:|
| ordinary updates | 71.94% | 89.36% | 81.46% |
| frozen gated action adapter | 57.25% | 99.23% | 98.48% |
| usage protection strength 3 | 64.84% | 91.88% | 77.50% |
| usage protection strength 10 | 59.05% | 97.57% | 94.31% |

The new generic usage mechanism stores a per-parameter exponential moving
average of gradient use. Recent use lowers plasticity; importance decays when
unused. The importance state is checkpointed. This directly tests the proposed
volatility idea, but the scalar-only rule exposes rather than solves the
stability-plasticity tradeoff.

## Conclusion and frontier

Successful:

1. decompose `next item` by anchor and query position;
2. audit rare logical subgroups rather than aggregate accuracy;
3. use target alignment to create dense, informative experience;
4. keep acquisition runs under one minute until both learning and retention
   move correctly.

Rejected:

1. sparse mixtures of maximally hard examples;
2. larger novelty weights;
3. acquisition followed by consolidation;
4. a frozen bolt-on adapter;
5. scalar gradient suppression as the complete solution.

The next experiment should be direction-aware protected plasticity: compare
the new-skill gradient with rehearsal gradients and redirect only components
that would increase old verified losses. It should reuse the successful
target-aligned bridge, retain the subgroup gates, and remain sub-minute.

