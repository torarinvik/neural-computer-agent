# Four-lifetime numerosity compounding — 2026-07-29

## Question

Can the existing magnitude→numerosity bridge advance its own numerosity
frontier using less new experience than its first acquisition, without adding
parameters or forgetting the inherited repertoire?

The parent had mastered a `22.4%` dot-appearance blend from 16 new lifetimes.
The new target was the next consistently failed rung, `23.0%`.

## Method

The learner continued the existing numerosity slot rather than appending
another specialist. The controller remained 406,456 parameters and 18,265
existing slot parameters were trainable. Each candidate received:

- four new logical lifetimes, or 24 verifier bits;
- sixteen optimizer updates over those four lifetimes;
- balanced replay of the `22.4%` numerosity frontier and all registered parent
  magnitude, relation, and unrelated skills;
- only RGB frames, opaque actions, and scalar verifier outcomes.

No semantic labels, task identifiers, correct actions, hidden state, or
unattempted outcomes entered training. Runs used PyTorch MPS locally.

## Pre-registered discovery and controls

| Seed / condition | New lifetimes | `23.0%` target | All gates |
|---|---:|---:|---:|
| 23711, real outcomes | **4** | **90.84%** | pass |
| 23711, shuffled outcomes | 4 | 87.08% | fail |
| 23712, real outcomes | **4** | **90.59%** | pass |
| 23712, shuffled outcomes | 4 | 83.82% | fail |

Real outcomes passed on 2/2 seeds; matched shuffled-outcome controls failed on
2/2. Relative to the 16-lifetime adjacent acquisition, this is a **75%
reduction in new experience** while advancing to a harder visual frontier.

## Independent target audit

Both real children and their frozen parent were evaluated on the same 32,768
unseen lifetimes.

| Controller | Normal | Counterfactual | Prediction flip | Missing second object | Accepted |
|---|---:|---:|---:|---:|---:|
| frozen parent | 89.80% | 89.80% | 79.71% | 55.10% | no |
| child 23711 | **90.59%** | **90.62%** | **81.32%** | 53.42% | yes |
| child 23712 | **90.94%** | **90.91%** | **81.96%** | 53.42% | yes |

This rules out a lucky discovery split: both children crossed the mastery and
causal-flip gates on one large, independently seeded target set while the
parent failed them.

## Full retention audit

The selected seed-23712 child was then audited on 8,192 fresh lifetimes per
condition:

- target accuracy: **90.98%**, versus parent **89.94%**;
- counterfactual and pixel-flip causality: passed;
- missing-second-object accuracy: **58.80%**, satisfying the causal-input gate;
- inherited `22.4%` numerosity frontier: improved by **1.07 points**;
- worst magnitude change: **−0.46 points**;
- worst relation-appearance change: **−0.83 points**;
- all unrelated tasks remained within the two-point parent-relative floor.

Every pre-registered target, causal, and retention gate passed.

## Measured floor

Two-lifetime training was intentionally tested but not promoted:

| Seed / condition | `23.0%` target | All gates |
|---|---:|---:|
| 23713, real outcomes | 90.15% | pass |
| 23713, shuffled outcomes | 88.30% | fail |
| 23714, real outcomes | 88.70% | fail |

Two examples were unstable at 1/2 real seeds. Four examples are therefore the
current replicated floor, not a claim that two can never work.

The first two-example shuffle attempt exposed a control bug: permuting two
balanced rows could leave the outcomes unchanged. The shuffler now guarantees
a changed label arrangement while preserving the exact label histogram, and
the invalid run is excluded from this record.

## Conclusion

This is the first directly measured **within-primitive compounding gain** in the
numerosity lineage. The controller reused an acquired magnitude/numerosity
representation to master a harder rung from four new experiences instead of
sixteen, with no new parameters and no registered forgetting.

The result does not yet show that each subsequent rung will continue becoming
cheaper, nor that numerosity learning accelerates unrelated primitives. The
next frontier is a prospective repeated continuation: advance another small
rung with the same four-lifetime budget, then test whether two lifetimes become
stable only after still more inherited experience.

## Selected artifact

- `unified_pair_numerosity_compounding_seed23712.pt`
- SHA-256:
  `467c732d5e0d49cdc84a70dedd4510e53baa73ca7a6dbe938b857f2c29e5317b`

Machine-readable discovery, control, independent-audit, and retention reports
are preserved in [`reports/`](reports/).
