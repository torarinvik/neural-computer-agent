# Cross-family repertoire breakthrough — 2026-07-29

## Question

Can the existing one-controller lineage acquire a genuinely different visual
primitive—not another single-glyph hidden mapping—using only rendered RGB, its
own opaque attempted action, and scalar verifier outcome, while retaining all
three inherited skills?

The new `pair_relation` primitive renders two simultaneous objects. Their
identity relation changes independently on every event. The uniquely correct
opaque action depends on whether the objects are the same or different. No
semantic relation label, task ID, correct unattempted action, coordinates, or
hidden state enters training.

## Audit caught the first shortcut

The first generator accidentally held the relation constant across a lifetime.
Normal and reversed accuracy appeared excellent, but blank-vision performance
failed the chance gate: one outcome let recurrence predict every later event
without reading the objects. That candidate was rejected.

The corrected generator balances same/different independently inside every
lifetime. Regression tests require both relations in every lifetime, exact
determinism, a pixel-level second-object counterfactual that flips every
answer, and held-out position changes that preserve logic.

## Result

The parent is `unified_three_skill_compounding_seed8413.pt`. A fresh
zero-output rectified skill residual was trained from attempted-action outcomes
with balanced behavioral rehearsal. A generic locality price of `1.0` made the
residual earn any disturbance to inherited behavior.

| seed | updates | new verifier bits | total bits incl. rehearsal | held-out bars | missing second object | all 3 retention gates |
|---:|---:|---:|---:|---:|---:|:---:|
| 9111 | 64 | 12,288 | 30,720 | 99.02% | 48.29% | pass |
| 9112 | 64 | 12,288 | 30,720 | 99.56% | 49.12% | pass |
| 9114 | 96 | 18,432 | 46,080 | 99.46% | 49.06% | pass |

All three bars audits also passed:

- at least 90% on a valid pixel-level counterfactual that changes only the
  second identity and flips every correct answer;
- at least 80% paired prediction flips;
- blank vision at chance;
- the second-object removal ablation at chance;
- held-out colors and positions;
- the immutable inherited controller remained bit-identical.

The two 64-update configurations crossed the full gate in 4.39–4.44 seconds on
the rented GPU. The curated checkpoint is
`artifacts/checkpoints/unified_pair_relation_repertoire_seed9112.pt`.

## Retention repair

Simply increasing replay from 16 to 32 lifetimes per old skill did not repair
the failing seed and made two retention gates worse. Pricing the new slot's
actual residual norm on old events did: seed 9111 moved from a failed XOR
retention audit to all gates passing while relational accuracy increased from
96.70% to 98.93%. This is another instance of targeting the physical mechanism
of interference rather than paying for more indiscriminate replay.

## Honest boundary and next rung

The relation is not yet contour-agnostic. Without training on those appearances,
the three promoted controllers reached only 25.76–26.20% on diamonds and
67.56–71.27% on disconnected dot pairs. This is a bounded negative, not hidden
inside the headline.

The next gradual dogfood rung is therefore:

1. retain the four mastered behaviors;
2. mix a small number of unique diamond pair events with bars;
3. measure verifier bits to stable diamond mastery against a matched fresh
   relation learner;
4. test zero-shot dot-pair transfer;
5. only then progress to delayed match-to-sample with distractors.

The target is an increasingly appearance-independent relation representation,
not a renderer-specific same/different policy.
