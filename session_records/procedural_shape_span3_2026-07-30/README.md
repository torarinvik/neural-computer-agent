# Procedural shape span 3: verified independent short-term memory

## Breakthrough

The inherited controller now remembers **three fully independent procedural
shape identities** and answers one arbitrary, visually cued ordinal query after
the sequence. The primary and replica both pass a pre-registered 95% gate on
every ordinal, not merely on aggregate accuracy.

| run | overall | item 1 | item 2 | item 3 | conflicting item 3 |
|---|---:|---:|---:|---:|---:|
| primary seed 28331 | 98.01% | 98.58% | 96.83% | 98.63% | 97.85% |
| replica seed 28431 | 97.51% | 98.78% | 95.95% | 97.80% | 96.78% |

Every item and candidate is procedurally rerendered at randomness 0.135.
The learner receives only RGB, its own opaque action, and scalar success or
failure. Identity, ordinal, correct action, and generator state remain private.

The current claim is deliberately bounded to **one query after a three-item
sequence**. The same checkpoint is 88.75% zero-shot with two sequential
queries and 85.80% with all three, so multi-query state preservation is the
next frontier.

## Why it finally worked

The successful path was measurement-driven:

1. The original four workspace slots were proven to be exact clones because
   zero initialization plus content-only softmax addressing never breaks
   symmetry. A generic addressed-workspace pilot broke the symmetry but did
   not improve behavior, so it was not promoted.
2. The first capacity bridge accidentally varied both independence and visual
   visibility. Removing the fade isolated memory capacity from perception.
3. Aggregate accuracy hid a shortcut on rare third-item rows. The evaluator
   now reports every presented ordinal, independently sampled third-item rows,
   and rows where item three actually conflicts with item one.
4. A disposable probe showed that all three identities were already 99%+
   decodable after event three. Storage was not the blocker.
5. A matched state-plus-query probe reached 98.94% with a small MLP, while a
   linear probe and shuffled labels stayed at chance. The missing operation
   was nonlinear binding.
6. A normalized relation adapter was inserted with exactly zero output, so
   inherited behavior was bit-identical before training.
7. Binary outcome completion stopped wasting information. With two mutually
   exclusive actions and one correct answer, the agent's attempted action plus
   scalar success/failure identifies the correct binary action exactly. This
   uses no hand label or game-state hook.
8. A learned generic gate let the new residual open where useful without
   overwriting the inherited reader. This removed the ordinal-two regression
   seen with an ungated adapter.

## Sample efficiency and compounding

The redundant-content bridge required 13,824 target verifier bits to reach its
stable gate. From that shared parent, the primary independent relation needed
11,520 target bits and the replica needed 15,360.

A fresh controller with the same architecture, allowed to train all of its
weights, remained at 49.87% after 15,360 target outcomes. Therefore the
conditional transfer advantage for acquiring the final independent relation
is conservatively greater than `15,360 / 11,520 = 1.33x`. This is evidence of
useful inherited representations, not yet evidence that every successive
capacity rung accelerates monotonically.

No examples were replayed; every training lifetime was newly generated.

## Causal and adversarial controls

Primary seed 28331:

- blank presentation: 49.89%
- all fast memory reset: 49.95%
- workspace disabled: 80.88%
- active recurrent state reset, workspace retained: 70.30%
- valid reversed-presentation rerender: 98.16%
- prediction flips on reversal-changed answers: 96.39%
- candidate counterfactual: 98.13%
- prediction flips under candidate counterfactual: 98.00%
- retained redundant span-3 bridge: 98.93%
- retained independent span 2: 98.87%

The independent training-seed replica passes the same gates. A fresh learner
stays at chance. Shuffling scalar outcomes prevents mastery: independent item
three is 54.69%, genuinely conflicting item-three rows are 47.36%, and old
skills degrade rather than being spuriously preserved.

## Honest limitations

- This is three-item recognition, not yet arbitrary-length working memory.
- It supports one post-sequence query robustly; multiple queries still alter
  recurrent state and degrade later answers.
- Binary outcome completion is exact only for two actions with exactly one
  correct answer. Larger action spaces need a generic bandit or elimination
  mechanism rather than pretending a failed action identifies one alternative.
- Generic slot addressing remains implemented and tested as an optional
  architecture, but was rejected for this milestone because it did not improve
  the measured learning curve.

## Curated checkpoints

- `artifacts/checkpoints/unified_procedural_shape_span3_seed28331.pt`
  (`8382f425fb5768d86b2ee90b01665c4f6b171c14b9890eddcabac9b6240f56ee`)
- `artifacts/checkpoints/unified_procedural_shape_span3_seed28431.pt`
  (`5f4e0e1c8e25db16e1241ba493b9e4b85fe978865cb71564dce3c4f408af01a6`)
