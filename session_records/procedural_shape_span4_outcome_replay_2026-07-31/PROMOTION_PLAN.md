# Promotion plan

The replicated interleaved replay result is a capability breakthrough, not yet
a promoted controller.  The next run uses the same controller and memory
interfaces, but writes a candidate checkpoint with
`probe_span4_outcome_replay.py --checkpoint-out`.

## Training arm

- Four-item target: one query with a third-to-fourth `next` relation.
- One extra query thought step; no verifier state is exposed.
- Only the zero-initialized action adapter is trainable.
- Four unique target batches, 256 examples each.
- Sixteen target replay updates per batch.
- Interleave every two target updates with one update on each of three old
  span-three streams.
- Reuse the observed action and scalar outcome; replay creates no new labels or
  hidden game-state inputs.

## Promotion gates

1. Candidate checkpoint loads with the ordinary controller loader.
2. Four-item held-out accuracy is at least 90%, and strict next-conflict
   accuracy is at least 85%.
3. All-memory-reset accuracy is near chance, demonstrating memory dependence.
4. Candidate and operation counterfactuals flip predictions on changed cases.
5. Each old span-three stream remains at least 95% and no stream drops more
   than two percentage points from its parent checkpoint.
6. The same gates pass on an independent seed and a fresh nuisance/render seed.

Only after all six gates pass may the candidate be copied into
`artifacts/checkpoints/` and described as promoted.  Until then it remains a
candidate and the parent checkpoint is unchanged.
