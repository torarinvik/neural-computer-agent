# Five-item relation readout breakthrough

## Result

Using the frozen `unified_next_error_balanced_primary_seed43601.pt`
controller, a disposable probe was given five-item episodes and one query
asking whether the fourth item matches the fifth item.  The probe saw only the
post-query hidden state or workspace; verifier labels were used only to train
the discarded probe.

| representation | normal held-out | shuffled-label held-out |
| --- | ---: | ---: |
| hidden, linear | 66.02% | 52.05% |
| workspace, linear | 50.49% | 49.32% |
| combined, linear | 66.21% | 50.68% |
| hidden, MLP | **93.95%** | 51.27% |
| workspace, MLP | 81.93% | 49.22% |
| combined, MLP | 92.77% | 51.76% |

The generator required 1,024 examples per split for span five.  The result is
representation evidence only: no controller weights changed and no
behavioral five-item skill is claimed yet.  It justifies the smallest next
behavioral arm: one pure `next` relation, one extra query thought step, and
the already validated two-to-one target/rehearsal replay schedule.

## Interpretation

Span five is not blocked by sensory or memory representation.  As in span four,
the useful signal is strongly nonlinear and the workspace alone is weaker than
the recurrent hidden state.  The next experiment must therefore test credit
assignment, not invent a new encoder or memory architecture.

## Behavioral replay curve

Using one batch of 1,024 unique target outcomes and interleaved replay of three
old span-three streams:

| target replay updates | overall span-5 | pure next | strict next conflict | old span-3 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 55.66% | 55.66% | 50.00% | 98.39% |
| 4 | 56.05% | 56.05% | 50.00% | 98.52% |
| 16 | 63.67% | 64.45% | 53.91% | 98.09% |
| 64 | 84.57% | **87.89%** | **78.13%** | 96.92% |
| 128 | 87.11% | **89.26%** | 79.30% | 96.22% |
| 2 batches × 64 | **94.82%** | **96.68%** | **94.53%** | **98.09%** |

The diversity-controlled arm passed the primary gates: all-memory-reset
accuracy was 51.27%, candidate counterfactual flip rate was 92.29%, and old
span-three retention was 98.09%.  An independent seed is still required before
promotion.

The independent replica also passed: 97.27% overall, 96.79% strict conflict,
50.10% after all-memory reset, 94.63% candidate flip rate, and 97.96% old
span-three retention.  The canonical-schema candidate is stored as
`artifacts/checkpoints/unified_procedural_shape_span5_replay_seed44906.pt`.

An extra robustness audit at nuisance level 0.17 scored 90.53% overall, 84.74%
strict conflict, and 92.71% old-skill retention.  This is not a failure of the
baseline result; it identifies the next frontier: train with gradual nuisance
augmentation and recover the retention/accuracy margin at the harder render.

## Microscopic curriculum result

The first three-level attempt (0.1350, 0.1351, 0.1352) used the old learning
rate and one rehearsal update; it degraded to 85.74% pure-next accuracy and
94.44% old retention.  We discarded that branch.  Keeping the exact same
0.0001 increments but using learning rate `3e-4` and four rehearsal updates
produced:

- training evaluation: 94.24% overall, 96.48% pure-next, 92.65% strict conflict,
  98.91% old span-three retention;
- fresh 0.1352 endpoint: 95.41% overall, 96.68% pure-next, 93.15% strict
  conflict, 50.68% memory-reset accuracy, 92.58% candidate flip rate, and
  98.57% old retention.

This is the first robust nuisance-curriculum result.  The candidate is
`artifacts/checkpoints/unified_procedural_shape_span5_micro1352_seed45301.pt`.
The staircase can now continue in further 0.0001 increments, retaining the
lower learning rate and stronger rehearsal recipe.

The next staircase (0.1353 → 0.1355) also passed its fresh endpoint audit:
95.80% overall, 96.48% pure-next, 93.25% strict conflict, 50.88% after
memory reset, 93.07% candidate flip rate, and 98.96% old-skill retention.
The resulting candidate is
`artifacts/checkpoints/unified_procedural_shape_span5_micro1355_seed45401.pt`.

The next rung (0.1356 → 0.1358) passed as well.  Its fresh 0.1358 endpoint
scored 96.58% overall, 97.85% pure-next, 96.01% strict conflict, 49.41% after
memory reset, 94.24% candidate flip rate, and 99.35% old-skill retention.  The
candidate is
`artifacts/checkpoints/unified_procedural_shape_span5_micro1358_seed45501.pt`.

This is a sample-efficiency signal: the model is learning from repeated use of
the same observed outcomes, not from additional verifier episodes.  It is not
yet promotion-grade (the current gate is 90% overall and 85% strict-conflict),
but the monotonic curve justifies a 128-update rung before changing the
architecture or increasing the number of unique outcomes.

## Adaptive gate result

The adaptive curriculum evaluated each level and trained only if a held-out
gate failed. At 0.1359, 0.1360, and 0.1361, the existing candidate already
passed all gates, so the run used **zero optimizer updates** at every level.
The full 0.1361 audit still passed: 95.90% overall, 97.85% pure-next, 95.53%
strict conflict, 48.83% after memory reset, 94.43% candidate flip rate, 93.16%
reverse flip rate, and 99.35% old-skill retention. The adaptive runner and
candidate are saved as
`experiments/unified_cognitive_controller/train_microscopic_gate.py` and
`artifacts/checkpoints/unified_procedural_shape_span5_adaptive1361_seed45601.pt`.

This is stronger evidence than a fixed-budget run: the system crossed three
new difficulty levels without spending additional optimizer updates.
