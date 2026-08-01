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

## Corrected staircase through 0.2000

The retention gate was corrected so that old span-three skill is always
evaluated at its mastered nuisance level (0.1358), rather than at the harder
new target level.  This separates catastrophic forgetting from ordinary
perceptual difficulty.  The gate then advanced in exact 0.0001 increments to
0.2000 with MPS, training only when a held-out level failed.  The final
0.1998, 0.1999, and 0.2000 levels all passed without additional optimizer
updates.  The promoted checkpoint is
`artifacts/checkpoints/unified_procedural_shape_span5_adaptive2000_final_corrected_mps_seed53301.pt`.

The final adversarial audits (two independent seeds, 4,096 episodes each)
gave 90.8%/90.2% span-five accuracy, 88.7%/89.3% strict next-conflict
accuracy, 99.86%/99.78% old span-three retention, and 49.5%/49.0% accuracy
after clearing all memory (chance).  Candidate counterfactual prediction
flips were 89.4%/89.3%; reverse-presentation flips were 85.1%/84.6%.
The operation-counterfactual flip rate was lower (33–35%), so operation
binding remains a secondary frontier even though the primary curriculum gate
and memory-dependence checks pass.

## Trainable-vision climb to 0.4000

The next climb exposed the practical recipe for moving beyond the 0.20
nuisance frontier without wasting updates:

- fresh target and rehearsal renderings on each replay chunk, avoiding
  memorization of one reused batch;
- the vision encoder unfrozen together with the action adapter, while old
  skills were checked at the mastered 0.1358 nuisance level; and
- a stable, repeated held-out gate (4,096 episodes per repeat) so a noisy
  gate could not force unnecessary training.

Using MPS, a 0.001 curriculum stride, batch size 2,048, one rehearsal update
per chunk, and at most 128 updates per level, the run passed every level from
0.351 through 0.400.  It used only 44 optimizer updates across all 50 levels;
most levels passed without training.  The promoted checkpoint is
`artifacts/checkpoints/unified_procedural_shape_span5_adaptive4000_trainvision_mps_seed58501.pt`,
and the gate trace is
`adaptive_351_400_trainvision_mps.json`.

Two independent 4,096-lifetime audits at 0.400 measured target accuracies of
89.7% and 91.2%; the strict conflicting-new-slot accuracies were 90.4% and
91.7%.  The old span-three skill remained at 98.5% and 98.7%.  Clearing all
memory reduced target accuracy to 49.4% and 50.6%, establishing causal memory
dependence rather than a fixed-weight shortcut.  Reverse-presentation
accuracy remained about 90%, while operation-counterfactual flips were only
72–73%; operation binding is therefore still a separate frontier.  The full
audit artifact is `adaptive_4000_full_audit_trainvision_mps.json`.

The efficient policy is now evidence-backed: use microscopic gates for
precision, widen to 0.001 only after the curve is stable, refresh experience
instead of replaying a fixed batch, and adapt the sensory encoder while
rehearsing mastered skills.  Further difficulty should preserve these gates
and address operation binding separately rather than spending more updates on
the already-passing nuisance ladder.

## Coarse-to-fine climb from 0.4000 to 0.8000

The 0.001/4,096-episode gate was too expensive at the harder frontier: it
spent most of its time evaluating nearly identical levels. The scheduler was
therefore extended to allow coarse strides that still resolve to 0.0001
increments. We used 0.005 exploration rungs in three isolated blocks, with a
2,048-episode repeated gate, fresh target/rehearsal batches, trainable vision,
and a 128-update cap per rung. Full 4,096-episode audits were reserved for
the final checkpoint.

All four blocks passed every gate:

- 0.4000 → 0.5000: 220 optimizer updates;
- 0.5000 → 0.6000: 220 optimizer updates;
- 0.6000 → 0.7000: 297 optimizer updates;
- 0.7000 → 0.8000: 517 optimizer updates.

The final 0.8000 checkpoint is
`artifacts/checkpoints/unified_procedural_shape_span5_adaptive800_coarse_trainvision_mps_seed58901.pt`.
At its final gate it scored 91.6% overall, 90.9% strict conflict, and 99.4%
old span-three retention.

Two independent 4,096-lifetime audits at level 0.8000 scored 91.3% and 91.8%
target accuracy, 91.6% and 91.8% conflicting-new-slot accuracy, and 99.4% and
99.5% old-skill retention. Clearing all memory returned target accuracy to
49.9% and 50.7%, confirming that the result depends on the learned workspace
rather than a fixed-weight shortcut. Reverse-presentation accuracy was 91.1%
and 92.3%. Operation-counterfactual flip rates remained 72.3–73.3%, so
operation binding is still the next architectural frontier. The audit is
saved as `adaptive_8000_full_audit_trainvision_mps.json`.

This establishes a validated nuisance climb from 0.1358 through 0.8000 while
preserving the older skill. The efficient schedule is coarse exploration
followed by fine endpoint auditing, rather than paying a large verifier cost
at every microscopic level.

## Operation-binding frontier diagnostics

The 0.8000 checkpoint remains the canonical promoted model; all experiments in
`operation_frontier/` are disposable copies and none replaces it.  The first
diagnostic used the existing operation-counterfactual audit: on a 2,048-episode
quick check, changing the operation cue changed the correct action on about
24.7% of queries, but the controller changed its action on only about 23–26% of
those changed cases.  Normal target accuracy stayed around 91–93%, while
clearing all memory stayed at chance.

A representation probe then separated the failure location.  A linear probe
decoded the operation from the current event embedding at 99.9% held-out
accuracy and from the post-query recurrent state at 96.8%; the pre-query hidden
state was at chance (49.4%).  The glyph is therefore present in perception and
is not being bound to the already-computed intention.

Three tiny adaptation pilots were run on copies with old-skill rehearsal:

- an action-adapter pilot with a relatively large step damaged span-three
  retention (about 99% to 73%) and was rejected;
- paired normal/counterfactual reward training with a generic intention-reading
  skill slot preserved retention (~99.4%) but left operation flipping at ~25%;
- a greedy outcome version and a longer paired run likewise preserved retention
  but produced no operation-binding gain.

A disposable dense diagnostic on the same generic slot did eventually raise
held-out operation flipping to about 60% after 128 repeated updates on a fixed
paired batch.  This proves the interface is expressive, but also shows that
the current scalar-outcome learner has not found the binding efficiently.  The
next high-ROI experiment is therefore a better self-generated credit signal or
operation-specific curriculum—not a redesign of the vision encoder—and every
candidate must still pass the old-skill retention and memory-reset audits.

## Span-six extension check

The canonical span-five checkpoint was evaluated on a sixth-item target at
nuisance 0.8 before any adaptation.  Across 4,096 balanced episodes it scored
73.0% overall and 70.8% on strict conflicting-new-slot queries; clearing all
memory returned 48.8%, confirming that the sixth-item signal is still using the
workspace rather than a fixed shortcut.  The read-only baseline is recorded in
`span6_baseline_seed821001.json`.

A 16-update adaptation using exact 4,096-example batches was stopped after one
slow update.  The partial-balance microbatch mode then reached 57.4% held-out
span-six accuracy after four mixed target/rehearsal updates, below the baseline,
while each update still took roughly 15–20 seconds on local MPS.  No span-six
candidate was promoted.  The next span extension should therefore use a
capacity curriculum (e.g. a carefully gated sixth-slot adapter or a smaller
controller) rather than fine-tuning the entire span-five controller at once.

The subsequent direct-query capacity bridge used one generic zero-output skill
slot, trained only from greedy action plus scalar verifier outcome, with the
canonical span-five controller frozen and rehearsed.  At zero sixth-item
independence, direct span-six accuracy rose from 65.3% to 83.7% after two short
blocks; span-five and span-three retention stayed about 91.7% and 99.5%, and
memory reset stayed at chance.  At independence 0.01, verifier-side weighting
of rare independent examples produced 90.2% on 41 independent cases in one
  seed, but three seed-disjoint audits gave 80.5%, 75.6%, and 78.0% on 41 cases
each (mean 78.0%).  At independence 0.02 the independent score was 85.4% on 82
cases, but this did not improve the seed-disjoint estimate.  These are useful
learning signals, not a promotion: aggregate direct-sixth accuracy remains
about 84–85% and the independent gate is still too noisy.  Stop climbing the
independence scalar; the next high-ROI experiment is a larger, seed-disjoint
0.01 gate with more diverse sixth-item experience, while preserving the
canonical span-five model.
