# Few-shot forward transfer through latent memory

This isolated experiment tests whether a small neural computer can reuse old latent knowledge to
learn a new primitive attention rule faster. The controller receives only RGB frames, PCM, its own
workspace, and its own latent external memory. Mappings, feature IDs, rules, and task metadata are
never model inputs.

Each deterministic lifetime contains:

1. one visual card demonstrating two colour-to-response associations chosen from eight actions;
2. old sensory queries measuring whether that mapping can still be solved;
3. a novel abstract cue whose independently hashed meaning is attend-left or attend-right;
4. zero, one, two, or four visual demonstrations of that cue in a two-object display;
5. future queries with new backgrounds, PCM, and irrelevant shape variation.

The mapping card becomes one latent memory row. A support event reads that row and may write a new
latent combining old response knowledge with the novel attention rule. This makes the benchmark a
minimal test of learned knowledge composition rather than multi-row search.

## Counterfactual objective and controls

Training compares the inherited branch with an empty-memory twin on exactly matched lifetimes. A
detached margin term rewards only reducing inherited one-/two-shot loss below the control; it cannot
earn reward by making the control worse. Old-task loss remains in the objective. Evaluation freezes
weights and compares inherited, empty, shuffled-value, unrelated-memory, and deterministic-garbage
conditions. The latter three preserve storage where applicable and test whether the contents are
causally used.

Run focused tests with:

```sh
python -m pytest experiments/forward_transfer_attention -q
```

Aggregate replicated reports with:

```sh
python -m experiments.forward_transfer_attention.summarize \
  'experiments/forward_transfer_attention/targeted_transactional_audit/seed_*.json'
```

## Replicated milestone (2026-07-21)

Four independently trained seeds were evaluated on 1,024 held-out lifetimes each after removing a
previous pseudorandom cue/rule correlation. Chance accuracy is 12.5%. Mean inherited-memory
accuracy rose from 42.48% at zero shots to 44.49% after one shot and 44.42% after two shots. The
one-shot and two-shot gains were positive for every seed. Mean few-shot AUC was 43.91%, versus
19.34% with empty memory; shuffled, unrelated, and garbage memory scored 13.44%, 12.49%, and
12.58%. Thus the gain depends causally on the learned memory contents rather than weights alone or
an accidental generator shortcut.

Compression produced a useful boundary result. Unconditionally retaining only the newest row
reduced five rows to one but cut AUC by 20.70 percentage points and destroyed old-task retention.
A transactional gate evaluated each proposed one-row rewrite against separate replay images. It
kept one row (80% fewer rows) and preserved mean old-task accuracy (43.14% before versus 43.66%
after), but accepted only 2.98% of proposals and reduced AUC by 1.19 points. In practice it usually
protected the original mapping by rejecting the new write, so this is safe forgetting, not yet
successful consolidation.

The raw reports and aggregate summary for this boundary experiment are in
`targeted_transactional_audit/`.

## Recursive learned consolidation milestone (2026-07-21)

The failed fixed rewrite was replaced by a 724,001-parameter latent consolidator. The controller is
frozen. After each visual support event, the consolidator receives only the current latent row and
the controller's new latent write, and recursively emits one replacement row. It is trained through
the frozen controller's behavior on visual replay and future queries, plus behavioral distillation
from the uncompressed memory. Targets supervise behavior but are never consolidator inputs. The
active compact branch therefore never secretly accumulates the five-row teacher history.

Four adapters were trained for the four independently trained controllers and evaluated on 1,024
held-out lifetimes each. Mean compact AUC was 45.93%, compared with 43.57% for five-row memory: the
one-row representation was 2.36 percentage points better while using 80% fewer rows. One-shot and
two-shot improvements over the shared zero-shot baseline were +4.71 and +5.37 points, positive for
every seed. Old-task accuracy after four recursive writes was 47.73%, versus 42.87% before supports
and 43.74% with full memory; retention improved on every seed.

A paired frozen-weight causal audit then replaced the compact row after each support. Empty memory
scored exactly 12.5% chance at one, two, and four shots; shuffled and deterministic-garbage rows
also stayed at chance. Intact compact memory scored 46.88%, 47.54%, and 47.13%. Its mean AUC
advantage over the strongest intervention was 25.57 points and was positive for every seed. This is
evidence that recursive learned consolidation both composes and denoises the latent experience,
instead of hiding performance in controller weights or retained rows.

Artifacts:

- `targeted_consolidator_replication/summary.json`: aggregate and per-seed metrics;
- `targeted_consolidator_replication/seed_*.pt`: four consolidator checkpoints;
- `targeted_consolidator_causal_audit/seed_*.json`: paired intervention reports.

The next scientific boundary is cross-controller and cross-task consolidation: determine whether
one shared consolidator can operate across independently learned latent spaces, then test whether
the compact representation improves acquisition of a genuinely different sensory primitive.

## Universal cross-controller and cross-primitive milestone (2026-07-21)

A 4×4 no-training transfer matrix applied every seed-specific consolidator to every independently
trained controller. Across 256 held-out lifetimes per pairing, off-diagonal AUC averaged 46.40%,
versus 46.77% on the native diagonal; all 16 pairings had positive one-shot gains. Adapter 23 was
selected from this small matrix as the single universal candidate. On a fresh 1,024-lifetime test
per controller it achieved 46.41% mean compact AUC versus 44.18% with full memory, and a +4.62
point one-shot gain that was positive for all four controllers. No shared retraining was required.

The same frozen universal adapter and frozen controllers were then tested on a new sensory
primitive. Its cue means attend to the circle or square rather than the trained left or right
position; colour order and position vary independently. On 1,024 held-out lifetimes per controller,
zero-shot accuracy was 42.86% and one-shot accuracy was 47.67% (+4.80 points, positive on every
controller). Compact AUC was 46.22%, versus 43.94% for five-row memory, while final old-task
accuracy rose from 43.95% to 47.20%.

No weights were updated on shape attention. A 256-lifetime paired intervention audit reduced
one-shot accuracy from 47.58% intact to 12.01% empty, 13.33% shuffled, and 11.77% garbage (chance
12.5%). Independent hashing and a 4,096-lifetime generator test rule out simple cue-bit prediction.
The result therefore demonstrates that one learned latent consolidation operation transfers across
both independently trained latent spaces and an unseen feature-selection primitive.

The small-batch progression and artifacts are preserved in `targeted_cross_controller_matrix/`,
`targeted_universal_candidate/`, `targeted_cross_primitive_pilot/`,
`targeted_cross_primitive_confirmation/`, and `targeted_cross_primitive_causal/`.

## Temporal representation milestone and routing boundary (2026-07-22)

Several apparent first/last results were rejected by a mandatory sequence-reversal audit. The audit
found and removed two generator shortcuts: a shape that marked frame position and seed-derived
audio/render variation correlated with query slot. In the strict generator, a given identity is
pixel-identical in either position, audio is silent, and reversal changes only visual stream order.
The adapted policy remained invariant under reversal. Empty, shuffled, and garbage memory still
collapsed its associative performance to roughly 12.5%, showing that memory use was real but the
claimed temporal operation was not.

A frozen supervised decode probe then localized the failure. It used 8,192 balanced training
examples and 4,096 held-out examples from the strict visual-only generator. On the original
controller—before any temporal training—a linear probe decoded the first identity with 100%
accuracy from the recurrent retrieval state and write key, 97.17% from the mean workspace, and
96.88% from the write value. A shallow MLP reached 100% on all four recurrent/write taps. The
explicitly order-neutral pooled sensory vector remained exactly 50% chance. The temporally adapted
controller gave the same qualitative result: 100% recurrent/MLP decoding, with a 97.31% linear
workspace score.

This establishes a sharper architectural boundary: visual stream order is already represented,
survives the recurrent controller, enters workspace, and reaches the memory-write interface. The
failure is downstream routing and credit assignment—the answer path does not bind the stored rule
to the available order feature. Further renderer or sensory-encoder curriculum work is therefore
low priority. The next targeted repair should expose or gate the recurrent order-bearing state in
the answer/read path, while preserving the successful associative consolidator and its retention.

Artifacts:

- `targeted_temporal_order_probe/original.json`: frozen original-controller probe;
- `targeted_temporal_order_probe/adapted.json`: frozen adapted-controller probe;
- `targeted_last_counterfactual_audit/`: paired reversal evidence rejecting the order-blind policy.

## Temporal rule-memory boundary (2026-07-22)

The first answer-routing repair did not learn under direct, reweighted, or staged temporal
curricula. A second frozen probe therefore tested the assumption behind that repair: whether the
demonstrated first/last rule was actually available in compact memory. It trained only diagnostic
linear and shallow-MLP decoders on 2,048 lifetimes and evaluated 1,024 held-out lifetimes. Neither
the compact key, value, concatenated row, nor the vector recalled for a future query exceeded the
52.15% held-out majority baseline reliably after zero, one, two, or four supports. Scores across
all taps and shot counts ranged from 50.59% to 53.61%, with no improvement as demonstrations were
added.

This revises the previous localization. The controller carries object order to its write
interface, but the demonstrated *rule binding* (which cue selects first versus last) is not
recoverable downstream. Answer routing alone cannot repair missing rule memory. The next decisive
probe separates the raw support write from the post-consolidation row; this distinguishes a writer
that never encodes the rule from a consolidator that erases it. Only that component should then be
trained, with the controller and answer policy kept frozen for the diagnostic phase.

Artifact: `targeted_temporal_rule_memory_probe/seed_23.json`.

The raw-versus-compact follow-up repeated the held-out probe at the write boundary. Raw support
write keys, values, and concatenated rows were also at chance after every shot count (50.59% to
53.61%, against the same 52.15% majority baseline). The compact row therefore is not erasing a
previously usable temporal rule: the frozen controller's writer never forms that rule binding in
the first place. This makes a writer-only behavioral adaptation the smallest justified repair;
the successful consolidator and answer policy can remain frozen during the first pilot.

Artifact: `targeted_temporal_rule_memory_probe/seed_23_raw_vs_compact.json`.

A forward pre-hook then probed the exact controller state consumed by the write projections. It was
also at chance: after one support both linear and MLP probes matched the 52.15% majority baseline;
after two and four supports they fell to 50.29%--51.95%. Thus the feedback frame is causally
upstream of writing, but the frozen controller never constructs the joint cue/order/outcome binding
in its final state. Linear write projections cannot manufacture a binding absent from their input.

The writer-only behavioral pilot nevertheless provided a useful negative control. Its ordinary
held-out temporal curve rose from 34.47% at zero shots to 47.85%, 48.44%, and 49.02% after one, two,
and four supports, while spatial and shape remained intact. The mandatory reversal audit rejected
this apparent gain: normal and correctly relabeled reversed streams had the identical
35.79%/47.22%/48.39%/48.97% curve, while reversal with deliberately stale labels was also virtually
identical. It learned a stronger order-blind association, not temporal reasoning.

Artifacts: `targeted_temporal_rule_memory_probe/seed_23_prewrite.json` and
`targeted_temporal_writer_pilot/`.

### Diagnostic discipline

For any future pipeline repair, probe both sides of the suspected boundary before training either
side. Establish that the required variable is recoverable at the producer output and consumer
input, include a causal counterfactual that must change behavior, and only then spend compute on a
repair. The next temporal design should add a generic learned binding/comparison operation before
the write interface, trained only through behavioral outcomes; it should not hard-code first/last
semantics.

## Tiny pre-write binding experiments (2026-07-22)

Per-frame probes found no hidden shortcut around the thought loop. Object-one, object-two, visual
feedback, and post-thought controller states all remained at the balanced first/last baseline.
Concatenating all three frame states did not reveal a reliable joint rule either: held-out linear
and MLP scores remained near 50%--53%, despite modest training-set fitting. The missing operation is
therefore relational composition across states, not merely selecting a better existing state.

An optional generic write binder was added as a zero-initialized, exactly behavior-preserving
residual. It attends across every sensory-event state and combines the attended result
multiplicatively with the final controller state; it has no task-specific inputs. Three 128-lifetime
pilots (`3e-4`, `1e-3`, `3e-3`) were used as an inexpensive learning-rate screen. The best ordinary
curve was 38.28%/45.31%/45.70%/42.19% at zero/one/two/four shots, but its rule was not decodable
from raw writes, compact rows, or recall. It was rejected before causal auditing.

A stronger capacity check repeated the same 32 temporal lifetimes for 30 optimizer steps with no
old-task penalty. It still plateaued at the known order-blind ceiling (roughly 48%--50% after
supports), so scaling this binder is unjustified. The next cheap fork is an explicitly sequential,
position-aware but task-agnostic relation module. It should first pass a disposable supervised
capacity probe; only then should its weights be trained through behavioral outcomes.

Artifacts: `targeted_temporal_binding_tiny_sweep/`, `targeted_temporal_binding_overfit/`, and
`targeted_temporal_rule_memory_probe/seed_23_joint_frame_probe.json`.

## Temporal rendering ceiling (2026-07-22)

The exact joint-state results motivating this check were chance-level against a 52.15% held-out
majority baseline. Concatenated object-one/object-two/feedback states scored 50.49% linear and
50.39% MLP after one support, 52.83% and 48.83% after two, and 52.44% and 48.44% after four. MLP
training accuracy rose only to 60.55%--64.16%, indicating modest fitting without transfer.

A disposable supervised pixel probe then received exactly four visible frames: the mapping card,
two ordered objects, and illuminated-answer feedback. The first overly compressed CNN was rejected
because it could not fit its own training set. A corrected high-resolution version reached 100% on
32 training lifetimes, proving basic capacity, but reached only 59.38% best accuracy on 128 held-out
lifetimes and was unstable. On 512 training lifetimes it collapsed to the 52.15% training majority
at both `3e-4` and `1e-3`; best held-out accuracy was 52.93%.

Separately, a deterministic decoder using only rendered pixels recovered the rule on all 4,096
held-out lifetimes. It traces the visible colour-to-button line, identifies the illuminated feedback
button, and compares it with the two ordered visible colours. This oracle is diagnostic only and is
never exposed to the agent. It establishes that the renderer is unambiguous; the remaining problem
lies somewhere in neural optimization or learned relational visual composition. Architecture is
not identifiable until a nontrivial training set is actually fitted.

Artifacts: `targeted_temporal_rendering_ceiling/` and
`probe_temporal_rendering_ceiling.py`.

### Perception integration boundary

Replacing the existing sensory encoder would reopen the retention result: that encoder currently
supports the stable spatial and shape primitives. A spatiotemporal candidate must therefore be
evaluated as a potential backbone before agent training. Freeze it and measure spatial-cue,
shape-cue, identity, and temporal-order decodability on matched held-out streams. If it preserves
all four, it may replace the old encoder; if not, it must be rejected or run as a parallel path,
whose extra routing and latency costs must be measured explicitly. No behavioral temporal result
justifies silently sacrificing the established spatial/shape floor.

Before building that candidate, run a matched-compute data-scaling curve on the existing corrected
probe at 4k, 16k, and 64k unique lifetimes. Compare it with any later patch-relation diagnostic at
the same data scale and optimizer-step budget, so success can be attributed to relational structure
rather than additional supervision or compute.

## Matched-compute data and architecture scaling (2026-07-22)

The corrected frame-vector probe was trained for approximately 1,024 optimizer updates at each
data scale. Increasing unique supervised lifetimes across two orders of magnitude produced no
learning: 4,096 examples reached 50.71% best train and 50.20% best held-out accuracy; 16,384 reached
50.39% and 50.20%; 65,536 reached 50.04% and 50.20%. The 64k chance-normalized held-out
early-learning AUC was 0.10%. Thus the 512-example failure was not explained by insufficient unique
data under this compute budget. It does **not** show that data scale cannot help after sufficient
optimization.

A matched patch-relation probe retained 8x8 spatial tokens across all four frames and used a
three-layer spatiotemporal transformer. At 4,096 examples and 1,024 updates it reached 50.71% best
train, 50.20% held-out, and 0.12% early-learning AUC. At 65,536 examples and 1,024 updates it reached
50.04% train, 49.80% held-out, and zero early-learning AUC. Both architectures underfit their own
training sets, so these runs compare optimization behavior only; they do not establish an
architectural ceiling or justify replacing the sensory encoder.

The corrected next fork is optimization-first. At fixed data, increase the update budget and use a
sane warmup/decay schedule until training accuracy rises meaningfully above chance. Decompose the
supervised diagnostic into three simultaneous heads—first visible identity, rewarded identity, and
their first/last conjunction—to verify label/loss wiring and localize which visual fact fails. Only
after at least one model fits the training set may its held-out result support an architecture or
curriculum conclusion.

Artifacts: `targeted_temporal_rendering_data_scale/` and
`targeted_temporal_patch_relation/`.

## Decomposed outcome-perception fork (2026-07-22)

A fixed-data optimization sweep trained the decomposed frame-vector diagnostic for 8,000 updates
on 4,096 lifetimes. The first-visible-identity head reached 100% held-out accuracy by update 960
and retained it. At update 8,000, however, rewarded identity reached only 52.86% training and
50.71% held-out accuracy; the composed first/last rule reached 53.05% training and 50.76%
held-out. The loss and first-identity result verify that batches, labels, gradients, and optimizer
updates are active. The feedforward architecture specifically fails to learn outcome attribution
from the thin colour-to-button mapping lines and illuminated feedback button.

That renderer accidentally embeds a line-following operation related to the Pathfinder family of
long-range visual-correspondence tasks inside the intended temporal-binding primitive. This is a
confound, not evidence that temporal order itself is hard. The preprocessing audit image is saved
as `targeted_temporal_rendering_ceiling/preprocessing_audit.png`; it shows that the cue survives
the 40x24 effective feature resolution, but only as a very thin trace.

A new frozen-controller probe then decoded the identity rewarded by the latest demonstration from
the controller's recurrent states. A shallow nonlinear decoder reached 81.15%, 87.30%, and 85.25%
held-out accuracy from the post-feedback state after one, two, and four demonstrations. The same
signal weakened to 65.82%, 72.36%, and 70.41% in the final state consumed by the write
projections. Linear probes were near chance. Therefore the recurrent controller performs much of
the serial outcome-perception operation that the feedforward pixel probe cannot learn, but the
nonlinearly represented result is partly lost before writing. The current blocker is construction
and preservation of the cue/order/outcome binding at the write interface, not absent outcome
information in the sensory stream.

Matched renderer controls used the same data, architecture, and 1,024-update budget. Both the
original 4-pixel lines and 12-pixel lines remained at chance, with 50.34% best held-out accuracy
and 0.60% chance-normalized early-learning AUC. Direct colour feedback on the illuminated button
remained flat through update 512, abruptly reached 99.73% held-out at update 640, and reached 100%
by update 768; its early-learning AUC was 44.02%. Direct visual outcome identity is therefore
rapidly learnable by this probe, while thicker lines alone do not remove the correspondence burden.
Direct colour feedback is the clean temporal-binding curriculum; line tracing remains a separate
cognitive primitive. Defaults remain bit-for-bit compatible with the original environment.

The abrupt positive-control transition is also a stopping-rule warning. Every fixed-budget
negative in this project means only “no learning within N updates”; it is not proof that the model
class cannot learn. This control showed no useful endpoint loss drift before its jump. A negative
that controls an architectural decision therefore needs a substantially wider budget, an explicit
positive control, or both. The 8,000-update thin-line result is stronger than the 1,024-update
result, but remains bounded in exactly this sense.

An exactly matched composition probe then targeted whether the rewarded identity occupied the
first stream position—the demonstrated per-support first/last rule. Linear and shallow nonlinear
probes stayed at chance from the post-feedback state, pre-write state, and concatenated observation
sequence. Post-feedback MLP held-out accuracy was 47.85%, 48.44%, and 50.39% after one, two, and
four demonstrations; best scores during fitting were only 51.37%--52.54%. A follow-up invalidated
the tempting interpretation that both ingredients were already co-located there: first identity
was only 51.37% decodable from the post-feedback state, while rewarded identity was 79.59%.

Finally, a per-thought-step rewarded-identity probe mapped the erosion before writing. For one,
two, and four demonstrations, post-feedback held-out accuracy was 81.15%/87.30%/85.25%. It fell
after thought step one to 76.56%/81.93%/75.78%, after step two to
68.46%/78.61%/75.20%, after step three to 67.77%/74.61%/71.97%, and after the fourth/final step to
65.82%/72.36%/70.41%.

The complementary first-identity curve shows a sharper event-boundary overwrite. First identity
was 95.41%--97.07% decodable immediately after object one and 92.29%--95.41% after object two, but
fell to 52.15%--55.66% when feedback arrived and stayed near chance through all thought steps. This
does not contradict the original temporal-order milestone: that probe labeled the first identity
of a two-frame future query and measured its recurrent retrieval/write states before any feedback
frame. It established that order is represented while the ordered sequence is current. The new
probe establishes that the first identity is not retained when the later feedback event is
processed.

A cross-event two-stage control used three disjoint lifetime splits. It decoded first identity
from the object-one state at 91.50% held-out and rewarded identity from the feedback state at
79.59%, then composed only those probe outputs. A learned tiny combiner reached 73.83%; direct
inequality of the two predicted identities reached 74.22%, and inequality of the true identities
verified label wiring at 100%. Sixteen shuffled-label calibrations averaged 52.94% (42.77%--68.95%);
the wide discrete null distribution is why a single shuffle is not an adequate calibration here.

The repair target is therefore preserve-and-bind across distributed event states. The smallest
candidate is a short generic buffer of recurrent snapshots at sensory-event boundaries, plus a
learned cross-event binder and write-source mixture. Its disposable supervised acceptance tests
are (1) retaining the ingredient-probe baselines at the binder input and write output and (2) making
the composed label held-out decodable above the multi-seed shuffled calibration. Any capability
claim still requires a fresh copy trained only through behavioral outcomes, followed by reversal
and spatial/shape-retention audits.

Artifacts: `targeted_temporal_rendering_ceiling/decomposed_4096_8k.json`,
`targeted_temporal_rendering_ceiling/preprocessing_audit.png`, and
`targeted_temporal_rewarded_identity/`, plus
`targeted_temporal_rewarded_identity_probe/{seed_23,rewarded_was_first_seed_23,thought_curve_seed_23,first_identity_retention_curve_seed_23,two_stage_binding_seed_23}.json`.

## Event-snapshot binder diagnostic (2026-07-22)

A two-layer transformer over the three event snapshots did not learn within 2,000 updates on
4,096 unique lifetimes. This is a bounded optimization result, not evidence of incapacity: later
pairwise models showed that 2,000 updates can sit near the task's ignition point. The simpler
candidate projects each event with shared weights and exposes all pairwise products and absolute
differences to a small MLP. At 1,024 lifetimes it fit 96%--97% of training data but generalized
only to 56%--58%, localizing the failure to small-data overfitting rather than missing capacity.

The diagnostic pass bar was pre-registered at 65% held-out. Cached sensory replay then separated
logical diversity from nuisance-render augmentation while keeping all renderings of a logical
lifetime on one side of the split. An explicit assertion and regression test enforce disjoint
logical IDs before augmentation. With the width-64 pairwise binder and direct-colour feedback:

- 256 logical lifetimes x 4 renderings (1,024 examples) reached 54.27% best held-out;
- 1,024 x 4 (4,096 examples) reached 76.54%;
- 4,096 x 1 (4,096 examples) reached 94.51%;
- 4,096 x 4 (16,384 examples) reached 96.31%.

Logical diversity is therefore the dominant lever; render augmentation is a useful accelerant and
invariance pressure, not a substitute. Successful runs repeatedly showed a flat valley followed by
a sharp transition: direct-colour pixels near update 640, the 16k snapshot run near update 1,000,
and the 4k-unique run near update 1,200--1,400. Future stopping rules must use supervised ignition
points plus training-loss/accuracy slope as leading indicators; chance-level held-out accuracy
before that point is not informative.

The earlier 73.83% two-stage result was never an information-theoretic ceiling. It was a lower
bound imposed by two separately trained shallow ingredient probes. End-to-end training extracts
more useful information from the same recurrent snapshots.

The exact 4,096 x 4 shuffled-label audit remained at chance (49.80% best held-out and 53.88% best
training). A first attempt at reversal merely swapped history-bearing recurrent tensors and
produced out-of-distribution garbage; it is retained as a malformed-control artifact, not treated
as evidence. The corrected counterfactual reversed the two object frames before replay through the
frozen controller while preserving the feedback frame byte-for-byte. Against pre-registered
90%/90%/90% gates, the binder achieved 95.02% normal accuracy, 95.61% correctly relabeled reversed
accuracy, and a 93.36% prediction-flip rate; stale-label accuracy was 4.39%.

Disposable ingredient probes independently validate that counterfactual. First identity decoded
at 99.44% normal and 99.22% reversed, with a 98.66% prediction-flip rate. Rewarded identity decoded
at 89.55% and 89.36%, with predictions preserved across reversal 99.22% of the time. Both ground-
truth construction invariants were 100%.

This proves only that a task-agnostic binder over frozen event snapshots *can* learn the temporal
relation under dense disposable supervision. It does not yet prove that the behavioral agent can
discover it. Further diagnostic binder forks stop here. The milestone ladder is now integration:
binder inside a fresh agent, rule decodability at write, survival through consolidation and recall,
behavioral few-shot gain under reward-only training, then sensory reversal and spatial/shape
retention audits. The original thin-line renderer remains the final graduation arm; direct colour
is the efficient temporal-binding curriculum.

Artifacts: `targeted_temporal_event_snapshot_binder/`,
`probe_temporal_event_snapshot_binder.py`, and
`probe_temporal_counterfactual_ingredients.py`.

## Event-binder integration pilots (2026-07-22)

The integrated three-event pairwise binder passed its exact no-op architecture gate and gradient
tests. A binder-only pilot on 32 repeated lifetimes and a binder-plus-reader pilot on 128 repeated
lifetimes both had healthy gradients and nonzero write residuals, but neither made the temporal
rule reliably decodable. Across joint checkpoints 5, 10, and 20, the best held-out result at raw
write, compact memory, or recall was 55.47%; held-out behavior did not improve with demonstrations.

These are plumbing and credit-path results, not negative evidence about reward-only discovery. The
supervised learning curve did not generalize at 1,024 examples and ignited only with at least 4,096
distinct logical lifetimes after a long flat valley. The behavioral pilots used only 32 or 128
repeated lifetimes, so their approximately 55% probe results are exactly what the measured data
curve predicts even under dense supervision.

The binder-only checkpoint also showed that forgetting can enter through changed stored data even
when old weights are frozen: several spatial/shape measures fell by roughly 2.4--4.1 points.
Balanced rehearsal preserved shape better in the joint pilot, while spatial remained close to the
pre-registered two-point boundary. A larger paired two-seed audit was scheduled before adding any
constraint.

The next economical branch is explicitly supervised-bootstrapped integration. A disposable rule
head reads the actual raw write key/value row and trains the real binder on at least 4,096 unique
temporal lifetimes; the reader trains jointly and spatial/shape rehearsal starts immediately. The
head is then discarded and behavioral fine-tuning plus reversal, corruption, retention, and
thin-line audits remain unchanged. This can establish a real supervised-bootstrapped capability,
but it does not close the stronger reward-only discovery question.

The first properly scaled bootstrap epoch used 4,096 distinct temporal lifetimes plus 4,096
balanced rehearsal lifetimes (256 updates). It passed its mechanistic gate: mean binder gradient
norm 0.307, residual RMS 0.075, stable spatial/shape rehearsal, and no numerical failures. The
temporary head remained at 48.74% mean accuracy with 0.7066 loss. A fresh independent 1,024-
lifetime probe reached 55.76% best at raw write, 54.98% at compact memory, and 55.76% at recall.
Because this is well before every previously measured ignition point, it is recorded as a healthy
pre-ignition checkpoint, not a negative. The optimizer and temporary-head state were resumed for
the remaining 768 updates.

The four-epoch run completed without ignition: the temporary head ended at 49.83% accuracy and
0.6944 loss, temporal behavior stayed at chance, and a fresh 2,048-lifetime probe scored 52.69%
best at raw write, 52.05% at compact memory, and 51.90% at recall. This does **not** yet falsify the
bootstrap. The balanced cycle means only 512 of the 1,024 total optimizer updates were temporal and
carried the auxiliary rule loss—below the earliest previous ignition at roughly 640 signal-bearing
updates and far below the 1,000--1,400 range of the cached binders. The self-contained checkpoint
preserves optimizer and temporary-head state for a staged continuation to 1,024 temporal updates.
