# Brain Workshop 4-back, protected 5-back, and protected 6-back continuation

This session tested the next protected-plasticity frontier after the verified
1-back → 4-back compounding rung. The controller was initialized from the same
inherited `nback3_rehearsal1_2_depth3` parent family used by the earlier ladder,
then trained on 4-back with the unchanged three-stream amodal bus:

```text
--modalities vision,audio,text
--target-modalities text
--factorized-output --factorized-reward
--external-history --external-history-depth 4
--per-stream-external-history
--external-memory-adapter-width 64
--per-stream-intention-adapter-width 64
```

The new implementation adds `--rehearsal-weights`, a verifier-side vector
matching `--rehearsal-n-backs`. The tested policy used weights **2, 0.5, 0.5**
for 1-, 2-, and 3-back. This is not a semantic label exposed to the controller;
it only changes how much private rehearsal loss is mixed into the optimizer.

## Result

| seed | 4-back at 256 updates | 4-back after continuation | 1-back retention | reset control | time-shuffle |
|---|---:|---:|---:|---:|---:|
| 47408 | **80.42%** | — | **94.29%** (parent 93.96%) | 49.12% | 51.56% |
| 47409 | 66.99% | **77.00%** at 320 updates | **94.61%** (parent 94.32%) | 49.07% | 51.78% |
| 47405 | **81.52%** | — | **94.56%** (parent 94.24%) | 49.07% | 51.32% |

All percentages above are eligible exact accuracy after the n-back warm-up,
unless noted otherwise. Each 256-update run sees 8,192 unique lifetimes and
65,536 target-stream verifier bits (8 trials × one target modality per
lifetime). The three initial runs average **76.31%** at that fixed budget.
The checkpoint retention audits use a frozen controller and a one-update
no-op continuation; history reset and time-shuffle controls remain near
chance. All three seeds therefore pass the 2-point 1-back retention gate and
the causal controls. The continuation is the important sample-efficiency
result: we spent another 64 updates only after a valid 256-update run had
crossed the 65% acquisition gate but remained below the desired mastery band.
That extra stage added 16,384 target bits and brought seed47409 to 77.00%; the
three final scores average 79.65%.

The fixed weighting is not universally better than uniform rehearsal: seed
47408 improved over its uniform-weight comparison, while seed47409 initially
lagged it and needed continuation. The promoted strategy is consequently
**gated continuation with retention audits**, not “always use these weights.”
Uniform rehearsal remains the robust baseline; per-rung weights are an
experimentally supported control for protecting a mastered rung when the
learning curve justifies it. The reusable verifier-side accounting and gate
are implemented in `audit_nback_continuation.py`: it reports bits-to-mastery,
requires positive held-out progress before continuation, and rejects a
capability claim if causal controls or retention fail.

## Artifacts

- `nback4_rehearsal123_w2_05_05_seed47408_256_inherited.json` and
  `nback1_retention_targeted_inherited_after_nback4_seed47408_256.json`.
- `nback4_rehearsal123_w2_05_05_seed47409_256_inherited.json`,
  `nback4_rehearsal123_w2_05_05_seed47409_64_continuation.json`, and
  `nback1_retention_targeted_inherited_after_nback4_seed47409_320.json`.
- `nback4_rehearsal123_w2_05_05_seed47405_256_inherited.json` and
  `nback1_retention_targeted_inherited_after_nback4_seed47405_256.json`.
- `nback4_continuation_audit_seed47405.json`,
  `nback4_continuation_audit_seed47408.json`, and
  `nback4_continuation_audit_seed47409.json` are generated verifier-side
  summaries; they train no parameters.
- `nback5_probe_inherited_seed47405_8.json` records the bounded fifth-back
  compatibility gate; its checkpoint is kept under `artifacts/checkpoints/`.
- Checkpoints with matching names under `artifacts/checkpoints/`.

Several earlier files in this directory are deliberately retained as negative
controls. Some were fresh rather than inherited, and others omitted the
factorized output or 64-wide RAM adapters; they must not be compared with the
compounding runs. This provenance distinction was caught before promotion.

## Fifth-back compatibility probe

The trainer and generic RAM bridge now accept a fifth-back rung with five
opaque external snapshots. Starting from the inherited seed47405 4-back
checkpoint, an 8-update probe completed in 34 seconds with live gradients and
4-back rehearsal between 76% and 95%. The 5-back eligible target score was
**46.88%** after training versus **47.66%** before; reset and time-shuffle
controls were 49.61% and 52.47%. The acquisition gate correctly rejected a
longer run.

This is deliberately a bounded result: the project's successful learners have
long ignition valleys, so eight updates cannot establish that 5-back is
unlearnable. It establishes that the depth-five interface is executable and
that there is no evidence yet to justify spending the longer budget. The next
5-back run should be triggered by a calibrated progress signal or a separate
sample-efficiency intervention, not by blindly scaling duration.

The trainer now records `batch_eligible_accuracy` separately from its legacy
warm-up-inclusive batch score (and does the same for rehearsal rungs). This
prevents the high no-target warm-up trials from masquerading as early learning
when calibrating a stopping policy.

## Fifth-back learning with a protected extension

The first eight-update depth-five probe was intentionally too short to test
learning. Two matched longer diagnostics separated capacity from learning
signal. A target-only supervised diagnostic, starting from the inherited
4-back checkpoint and using 8,192 unique lifetimes over 256 updates, reached
**93.03%** eligible held-out 5-back accuracy. Reset and time-shuffle controls
were **50.00%** and **49.74%**. This is a verifier-label ceiling probe, not a
claim that the controller discovered the skill from reward.

We then tested a genuinely protected extension. `--freeze-inherited-history`
freezes the controller, encoders, decoder, and all inherited RAM-adapter
columns; only the appended fifth-history input columns can receive gradients.
After the same 256-update supervised diagnostic budget, the protected adapter
reached **60.55%** eligible held-out 5-back accuracy, with reset **49.35%** and
time-shuffle **51.43%**. The inherited 1-back skill was unchanged at
**94.08%** in a 512-lifetime, zero-learning-rate retention audit; reset history
was **50.08%**. This demonstrates that a new temporal skill can be acquired
through an additive RAM-side path without changing the inherited weight path,
though the protected adapter is less sample-efficient than the task-only
supervised ceiling.

Finally, eight updates of verifier-reward fine-tuning on that protected adapter
(with 1/2/3/4-back rehearsal) raised 5-back held-out accuracy from **60.55%** to
**78.91%**. Reset and time-shuffle controls remained **49.35%** and **52.34%**;
the post-fine-tune 1-back retention audit remained **93.53%**. This is the
current strongest result: supervised initialization discovers the new
representation, then a small reward-only continuation improves it while the
old skill stays intact. It is a supervised-bootstrapped reward result, not yet
reward-only discovery from a cold start.

An additional eight-update reward-only continuation, using a fresh verifier
seed rather than repeating the first batches, raised the protected adapter to
**80.86%** eligible 5-back accuracy. A larger no-update evaluation covered
3,072 target-bearing episodes and gave reset **50.10%** and time-shuffle
**50.78%**. The 1-back retention audit remained **93.14%** over 512
lifetimes, with reset **49.94%**. The improvement from 78.91% to 80.86% did
not meet the pre-registered +5-point continuation gate, so training stopped
there; this is a verified efficiency curve point, not an excuse to scale
compute indefinitely.

## Sixth-back compounding rung

The trainer now supports a sixth opaque RAM snapshot and an explicit parent
depth override for extending an already protected checkpoint. A target-only
6-back ceiling diagnostic reached **80.86%** after 256 updates, but its
1-back retention fell to **61.16%**, so it is rejected as a capability result.

The protected sixth-back diagnostic opened only the appended depth-six input
columns. It reached **57.42%** eligible held-out 6-back accuracy from a
48.63% parent, with reset **50.00%** and time-shuffle **49.41%**. 1-back stayed
at **93.28%**, but 5-back fell to **72.66%** because sixth-history features are
also present on later 5-back trials. This localized the remaining problem to
task-conditional interference rather than weight overwriting.

The decisive continuation used eight verifier-reward updates on that protected
branch with 5-back rehearsal. 6-back rose to **72.27%** on the 256-lifetime
check and **71.24%** on a larger evaluation with 2,048 target-bearing trials.
Reset and
time-shuffle were **50.00%** and **49.95%**. The earlier rungs were retained:
5-back **80.86%** and 1-back **92.75%** in zero-learning-rate audits. This is
the first protected 1→5→6 compounding result, honestly labeled as a
supervised-bootstrapped reward continuation rather than cold-start reward
discovery.

A fresh eight-update continuation reduced the 5-back rehearsal weight to 0.25,
keeping the inherited path frozen. It raised 6-back from **72.07%** to
**78.13%** on the check evaluation and to **77.34%** on a larger evaluation
with 2,048 target-bearing trials. Reset and time-shuffle were **50.00%** and
**49.66%**. Retention remained 5-back **80.86%** and 1-back **92.22%**. The
continuation gate accepted this rung, then stopped rather than extending
unconditionally.

Two short follow-up forks were rejected rather than promoted. Repeating the
same 0.25-weight continuation from the 77.34% checkpoint fell slightly to
**77.15%**; a router-only reward continuation also fell to **77.15%**. A
zero-initialized stacked relation/router branch was then tested from the best
checkpoint with dense verifier labels. It degraded 6-back from **76.37%** to
**65.04%** at 128 updates, while its reset and time-shuffle controls remained
at **50.00%** and **50.98%**. The same branch trained from the 5-back parent
never rose above chance (49.61%). These are clean negative results: healthy
gradients and falling loss do not justify a new routing module when target
accuracy or retention is not improving.

The promoted checkpoint also passed a larger no-update audit: **77.16%**
eligible 6-back accuracy over **32,768 target-bearing trials**, with history
reset **50.00%** and time-shuffle **49.21%**. This is slightly below the
2,048-trial 77.34% estimate but confirms the same causal signal at much higher
precision. The current frontier is therefore not another blind continuation;
it is a task-conditional memory gate that can preserve the sixth-history
signal on 6-back trials without making the same feature perturb 5-back trials.

## Seventh-back protected compounding breakthrough

The next gradual rung extends the same generic RAM bridge to seven opaque
snapshots. An eight-update compatibility probe correctly stayed at chance, and
a full 256-update protected diagnostic reached **54.30%** eligible 7-back from
a **47.66%** parent. That was enough evidence to use the established
supervised-bootstrap then reward-continuation recipe, but not enough to claim
mastery by itself.

The first reward continuation with 6-back rehearsal weight 0.25 rose from
**55.47%** to **58.59%**, but the 6-back retention audit fell to **71.44%**;
that branch was rejected. Increasing only the verifier-side 6-back rehearsal
weight to **1.0** fixed the interference: 7-back rose from **57.81%** to
**68.75%** in eight reward updates. A larger no-update evaluation measured
**68.36%** over **8,192 target-bearing trials**, with history reset **50.00%**
and time-shuffle **55.18%**.

The retention ladder passed on independent 8,192-trial audits: 6-back
**76.32%**, 5-back **80.14%**, and 1-back **91.99%**. Their reset and
time-shuffle controls remained near chance. This is the first protected
1→5→6→7 compounding result. It is still a supervised-bootstrapped
reward-continuation claim, not cold-start reward-only discovery. The next
frontier is 8-back or transfer to a different cognitive primitive, with the
same retention and causal gates.

## Eighth-back protected representational rung

The generic RAM bridge now accepts eight opaque snapshots (the task uses ten
trials so the warm-up remains separate from the target-bearing region). An
eight-update reward compatibility probe stayed at chance, so the established
256-update dense diagnostic was the only justified escalation. It reached
**60.55%** eligible 8-back from a **46.88%** parent, with reset **50.00%** and
time-shuffle **44.92%**. A subsequent eight-update reward continuation did not
improve that point and was stopped by the progress gate.

The new eighth-history path nevertheless passed the complete retention ladder
on independent 8,192-trial audits: 7-back **68.55%**, 6-back **76.07%**,
5-back **80.11%**, and 1-back **91.95%**. Reset and time-shuffle controls
remained near chance at every rung. This is a protected eighth-back
representational breakthrough, not a claim of reward-only mastery; the next
high-ROI question is how to make reward fine-tuning discover the already
decodable eighth-back relation without sacrificing the ladder.

## Next frontier

The system now has a replicated, causal **learn → check → continue** loop for a
harder cognitive primitive while retaining the mastered one. The new protected
extension adds a second loop: **freeze inherited path → train appended RAM
columns → reward-fine-tune → audit retention**. The next frontier is 8-back or
transfer to a genuinely different primitive. Any future cold-start reward claim
must still pass the same reset, time-shuffle, and full-ladder retention audits.

## Eighth-back acquisition and retention audit (2026-08-02)

The initial 8-back representational rung was real: the dense diagnostic improved
eligible accuracy from **46.88% to 60.55%**, while history reset was **50.00%**
and time-shuffle was **44.92%**. Three correctly matched supervised continuation
blocks then crossed the local acquisition threshold, reaching **77.15%** on the
third block (reset **50.00%**, shuffle **52.34%**). This is evidence that the
generic RAM bridge can learn the eighth relation; it is not yet a protected
continual-learning result.

The retention audit exposed the actual blocker. Relative to the original
retention-safe 8-back checkpoint, the matched continuation branch raised 8-back
from **60.69% to 78.96%** on the saved 10,240-trial checkpoint audit but reduced
1-back from **90.40% to 81.26%**, 5-back from **71.89% to 53.87%**, and 6-back
from **68.48% to 55.27%**. The branch is therefore rejected despite its
impressive new-task score. This is a causal retention failure, not a
measurement artifact.

Two repairs were tested in short, pre-registered forks. Multi-rung supervised
rehearsal (1/5/6/7-back at equal weight) improved 8-back from **73.05% to
77.93%**, but the original-safe baseline audit still showed 1-back **90.40% →
81.26%**, 5-back **71.89% → 53.87%**, and 6-back **68.48% → 55.27%**. An
append-only relation residual that froze the inherited controller/bridge path
reached only **59.96% from 58.40%** (+1.56 points), below the +5-point
escalation gate; reset was **50.00%** and time-shuffle **47.66%**. Both branches
are rejected. The reusable lesson is that replaying old labels is not enough
when the trainable RAM-to-intention path remains shared; the next repair must
make the new relation conditional or physically isolated before more updates
are purchased.

A matched full-history residual was also tested from the same safe parent. It
fit its tiny diagnostic batches (late batches exceeded 90% eligible accuracy)
but stayed at **59.18% → 59.38%** on the held-out check, with reset **50.00%**
and shuffle **52.15%**. This rules out simply widening the isolated residual:
the next bottleneck is conditional skill selection or an actually separate
skill/memory path, not residual capacity.

Two final conditional variants were held to the same 64-update gate. A full
history residual with a learned history router stayed at **59.96% → 59.77%**;
conditioning the residual on the raw history did not make it selective. A
zero-initialized branch conditioned on the controller's previous scalar reward
and feedback flag reached **60.35% → 60.74%**, with reset **50.00%** and
time-shuffle **47.46%**. Neither cleared the gate. Together these tests rule
out the cheap explanations “the residual needs more width” and “the last
reward is enough task context.” A future skill-selection design must provide a
stable task/context representation learned from demonstrations or long-term
memory, then prove that selection causally before spending a longer run.

## Task-context localization (2026-08-02)

Before adding another skill-selection mechanism, a disposable probe measured
whether task identity is present in experience at all. A five-way classifier
over raw text stimulus sequences reached **55.66%** held-out accuracy (chance
**20%**); adding the verifier's feedback target history raised it to
**74.61%**. The shuffled-label control was **19.73%**, confirming real signal
and not a split artifact. This is verifier-side diagnostic evidence only:
targets are not exposed to the agent.

The actual action/reward-conditioned branch, which receives the agent's opaque
previous action and scalar reward rather than the privileged target, improved
8-back only **60.55% → 60.94%** at 64 updates (reset **50.00%**, shuffle
**53.32%**). The gap between privileged feedback decoding and the near-flat
causal branch localizes the next problem: one noisy outcome is not a stable
task representation. The next experiment should store a short, task-agnostic
trajectory of `(opaque event, opaque action, scalar outcome)` in working memory
and test whether that context becomes decodable before training another answer
path. This is the first evidence-backed justification for a demonstration-
conditioned skill selector.

The frozen-agent trajectory probe made the distinction sharper. With 256
lifetimes per ring, action/reward-only context decoded task identity at
**60.16%** held-out (sensory-only **41.80%**, combined **61.72%**). Scaling to
1,024 lifetimes raised action/reward-only to **66.11%** (sensory-only **57.03%**,
combined **64.84%**), so the signal is genuine but data-hungry. Feeding the
same opaque action/reward history directly into an isolated answer residual did
not exploit it: depth 4 fell **61.33% → 59.38%**, and depth 8 fell
**62.50% → 59.18%**. Both reset controls stayed at **50.00%** and shuffle
controls near chance.

The first zero-label representation probe was also negative. Predicting the
mean future return from an eight-record context produced only **44.14%** task
decoding versus **62.11%** for the raw action/reward trajectory; a shuffled-
return control was **41.02%**. Predicting the complete future opaque
action/reward trajectory improved the self-supervised latent only to **48.05%**
versus **55.86%** raw (shuffled-target control **41.02%**). Naive return or
future-trajectory prediction therefore does not preserve the task context
needed for skill selection. The next high-ROI representation test should use
episodic trajectory retrieval or a contrastive objective that preserves
episode-level relation structure, rather than another scalar-prediction head.

The episodic retrieval check was negative as well: standardized nearest-neighbor
matching over complete opaque action/reward trajectories reached only **44.14%**
with k=9 (k=1 **37.11%**, k=5 **41.80%**), while the shuffled-label control was
**17.97%**. Exact trajectory storage therefore does not solve skill selection
by surface similarity. The remaining high-ROI representation candidate is a
relational/contrastive context encoder that preserves episode-level structure,
not a larger nearest-neighbor bank.

This localizes the next bottleneck to credit assignment/representation
learning, not missing information or RAM capacity. The next probe should use
episodic retrieval or a contrastive episode objective that preserves relation
structure, then test whether its frozen representation linearly decodes task
context. Only after that representation gate passes should it drive the answer
branch. This keeps the eventual path zero-label while avoiding another
answer-loss-only fork.

The high-precision audit is saved as
`nback8_multirung_retention_audit_seed48300.json`, and the reusable evaluator is
`experiments/unified_cognitive_controller/audit_nback_checkpoint_retention.py`.
The current frontier is therefore **retention-safe 8-back acquisition**, not
another unbounded continuation. Any new branch must first pass a tiny gain gate
and then the original-safe full ladder (1/5/6/7/8), with reset and time-shuffle
controls, before it can be promoted.

## Zero-label context-representation probes (2026-08-02)

The raw action/reward trajectory contains task context, but the first family of
generic self-supervised encoders did not preserve it. A contrastive predictor
trained to match a prefix with its future suffix reached **35.94%** held-out
five-way task decoding at depth five, versus **63.67%** for the raw trajectory;
the shuffled-suffix control was **26.56%** (chance is 20%). The same pattern
held across prefix depths 2/4/6: predictive latents were **27.34/35.16/42.19%**
while raw trajectories were **53.13/57.81/59.38%** and controls were
**27.34/35.16/36.72%**. Future behavior is therefore not a useful generic
training target for this context.

Three causally closer objectives were tested as well. Recurrent next-outcome
predictors decoded only **40.63–44.14%** (shuffled-outcome controls
**31.64–39.06%**) at 256–1,024 updates; two
masked-view alignment reached **31.25%** versus raw **59.77%** (shuffled-pair
**34.77%**); and temporal-intact-versus-time-shuffled training reached
**29.69%** versus raw **53.52%** (shuffled-consistency **36.33%**). A masked
event predictor (current sensory plus prior opaque action/outcome, predicting
the current action and outcome) reached **39.06%** versus raw **62.11%**, with
its shuffled-target control at **28.13%**. Extending that same objective to
1,024 updates did not reveal an ignition transition: **33.20%** versus raw
**59.77%**, control **27.73%**.

These are diagnostic-only, verifier-side labels; no ring/task label entered
any representation loss. The controls and the phase-transition-sized run rule
out a simple optimization-budget explanation. The conclusion is not that
task context is absent—raw trajectories decode it—but that predicting generic
future events, aligning noisy views, or detecting temporal corruption discards
the relation needed for skill selection. The next high-ROI branch is therefore
an explicit **relation-aware episodic memory gate** whose writes and reads are
trained/evaluated causally, rather than another generic predictive head. It
must still pass the existing reset, time-shuffle, and full 1/5/6/7/8-back
retention ladder before any longer run.
