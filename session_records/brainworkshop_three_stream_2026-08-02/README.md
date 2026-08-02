# Brain Workshop 4-back and protected 5-back continuation

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

## Next frontier

The system now has a replicated, causal **learn → check → continue** loop for a
harder cognitive primitive while retaining the mastered one. The new protected
extension adds a second loop: **freeze inherited path → train appended RAM
columns → reward-fine-tune → audit retention**. The next frontier is to make
the fifth-back adapter as sample-efficient as the 93% supervised ceiling,
preferably by learning a task-agnostic progress/stop signal and by testing a
new cognitive primitive after the 5-back rung. Any future cold-start reward
claim must still pass the same reset, time-shuffle, and retention audits.
