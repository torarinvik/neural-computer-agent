# Protected-plasticity successor frontier — 2026-08-04

This diagnostic tested a generic protected-plasticity update on the archived
successor-slot trainer. Each epoch first accumulated a gradient from protected
rehearsal rows, then applied only fresh-target updates after projecting away
the opposing component. An optional post-AdamW projection and a verifier-
visible behavioral-retention rejection were also tested.

The learner boundary remained unchanged: latent controller features, opaque
attempted actions, and scalar attempted-action outcomes only. No correct
unattempted action, task ID, or semantic field entered the trainer.

## Result

The strongest single arm passed every gate at one seed but did not replicate
the retention gate:

| Arm | Complement causal gain | Span-9 retention | Span-10 retention | Decision |
| --- | ---: | ---: | ---: | --- |
| Protected, penalty .03, seed 93748 | **+10.06 pp** | **−1.80 pp** | **−1.24 pp** | safe single seed |
| Protected, penalty .03, seed 93750 | +9.01 pp | −4.50 pp | −3.47 pp | reject |
| Protected + behavior rejection .005, seed 93750 | +6.86 pp | −2.53 pp | −1.78 pp | reject |
| Protected + loss rejection .001/.01, seed 93750 | ≈0 pp | not useful | not useful | over-constrained |

The seed-93748 audit also returned blank-cue accuracy 46.46% and full-reset
accuracy 50.00%. The behavior-rejection arm accepted 559 of 640 target
updates; the stricter loss-rejection arms accepted only 52 and 62 updates,
respectively, and stayed at chance on the new primitive.

The generic reference direction therefore supplies a real causal learning
signal, but it is not a reliable retention mechanism. The remaining
bottleneck is context-selective protection: an aggregate old-task gradient
does not identify which parts of a new slot should remain plastic on a new
context. This branch is not promoted and must not replace the existing
retention recipe.

## Accounting

Each full arm used 256 fresh logical lifetimes and 2,560 unique verifier bits
(the persisted replay buffer is not charged as new experience). The ordinary
control used 1,024 optimizer updates; the protected path used 640 target
updates plus 384 rehearsal-gradient evaluations. Training time was about
6.1–9.5 seconds on MPS, excluding the independent 1,024-lifetime audit.

## Audit correction

The historical `span11_binary_hidden_criticbinary_93712.json` configuration
records `test_operation=complement`, but its stored forward/reverse audit
fields match the mixed-operation audit. Re-running with the explicit current
operation showed that the complement benchmark is a distinct primitive and
must be audited with `--test-operation complement`; the old report is retained
as historical evidence, not used for a new promotion claim.

The next useful experiment is a context-selective, verifier-visible retention
constraint with the same zeroed-slot, shuffled-outcome, blank/reset, and old-
skill gates. Do not scale the span-eleven branch from this result alone.
