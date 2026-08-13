# Held-out external eviction transfer rejection

This directory preserves the matched fresh-learner audit for a learned
external-capability eviction policy. It is negative evidence and is not a
promotion record.

The source cohort contains six mastered external compute files. A held-out
n-back-2 file is added cold, while the related n-back-3 source is reactivated
through a verifier-gated retention probe. The controller and learned event
frontend are frozen. The inherited policy and a fresh policy receive the same
fresh verifier probes and the same optimizer budget; replayed examples are
zero.

The first residual-only implementation discarded the frozen base score and
therefore tied the fresh policy at 1.0000. The corrected implementation adds
the isolated route residual to the frozen base. It exposes negative transfer:
seeds 17 and 18 both give inherited transfer accuracy 0.0000 versus 1.0000
for the matched fresh policy. Direct source/target mastery, target retention,
artifact integrity, frozen digests, and zero replay all pass, so this is a
policy-transfer failure rather than a file-learning or retention failure.

The n-back-4 target was not used because the current external compute ABI's
four-event history cannot represent the required five-event window; the
held-out n-back-2 target is ABI-compatible and absent from the source schedule.

The follow-up adapter uses a scale-normalized frozen prior plus an explicit
residual gain, so a new route can correct a badly scaled inherited logit
without changing unknown-route fallback. It improved seed 17 to 0.7500
held-out transfer accuracy, but seed 18 remained at 0.0000; the replicated
transfer gate still rejects inherited policy transfer. The next experiment
should improve permutation-invariant context/candidate representations and
use a safety-gated route adapter that cannot harm a new family before it earns
verifier evidence.

The detailed scale-stable follow-up reports are
`seed17_scale_stable_residual.json` and
`seed18_scale_stable_residual.json`.
