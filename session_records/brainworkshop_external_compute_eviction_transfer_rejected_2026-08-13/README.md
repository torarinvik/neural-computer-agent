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

The first set-relative representation follow-up was also replicated at the
same calibrated rung. It preserved exact candidate-row permutation behavior,
but both seeds still produced `0.0000` inherited transfer versus `1.0000`
fresh transfer. This rejects set-relative normalization alone as the missing
transfer mechanism; the next adapter must use behaviorally grounded evidence
and a verifier safety gate, not merely a more invariant view of raw weights.

The detailed scale-stable follow-up reports are
`seed17_scale_stable_residual.json` and
`seed18_scale_stable_residual.json`. The set-relative reports are
`seed17_set_relative_residual.json` and
`seed18_set_relative_residual.json`.

The verifier-gated variant on seed `17` kept the residual behind the frozen
fallback for its first four probes, promoted it only after four consecutive
non-inferiority observations, and recorded no harmful probe across all eight
transfer updates. This is a narrow safety result: inherited transfer remained
`0.0000` versus `1.0000` fresh transfer, so the gate prevents additional harm
but does not create reusable eviction knowledge. It is archived as
`seed17_set_relative_safety_gated.json`.

The candidate-order control was then replicated across both seeds. Feature
rows and their verifier outcomes were permuted together on every transfer
update; inherited transfer remained `0.0000` on both seeds, while the fresh
baseline measured `1.0000` and `0.7500`. This removes the fixed physical-slot
tie as an explanation for the negative result. The control reports are
`seed17_set_relative_permuted.json` and
`seed18_set_relative_permuted.json`.
