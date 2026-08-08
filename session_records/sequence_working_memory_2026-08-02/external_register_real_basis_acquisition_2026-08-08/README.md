# Real external-register basis acquisition — 2026-08-08

Three opaque source primitives (`rotate`, `global_parity`, `complement`) were
trained into independent external basis slots. Their fresh verifier outcome
matrix was used to update the compatibility prior once, then a held-out
`prefix_parity` acquisition was routed through the live register scheduler.

Both seeds produced distinct source outcome rows, preserved fresh-verifier
admission as the authority, and correctly found no passing existing basis for
the unseen target. The target therefore requested growth rather than being
incorrectly reused. No replayed examples were used.

This promotes real multi-slot opaque acquisition and no-false-admission
behavior. It does not yet demonstrate positive transfer to a genuinely new
primitive; the correct result here is verified growth.

## Growth execution follow-up — rejected

The selected slot-3 growth branch was then trained on held-out `prefix_parity`.
Both seeds reached high final target accuracy (`0.9766` and `0.9453`) and
retained all source capabilities, with the old basis digests unchanged.
However, neither reached a stable-prefix threshold, and shuffled-outcome
controls remained above the rejection floor (`0.9922` and `0.9531`). The
growth result is therefore rejected for promotion. The next bottleneck is
causal credit/verification dependence in new-slot acquisition, not retention
or append-only capacity.

The causal follow-up switched only new-slot training to `attempted_bce`, so
the optimizer received delivered scalar outcomes rather than verifier-private
correct-action utilities. Shuffled-training controls then collapsed to
`0.4766` and `0.5000`, confirming causal dependence. Normal target accuracy
remained `0.9375` and `0.9063`, with source retention intact, but stable-prefix
promotion still failed. The corrected result remains rejected for stability,
while the credit-path repair itself is retained.

## Staged scalar-credit follow-up — rejected

The next rung used a two-stage curriculum: a short-span warmup followed by
full-span target training, with source retention checked after warmup and
again after growth. Seed `69316` reached `0.8125` final target accuracy and
seed `69317` reached `0.8320`; both retained all source skills, rejected
shuffled training, and left old basis digests unchanged. Neither produced a
stable full-span prefix, and the reward-shuffled control remained too strong.
The staged curriculum therefore does not promote new-skill acquisition yet.
It establishes that the current failure is not simply catastrophic forgetting:
the remaining blocker is reliable scalar-credit learning and control
sensitivity on the full target distribution.

The audit also corrected propensity accounting: sampled actions use an
epsilon-smoothed behavior policy, and the exact smoothed propensity is now
carried in the opaque action record and used by policy-gradient credit.

## Exact-propensity REINFORCE follow-up — rejected

Replacing attempted-outcome BCE with exact-propensity REINFORCE did not
produce stable full-span acquisition in either seed (`69316`: `0.8281`,
`69317`: `0.7969` final accuracy; both had no stable prefix). Source
retention and shuffled-training rejection remained intact. This path is
therefore retained as a valid baseline, not promoted as the solution.

## Fixed-baseline policy-gradient follow-up — rejected

A second scalar-only policy-gradient path used an action-independent `0.5`
verifier baseline and a small entropy floor, avoiding the batch-centered
advantage used by the earlier REINFORCE path. It still failed stable full-span
acquisition (`69316`: `0.8164`; `69317`: `0.7813`) while retaining source
skills and rejecting shuffled training. The estimator is retained as a valid
option, but the bottleneck is now localized to routing credit through the
new basis and decoder representation.

## Eligibility-trace scalar credit follow-up — rejected

Discounted return-to-go credit was added so each delivered outcome could
credit earlier selected actions through the sequence. It performed worse than
the one-step estimators (`69316`: `0.7930`; `69317`: `0.7656` final accuracy;
neither had a stable prefix). Retention and shuffled-training rejection still
passed. Temporal credit accumulation is therefore not promoted as the default
new-slot learner.

## Basis-focused acquisition follow-up — rejected

The new slot was given a dedicated scalar-credit phase with the decoder
frozen between warmup and final joint training. The focus phase remained near
chance-plus (`69316`: `0.6563`, `69317`: `0.6563`), and final target accuracy
was `0.8477` and `0.8086`, with no stable prefix. Source retention and causal
shuffle controls passed. Isolating basis updates does not solve acquisition;
the next bottleneck is the representational interface between the frozen
controller/register and a freshly added computation slot.

## Near-identity fresh-slot initialization — rejected

A fresh basis slot was initialized as a gated near-identity residual to avoid
perturbing the established register before learning. It did not improve the
causal growth rung (`69316`: `0.7578`; `69317`: `0.8164`), and one seed lost
source retention. The production initialization remains unchanged; the
interface problem requires a learned representation path, not only safer
initialization.

## Bounded learned-event window follow-up — positive mechanism, not yet promoted

The external register v4 now carries a bounded window of standardized learned
event tensors and masks it through quiet ticks. New computation slots can read
that window alongside the register and opaque instruction, while raw modality
formats remain outside the boundary. In the two-seed causal audit, final
target accuracy rose to `0.9180`/`0.9063`, source retention was
`0.9570`/`0.9336`, shuffled-training remained below threshold, and old basis
digests were unchanged. The full-span progress checkpoints still failed the
stable-prefix gate, so this is a promoted interface mechanism and a positive
direction—not yet a promoted continual-learning capability.

The longer 256-update follow-up reached final accuracy `0.9980`/`0.9766`,
but full-span checkpoints later fell to `0.5684`/`0.6133`. This confirms that
the event window improves access to task information without solving protected
continual acquisition; final-sample accuracy must not be used as the mastery
criterion.

An attempted length-9 window retained the full rendered episode but performed
worse: full-span progress ended at `0.6289`/`0.6055` and both candidates were
rolled back. The shorter length-4 window remains the empirically stronger
representation; more context is not automatically more useful.

The event-window state also now preserves the entire prior window on quiet
ticks instead of shifting and duplicating its final token. The corrected
two-seed audit produced the same all-active scores (`0.9180`/`0.9063`) and
rolled both unstable candidates back, confirming that this was a state
correctness repair rather than an unverified capability gain.

## Verifier-gated consolidation/rollback

External growth now uses the shared retention gate transactionally. An
unstable candidate is removed from the newest slot, its opaque instruction is
restored, and previously mastered slots remain untouched; only a candidate
that passes stable-prefix mastery and the retained-capability floor is frozen
as consolidated. A rollback smoke audit rejected an unstable candidate with
candidate prefix minimum `0.5` and confirmed `rollback_applied: true`.

## Action-independent actor–critic follow-up — rejected

A trainer-only value head was added to estimate expected scalar verifier
success from the learned register, reducing policy-gradient variance without
correct-action labels. Across two seeds it reached `0.8320`/`0.8281` with no
stable prefix. Source retention and causal controls passed, but it did not
beat attempted-outcome BCE; the value head remains an optional experiment,
not the default learner.
