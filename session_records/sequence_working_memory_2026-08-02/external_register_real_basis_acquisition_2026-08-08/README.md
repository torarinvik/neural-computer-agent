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
