# Challenger consistency diagnostic — pre-registration

## Problem

Fresh confirmation reliably prevents mastered-policy corruption, and the disk
store is sound. The remaining failure is sensitivity: on seed 7973 the gap
challenger never produced a positive proposal within 720 verifier bits.

## Same-data diagnostic

Replay seed 7973 with the same initialization, randomized attempted actions,
720-bit budget, global-centered estimator, and mandatory fresh confirmation.
Change only challenger learning rate:

- `0.001`;
- established baseline `0.003` (already observed);
- `0.006`;
- `0.010`.

This is a diagnostic, not a capability replication. The candidate is selected
only from learner-visible proposal evidence:

1. earliest positive proposal lower bound;
2. then largest positive proposal lower bound;
3. private audited performance is reported but cannot select the candidate.

Any proposal still requires an independent positive confirmation block before
deployment. If no alternative rate produces a positive proposal within 720
bits, learning-rate tuning is closed and the next candidate is a parallel
challenger population. If one wins, it must pass two fresh seeds before
persistent integration is retried.

## Learning-rate result

No alternative produced a positive proposal. Final lower bounds were:

- `0.001`: `−0.0283`;
- `0.003`: negative in the original seed-7973 run;
- `0.006`: `−0.0114`;
- `0.010`: `−0.0088`.

Learning-rate tuning is closed.

## Shared-experience population fork

The next diagnostic keeps seed 7973, learning rate `0.003`, randomized actions,
incumbent, budget, and confirmation rule fixed. Four gap challengers use
different randomly initialized latent bases (seed offsets `11001`–`11004`) but
consume the identical logged attempted outcomes. The output layer remains
zero-initialized, so every challenger begins with the same action policy.

Selection is pre-registered from proposal evidence only: earliest positive
proposal, then largest lower confidence bound. Only that selected candidate's
fresh confirmation is interpreted. Private audit performance and other
candidates' confirmation results cannot select the winner. If no candidate
proposes, the population branch closes. A selected-and-confirmed candidate
must still pass two fresh seeds before persistent integration.

## Population result

None of the four latent initializations produced a positive proposal. Their
final lower bounds ranged from `−0.0149` to `−0.0442`. Initialization diversity
is not the limiting factor on this stream, so the population branch is closed.

## Doubly robust evidence fork

The next candidate changes only the promotion estimator. A two-fold
cross-fitted ridge model predicts attempted utility as a function of the four
generic context features and action. Promotion uses a doubly robust
policy-difference estimate: predicted incumbent/challenger values plus
inverse-propensity residual corrections.

Each record's predictions come from a fold that did not train on its outcome.
The estimator uses only attempted outcomes and remains unbiased under the
known randomized logger even if its outcome model is imperfect. Seed 7973 is
replayed diagnostically with all learning and confirmation settings unchanged.
It must produce a positive proposal without a mastered-policy promotion. A
pass requires two fresh seeds before persistent integration; a failure closes
this estimator.
