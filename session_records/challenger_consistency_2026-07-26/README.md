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
