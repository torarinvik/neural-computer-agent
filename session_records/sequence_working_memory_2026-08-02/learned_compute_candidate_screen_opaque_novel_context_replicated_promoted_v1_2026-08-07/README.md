# Generalizing learned compute-candidate screening (2026-08-07)

`LearnedComputeCandidateScreen` is the parametric memory-side screen for
opaque external compute. It consumes learned event queries and opaque
candidate keys, learns from attempted scalar verifier outcomes, and returns
an ordering only. Its factorized scorer is disabled at cold start and fresh
verifier admission remains mandatory.

The two-seed audit uses six opaque candidates. Four candidates receive fresh
outcome training; two remain outcome-unseen as a cold-start diagnostic. Fresh
novel contexts over the four known candidates route at `1.0000` for both
seeds, versus `0.2500` append-order cold start. Candidate permutation, exact
reload, frozen-core, and reward-shuffled null gates all pass. The outcome-
unseen candidates route at `0.0000` for both seeds, so the result is not
claimed as generalization to unseen computation.

A matched registry-family control is decisively rejected: novel-context
accuracy remains `0.2500`, candidate permutation is `0.2500`, and it does not
beat append-order cold start. This keeps the promotion scoped to the opaque
event representation and identifies representation transfer across the
registry family as an open bottleneck.

This promotes learned query generalization for known external candidates and
zero-replay screen adaptation. It does not establish arbitrary new
computation, unseen-candidate cold start, unrestricted memory growth, or
general continual learning. Full accounting is in
`sample_efficiency_ledger.json`; report checksums are in `SHA256SUMS`.
