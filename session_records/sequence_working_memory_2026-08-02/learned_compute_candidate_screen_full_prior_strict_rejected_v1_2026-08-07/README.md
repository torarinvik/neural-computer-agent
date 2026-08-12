# Full append-prior strict control rejected (2026-08-07)

This matched control copies the full mastered factorized screen into each
append stage. It is compared with the query-path prior at the same bank-20,
five-stage, 32-update boundary under the new strict per-candidate verifier.

Full transfer repairs the unseen extension on both seeds: unseen routing is
`1.0000/1.0000` and every unseen target clears the mastery floor. It does not
repair the frozen base on seed `69317`, whose known-target routing remains
`0.8542` with per-target holes at `0.7` and `0.0`. The replicated promotion
therefore remains rejected. The candidate-key diagnostics are unchanged from
the query-path control, confirming that the remaining failure is source-screen
mastery rather than append initialization.

This result supports testing more effective source training or alignment, but
does not justify claiming that full inheritance solves continual growth.
