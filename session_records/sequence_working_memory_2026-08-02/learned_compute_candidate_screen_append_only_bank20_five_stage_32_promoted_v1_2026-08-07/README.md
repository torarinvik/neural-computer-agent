# Bank-20 five-stage append boundary (2026-08-07)

This audit increases sequential depth while keeping every append stage at two
opaque candidates: twenty total candidates, ten mastered base candidates, and
five isolated extensions. The controller and base screen remain frozen; each
later extension activates only after cumulative scalar-verifier failure, and
no examples are replayed.

At 32 calibration updates per stage, both fresh and full query-prior controls
pass both seeds. Fresh unseen routing is `1.0000/0.8958`; query-prior routing
is `1.0000/0.8958`. Known-context retention, stage-local permutation, exact
reload, frozen-core, reward-shuffled null, and zero-replay gates pass.

This promotes replicated five-stage bounded append growth with ten unseen
logical candidates. The matched fresh control means it does not establish a
prior-efficiency gain. Physical external state still grows linearly with each
stage, and this is not unrestricted memory growth, learned consolidation,
arbitrary new computation, or general continual learning.

Raw reports, accounting, and checksums are included here.
