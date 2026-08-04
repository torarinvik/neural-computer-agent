# v54 optimizer-state reset diagnostic

v54 tests whether resetting the write-policy output at the parent-to-retention
transition is being undone by stale Adam moments. The reset now clears the
optimizer state for the generic write-policy parameters as well as restoring a
neutral output prior.

Matched seed-19 runs use 1,024 parent updates and 256 retention updates. Both
arms pass the narrow single-run retention gate with identical causal metrics,
but the reset arm reaches stable threshold later (`23,040` versus `17,920`
verifier bits). It is therefore a correctness fix for phase isolation, not a
promotion candidate or a learned capability gain. No transfer or persistent
memory claim is made.

The default protocol remains unchanged. See the paired JSON reports and the
sample-efficiency ledger in this directory.
