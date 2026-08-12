# Inherited learned-screen prior rejected (2026-08-07)

This matched control copied the mastered base address blueprint into each
new extension before the same 128-update mixed `[1, 2]` calibration. The
copy-on-write boundary preserved known routing and the frozen base, but new
candidate acquisition reached only `0.3333` and `0.6667` across the two seeds;
stage permutation also failed.

The result rejects blind inheritance of the base weights as a sample-
efficiency shortcut. Fresh extension initialization remains canonical. The
copy-on-write mechanism is retained only as an explicitly controlled,
unpromoted diagnostic for future selective-prior designs. Report checksums
are in `SHA256SUMS`.
