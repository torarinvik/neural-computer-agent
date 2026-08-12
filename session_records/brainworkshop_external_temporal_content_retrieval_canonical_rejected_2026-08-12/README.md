# Canonical external temporal content retrieval rejection

The v2 content-address experiment now uses `ExternalTemporalAddressIndex` and
the canonical history bridge instead of the old random basis decoder. Storage
qualification passed: two routes were written, unknown keys missed, reload
preserved noisy-key hits, checksum corruption was rejected, clear removed
hits, the controller and event encoder stayed frozen, and replay was zero.

The learned source and target route/readout gates did not reach mastery at the
512-update rung (source `0.7711`, target `0.7630`, noisy source `0.7445`, noisy
target `0.7630`). The experiment is therefore rejected. The failure isolates
the next bottleneck as outcome-only query-conditioned route/readout learning,
not address-index storage. No historical v1 promotion is carried forward as
current canonical evidence.
