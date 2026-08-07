# Shared-screen unseen-candidate calibration (rejected, 2026-08-07)

This control attempted to acquire two outcome-unseen candidates by applying
64 fresh scalar-outcome updates to the same learned screen that already
protected four known candidates.

The unseen candidates were acquired at `1.0000` for both seeds, but the known
candidate novel-context route collapsed from `1.0000` to `0.2083/0.2500` and
candidate permutation fell to the same levels. The frozen controller and
zero-replay gates remained intact, but known screen memory was catastrophically
overwritten.

This rung is rejected. It proves that shared parametric screen adaptation is
not a safe continual-memory update. The next implementation target is an
append-only isolated screen extension with frozen prior routing and an explicit
confidence/failure gate.
