# Reusable growth prior at 128 updates: rejected (2026-08-06)

This is the reduced-budget control for the copy-on-write external growth
prior with the capability-specific score head reset to neutral. It reaches
phase one and phase two, but both seeds fail final length-twelve route
recovery at `0.71875` and `0.734375`; seed 69317 also fails the full-bank
retention gate. The causal, credit, permutation, antithetic null, and
zero-replay controls still pass.

The result rejects the claim that prior reuse alone halves the required
acquisition budget. Full 256-update prior runs remain promoted for safe reuse,
while late-shift acquisition depth and prior calibration remain open.
