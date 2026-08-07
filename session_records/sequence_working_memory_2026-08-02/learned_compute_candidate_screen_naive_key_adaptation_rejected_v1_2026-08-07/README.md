# Naive joint key adaptation rejected (2026-08-07)

This control optimizes the mastered base screen and all candidate addresses
together from the same scalar-outcome ranking loss at the bank-20/five-stage,
512-update boundary. It is intentionally unsafe: the base address memory is
not isolated from new acquisition.

The result confirms the failure mode. Both seeds collapse to `0.5` unseen
routing with multiple per-target holes; the hard seed also loses known-context
mastery to `0.8125`. The key representation becomes more diverse, but the
mastered route is no longer stable. This update policy is rejected and removed
from the canonical training path.
