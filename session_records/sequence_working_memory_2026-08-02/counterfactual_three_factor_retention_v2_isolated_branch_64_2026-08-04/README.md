# Retention write-intervention rejection — 2026-08-04

This diagnostic used the parent-acquisition three-factor trainer followed by
64 parent-protected retention updates. The retention intervention was corrected
so only the selected branch differed: all non-branch positions were forced to
skip in both common-random arms.

The parent remained intact (`0.9922` mastered-primitive retention), but the
target-conditioned write policy did not improve: target-first recall was
`0.3945` and target-last was `0.9922`. This rejects the intervention as a
retention solution. The remaining credit problem is specifically the negative
utility of overwriting a target already held in memory; a branch where the
target was never retained cannot provide that negative signal.

See `report.json` and `sample_efficiency_ledger.json` for full accounting.
