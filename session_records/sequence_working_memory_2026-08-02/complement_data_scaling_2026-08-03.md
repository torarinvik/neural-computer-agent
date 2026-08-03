# Complement data/retention curve — 2026-08-03

This record follows the replicated 256-lifetime complement result. Every row
starts from the frozen span-nine/span-ten parent, appends one zero-output slot,
and is judged on an independent 1,024-lifetime audit. The causal number is
child accuracy minus the same checkpoint with the appended slot zeroed. An old
span fails if either retention change is below −2 points.

| Arm | Fresh lifetimes | Causal gain | Span 9 Δ | Span 10 Δ | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 93748 | 256 | +9.12 pp | −0.01 | −0.04 | safe |
| 93750 | 256 | +7.51 pp | −1.23 | −0.86 | safe |
| 93755 | 512 | +10.94 pp | −0.04 | −0.06 | safe |
| 93756 | 1,024 | +16.15 pp | −2.68 | −2.08 | reject |
| 93757, penalty .03 | 1,024 | +15.47 pp | −1.80 | −1.39 | safe, single seed |
| 93758, penalty .03 | 1,024 | +16.89 pp | −2.11 | −1.11 | reject |
| 93758, penalty .05 | 1,024 | +16.31 pp | −1.81 | −1.08 | safe, single seed |
| 93759, penalty .05 | 1,024 | +15.58 pp | −2.49 | −2.47 | reject |
| 93760, penalty .1 + replay 2× | 1,024 | +17.16 pp | −2.64 | −2.09 | reject |
| 93761, provenance 1× | 1,024 | +17.17 pp | −2.17 | −1.89 | reject |
| 93762, 512+512 old rehearsal | 1,024 | +11.62 pp | +0.00 | +0.00 | safe, lower utility |
| 93763, provenance 10× | 1,024 | +18.67 pp | −1.88 | −1.81 | safe, single seed |
| 93764, provenance 10× | 1,024 | +17.04 pp | −5.04 | −4.08 | reject |
| 93765, provenance 10× | 1,024 | +16.77 pp | −3.21 | −2.17 | reject |

All rows used the same visible complement cue, scalar outcomes, and binary
outcome-only loss. Cue-blank and complete-memory-reset controls stayed at
chance for the reported candidates. The machine-readable reports are the
matching `complement_*.json` and `*_audit.json` files in this directory.

## What the curve says

More unique target lifetimes reliably increase complement accuracy from roughly
58–60% at 256 to 61.5% at 512 and 66–69% at 1,024. However, the retention gate
becomes seed-sensitive at 1,024: soft residual penalties, replay weighting,
ordinary provenance supervision, and even a stronger provenance weight do not
yet guarantee old-skill preservation. Extra old-task rehearsal preserves the
old skills but drops the new primitive to 62%, which is a poor sample-efficiency
tradeoff.

The correct current claim is therefore **data helps, but the new slot's
context-selective plasticity is not yet reliable**. No 1,024 checkpoint is
promoted. The best single arm (`93763`, 69.19%) is archived only as a
diagnostic candidate; two matched replications (`93764`, `93765`) fail the
retention gate. Do not scale to 2,048 until the retention variance is reduced.

## Continuation bookkeeping fix

The existing-slot continuation path now removes an inherited final-slot logit
from fresh buffers and from replay buffers whose provenance matches the parent
checkpoint. Its replay residual/gate/logit penalties are trust-region penalties
around the parent's behavior, rather than absolute penalties that would erase
an already learned slot. A regression test covers the exact subtraction. The
first continuation runs before this fix were invalid because they double-counted
or penalized the inherited slot; they are not scientific results.

The next high-ROI branch is a genuinely context-selective, retention-aware
plasticity constraint, evaluated with the same zeroed-slot, cue-blank, reset,
and old-skill gates. More data alone is not yet justified.

The follow-up retention controls are recorded separately in
`complement_retention_controls_2026-08-03.md`; they confirm that extra
rehearsal, gate-only calibration, and a smaller slot do not yet solve the
leakage.
