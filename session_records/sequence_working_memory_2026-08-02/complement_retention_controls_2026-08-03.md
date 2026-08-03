# Complement retention-control follow-ups — 2026-08-03

After the 1,024-lifetime curve showed seed-sensitive old-skill leakage, four
small controls tested different explanations. None is promoted.

| Arm | Complement | Span 9 Δ | Span 10 Δ | Reading |
| --- | ---: | ---: | ---: | --- |
| 93766, provenance 10× + 256 old rehearsal | 67.38% | −5.56 pp | −3.86 pp | more rehearsal at this budget is not sufficient |
| 93767, gate-only old calibration | 51.18% | — | — | erases the new skill |
| 93768, gate calibration + fresh preservation | 53.18% | — | — | gate distributions overlap |
| 93769, 64-wide successor slot | 62.23% | −3.47 pp | −2.83 pp | smaller capacity does not solve leakage |

The exact machine-readable reports are the corresponding
`complement_1024_*93766–93769*.json` files in this directory. All checks used
the same zeroed-slot, cue-blank, reset, and old-skill audits.

## Conclusion

The new complement information is learnable and causally used, but the current
successor slot does not reliably know when its residual should be active on
unseen old-task contexts. More target data raises new-task accuracy; generic
provenance gates, absolute penalties, gate-only calibration, and simply
shrinking the slot do not robustly preserve old skills. Extra rehearsal can
preserve old behavior only at a substantial new-task sample-efficiency cost.

The next architectural experiment should therefore expose a genuinely
task-agnostic context/novelty representation to the plasticity gate (or use an
explicit verifier-side promotion/rejection population), with a pre-registered
retention gate. Do not spend more compute on 2,048 target lifetimes until that
gate is measurably selective.
