# Outcome-only n-back-5 external-history depth selection and maintenance (2026-08-13)

This audit gives an isolated persistent memory-side policy candidate active
query counts `1..6` for the five-file external computation bank. The policy
sees only attempted-lifetime scalar accuracy and exposes a depth only after a
stable-prefix mastery gate. The controller, event frontend, and external file
parameters are not updated during calibration.

The bank is `symbol_parity`, `triplet_parity`, `parity2`, `switch_binary`, and
`nback5`, with a six-record shared flattened event window. Query count `q`
means `q - 1` preceding records plus the current event. Across seeds 17 and
18, n-back-5 selected query count 6; all files selected a stable depth and
retained mastery on four fresh lifetimes. The selected profiles were
`[1,4,2,2,6]` and `[1,3,2,2,6]`, showing that the policy chooses from evidence
rather than relying on a fixed hand-written profile.

The maintenance test injected four patient scalar failures into the first
file's selected depth 1. Both seeds demoted it, probed fresh depth 2, and
re-promoted that replacement at 1.0 on eight fresh lifetimes. Policy reload,
controller/frontend immutability, file immutability, shuffled-outcome
fail-closed behavior, and zero replay all passed.

This promotes outcome-only active-depth selection and replay-free maintenance
through n-back-5. It is still a bounded external policy: it does not establish
learned neural depth inference, learned compression, unrestricted memory
growth, arbitrary program induction, or general continual learning. Full
reports and accounting are in `seed17_report.json`, `seed18_report.json`, and
`sample_efficiency_ledger.json`.
