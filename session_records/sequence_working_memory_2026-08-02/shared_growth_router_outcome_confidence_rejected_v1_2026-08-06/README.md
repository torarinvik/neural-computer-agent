# Shared growth router: outcome-confidence acquisition rejection (2026-08-06)

This is a matched control against the promoted 6→8→10→12 shared-router rung.
It keeps the controller, query representation, candidate bank, hidden width,
and 8,192-update budget fixed, but replaces round-robin target acquisition with
an outcome-driven opaque confidence scheduler.

## Result

The control is rejected. On seed 69316, phase minima fall to `0.9063`,
`0.6719`, and `0.7188`, compared with the promoted round-robin seed's
`0.9844`, `0.9688`, and `0.9219`. Old-route retention, causal credit,
permutation, reward-shuffled, and zero-replay controls remain informative, but
new-route recovery and retention/reversal fail.

The failure is mechanistic: concentrating shared-router updates on currently
weak candidates changes the shared representation and destabilizes candidates
that had already acquired usable routes. A confidence score alone is not a
retention mechanism. The scheduler is not part of the production package.

## Accounting

The run uses the same 32-capability, three-shift schedule and records
`replayed_examples: 0`. It is a diagnostic rejection, not a promotion or a
claim of improved sample efficiency.

Evidence is in `report_seed69316.json`; its checksum is recorded in
`SHA256SUMS`.
