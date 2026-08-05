# Three-factor parent credit — 2026-08-04

This rung separates probe action, write/skip, and recall action with
common-random verifier pairs. The recall factor uses a differentiable memory
transaction for the learned value path, while `memory_write_gradient=false`
detaches the forced write-gate gradient. The controller receives only opaque
events, opaque actions, and scalar outcomes.

Seed 17 reaches `0.9805` retention on mastered primitives and `0.7441` mean
retention on four unseen event-token pairs. Intact recall is `0.7871` and
reverse-order recall is `0.7422`. The run is not promoted: target-first is
`0.4688`, target-last is `0.9961`, and the stable retention gate is not met.
The matched reward-shuffled control remains at chance.

This is a positive parent/value-credit mechanism, not a general retention
result. The exact accounting is in `report.json` and
`sample_efficiency_ledger.json`.
