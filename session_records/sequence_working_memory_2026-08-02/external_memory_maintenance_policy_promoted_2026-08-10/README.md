# Learned external-memory maintenance policy

This archive records the promoted finite-budget maintenance rung added on
2026-08-10. A replaceable policy selects among `grow`, `share`, `compress`,
and `defer` from generic storage telemetry and consumes one scalar verifier
utility per decision. The controller is frozen, replay is zero, and the policy
does not see task labels, modalities, semantic IDs, or protocol actions.

Three seeds passed the matched fresh, reward-shuffled verifier, all-actions,
frozen-controller, replay-free, and one-update-per-utility gates. Held-out
utility was `0.50`, `0.75`, and `0.75` for trained policies versus `0.25` for
both fresh and shuffled controls on every seed.

This is a bounded action-selection result. It does not establish a learned
verifier, autonomous equivalence discovery, unrestricted memory growth, or
general continual learning. The source experiment is
`experiments/external_memory_maintenance_policy/`; the implementation is the
versioned `ExternalMemoryMaintenancePolicy` boundary plus router integration.

Reports and the accounting ledger are checksummed in `SHA256SUMS`.
