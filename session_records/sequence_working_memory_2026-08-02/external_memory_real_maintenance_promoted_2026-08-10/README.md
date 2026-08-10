# Real external-memory maintenance policy

This archive records the first maintenance audit whose utility comes from
actual external-bank transactions rather than synthetic action labels.

The policy selects `grow`, `share`, `compress`, or `defer` from generic bank
telemetry. Utility is produced by retention-verified capacity growth,
held-out-equivalent factual parameter sharing, compressed-byte savings, or a
verified no-op. Three seeds pass fresh, reward-shuffled, action-shuffled,
real-transaction, compression-byte, growth, persistence, mutating-probe
atomicity, frozen-controller, replay-free, and one-update-per-utility gates.

Held-out utility is `0.95` on all three trained seeds versus `0.70`, `0.7375`,
and `0.70` for matched fresh controls. The runs save `5664`, `5472`, and
`5616` bytes through retained compression candidates. The action-shuffled
held-out controls score `0.25`, `0.70`, and `0.2375`, respectively, versus
`0.95` for the trained policy.

This promotes learned maintenance selection over real external-memory
receipts. It does not establish learned verifier design, autonomous candidate
equivalence discovery, unrestricted memory growth, or general continual
learning.
