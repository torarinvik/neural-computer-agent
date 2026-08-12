# Replay-free external-history depth maintenance (2026-08-12)

This audit extends outcome-only depth selection with maintenance. After the
policy protects the minimal profile `[1, 3, 2, 2, 5]`, the selected depth for
the first file is exposed to four patient failures. The policy demotes that
stale candidate, skips it during re-probing, and evaluates the next candidate
from fresh scalar outcomes. The external file, decoder, controller, and event
frontend remain unchanged.

Across seeds 17 and 18, depth 1 was demoted and depth 2 was promoted as the
replacement. The replacement reached `1.0000` on all eight calibration
lifetimes, while every initially selected depth retained `1.0000` on four
fresh lifetimes. Exact policy reload, controller/frontend immutability, file
immutability, shuffled-outcome fail-closed, and zero-replay gates passed.

This promotes replay-free stale-depth demotion and replacement selection as a
memory-side maintenance contract. The reversal stimulus is an explicit
scalar-outcome control; this does not establish unconstrained nonstationary
learning, learned compression, unrestricted memory growth, or general
continual learning.
