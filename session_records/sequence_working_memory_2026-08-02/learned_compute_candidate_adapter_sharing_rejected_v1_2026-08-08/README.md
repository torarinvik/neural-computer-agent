# Verifier-gated adapter-sharing audit — rejected

Date: 2026-08-08

This audit tested whether a frozen intention adapter could be shared by a new
recurrent compute module. Each candidate was trained only through fresh
verifier outcomes; the adapter was frozen during candidate training. Failed
candidates were removed and their stochastic state was restored before the
least-growth fallback, so rejection could not contaminate later training.

Both independent seeds preserved the frozen controller, old bindings, reload,
memory-corruption recovery, and no-replay controls. Both also found a passing
adapter-sharing candidate for the third capability. The promotion was rejected
because the first incompatible capability fell below the strict mastery floor:

| seed | final behaviors | adapter-sharing result | promoted |
| --- | --- | --- | --- |
| 69316 | `1.000 / 0.684 / 1.000` | third capability reused adapter 0 | no |
| 69317 | `1.000 / 0.656 / 0.867` | third capability reused adapter 0 | no |

The matched permutation control `[2, 1, 0]` was also order-sensitive: seed
`69316` promoted at `1.000 / 0.820 / 0.934` with all three bindings sharing
adapter 0, while seed `69317` failed at `1.000 / 0.660 / 1.000`. This is
evidence that acquisition order changes the learning curve, not evidence of a
general adapter-sharing capability.

The evidence supports compatible adapter sharing as a real, verifier-gated
mechanism, but not yet as a replicated continual-growth improvement. The next
work should improve the learning/growth path for incompatible capabilities and
add a matched permutation control before any promotion claim.

Reports are covered by `SHA256SUMS`.
