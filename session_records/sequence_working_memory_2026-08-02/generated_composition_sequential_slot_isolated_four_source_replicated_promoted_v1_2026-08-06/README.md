# Four-source replay-free slot-isolated growth

This pressure test extends the external memory boundary through four
sequential source procedures: source `0`, then `2`, `3`, and `4`. Each source
is learned from fresh verifier outcomes in an independent neural slot. The
slot is appended into one physical artifact row under an opaque alias, and
the preceding slots remain immutable. A fresh target is then acquired from
the first retained slot and admitted by capacity growth.

The runtime-private grammar was:

1. `reverse -> adjacent_xor -> complement -> prefix_parity`
2. `prefix_parity -> global_parity -> rotate -> complement` (target)
3. `global_parity -> reverse -> adjacent_xor -> rotate`
4. `complement -> prefix_parity -> reverse -> global_parity`
5. `rotate -> global_parity -> complement -> adjacent_xor`

Across seeds `69316` and `69317`, all three sequential rewrites passed fresh
retention and reload. The four source aliases resolved to one physical row;
source behavior after the final reload was:

| seed | source 0 | source 2 | source 3 | source 4 | inherited target bits | fresh target bits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 69316 | 0.9570 | 1.0000 | 0.9531 | 1.0000 | 2,048 | 14,336 |
| 69317 | 0.9805 | 1.0000 | 0.9844 | 1.0000 | 2,048 | 8,192 |

The final physical payload was `1,341,824` bytes, equal to the sum of the
four isolated source payloads. This is capacity-safe slot isolation, not
neural weight compression. The grown target reloaded at `1.0000` in both
replicas. Alias-specific reversal released and recovered the selected alias
while the shared physical row remained protected by the other source aliases;
the separate target row released and recovered independently. Checksum
corruption was rejected, the frozen controller digest was unchanged, and
replayed examples were zero.

The short rung was rejected before growth because the first source was not
protected; the transaction refused to evict it. A dense shared-weight
expansion control was also rejected: with source 0 mastered at `0.9531`,
training a new slot and route on source 2 alone drove source 0 to `0.6250`
while source 2 reached `1.0000`. This identifies route isolation—not merely
freezing old tensors—as the key no-replay requirement.

This promotes bounded replay-free external slot growth through four sources.
It does not establish unrestricted memory growth, dense shared-weight
consolidation, arbitrary program induction, or general continual learning.
