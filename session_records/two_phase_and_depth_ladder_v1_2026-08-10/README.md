# Two-phase null + depth ladder (F136, F137)

Two-phase (oracle-built plant frozen, reader trained through it):
0.4973 per-bit, worse than joint 0.5283. Binding needs the entry in a
narrow region; task loss cannot find it.

Depth ladder under re-attention: gap +0.030 / +0.026 (depth 1),
+0.045 (depth 2), +0.000 (depth 4). Non-monotonic — a knee, not decay.

The two architectures fail oppositely: re-attend reads a little and
executes not at all (0.5548 ceiling); bind executes almost perfectly
(0.9983) and reads not at all.
