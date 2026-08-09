# Replay-free affine transition memory — promoted bounded primitive

Across seeds `13011` and `13012`, the sufficient-statistics memory consumed
`12` opaque transition rows exactly once, stored only normal/target matrices,
and predicted four held-out rows without raw-evidence replay. Both seeds
passed train prediction, held-out prediction, one-pass sample count, raw-row
absence, and exact persistence gates.

Held-out errors were `1.84e-13` and `2.58e-14`. The result uses zero optimizer
updates and zero replayed examples; its updates are streaming sufficient-
statistics updates. This promotes a narrow affine external-memory primitive,
not general continual learning, nonlinear rule induction, or arbitrary new
computation.

Reports are protected by `SHA256SUMS`.
