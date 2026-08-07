# Spatial-binding extra-calibration control rejected (2026-08-07)

The spatial-binding frontend was given twice the append calibration budget
(64 updates per stage) after the 1024-update source/full-prior regime was
established. This isolates whether its weaker hard-seed result is merely local
extension under-training.

It is not. Seed `69317` remains at `0.8125` unseen routing with two targets at
`0.0`, exactly matching the 32-update spatial control; seed `69316` remains at
`1.0000`. The extra calibration updates therefore add cost without repairing
the static query/key alignment failure. The spatial frontend remains a
diagnostic blueprint, not a promoted acquisition path.
