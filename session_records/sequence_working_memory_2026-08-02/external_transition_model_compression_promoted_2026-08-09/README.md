# Promoted verified external transition-model compression

Two seeds (`70211`, `70212`) evaluate compressed storage candidates by
decompressing them into independent runtime banks and applying the same held-
out source/target retention probe. Float16 reduced model-state bytes from
`26,944` to `13,472`; int8 reduced them to `6,784`. Both passed: float16 loss
deltas stayed below `4e-8`, and int8 below `9e-6`. Int4 reduced storage to
`5,176` bytes but failed the `1e-4` retention tolerance, with loss drift from
`2.7e-4` to `7.6e-4`.

The controller and compression path performed zero optimizer updates, context
and alias metadata persisted, and codec acceptance was caller-owned and
verifier-gated. The adaptive selector chose int8, the smallest accepted codec,
in both seeds. This promotes storage compression only, not live reduced-
precision reasoning or arbitrary learned computation.
