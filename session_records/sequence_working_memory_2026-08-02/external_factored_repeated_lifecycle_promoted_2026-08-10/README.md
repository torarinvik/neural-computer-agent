# Repeated factored-memory growth and eviction

This three-seed lifecycle audit promotes seven nonlinear regimes through two
independent capacity expansions and two verified middle-slot evictions. The
surviving opaque IDs remain `(0, 2, 4, 5, 6)`; all surviving regimes route
before and after each mutation, partial reads remain non-mutating, and the
final state restores with the same IDs and routes.

The final residual bank also passes the selected `float16_stats` compression
round-trip retention probe. The base model, controller, and context encoder
remain frozen, and no old-regime examples are replayed for learning.

This promotes repeated bounded memory lifecycle behavior. It does not establish
unrestricted memory growth, multimodal grounding, arbitrary new computation,
or general continual learning.
