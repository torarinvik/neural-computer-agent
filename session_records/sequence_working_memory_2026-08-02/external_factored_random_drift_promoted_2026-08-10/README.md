# Randomized partial evidence plus factual drift

This archive records the five-seed promotion of the composed factored
external-memory pressure test. Four opaque nonlinear regimes were admitted
through independent randomized seven-row windows. Each retained slot then
received a disjoint randomized drift update containing `4/8` rows; the other
four rows were held out. The prior regime's held-out rows were used only as a
retention gate, with zero old-regime replay during drift.

All initial and drift versions promoted on seeds `84041` through `84045`.
Random partial reads routed the correct slots, mixed evidence was ambiguous,
initial and drift held-outs stayed within tolerance, state persistence was
exact, and the controller, base, and context encoder remained unchanged.

This is a bounded composition result for replay-free randomized missingness
and gradual factual drift. It is not evidence of learned semantic identity,
open-world version formation, unrestricted memory growth, or general
continual learning.
