# High-noise partial-evidence disjoint routing

This three-seed stress test raises the observed state and next-state noise to
standard deviation `0.04`, twice the earlier promoted noisy-partial rung. The
router receives only target-covering partial evidence, no regime labels or row
metadata, and alternates all four disjoint regimes for two complete rounds.

All seeds admitted each novel regime once, reused it correctly, retained and
mastered every source slot, preserved persistence, kept old-slot optimizer
updates at zero, and beat matched fresh target learners. The controller stayed
frozen and the wrong-context factual control remained positive.

This promotes robust learned-context routing at one stronger synthetic noise
level. It does not establish arbitrary missingness, real multimodal noise,
unrestricted memory growth, or general continual learning.
