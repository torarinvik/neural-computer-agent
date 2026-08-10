# Disjoint factual-model compounding with explicit no-agent floors

This three-seed promotion reruns the disjoint-dynamics policy-free compounding
audit with an explicit verifier-only random-intention control. Two source
dynamics are learned first; two genuinely disjoint target dynamics are then
acquired sequentially with copy-on-write factual challengers and matched fresh
learners.

All seeds mastered every source and target, retained every prior slot with
byte-stable digests, kept the controller frozen, used zero old-regime replay,
and selected warm or fresh initialization through the bounded factual probe.
The no-agent floor received 768 random verifier trials per seed and remained
well below the 0.8 mastery threshold (target means 0.15–0.20 in the archived
runs). The planner's result therefore clears an explicit random-action floor,
not merely an unmeasured baseline.

This promotes a bounded disjoint factual-model compounding result with fresh
and no-agent controls. It does not establish unrestricted memory growth,
arbitrary new computation, or general continual learning; the five-seed
challenger population remains documented separately because one seed failed
the cumulative-cost gate.
