# Open-world redundancy discovery and compression

This experiment runs one evolving `ExternalTransitionRouteMemory` per seed for
50 latent lifetimes under a 1,000-attempt budget. It begins with mastered
anchors and one distractor but no duplicate pair. Each latent route is
introduced once, then a noisy second observation arrives; the planner must
preserve both observations, discover their redundancy, and select compression.
The stream also includes growth pressure and a reversed coordinate schedule.
The verifier assigns full utility to compression, lower utility to
admission/eviction, and `0.65` utility to growth.

The planner receives only the current opaque candidate bank and generic
protection/availability facts. Accepted proposals mutate the same memory via
copy-on-write verifier-gated transactions; rejected proposals do not mutate
it. Retention prefixes are checked after every event, with zero replay and one
online planner update per unique verifier utility.

This promotes persistent bounded open-world redundancy discovery and
cost-sensitive compression. It does not establish unrestricted growth,
universal economic planning, autonomous verifier design, or general continual
learning.
