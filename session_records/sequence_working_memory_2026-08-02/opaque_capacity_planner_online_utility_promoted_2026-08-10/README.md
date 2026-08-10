# Promoted: online verifier-utility learning for opaque capacity planning

Four independent seeds trained `OpaqueCapacityPlanner` online using one
scalar verifier utility per exploratory maintenance proposal. Each episode
used a fresh opaque four-row bank containing one redundant pair. The planner
learned to select consolidation and the correct pair, then transferred to a
held-out stream without replay.

All seeds passed the promotion gates: stable online utility at least `0.95`,
held-out trained transfer `1.0`, trained transfer at least `0.2` above a fresh
planner, frozen controller, exactly 600 updates for 600 unique verifier
utilities, and zero replay. The fresh baseline ranged from `0.0` to `0.42`.

This is a causal capability result for one bounded maintenance regime. It does
not establish universal policy learning, autonomous verifier design, unbounded
memory growth, or general continual learning. The next pressure test must mix
all maintenance actions and use longer interfering route-memory streams.
