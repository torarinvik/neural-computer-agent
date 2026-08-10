# Persistent route-memory compression with interference and cost

This experiment runs one evolving `ExternalTransitionRouteMemory` per seed.
It begins with mastered anchors and two redundant pairs, then introduces
important routes, unprotected distractors, and pressure events that make
growth costly but sometimes necessary. Repeated compression opportunities and
the event schedule reversal test whether the policy learns to spend capacity
only when compression cannot safely create room.

The planner receives only the current opaque candidate bank and generic
protection/availability facts. Accepted proposals mutate the same memory via
copy-on-write verifier-gated transactions; rejected proposals do not mutate
it. Retention prefixes are checked after every event, with zero replay and one
online planner update per unique verifier utility.

This is a persistent-stream pressure test, not a claim of unrestricted memory
growth or general continual learning.
