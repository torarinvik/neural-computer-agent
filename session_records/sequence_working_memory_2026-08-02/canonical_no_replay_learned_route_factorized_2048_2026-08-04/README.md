# Learned opaque address discovery with no-replay retention — 2026-08-04

This is the promoted learned-routing successor to the occupancy-address
audit. Span-2, span-3, and span-4 capability files are acquired over a frozen
canonical parent. Their memory keys are random opaque vectors. A
`FactorizedOpaqueAddressRouter` learns query/key compatibility from fresh
attempted-row scalar outcomes.

Primary seed `69413` passed every gate:

- learned route accuracy: `100%`;
- candidate-row permutation accuracy: `100%`;
- reward-shuffled route accuracy: `33.3%` three-way chance;
- selected execution: span 2 `100%`, span 3 `98.96%`, span 4 `86.72%`;
- stable span-4 acquisition: `16,384` verifier bits;
- replayed examples: `0` for capability and route training;
- frozen parent core: exact bit identity.

The route receives only the controller hidden query and opaque candidate keys.
No occupancy feature, span label, semantic task ID, or correct unattempted row
is exposed. This closes the narrow hand-shaped routing bottleneck, but the
router still requires substantial fresh verifier evidence and does not claim
general program discovery.

Harness:
`experiments/working_memory_continuous/canonical_no_replay_learned_route.py`.
The independent replica is in
`../canonical_no_replay_learned_route_factorized_2048_replication_seed69415_2026-08-04/`.
