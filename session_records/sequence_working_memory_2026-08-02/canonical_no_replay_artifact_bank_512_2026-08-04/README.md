# Canonical no-replay artifact-bank retention — 2026-08-04

This is the promoted no-replay continual-learning result. The parent
controller first acquires span 2. Span 3 and span 4 are then acquired in
independent generic growth slots over the frozen parent. Each slot is stored
as an opaque executable artifact in `ExecutableArtifactMemory`; a
memory-side context router selects one artifact rather than summing unrelated
skills.

Both the controller core and previously acquired artifacts remain untouched
while later artifacts are trained. No old-span example is replayed into any
controller optimizer update. The route key uses only the controller's opaque
context state and event-window occupancy; it does not receive a span label,
correct action, or raw frame.

Primary seed `69316` passed every gate:

- route accuracy: `100%` for spans 2, 3, and 4;
- final selected accuracy: `100%`, `100%`, and `87.5%`;
- stable span-4 threshold: `12,288` verifier bits;
- blank-sequence control: `51.2%`;
- replayed examples: `0`;
- frozen parent core: exact bit identity.

The result is narrow: retention is achieved through isolated external
capability files and routing, not through uninterrupted shared-weight SGD.
General task discovery, arbitrary program induction, and transfer to natural
modalities remain unqualified.

The executable harness is
`experiments/working_memory_continuous/canonical_no_replay_artifact_bank.py`.
The independent replica is in
`../canonical_no_replay_artifact_bank_replication_2026-08-04/`.
