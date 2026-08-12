# Promoted temporal executable-artifact alias consolidation

This promotion composes the frozen temporal capability path with the
replaceable `ExecutableArtifactMemory` backend. Three learned opaque temporal
routes are materialized as external route artifacts. One route is represented
twice under an exact key and an independently learned nearby key; a generic
memory-side consolidation policy must select that redundant pair from four
physical rows.

The independent verifier then reduces the bank from four physical rows to
three, retaining both source aliases and the two distinct routes. Rejected
consolidation is copy-on-write and leaves the source manifest byte-stable.
Reload and artifact-hash corruption controls run after the accepted rewrite.
The controller, event encoder, and temporal capability file remain frozen and
no route-training stream is replayed.

Seeds `17`, `18`, and `19` pass all `15/15` gates. Learned pair selection is
`1.0000` across all 24 physical-order permutations; reward-shuffled controls
are `0.0000`, and untrained controls are `0.1667`. The accepted rewrite saves
one physical row while exact and alias route retention remains at `1.0000`.

Per seed: `168,576` unique temporal verifier bits, `1,344,000` policy
verifier bits, `16` alias-retention verifier bits, `3,000` optimizer updates,
48,024 policy logical lifetimes, four physical rows before, three after, and
zero replayed examples.

This promotes narrow learned external alias consolidation, not arbitrary
semantic compression, unrestricted memory growth, arbitrary new computation,
or general continual learning. Raw reports are `seed-17.json`, `seed-18.json`,
and `seed-19.json`.
