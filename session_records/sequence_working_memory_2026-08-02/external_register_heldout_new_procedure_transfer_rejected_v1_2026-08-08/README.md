# Held-out new-procedure transfer — rejected

This audit acquired four verifier-private source procedures, froze the shared
external register interpreter, and trained only a new opaque instruction code
and decoder for a fifth unseen procedure. A matched fresh interpreter of the
same size received the same target outcomes.

The depth-four rung was rejected because source acquisition was undertrained.
Reducing only program depth to two made seed `69316` valid: inherited target
accuracy was `0.9531`, fresh accuracy `0.9844`, and both stable prefixes were
`12,288` verifier bits. Seed `69317` initially left one source at `0.7617`.
Increasing only source updates from `256` to `384` repaired that source gate,
but inherited target still required `12,288` bits versus `8,192` fresh.

All valid runs passed source retention, target mastery, reward-shuffled and
missing-evidence controls, exact reload, checksum-corruption rejection,
frozen-parent equality, and zero replay. The strict positive held-out
transfer gate therefore remains rejected.

Conclusion: one new opaque instruction code is not yet a reusable blueprint
for an entire unseen procedure. The next experiment should assemble a new
program from already learned instruction data.
