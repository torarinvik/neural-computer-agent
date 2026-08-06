# Protected artifact capacity growth v1 (2026-08-06)

This is a memory-boundary pressure test, not a skill-learning claim. A
two-row executable-artifact bank receives eight fresh successful verifier
outcomes for each of its two opaque rows. Once both rows are protected, a
third write is attempted and must refuse eviction. The source is then copied
into a separate three-row verified store with its opaque retention state, and
the third artifact is admitted there.

Seeds `69316` and `69317` both passed:

- full protected write refused explicitly;
- source capacity, rows, version, and protection state unchanged;
- retention transferred to the grown store;
- new artifact admitted only after growth;
- all three artifacts recovered after reload;
- zero optimizer updates and zero replayed examples.

Each seed used `16` fresh retention observations and recorded `16` unique
verifier bits and logical lifetimes. This qualifies the explicit refusal →
capacity growth → admission boundary. It does not establish learned capacity
planning, compression, arbitrary new skill acquisition, or general continual
learning.
