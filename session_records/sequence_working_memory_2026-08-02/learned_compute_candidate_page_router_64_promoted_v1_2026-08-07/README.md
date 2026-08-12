# Learned opaque page router at 64 candidates promoted (2026-08-07)

This audit replaces physical source-page lookup with a learned external page
router. Three independently trained normalized source pages protect 30 opaque
source candidates. The router receives a learned query and an opaque page
summary, then learns only from the scalar verifier outcomes of attempted local
pages. The controller remains frozen throughout.

Both independent seeds pass strict candidate and page retrieval at `1.0000`,
every source candidate and page is mastered, page order can be permuted without
loss, reward-shuffled outcomes produce the null, reload is exact, the frozen
controller is unchanged, and no examples are replayed. Each run spends 2,592
optimizer updates, 221,952 unique verifier bits, and 221,568 logical
lifetimes.

This promotes learned bounded page addressing. It does not claim arbitrary
memory growth, learned representation selection, unseen append integration, or
general continual learning. The next pressure test is to combine learned page
retrieval with verifier-gated append-only pages and then measure retention while
new pages are added.
