# Opaque artifact capacity planning — 2026-08-06

Status: promoted bounded learned admission planning.

This audit adds the missing memory-side decision primitive after protected
capacity growth. An outcome-trained `OpaqueCapacityPlanner` receives only an
incoming learned key/value, an unordered bank of learned row summaries, and
generic protection/transaction availability facts. It proposes one of four
storage actions: admit into free capacity, evict an unprotected row,
consolidate a pair, or grow the bank.

Across seeds 69316 and 69317, the planner learned the opaque action choice and
the ambiguous eviction-versus-consolidation decision. Candidate permutation
was invariant, while the reward-shuffled control reached only `0.545` on the
ambiguous choice audit. A fully protected artifact bank was forced to choose
growth; the source bank remained byte-for-byte unchanged, retention transferred
to the grown bank, and all three artifacts reloaded successfully. Replay and
controller updates were zero.

This promotes a bounded learned admission-planning mechanism. Protection
masking, executable behavior verification, and the final storage transaction
remain explicit. It does not establish learned consolidation of arbitrary
procedures, unrestricted memory growth, arbitrary new computation, or general
continual learning.

Evidence is in `report_seed69316.json` and `report_seed69317.json`.
