# External temporal capacity scheduling — promoted 2026-08-12

This record composes the canonical persistent content-addressed temporal
memory with the replaceable opaque capacity planner. The planner is trained
from scalar utility on generic candidate banks, then transferred to a live
four-row memory. Two distinct opaque addresses each arrive with a redundant
alias; two new addresses must be admitted, so the planner must consolidate a
redundant pair before each admission. The live stream is repeated after a
physical row-order reversal.

Both seeds passed every gate:

- held-out admit, evict, consolidate, and grow utility reached `1.0`;
- the trained planner improved nontrivial consolidation transfer over fresh
  policies (`1.0` versus `0.15625`/`0.25`);
- forward and reversed streams each completed two compactions and two
  admissions under a fixed four-row external budget;
- four distinct routes survived every intermediate verifier probe and exact
  persistence reload;
- checksum corruption was rejected;
- the controller and learned event encoder stayed byte-stable; and
- replayed examples were zero.

The policy view is generic `MemoryCandidates` data padded with unoccupied,
zero-filled rows. The compaction candidate is still accepted only by an
independent route verifier and committed through the versioned
`replace_from_candidates` transaction.

This promotes bounded replay-free capacity scheduling and sequential
verifier-gated multi-row compaction in the canonical temporal-memory path. It
does not establish arbitrary shared-structure compression, semantic
equivalence discovery, unbounded memory, autonomous verifier design, or
general continual learning. The next pressure test is a learned multi-row
shared-structure representation that reduces physical storage for genuinely
distinct but compositional capabilities, followed by longer nonstationary
streams and retention-adjusted regret.

Reports: `seed-17.json`, `seed-18.json`.
