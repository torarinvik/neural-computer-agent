# Retention-safe artifact consolidation — 2026-08-05

Status: promoted narrow retention-aware logical compaction boundary.

Two independently acquired growth artifacts were first behavior-verified in a
temporary candidate bank. The source rows were then marked mastered through
eight fresh scalar retention probes each. A final consolidation was allowed
only with eight fresh candidate probe scores and an empty retained-row set;
the replacement inherited the candidate evidence into its opaque retention
ledger. The source bank was never mutated.

Both canonical seeds 69316 and 69317 passed:

- source rows `2` -> consolidated rows `1`, saving one row;
- source capabilities protected before consolidation;
- replacement protected after consolidation and persistent reload;
- opaque aliases `0` and `1` routed to one physical row;
- behavior preserved for the parent, span-3, and span-4 procedures;
- frozen controller digest unchanged;
- checksum corruption rejected;
- rejected candidate not adopted;
- zero consolidation optimizer updates and zero replayed examples.

The short 64-update control was rejected because its candidate retention
probes did not establish stable mastery. The retention gate therefore remains
a real promotion boundary rather than a reporting-only flag.

This promotes retention-aware behavior-verified logical compaction. It does
not establish byte-level compression, learned consolidation policy, arbitrary
new computation, unrestricted memory growth, or general continual learning.
