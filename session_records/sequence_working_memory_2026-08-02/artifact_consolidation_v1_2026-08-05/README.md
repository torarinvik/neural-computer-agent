# Verified artifact consolidation — 2026-08-05

Status: promoted narrow memory-side compaction boundary.

Two independently acquired direct-growth artifacts were stored as two source
rows, then composed into one tensor-only row with two opaque address aliases.
Each alias carries an opaque executable view identifier; the generic caller
projects only that namespace into the frozen controller’s growth boundary.
This avoids the rejected naive composition that executed both procedures at
once and changed their behavior.

Both 512-update seeds passed the behavior-only promotion gates:

- source rows: `2`; consolidated rows: `1`; rows saved: `1`
- alias views: `0` and `1`, both resolving to the same physical row
- mastered parent retention: `1.000` for both seeds
- span-3 retention: `+0.0052/+0.0000` versus the independently loaded artifact
- span-4 retention: `+0.0234/+0.0039` versus the independently loaded artifact
- frozen-core digest: unchanged for both seeds
- persistent reload: behavior preserved and artifact bytes exact
- checksum corruption: rejected; rejected candidate: not adopted
- optimizer updates during consolidation: `0`; replayed examples: `0`

The initial 64-update naive composition is retained as a rejected mechanism:
executing both slots simultaneously reduced parent accuracy from `0.875` to
`0.750` and span-3 accuracy from `0.792` to `0.672`. The result establishes
that storage compaction and execution routing are separate contracts.

This promotes behavior-verified logical compaction with opaque views. It does
not establish byte compression, arbitrary new computation, unrestricted
procedure induction, or general continual learning. Independent capabilities
must remain append-only unless a held-out verifier proves a routed compact
representation equivalent.
