# Vectorized external-history lookup (2026-08-12)

`ExternalTemporalHistoryMemory` now resolves relative and absolute scoped
positions through a vectorized sorted position index. The public storage ABI is
unchanged: records remain append-only, positions remain opaque, and missing
records still return explicit `present=False` masks. Metadata fields are
gathered through the same record index.

The old implementation performed nested Python loops over scopes, queries, and
matching records. On a deterministic CPU benchmark with 32 scopes, 1,000
records per scope (32,000 records total), width 16, and four relative queries
per scope:

- nested Python scan: `0.00510121 s` per read;
- vectorized position index: `0.00120285 s` per read;
- speedup: `4.2409x`;
- values, presence masks, and resolved positions matched exactly.

This is an execution-path scalability improvement, not a learned-capability
promotion. It does not establish unrestricted memory growth or general
continual learning.
