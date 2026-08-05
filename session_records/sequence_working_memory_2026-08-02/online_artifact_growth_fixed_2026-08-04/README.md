# Online external-artifact growth audit

This storage-boundary audit appends two acquired working-memory artifacts to
one deployed `ExecutableArtifactMemory` instance rather than constructing the
bank with both rows at once.

The first artifact was written, its hot cache was evicted, and it was cold
reloaded. The second artifact was then appended into a free row. The resulting
bank was reloaded, compacted into a two-row store, reloaded again, and checked
for exact tensor and SHA-256 preservation. Corruption rejection was tested on
the compacted store.

All append, cold-reload, eviction, compaction, hash-preservation, and
corruption gates passed. Canonical random opaque keys were used for the bank.
The audit also exposed and motivated a fix to the legacy acquisition helper:
future source addresses now incorporate the complete rendered history and
operation cue rather than only the first frame.

Report: `report.json`.
