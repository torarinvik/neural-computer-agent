# Verified artifact consolidation

This audit composes two independently acquired canonical growth artifacts into
one tensor-only artifact row. The consolidated row carries two opaque address
aliases and two opaque executable views, so the memory-side resolver can reach
the same file from either public-context query while the caller activates only
the matched namespace. The shared controller remains frozen after acquisition.

The candidate bank is built transactionally. An external verifier loads the
candidate through both aliases and evaluates both verifier-private procedures
plus the mastered parent procedure. A failed candidate is not adopted. The
source bank is never mutated.

This is a behavior-preserving logical compaction result, not a claim that the
tensor payload itself has been compressed. The naive unrouted composition is a
rejected control. Byte-level compression, arbitrary procedure induction, and
broad continual learning remain open.

Example:

```bash
uv run python -m experiments.artifact_consolidation_amodal.train \
  --updates 512 --seed 69316 \
  --report-out /tmp/artifact-consolidation-seed69316.json
```

The current trainer adds a two-phase retention gate. It first verifies a
candidate without adopting it, records eight fresh held-out retention probes
for each source capability, and then permits consolidation only when the
protected source rows and the replacement both satisfy the opaque retention
ledger. The promoted two-seed evidence is in
`session_records/sequence_working_memory_2026-08-02/artifact_consolidation_retention_v2_2026-08-05/`.

The 64-update schedule is retained as a rejected control when candidate probe
scores fail stable mastery. This is retention-aware logical compaction, not
learned byte compression or general continual learning.
