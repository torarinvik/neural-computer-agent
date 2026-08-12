# Outcome-only control-flow fragment splice promotion

This archive records the four-seed audit for reusable multi-instruction
external fragments. The search selects an opaque parent file, insertion
boundary, and fragment from scalar verifier outcomes; it stores no verifier
rows, task labels, or controller parameters.

Command:

```text
.venv/bin/python experiments/recipe_expressibility/control_flow_fragment_splice.py \
  --seeds 31 32 33 34 \
  --report-out session_records/control_flow_fragment_splice_promoted_2026-08-12/report.json
```

Results:

- all 8 warm arms (four seeds, forward/reversed verifier-state order)
  materialized two held-out-perfect assemblies;
- protected source files were retained, search and memory state reloaded
  exactly, checksum corruption was rejected, and empty evidence caused no
  write;
- a matched fresh control passed all target behavior gates;
- shuffled-feedback controls admitted no candidate;
- accounting charged `2,105` unique verifier bits and `421` logical lifetimes,
  with zero replay and zero optimizer updates.

The verifier intentionally promotes behavior, not a hand-assigned
parent/fragment provenance. Equivalent file arrangements remain valid and
selected slots are diagnostic only.

This promotes bounded outcome-only multi-instruction fragment splicing and
behavioral reuse. It does not establish efficient arbitrary program synthesis,
unrestricted memory growth, arbitrary new computation acquisition, or general
continual learning.
