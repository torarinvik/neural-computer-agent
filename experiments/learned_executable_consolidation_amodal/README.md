# Learned executable-artifact consolidation

This is the next pressure test after latent consolidation. Four independently
trained executable growth artifacts are stored as opaque rows. The canonical
memory-side consolidation policy selects a pair from learned key/value
summaries, while the artifact store composes their executable views.

Every proposed rewrite runs through one immutable transaction: the candidate
bank is built, a fresh behavior-only outcome probe establishes replacement
mastery, and the retention gate runs before the external verifier can adopt
it. Protected source rows therefore never need to be weakened to make a
candidate measurable. The chain repeats until four procedures occupy one
physical row. Reload, alias routing, frozen-core equality, and checksum
corruption are independent final gates.

After compaction, an outcome-trained permutation-equivariant router acquires
the four opaque executable addresses. Its selected candidate is resolved
through the generic memory promotion path before execution; direct semantic
view lookup is not used by the routed behavior audit.

```bash
PYTHONPATH=src uv run python -m experiments.learned_executable_consolidation_amodal.train \
  --updates 1024 --policy-updates 512 --route-updates 2048 --audit-count 64 \
  --seed 69316 --report-out /tmp/learned-executable-consolidation.json
```

The 512-update control is deliberately rejected when the candidate retention
prefix does not reach the `.75` mastery threshold. The promoted two-seed rung
uses 1,024 artifact-acquisition updates and keeps zero replay and zero
controller updates during consolidation.
The route learner uses the established 2,048-update acquisition budget; a
short 128-update smoke remains below the 90% route/permutation gate.

This remains a bounded, behavior-verified executable-memory result. It does
not claim arbitrary program induction, learned byte compression, unrestricted
memory growth, or general continual learning.
