# Learned eviction utility for a six-file external archive (2026-08-13)

Status: **promoted bounded learned external-memory lifecycle**.

This rung extends the promoted hot/cold external-compute lifecycle from four
files and a hand-designed reliability/recency victim rule to six independently
mastered opaque files behind a three-slot executable hot cache. The remaining
files are checksummed cold artifacts indexed by learned event-tensor keys.

The memory-side eviction policy receives only the incoming learned event tensor
and fixed-width opaque artifact descriptors. It learns disposability from
paired fresh verifier outcomes. Stable-prefix protection remains a separate
verifier-owned gate. Route-side adaptation uses at most two updates and ignores
weak fresh utility differences below `0.15`, preventing low-confidence outcomes
from rewriting the learned policy.

| Measure | Seed 17 | Seed 18 |
| --- | ---: | ---: |
| Direct mastery of all six files | 1.0000 | 1.0000 |
| Minimum cold/reactivated accuracy | 1.0000 | 1.0000 |
| Learned calibration | 36/48 | 45/48 |
| Shuffled-utility calibration | 15/48 | 3/48 |
| Learned route selection | 9/11 | 8/13 |
| Source retention | 1.0000 | 1.0000 |
| Replayed examples | 0 | 0 |

Both seeds passed every promotion gate. The hot cache stayed at three physical
slots while the archive held six files. Unknown keys failed closed without
cache mutation; archive and artifact checksum corruption were rejected; archive
and policy reloads were exact; active artifacts matched immutable snapshots;
and the controller and event frontend remained byte-identical.

This promotes outcome-only learned eviction utility for a bounded six-file
archive. It does not establish unrestricted memory growth, learned compression,
semantic conflict resolution, arbitrary program induction, or general
continual learning. The next pressure test is learned archive compaction and
utility transfer to genuinely new artifact families, with a fresh-learner
transfer baseline.

Implementation:
`experiments/brainworkshop_canonical/external_compute_learned_eviction_scale.py`.
