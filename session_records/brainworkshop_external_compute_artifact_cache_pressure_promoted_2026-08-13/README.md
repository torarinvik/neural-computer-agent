# External compute artifact archive under hot-cache pressure (2026-08-13)

Status: **promoted bounded external-compute lifecycle**.

This audit tests whether learned external computations can grow beyond the
active execution cache without catastrophic forgetting. A frozen controller,
event frontend, and shared register interpreter acquire four opaque files:
`symbol_parity`, `triplet_parity`, `parity2`, and `switch_binary`. The physical
hot cache has two slots. The other learned files are stored as portable,
checksummed external artifacts behind `EpisodicBindingArtifactIndex`.

The route key is a learned event tensor. The archive sees only opaque artifact
handles, normalized learned keys, and scalar verifier outcomes. A generic
reliability/recency policy chooses an unprotected resident for replacement;
`reactivate_verified` commits the replacement only after a fresh held-out
probe. The source file is protected by a stable prefix before later candidates
arrive.

Both independent promotion seeds passed every gate:

| Measure | Seed 17 | Seed 18 |
| --- | ---: | ---: |
| Direct accuracy for all four files | 1.0000 | 1.0000 |
| Minimum cold/reactivated accuracy | 1.0000 | 1.0000 |
| Successful cache replacements | 7 | 7 |
| Source retention | 1.0000 | 1.0000 |
| Unknown-key cache mutation | no | no |
| Shuffled-outcome control maximum | 0.4688 | 0.5625 |
| Replayed examples | 0 | 0 |

The hot cache remained fixed at two executable slots while the archive held
four files. Archive reload was exact; archive and artifact checksum corruption
were rejected; active files matched their immutable snapshots; and the
controller/frontend digests remained unchanged. Each seed used 256 optimizer
updates per file, two held-out retention lifetimes, 16 source protection
lifetimes, and seven successful reactivations.

This is bounded hot/cold lifecycle management, not unrestricted memory growth,
learned compression, learned eviction utility, arbitrary program induction,
or general continual learning. The next bottleneck is learning a reusable
eviction utility from fresh verifier outcomes and scaling beyond the fixed
four-file schedule.

Implementation: `experiments/brainworkshop_canonical/external_compute_artifact_cache_pressure.py`.
