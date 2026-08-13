# Append-only external nback32 growth (2026-08-13)

Status: **promoted bounded continual-memory/computation growth**.

This audit tests the CPU/files hypothesis on the canonical Brain Workshop
boundary. A frozen controller and frozen event frontend feed a generic
external indexed-history compute file. The source file first masters `nback16`.
It is then frozen while a fresh external file learns `nback32`; the controller,
frontend, and source file are never replayed or updated during the extension.

The two independent seeds (`17`, `18`) both passed every substantive gate:

| Gate | Seed 17 | Seed 18 |
| --- | ---: | ---: |
| Source nback16 before extension | 1.0000 | 0.9992 |
| New-file nback32 | 1.0000 | 1.0000 |
| Source nback16 retention | 1.0000 | 0.9992 |
| Missing-history maximum | 0.6563 | 0.7813 |
| Corrupted-history maximum | 0.6625 | 0.6344 |
| Action-shuffled maximum | 0.5219 | 0.5000 |
| Shuffled-outcome training control maximum | 0.2594 | 0.2594 |

Both runs used 256 source updates and 256 target updates, batch size 32,
four held-out lifetimes, zero replay, and `81,920` unique verifier bits per
learned file. The controller and event encoder digests were unchanged, and
the source file digest was unchanged during target acquisition.

The post-training reward-shuffle diagnostic remained near mastery. That is not
counted as a failure: this external reader intentionally consumes rendered
event history and not scalar reward, so shuffling reward after a file has been
learned is noncausal. The valid negative control trains a fresh file on
shuffled outcomes and fails near chance.

This promotes a bounded append-only external capability seam and replay-free
retention of the tested old skill. It does **not** establish general continual
learning, unrestricted memory growth, automatic open-ended program induction,
learned compression, or arbitrary computation. The next bottleneck is routing
among an unbounded growing bank and learning when to allocate/protect/compact
files without a hand-selected slot.

Implementation: `experiments/brainworkshop_canonical/external_compute_append_only_depth_growth.py`.
The rejected relation-aware nback32 probe remains a diagnostic only; it did not
improve on the flat indexed reader and is not part of this promotion claim.
