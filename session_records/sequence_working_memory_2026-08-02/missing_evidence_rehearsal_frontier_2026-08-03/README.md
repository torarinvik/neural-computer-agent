# Missing-evidence rehearsal frontier (2026-08-03)

## Claim

Protected missing-evidence rehearsal is the first continuation in this
branch to produce a replicated, control-safe span-11 gain at the higher-power
audit size. It is a frontier promotion for further mastery, not a mastery
claim.

The recipe starts from the independent span-11 parent `996033` and trains
the existing skill slot with scalar outcome supervision on four disjoint
stream types:

- 512 target span-11 lifetimes;
- 512 span-10 rehearsal lifetimes;
- 512 span-9 rehearsal lifetimes;
- 512 blank/missing-evidence span-11 lifetimes.

The missing-evidence stream is protected as rehearsal and is charged to the
sample-efficiency budget. It contains no privileged correct-action labels.

## Evidence

The two independent 1,024-lifetime audits both passed acquisition, causal
slot-contribution, retention, blank, and reset gates:

| Seed | Child − parent | Span-10 retention | Span-9 retention | Blank | Reset |
| --- | ---: | ---: | ---: | ---: | ---: |
| 996046 | +2.43 pp | +0.25 pp | −0.78 pp | 49.02% | 50.35% |
| 996047 | +2.65 pp | −0.06 pp | −0.61 pp | 48.88% | 50.31% |

The higher-power 4,096-lifetime audit for seed `996047` also passed every
gate:

- child accuracy: 70.4346%; parent and zeroed-newest-slot accuracy: 68.6657%;
- causal gain: **+1.7689 pp**;
- 95% paired-lifetime bootstrap interval: **[+1.4782, +2.0531] pp**;
- span-10 retention change: +0.3589 pp;
- span-9 retention change: −0.9142 pp;
- blank sequence: 49.7559%; all-memory-reset: 49.6671%;
- accepted: true.

For comparison, the prior accepted `996033` continuation reached +1.00 pp
on its 4,096-lifetime audit. The short 128-lifetime screen was intentionally
not used as promotion evidence because one independent seed collapsed at
1,024 lifetimes; this record relies on the replicated 512-lifetime runs and
the 4,096-lifetime verification.

## Accounting and boundary

There are 2,048 unique logical lifetimes and 20,992 unique scalar verifier
bits: 5,632 target bits, 5,120 span-10 rehearsal bits, 4,608 span-9
rehearsal bits, and 5,632 missing-evidence control bits. The run performed
1,312 optimizer updates (32 epochs × 41 batches). The 1,024 old-span
rehearsal lifetimes are counted as replayed examples; the 512 missing-
evidence lifetimes are separately charged as protected controls. Online
latency and fresh-learner transfer were not measured.

The audit objective score remains negative because the run did not earn replay
savings and paid for the full rehearsal/control set. Therefore this is not
an autonomous replay-efficiency stop and not a 90% threshold/mastery result.
The next experiment should retain the missing-evidence gate while testing
whether target diversity or a smaller per-output curriculum improves the
gain without relaxing retention.

## Receipts

- Training reports: `span11_blank_rehearsal_996046.json`,
  `span11_blank_rehearsal_996047.json`.
- Replication audits: `continual_blank_rehearsal_996046_count1024.json`,
  `continual_blank_rehearsal_996047_count1024.json`.
- Higher-power audit: `continual_blank_rehearsal_996047_count4096.json`.
- Curated checkpoint:
  `artifacts/checkpoints/span11_missing_evidence_rehearsal_seed996047.pt`.
- SHA-256: `ff1fb764c51e5cb3d29c8609e8c3ad061c7ed5856762c4906974d5aa62018c52`.
