# Multi-feature online utility milestone — 2026-07-26

## Result

The unified controller now adapts two generic memory-utility coefficients
online: access frequency and verified outcome reliability. Reliability is
computed from per-row success and failure counts with a one-success/one-failure
prior. It is not a task or semantic label.

Two independent runs passed:

| Seed | Seconds | Verifier bits | Old equal | Reliability dominant | Old return | All equal |
|---:|---:|---:|---:|---:|---:|---:|
| 6932 | 29.37 | 196,608 | 89.75% | 78.22% | 88.48% | 87.45% |
| 6938 | 29.34 | 196,608 | 88.67% | 88.43% | 84.72% | 83.35% |

Each used 48 updates, no replay, no learner-visible boundary, no optimizer
reset, and no utility or correct-action labels. Binary mapping and four-rule
retention passed. Only the two-coefficient residual changed.

## Mechanism

A symmetric Rademacher direction creates three temporary candidates: move
positive, stay put, or move negative. All act on the same fresh memory banks.
Only the candidate with the best later verified behavior survives. The center
candidate fixed harmful drift observed in an earlier forced plus/minus race.

## Controls and rejected paths

- A write-strength coefficient was rejected because the inherited controller
  already saw write strength; the new coefficient added only 2.93 target
  points.
- The exact reward-shuffled reliability run damaged old-equal performance,
  returned to the old mixture at only 75.93%, and ended all-equal at 64.31%.
  It failed and saved no checkpoint.
- Ablating only reliability from selected seed 6932 reduced the
  reliability-dominant phase from 78.22% to 55.27% and all-equal from 87.45%
  to 63.67%.

## Physical memory

Persistent memory schema v3 stores access, success, and failure counts. Outcome
attribution follows ordinary content addressing. Counters survive save/reload,
copy into active memory, grow with storage, reset on replacement, and default
to zero when older schemas load.

The selected controller passed a 1,024-bank audit:

- learned 96.21%, visible oracle 96.35%, full oracle 97.18%;
- 6,144 rows before and after, zero growth;
- all 1,024 complete histories survived save/reload;
- learned correct eviction 89.65%;
- age/frequency/reliability corruption reduced correct eviction by
  50.29/60.55/30.18 points and behavior by 3.11/6.75/2.56 points.

## Checkpoints

Selected:
`artifacts/checkpoints/unified_memory_multifeature_reliability_seed6932.pt`

SHA-256:
`bb5cd158c08f4b92061aca7bfae0751d4e18408e8e37f53cac13dffaed8ac9f4`

Replica:
`artifacts/checkpoints/unified_memory_multifeature_reliability_seed6938.pt`

SHA-256:
`0342a8266bde7bc5a0f79004792ce29668f758904aa954755b7bf7130993730d`

## Honest boundary and next atom

Training still uses tensorized histories and then audits the result on physical
disk. The next experiment should run the adaptation loop itself on a tiny
evolving physical stream and compare its coefficient trajectory against the
tensor arena at matched verifier bits.
