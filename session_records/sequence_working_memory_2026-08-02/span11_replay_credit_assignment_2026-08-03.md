# Span-eleven replay-credit sweep — 2026-08-03

This record is diagnostic, not a promotion. Every arm starts from the frozen
span-nine/span-ten parent and appends a zero-output slot. Old spans must remain
within two percentage points; the new slot must beat its zeroed-slot control by
five points to promote. Correct actions are never stored: the buffer contains
latent features, the attempted opaque action, and its scalar verifier outcome.

## Key arms

| Seed / arm | Target gain | Causal gain | Retention Δ span 9 / 10 | Reading |
| --- | ---: | ---: | ---: | --- |
| 93701, random replay, 64×4 | +1.10 pp | +1.10 pp | −0.17 / −1.17 pp | First non-flat signal; below bar |
| 93702, 128×8 | −0.67 pp | −0.67 pp | −4.64 / −2.42 pp | More data alone fails |
| 93703, replay penalties, 64×8 | +0.78 pp | +0.78 pp | −0.39 / −0.20 pp | Safe but small |
| 93705, detached critic policy | +1.03 pp | +1.03 pp | −2.78 / −1.37 pp | Retention gate failure |
| 93706, 256 target | +0.36 pp | +0.36 pp | −3.26 / −2.03 pp | Data scaling fails |
| 93707, provenance gate | −0.11 pp | −0.11 pp | −0.56 / −1.09 pp | Gate alone does not learn |
| 93708, binary complement | +0.46 pp | +0.46 pp | −1.39 / −0.82 pp | Shuffled control collapses |
| 93710, hidden gate | +0.60 pp | +0.60 pp | −1.69 / +0.08 pp | Best safe before critic-complement |
| 93711, hidden gate, 256 target | +0.71 pp | +0.71 pp | −2.52 / −1.68 pp | Larger data breaks retention |
| **93712, hidden + critic complement** | **+0.89 pp** | **+0.89 pp** | **−1.04 / −0.43 pp** | **Strongest safe arm; below bar** |
| 93713, same, 32× reuse | +0.32 pp | +0.32 pp | −0.09 / −0.35 pp | More reuse does not help |
| 93715, on-policy replay | −0.21 pp | −0.21 pp | −1.78 / −0.94 pp | Distribution shift not solved |
| 93717, no-distractor rung | +0.99 pp | +0.99 pp | −1.22 / +0.78 pp | Gain disappears with distractors |
| 93720, one-distractor rung | −0.78 pp | −0.78 pp | −1.87 / +0.04 pp | Not a reliable adjacent rung |

The 93708 shuffled-outcome control reached −12.14 points on span 11 and lost
old skills, while zeroing the new slot returned the truthful child to the
parent. The 93712 truthful arm likewise returned to the parent when every new
slot module was zeroed. Thus the small gains are causal and reward-dependent,
but not yet useful enough to promote.

## Interpretation and next branch

The independent frozen input probe already decoded the next action at 84.66%
with a linear head and 87.71% with a small MLP; independent random labels were
50.57%. The replay sweep therefore does not justify a new encoder. The
remaining bottleneck is converting scalar outcome credit into a stable action
residual while retaining old spans, and the hard span-eleven rendering is a
large leap. The next experiment should be a smaller, explicitly staged
per-output primitive (or a new adjacent cognitive primitive) with the same
replay, zeroing, shuffled-outcome, and retention audits. Do not spend a long
run on the current span-eleven recipe without a new leading indicator.

The two strongest unpromoted artifacts and their source buffer are archived as
`artifacts/checkpoints/span11_replay_credit_candidate_seed93712.pt` and
`artifacts/memory/span11_replay_buffer_seed93703.pt`.
