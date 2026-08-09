# Disjoint-dynamics model compounding — rejected promotion

This audit imported the exported games session's strongest architectural
lesson: factual transition models should be extended and searched, while
task-specific policy preferences should not be treated as universal. Two
genuinely disjoint source dynamics were learned first. Two disjoint target
dynamics were then acquired sequentially with a frozen controller and an
external model bank.

Each target used a copy-on-write factual challenger. A transfer candidate was
initialized from the previous model, a fresh candidate from new weights, and
both received the same bounded current-target probe. Only the winning
candidate was appended to the live bank. Earlier slots were never replayed or
updated. The challenger protected the live source slot and made negative
transfer reversible, but it did not make every seed cheaper.

| seed | warm cumulative updates | fresh cumulative updates | result |
| ---: | ---: | ---: | --- |
| 70411 | 155 | 158 | pass |
| 70412 | 123 | 146 | pass |
| 70413 | 122 | 136 | pass |
| 70414 | 146 | 155 | pass |
| 70415 | 141 | 139 | reject |

All five seeds mastered the source and target regimes, retained every prior
slot at `1.0` with byte-stable digests, kept the controller frozen, and used
zero old-regime replay. The promotion gate correctly failed because seed
`70415` had higher cumulative warm cost than its matched fresh control.

This is a useful bounded mechanism and a rejected promotion, not evidence of
general continual learning. The strongest supported result is that factual
model compounding with verifier-gated transfer/fresh selection is promising on
disjoint dynamics, but the selection probe is not yet a reliable predictor of
full acquisition cost. The next experiment should improve the challenger
criterion or test a broader disjoint family without silently relaxing the
cost gate.

Reports are protected by `SHA256SUMS`.
