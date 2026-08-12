# Interleaved active-discovery scheduling

This screen compares the existing post-training active probe with an
`active_interleaved` schedule. The interleaved arm uses the same single
model-disagreement probe lifetime, but inserts it after the first target
lifetime when an isolated provisional candidate already exists. If staging
happens later, it uses the existing post-training position.

| target arm | post-training active | interleaved active |
| --- | ---: | ---: |
| same-cue n-back-3 | 16/24 | 16/24 |
| same-cue n-back-4 | 15/24 | 14/24 |
| same-cue n-back-5 | 14/24 | 14/24 |
| same-cue aggregate | 45/72 | 44/72 |
| different-cue n-back-5 | 13/24 | 17/24 |
| all active arms | 58/96 | 61/96 |

The change is retained as an opt-in transfer schedule because it improves the
changed-cue pressure test at identical transition-row cost. It is not made a
global default because same-cue n-back-4 regressed and the aggregate result
does not establish universal improvement.
