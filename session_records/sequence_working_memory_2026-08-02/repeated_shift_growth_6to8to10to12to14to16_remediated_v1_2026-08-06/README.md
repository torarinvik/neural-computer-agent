# Remediated persistent growth to 62 capabilities (2026-08-06)

This audit extends the persistent isolated route/credit boundary through a
fifth temporal shift: length 6 → 8 → 10 → 12 → 14 → 16, ending at 62 opaque
capabilities. After the normal replay-free acquisition pass, a fresh
confidence probe samples each route 32 times. Only modules whose minimum
probe selection is below `.80` receive one additional 256-update training
block from fresh verifier lifetimes; earlier modules and the controller stay
frozen.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| total capabilities | 62 | 62 |
| shift-1/2/3/4/5 route floors | 0.9219 / 0.8906 / 0.9219 / 0.8594 / 0.8906 | 0.8906 / 0.9063 / 0.8594 / 0.8750 / 0.8594 |
| weak modules remediated | 1 | 4 |
| old route / permutation | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| full-bank protection / reversal / recovery | pass | pass |
| persistent route/credit reload | pass | pass |
| corruption rejection | pass | pass |
| replayed examples | 0 | 0 |

The no-remediation matched control is retained in `controls/`: seed `69317`
passes route, credit, persistence, and causal gates but fails the hard
retention gate with one unprotected row. The remediation repairs only the
weak external modules and restores full retention without replay. This is a
confidence-aware acquisition result, not a weaker retention threshold.

## Accounting

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| unique verifier bits | 4,237,688 | 4,262,264 |
| unique logical lifetimes | 676,344 | 688,632 |
| optimizer updates | 41,216 | 41,984 |
| remediation updates | 256 | 1,024 |
| persistence verifier bits / updates / replay | 0 / 0 / 0 | 0 / 0 / 0 |
| wall seconds | 47.30 | 49.00 |

The 121 route/credit state files per seed reload through
`PersistentOpaqueStateStore`; their digests and all gates are recorded in the
reports. This promotes bounded, persistent, replay-free growth to 62
capabilities. It does not establish unbounded expansion, arbitrary program
induction, fresh-learner transfer, or general continual learning.

Reports, the control, and the accounting ledger are in this directory;
`SHA256SUMS` verifies the archived evidence.
