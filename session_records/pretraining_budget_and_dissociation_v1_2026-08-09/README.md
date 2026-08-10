# Pre-training budget, schema support, and the double dissociation (F80-F82)

## Break-even has an interior optimum (pool 4096, 2 seeds each)

| pre-train | novel read | mastered | acquisition | cold | saving | break-even |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2500 | 0.308 | 0/16 | 540.6 | 50.0 | -490.6 | never |
| 5000 | 0.361 | 0/16 | 489.6 | 50.0 | -439.6 | never |
| 10000 | 0.692 | 1.5/16 | 214.6 | 50.0 | -164.6 | never |
| 15000 | 0.837 | 4.0/16 | 97.9 | 50.0 | -47.9 | never |
| 20000 | 0.907 | 5.5/16 | 34.3 | 50.0 | +15.7 | 1278 |
| **40000** | 0.972 | 9.5/16 | 7.2 | 50.0 | +42.8 | **936** |
| 80000 | 0.990 | 10.0/16 | 5.2 | 50.0 | +44.8 | 1786 |

Saving is capped at cold's 50, and 40000 already captures 86% of it, so the
optimum is interior and ~936 families is a structural floor for this
configuration. The budget axis is exhausted.

Both of the ledger's stated next steps were wrong: 20000 was NOT padding
(below it, acquisition is worse than cold), and lengthening pre-training
improves break-even rather than worsening it.

## Double dissociation (F81), imported from the parallel Codex session

| bank | novel read accuracy |
| --- | ---: |
| present | 0.907 |
| withheld (zero entry) | 0.236 |
| corrupted (wrong family's entry) | 0.037 |

Present -> mastery, withheld -> chance, corrupted -> chance, plant frozen
throughout. The residual 0.236 is not noise: a zero entry is neutral, so the
plant falls back on its generic structural prior. Of the 0.907, **0.236 is
structure in frozen weights and 0.671 is content from the bank** — the
architecture's central split, measured.

## Schema support (F79 corrected)

`--wide` (two-slot ops + permutation spaces), pool 4096, 20000 updates:
toggle 0.096 -> 0.306, perm 0.521 -> 0.708, dial 0.775 -> 0.863. The floor
lifts, but the harder distribution un-crosses the cost gate at that budget
(acq 81.3 vs cold 57.9).

Separately, the narrow generator at 80000 updates reads `perm` at 0.965 (from
0.521 at 20000), so F79's "diversity buys nothing outside the schema" was an
undertrained-reader artifact, not a boundary. `toggle` at 0.272 remains the
genuine hard case.
