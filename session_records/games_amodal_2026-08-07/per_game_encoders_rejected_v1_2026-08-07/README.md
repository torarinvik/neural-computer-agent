# Rejected: per-game screen encoders (2026-08-07)

The best-performing battery configuration the program has produced, and
inadmissible. Each game received its own screen encoder (`--per-game-encoders`
on `two_speed_battery`), motivated by F29's finding that a shared frontend
couples every game's representation.

## What it bought

| readout | shared encoder | per-game encoders |
| --- | ---: | ---: |
| choiceA / choiceB (69316) | 0.72 / 0.31 | 1.00 / 1.00 |
| choiceA / choiceB (69317) | 0.78 / 0.22 | 1.00 / 1.00 |
| worst forgetting | 0.203 / 0.219 | 0.048 / 0.062 |
| mean solo ratio (69317) | 0.61 | 0.84 |

## Why it is rejected

| twin cross-feed | shared | per-game |
| --- | ---: | ---: |
| `choiceA <- choiceB` | 0.19 / 0.20 | **1.000 / 1.000** |

Feeding a twin its opposite's fragments should invert behaviour toward
0.000. At 1.000 the fragments are irrelevant: each game's encoder had
become its own program. The twins were "solved" by giving each twin a
private model — the per-game-model outcome this research program exists
to avoid, and the specific thing ruled out at its outset.

The amodal design's N encoders mean one per MODALITY (screen, sound,
text), never one per task. A per-task encoder is weight-stored skill
using the architecture's vocabulary.

## Standing consequences

1. Any per-game trainable component absorbs the skill if allowed to.
2. Performance gains are not evidence of architectural progress; the
   cross-feed audit is. This configuration improved four metrics and
   destroyed the only one that distinguishes a memory bank from a
   collection of per-game models.
3. Every future improvement must report cross-feeding in the same table
   as its gains.

The flag remains in the code, defaulting off, as the reproducible
counter-example.
