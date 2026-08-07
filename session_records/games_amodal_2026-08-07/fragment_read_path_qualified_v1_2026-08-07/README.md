# Qualified: the fragment read path specifies which policy runs (2026-08-07)

The first demonstration in this program that an opaque bank entry loaded
into the controller's sketchpad *selects which competence executes*, not
merely that context helps. Task: `choice` twins — two variants whose
observations are distributionally identical (one type-A item and one
type-B item adjacent to the avatar) with opposite rules (A: take type-A;
B: take type-B). Navigation costs one step, so the only learnable content
is which type to take, and it is unavailable from the screen. The
selector is an oracle here (each twin always receives its own distinct,
salience-matched fragments): this rung isolates the READ path.

Command (per seed):

```bash
uv run python -m experiments.games_amodal.fragment_bank \
  --seed <seed> --suite twins --oracle-selection --warm-updates 0 \
  --updates 900 --batch-size 32 --steps 48 \
  --ignorance-weight 0.5 --ignorance-every 3
```

## Result

| condition (mastery) | seed 69316 A/B | seed 69317 A/B |
| --- | ---: | ---: |
| own fragments | **1.000 / 1.000** | 0.227 / **1.000** |
| bank withheld | 0.297 / 0.313 | 0.094 / 0.555 |
| noise decoy (matched norm) | 0.164 / 0.430 | 0.227 / 0.234 |
| cross-fed (other twin's fragments) | **0.000 / 0.000** | **0.000** / 0.234 |

Seed 69316 is the complete result: one fixed plant holds both
contradictory policies at once; withholding the bank collapses both to
chance; noise collapses them; and cross-feeding drives mastery to exactly
zero — systematically wrong, far below chance. Below-chance cross-feeding
is the decisive signature that the fragment specifies the policy.

Seed 69317 is a winner-take-all failure: `choiceB` took the shared plant
(1.000) and `choiceA` was left at 0.227. Where competence exists the
signature still holds (choiceA cross-fed = 0.000). Joint acquisition of
contradictory contexts is therefore seed-unstable, and this rung is
qualified rather than promoted.

## The eight failures that produced it

`rejected_probe5/7/8.json` and the design laws F1-F9 in
`docs/MEMORY_BANK_DESIGN.md` record the conditions discovered one failure
at a time: ambiguity (F1), no passive escape (F2), staged vs simultaneous
contexts (F5, narrowed by F8), survivable error (F6), isolation of the
decision from motor difficulty (F7), fragment salience matched to event
scale and no blind warm-up (F8). The final experiment succeeded only with
all of them satisfied simultaneously.

## Claim boundary

Promoted-in-principle: skill-as-context in the event window is a
sufficient read mechanism for a fixed-size plant — no weight patching, so
the architecture's storage rule survives. Not promoted: seed-robust joint
acquisition (1/2 seeds complete); learned selection — the outcome-REINFORCE
selector collapsed to identical picks for both variants (probe 8), which
is the literature's predicted routing failure and is now the isolated
next problem; and anything about sharing, compounding, or composition,
which this two-context rung does not test.

Standing methodological consequence: every future bank claim must report
the cross-fed condition. A withheld-only audit cannot distinguish
"context enables competence" from "context specifies competence", and
only the latter supports the memory-bank architecture.
