# Founding holdout promotion (2026-08-15)

Status: **promoted for bounded Neural Workshop header transfer and
first-time depth invention**.

This record consumes the one-use holdout lease
`brainworkshop-founding-holdout-2026-08-15`. Development seeds 94017 and
97017 remain probation. The holdout population is the unused seed block
110017, 111017, 112017. The skip-shallower protocol and both promotion
gates were frozen before those seeds ran.

## What was tested

A warm bank already holds verified 2-cell 1-back and 2-back files. Public
headers are resolved without `n_back`: exact or same-slot invariant
retrieve, else try existing files, else compose `PREVIOUS` one step
deeper. Shallower leftovers are not retried. A matched fresh learner
starts from the same primitive and the same policy.

Header transfer is 3-cell 3-back after 3-back has already been invented:
the warm bank retrieves the same-slot file. First-time depth invention is
2-cell 3-back: the warm bank fails the existing 2-back file and composes
depth 3, while the fresh learner climbs 1 then 2 then 3.

## Holdout results

| Seed | Header warm/fresh | Header ratio | Header kind | Depth warm/fresh | Depth ratio | Depth kind | Retention min |
| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 110017 | 23 / 97 | 4.22× | invariant | 82 / 116 | 1.41× | compose | 0.968 |
| 111017 | 20 / 82 | 4.10× | invariant | 71 / 119 | 1.68× | compose | 1.000 |
| 112017 | 26 / 89 | 3.42× | invariant | 67 / 125 | 1.87× | compose | 0.955 |

Combined header transfer is 268 / 69 unique bits (`3.88×`). Combined
first-time depth is 360 / 220 unique bits (`1.64×`). Warm was strictly
faster on every seed for both arms. Source 1-back, 2-back, and 3-back
then retrieved exactly.

## Controls

Every holdout seed failed the reject controls on 3-cell 3-back:

| Control | 110017 | 111017 | 112017 |
| --- | ---: | ---: | ---: |
| Wrong-depth 2-back file | 0.294 | 0.194 | 0.265 |
| Missing history | 0.000 | 0.000 | 0.000 |
| Reversed actions | 0.000 | 0.000 | 0.000 |

Shuffled learner-visible rewards stayed at `0.958/1.000/1.000`. That arm
is a diagnostic, not a gate: frozen execution is scored by the private
verifier, so corrupting the learner reward cannot change the program.

Controller, program, and replay updates were zero. The campaign used 1,694
unique verifier bits across 51 logical lifetimes and 288 wall seconds.

## Why this is a promotion

- distinct development and promotion populations;
- one-use holdout claimed once and rechecked by
  `scripts/verify_promotion_record.py`;
- three holdout replicates, each passing the frozen gates;
- required reject controls present and below threshold;
- no workarounds after the protocol was frozen.

## Limits

This does not claim autonomous general program induction, unrestricted
memory growth, a complete executive ISA, desktop system-audio Dual, or
that first-time invention is cheaper than header transfer. First-time
depth is cheaper than a matched climb because the warm bank does not
recompose depths it already owns.

## Run

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.founding_promotion \
  --neural-workshop /absolute/path/to/neural-workshop \
  --claim-holdout \
  --output-dir session_records/brainworkshop_founding_holdout_2026-08-15
```

The lease in `holdout-ledger.jsonl` is already consumed. Re-running with
`--claim-holdout` must fail closed.
