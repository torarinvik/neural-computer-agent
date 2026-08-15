# Dual acquisition holdout promotion (2026-08-15)

Status: **promoted for Neural Workshop Dual 1-Back acquisition and
composed Dual 2-Back execution**.

This record consumes the one-use holdout lease
`brainworkshop-dual-holdout-2026-08-15`. Development seeds 99117 and
99217 remain probation. The holdout population is the unused seed block
113017, 114017, 115017. The blank-file Dual protocol and the promotion
gate were frozen before those seeds ran.

## What was tested

A fresh recursive address file starts uniform. It may update only from
public Dual pixels and public Dual PCM. After the first stable Dual
1-Back prefix the same primitive is composed once and executed on Dual
2-Back with the controller frozen. Mixed Dual labels are packed
exact-match credit. Wrong-depth executes the uncomposed 1-Back file on
Dual 2-Back.

This is Dual acquisition and composition. It is not a Dual 2-Back
bits-to-threshold transfer ratio.

## Holdout results

| Seed | Dual 1-Back bits | Last packed | Retention | Warm Dual 2-Back | Fresh Dual 2-Back bits |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 113017 | 29 at `0.862` | `0.862` | 1.000 | 1.000 / 0 updates | 27 |
| 114017 | 45 at `1.000` | `1.000` | 1.000 | 1.000 / 0 updates | 23 |
| 115017 | 88 at `0.968` | `0.968` | 1.000 | 1.000 / 0 updates | 56 |

Every stimulus produced one vision event and one audio event.

## Controls

Every holdout seed failed the reject controls:

| Control | 113017 | 114017 | 115017 |
| --- | ---: | ---: | ---: |
| Wrong-depth 1-back file on Dual 2-back | 0.086 | 0.100 | 0.040 |
| Missing history | 0.000 | 0.000 | 0.000 |
| Reversed packed actions | 0.017 | 0.017 | 0.050 |

Shuffled learner-visible rewards stayed at `0.393/0.615/0.500`. That arm
is a diagnostic, not a gate.

Controller optimizer updates and replay were zero. The campaign used 835
unique verifier bits across 28 logical lifetimes and 154 wall seconds.

## Why this is a promotion

- distinct development and promotion populations;
- one-use holdout claimed once and rechecked by
  `scripts/verify_promotion_record.py`;
- three holdout replicates, each passing the frozen Dual gate;
- required reject controls present and below threshold;
- no workarounds after the protocol was frozen.

## Limits

This does not claim a cheaper Dual 2-Back search, autonomous program
induction, unrestricted memory growth, a complete executive ISA, or a
measured desktop Dual lifetime. Desktop ScreenCaptureKit Dual remains
optional human-parity I/O, not a trainer.

## Run

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.dual_promotion \
  --neural-workshop /absolute/path/to/neural-workshop \
  --claim-holdout \
  --output-dir session_records/brainworkshop_dual_holdout_2026-08-15
```
