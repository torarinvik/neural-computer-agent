# Dual acquisition and transfer

This record is **probation**, not a promotion. It closes two holes: Dual
had no blank-file learning curve, and mixed Dual feedback was half-credit
on a packed four-way action.

## What changed

A Dual trial with two public labels now maps to packed exact-match credit:
learner reward is `1` only when every visible label is positive. Mixed
`0.0` stays `0` for the program file, while the audit scalar remains the
label average.

A blank recursive address file then learns Dual 1-Back from public pixels
and public PCM. After the first stable prefix the same primitive is
composed once and evaluated on Dual 2-Back with the controller frozen.

## Rendered Dual (clean-room RGB + waveform)

| Seed | Dual 1-Back bits | Retention | Warm Dual 2-Back | Fresh Dual 2-Back bits |
| ---: | ---: | ---: | ---: | ---: |
| 99017 | 47 at `0.851` | 1.000 | 1.000 / 0 updates | 92 |
| 99117 | 94 at `0.915` | 1.000 | 1.000 / 0 updates | 46 |

Controls on Dual 1-Back stayed below threshold: reversed `0.13/0.04`,
missing history `0.23/0.32`, shuffled reward `0.60/0.40`.

## Neural Workshop Dual (public RGBA + PCM)

| Seed | Dual 1-Back bits | Retention | Warm Dual 2-Back | Fresh Dual 2-Back bits |
| ---: | ---: | ---: | ---: | ---: |
| 99117 | 95 at `0.957` | 1.000 | 1.000 / 0 updates | 49 |
| 99217 | 49 at `1.000` | 1.000 | 0.944 / 0 updates | 51 |

Controls: reversed packed `0.03/0.08`, missing history `0.00/0.00`,
shuffled `0.56/0.59`. Every stimulus still produced one vision event and
one audio event. Controller optimizer updates and replay were zero.

This is Dual acquisition and composed 2-Back execution. It is not a
bits-to-threshold transfer ratio on the same 2-Back climb, and it is not
a holdout promotion.

## Why this is not promoted

- two seeds per substrate, no Dual holdout lease in this record;
- Dual 2-Back warm path is composition, not a cheaper 2-Back search;
- desktop screen-capture Dual is optional I/O, not a trainer.

The later unused three-seed lease is
`session_records/brainworkshop_dual_holdout_2026-08-15/`.

## Run

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.rendered_dual_transfer_pilot \
  --steps 48 --sessions 6 --seed 99017

PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_dual_acquisition_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --trials 60 --sessions 6 --seed 99117
```
