# Neural Workshop Dual live path

This record is **probation**, not a promotion. It closes the hole that
Physical Neural Workshop Dual was not a valid learner path because the
public observation was pixels only.

## What changed

Neural Workshop now publishes the queued stimulus waveform on the public
observation as `{audio_pcm, audio_rate, audio_channels, audio_sample_width}`.
That is the sound a human would hear, not a letter index. The live adapter
encodes those samples as a second amodal event, binds vision then audio,
and packs the frozen two-way decoder onto the two public ports. Privileged
keys (`letter`, `current_stim`, `audio` as an ID, `n_back`, …) fail closed.

## What was tested

The same frozen `PREVIOUS` composition used on rendered Dual executed
Neural Workshop Dual 1-Back and Dual 2-Back with `learn=False`. A one-step
program on 2-back is the wrong-depth control.

| Seed | Dual 1-Back | Dual 2-Back | Wrong depth | Audio events | Vision events |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 98017 | 1.000 (29 bits) | 1.000 (20 bits) | 0.091 (33 bits) | 60 / 60 / 60 | 60 / 60 / 60 |
| 98117 | 1.000 (20 bits) | 1.000 (20 bits) | 0.109 (32 bits) | 60 / 60 / 60 | 60 / 60 / 60 |

Controller, program, and replay updates were zero on both seeds. Action
count is 4 from two packed binary decisions; the decoder still has two
keys.

## Why this is not promoted

- two seeds, no holdout lease;
- execution of an existing temporal program. Blank-file Dual acquisition
  is in `session_records/brainworkshop_dual_acquisition_2026-08-15/`;
- desktop screen-capture Dual is optional I/O against Neural Workshop, not a trainer.

Retain the blueprint. Do not treat an isolated threshold as mastery.

## Run

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.neural_workshop_dual_live_pilot \
  --neural-workshop /absolute/path/to/neural-workshop \
  --trials 60 --seed 98017
```
