# Brain Workshop gym integration — 2026-08-01

The upstream [Brain Workshop 5 repository](https://github.com/brain-workshop/brainworkshop)
is a Python/Pyglet dual N-back application with position, audio, multi-stimulus,
crab, interference, and configurable keypress modes. It is GPL-2.0, so this
project does not vendor its GUI, assets, or source code. Instead, this folder's
headless gym is a deterministic clean-room training surface with the same
useful first rung: simultaneous visual position and audio streams, two-bit
match keypresses, n-back targets, and real-time latency scoring.

The upstream reference was inspected at commit
`3476f724eb623b6e39605bd7a7e3df245787e73a`.

## Contract

The learner receives only an RGB frame, an audio waveform, and scalar verifier
reward after acting. The generated target mask, sequence, and seed remain
verifier-private. `POSITION_MATCH=1` and `AUDIO_MATCH=2` are output protocol
bits; the keypress decoder can map them to any physical key codes later.

The initial curriculum is deliberately small: eight-way position/audio
symbols, n-back 1–2, 1-second trials, and gradual match/interference expansion.
The event encoders emit independent amodal events so vision-only, audio-only,
and simultaneous-stream conditions can be compared without changing the
controller width.

## Smoke result

`smoke_audit.json` passes deterministic replay, perfect oracle scoring, a
strictly small speed bonus, random/no-op/inverted adversarial controls, two
independent event streams, and target privacy. This is a gym plumbing result,
not yet a learned-agent or compounding sample-efficiency claim.
