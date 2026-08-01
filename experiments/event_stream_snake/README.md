# Event-stream Snake proof of concept

> **Historical experiment:** The streamer/listener arrangement below tests one
> component hypothesis; it is not the target system architecture. See the
> [canonical amodal N-to-M specification](../../docs/AMODAL_N_TO_M_ARCHITECTURE.md).

This experiment intentionally overfits one audiovisual Snake environment to test a
narrow question: can a trainable event adapter improve a completely frozen SmolVLM2
listener while emitting fewer sensory tokens at lower latency?

The runtime accepts only synchronized raw pixels and PCM. Sensor sampling runs three
times faster than game actions, producing repeated frames and literal silence. Apple
collection produces a rendered PCM chirp. No game state, event flag, teacher action,
reward, or environment callback enters the listener.

Three controls share the same frozen listener:

- `dense`: emit vision and audio at every sensor tick.
- `fixed`: emit vision on raw pixel change and audio above an RMS threshold.
- `learned`: train representations and straight-through emission gates, then physically
  compact emitted events during rollouts.

Training begins with teacher action supervision through the frozen LLM. The learned
variant then receives short-segment policy-gradient updates from apples, survival,
death, and a deliberately small emission-cost term. Task success dominates efficiency.

```sh
python -m unittest experiments.event_stream_snake.test_event_stream

python -m experiments.event_stream_snake.train \
  --mode learned --local-files-only --device cuda
```

Admission requires higher apples per episode than the dense and fixed controls, fewer
tokens, lower p50/p95 latency, and a causal performance drop when relevant events are
removed. This folder is independent of `experiments/sensory_codec/`.
