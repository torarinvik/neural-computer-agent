# Continuous noisy reflex arena

> **Historical experiment:** The streamer/listener arrangement below tests one
> component hypothesis; it is not the target system architecture. See the
> [canonical amodal N-to-M specification](../../docs/AMODAL_N_TO_M_ARCHITECTURE.md).

This isolated experiment extends the one-step reflex proof into longer, noisy
sensor windows. Relevant audiovisual target/hazard cues are mixed with irrelevant
screen flashes and sounds. Held-out trials shift cue appearance, audio frequencies,
themes, and onset timing.

The frozen SmolVLM2 listener receives only embeddings derived from raw frames and
PCM. Labels and `relevant_ticks` exist solely for training/evaluation audits and are
never passed into the controller.

Four controls share the listener and data: dense, fixed change thresholds,
content-aware learned gates, and deterministic random gates. Random gating can be
assigned the learned streamer's measured token budget. A listener-only benchmark
also compares repeated causal prefill against persistent KV caching.

The learned content gate is stabilized against raw frame-delta and audio-energy
events before task reward refines it. These targets are computed solely from the
sensor stream and contain no game-state or relevance annotations.

```sh
python -m unittest experiments.continuous_reflex.test_continuous
python -m experiments.continuous_reflex.train --mode learned --device cuda
```

See [RESULTS.md](RESULTS.md) for the confirmed three-seed comparison. The fixed
gate retains nearly all dense accuracy with about 80% fewer tokens. The learned
gate does not beat fixed or random controls in this version.
