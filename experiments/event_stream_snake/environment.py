from __future__ import annotations

from collections import deque

import numpy as np

from experiments.sensory_codec.snake import SnakeEnv


def event_audio(ate: bool, samples: int = 64) -> np.ndarray:
    """Render observable PCM; silence is literal zeros, never a symbolic label."""
    if not ate:
        return np.zeros(samples, dtype=np.float32)
    phase = np.linspace(0, 10 * np.pi, samples, endpoint=False)
    envelope = np.linspace(1.0, 0.15, samples)
    return (0.65 * np.sin(phase) * envelope).astype(np.float32)


def make_event_dataset(samples: int, sequence: int, size: int, seed: int,
                       sensor_ticks: int = 3) -> dict[str, np.ndarray]:
    """Teacher trajectories sampled faster than actions to contain real redundancy."""
    if sequence < sensor_ticks:
        raise ValueError("sequence must cover at least one action interval")
    env = SnakeEnv(size=size, seed=seed)
    frames: list[np.ndarray] = []
    audio: list[np.ndarray] = []
    actions: list[int] = []
    frame_history: deque[np.ndarray] = deque(maxlen=sequence)
    audio_history: deque[np.ndarray] = deque(maxlen=sequence)
    while len(frames) < samples:
        env.reset()
        first = env.observe()
        frame_history.clear()
        audio_history.clear()
        for _ in range(sequence):
            frame_history.append(first.copy())
            audio_history.append(event_audio(False))
        for _ in range(300):
            action = env.teacher_action()
            _, done = env.step(action)
            current = env.observe()
            sound = event_audio(env.state.ate_last)
            # Only the first sensor tick observes the new transition and sound;
            # later ticks are unchanged video plus silence.
            for tick in range(sensor_ticks):
                frame_history.append(current.copy())
                audio_history.append(sound.copy() if tick == 0 else event_audio(False))
            if not done:
                frames.append(np.stack(frame_history))
                audio.append(np.stack(audio_history))
                actions.append(env.teacher_action())
            if done or len(frames) >= samples:
                break
    return {
        "frames": np.stack(frames).astype(np.float32),
        "audio": np.stack(audio).astype(np.float32),
        "action": np.asarray(actions, dtype=np.int64),
    }
