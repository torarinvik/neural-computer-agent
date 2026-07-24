from __future__ import annotations

import numpy as np


DIRS = np.asarray([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int64)
OPPOSITE = (2, 3, 0, 1)
MODALITIES = ("vision", "audio", "both")


def directional_pcm(direction: int, hazard: bool, samples: int = 64) -> np.ndarray:
    """Raw mono PCM: frequency conveys direction; waveform shape conveys kind."""
    phase = np.linspace(0, (3 + 2 * direction) * 2 * np.pi, samples, endpoint=False)
    carrier = np.sign(np.sin(phase)) if hazard else np.sin(phase)
    envelope = np.linspace(1.0, 0.25, samples)
    return (0.55 * carrier * envelope).astype(np.float32)


def render_frame(size: int, direction: int | None, hazard: bool,
                 theme: int = 0) -> np.ndarray:
    image = np.zeros((4, size + 2, size + 2), dtype=np.float32)
    image[0, 0, :] = image[0, -1, :] = 1.0
    image[0, :, 0] = image[0, :, -1] = 1.0
    center = np.asarray((size // 2, size // 2))
    image[2, center[0] + 1, center[1] + 1] = 1.0
    if direction is not None:
        row, col = center + DIRS[direction] * max(2, size // 3)
        channel = 1 if hazard else 3
        # Large enough to register as a visual event, but still just raw pixels.
        row, col = row + 1, col + 1
        image[channel, row - 1:row + 2, col - 1:col + 2] = 1.0
    if theme:
        image *= np.asarray([1.0, 0.5, 0.7, 0.6], dtype=np.float32)[:, None, None]
    return image


def make_reflex_dataset(samples: int, sequence: int, size: int, seed: int) -> dict[str, np.ndarray]:
    if sequence < 4:
        raise ValueError("sequence must be at least four sensor ticks")
    rng = np.random.default_rng(seed)
    frames, audio, actions, modalities, hazards = [], [], [], [], []
    order = rng.permutation(samples)
    for index in range(samples):
        correct_action = int(order[index] % 4)
        hazard = bool((order[index] // 4) % 2)
        modality_id = int((order[index] // 8) % len(MODALITIES))
        cue_direction = OPPOSITE[correct_action] if hazard else correct_action
        theme = int(rng.integers(2))
        baseline = render_frame(size, None, hazard, theme)
        visual = render_frame(size, cue_direction, hazard, theme)
        onset = int(rng.integers(max(1, sequence // 2), sequence))
        frame_sequence = []
        audio_sequence = []
        for tick in range(sequence):
            visual_active = tick >= onset and modality_id in (0, 2)
            audio_active = tick == onset and modality_id in (1, 2)
            frame_sequence.append((visual if visual_active else baseline).copy())
            audio_sequence.append(
                directional_pcm(cue_direction, hazard) if audio_active
                else np.zeros(64, dtype=np.float32))
        frames.append(np.stack(frame_sequence))
        audio.append(np.stack(audio_sequence))
        actions.append(correct_action)
        modalities.append(modality_id)
        hazards.append(int(hazard))
    return {
        "frames": np.stack(frames),
        "audio": np.stack(audio),
        "action": np.asarray(actions, dtype=np.int64),
        "modality": np.asarray(modalities, dtype=np.int64),
        "hazard": np.asarray(hazards, dtype=np.int64),
    }
