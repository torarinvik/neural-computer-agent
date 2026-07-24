from __future__ import annotations

import numpy as np

from experiments.event_stream_reflex.environment import DIRS, OPPOSITE


MODALITIES = ("vision", "audio", "both")


def cue_pcm(direction: int, hazard: bool, held_out: bool, samples: int = 64) -> np.ndarray:
    # Held-out tones shift within the same ordered frequency bands rather than
    # redefining the direction-to-frequency semantics.
    cycles = (3.3 if held_out else 3.0) + direction * 2.0
    phase = np.linspace(0, cycles * 2 * np.pi, samples, endpoint=False)
    carrier = np.tanh(3 * np.sin(phase)) if hazard else np.sin(phase)
    envelope = np.linspace(0.9, 0.2, samples) if held_out else np.linspace(1.0, 0.25, samples)
    return (0.55 * carrier * envelope).astype(np.float32)


def distractor_pcm(kind: int, samples: int = 64) -> np.ndarray:
    rng = np.random.default_rng(9000 + kind)
    if kind % 2:
        return (rng.normal(0, 0.16, samples)).astype(np.float32)
    phase = np.linspace(0, (12 + kind) * 2 * np.pi, samples, endpoint=False)
    return (0.13 * np.sin(phase)).astype(np.float32)


def base_frame(size: int, theme: int, held_out: bool) -> np.ndarray:
    image = np.zeros((4, size + 2, size + 2), dtype=np.float32)
    image[0, (0, -1), :] = 0.7
    image[0, :, (0, -1)] = 0.7
    center = size // 2 + 1
    image[2, center, center] = 1.0
    palettes = (
        np.asarray((1.0, 0.55, 0.75, 0.65)),
        np.asarray((0.75, 0.9, 0.55, 0.8)),
        np.asarray((0.6, 0.7, 1.0, 0.5)),
    )
    palette = palettes[(theme + (1 if held_out else 0)) % len(palettes)]
    return image * palette[:, None, None]


def add_cue(image: np.ndarray, direction: int, hazard: bool, held_out: bool) -> np.ndarray:
    result = image.copy()
    size = image.shape[-1] - 2
    center = np.asarray((size // 2 + 1, size // 2 + 1))
    row, col = center + DIRS[direction] * max(2, size // 3)
    radius = 1
    channel = 1 if hazard else 3
    result[channel, row - radius:row + radius + 1, col - radius:col + radius + 1] = 1.0
    if held_out:
        # Held-out hollow center changes appearance without changing semantics.
        result[channel, row, col] = 0.25
    return result


def add_visual_distractor(image: np.ndarray, row: int, col: int, kind: int) -> np.ndarray:
    result = image.copy()
    channel = kind % 2  # never uses the agent channel; may resemble cue color.
    result[channel, row:row + 2, col:col + 2] = 0.35 + 0.1 * (kind % 3)
    return result


def make_dataset(samples: int, sequence: int, size: int, seed: int,
                 split: str = "train", distractors: int = 5) -> dict[str, np.ndarray]:
    if split not in ("train", "heldout"):
        raise ValueError(split)
    if sequence < 12:
        raise ValueError("continuous windows need at least 12 ticks")
    held_out = split == "heldout"
    rng = np.random.default_rng(seed)
    order = rng.permutation(samples)
    frames = np.empty((samples, sequence, 4, size + 2, size + 2), dtype=np.float32)
    audio = np.zeros((samples, sequence, 64), dtype=np.float32)
    actions = np.empty(samples, dtype=np.int64)
    modalities = np.empty(samples, dtype=np.int64)
    hazards = np.empty(samples, dtype=np.int64)
    relevant_ticks = np.zeros((samples, sequence, 2), dtype=np.float32)
    for index, value in enumerate(order):
        action = int(value % 4)
        hazard = bool((value // 4) % 2)
        modality = int((value // 8) % 3)
        direction = OPPOSITE[action] if hazard else action
        baseline = base_frame(size, int(rng.integers(3)), held_out)
        frames[index] = baseline
        onset_low = sequence // 3 if not held_out else sequence // 4
        onset_high = sequence - 2 if not held_out else sequence - 1
        onset = int(rng.integers(onset_low, onset_high))
        if modality in (0, 2):
            frames[index, onset:] = add_cue(baseline, direction, hazard, held_out)
            relevant_ticks[index, onset, 0] = 1.0
        if modality in (1, 2):
            audio[index, onset] = cue_pcm(direction, hazard, held_out)
            relevant_ticks[index, onset, 1] = 1.0
        occupied = {onset}
        for nuisance in range(distractors):
            tick = int(rng.integers(1, sequence - 1))
            while tick in occupied:
                tick = int(rng.integers(1, sequence - 1))
            occupied.add(tick)
            if nuisance % 2 == 0:
                row = int(rng.integers(1, size))
                col = int(rng.integers(1, size))
                flash = add_visual_distractor(frames[index, tick], row, col, nuisance)
                frames[index, tick] = flash
            else:
                audio[index, tick] = distractor_pcm(nuisance + (7 if held_out else 0))
        actions[index] = action
        modalities[index] = modality
        hazards[index] = int(hazard)
    return {
        "frames": frames,
        "audio": audio,
        "action": actions,
        "modality": modalities,
        "hazard": hazards,
        "relevant_ticks": relevant_ticks,
    }
