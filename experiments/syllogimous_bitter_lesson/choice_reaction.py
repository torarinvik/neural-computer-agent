from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from experiments.syllogimous_latent_agent.data import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PCM_SAMPLES,
    PublicEpisode,
    SAMPLE_RATE,
)
from experiments.syllogimous_realtime.environment import Action, XorShift64


MAX_CHOICES = 8
TARGET_COLORS = (
    (239, 71, 111), (57, 189, 248), (93, 220, 132), (255, 194, 74),
    (180, 111, 255), (255, 125, 65), (75, 224, 216), (236, 93, 224),
)


@dataclass(frozen=True)
class ReactionDifficulty:
    choices: int
    distractors: int = 0
    delay_frames: int = 0
    audio_distractors: int = 0
    target_like_distractors: int = 0
    temporal_distractors: int = 0

    def __post_init__(self) -> None:
        if not 2 <= self.choices <= MAX_CHOICES:
            raise ValueError(f"choices must be between 2 and {MAX_CHOICES}")
        if min(self.distractors, self.delay_frames, self.audio_distractors,
               self.target_like_distractors, self.temporal_distractors) < 0:
            raise ValueError("difficulty counts cannot be negative")


def _positions(count: int) -> tuple[tuple[int, int], ...]:
    margin = 18
    all_positions = tuple(
        (int(round(margin + index * (IMAGE_WIDTH - 2 * margin) / (MAX_CHOICES - 1))), 55)
        for index in range(MAX_CHOICES))
    return all_positions[:count]


def _blank_frame(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    background = tuple(int(value) for value in rng.integers(4, 20, size=3))
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4), radius=6,
                           outline=(55, 62, 75), width=2)
    draw.ellipse((77, 42, 83, 48), fill=(150, 150, 150))
    return image


def _audio_frame(seed: int, target: int | None, distractors: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time_axis = np.arange(PCM_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    pcm = np.zeros(PCM_SAMPLES, dtype=np.float32)
    if target is not None:
        pcm += 0.08 * np.sin(2 * np.pi * (330 + 70 * target) * time_axis)
    for _ in range(distractors):
        frequency = int(rng.integers(170, 1200))
        phase = float(rng.random() * 2 * np.pi)
        pcm += 0.012 * np.sin(2 * np.pi * frequency * time_axis + phase)
    return (pcm * np.exp(-9.0 * time_axis)).astype(np.float32)


def generate_choice_reaction_episode(seed: int, difficulty: ReactionDifficulty,
                                     *, heldout: bool = False) -> PublicEpisode:
    """Render a deterministic public choice-reaction trial from pixels and PCM.

    The target is the bright filled light. Irrelevant hollow lights and audio
    tones are independently randomized. The returned bookkeeping arrays exist
    only for the shared batch container and are never inputs to the agent.
    """
    rng = XorShift64(seed ^ (0xD1B54A32D192ED03 if heldout else 0))
    target = rng.integer(0, difficulty.choices)
    positions = _positions(difficulty.choices)
    frames: list[np.ndarray] = []
    pcm: list[np.ndarray] = []

    context_frames = 1 + difficulty.delay_frames
    for index in range(context_frames):
        context = _blank_frame(seed * 31 + index)
        if index > 0 and index <= difficulty.temporal_distractors:
            # A salient premature cue at a wrong response location. It lacks
            # the double white confirmation ring used by the true GO target.
            false_target = (target + 1 + rng.integer(0, difficulty.choices - 1)) \
                % difficulty.choices
            false_x, false_y = positions[false_target]
            context_draw = ImageDraw.Draw(context)
            context_draw.ellipse((false_x - 8, false_y - 8,
                                  false_x + 8, false_y + 8),
                                 fill=TARGET_COLORS[false_target])
        frames.append(np.asarray(context, dtype=np.uint8).copy())
        pcm.append(_audio_frame(seed * 47 + index, None,
                                difficulty.audio_distractors))

    image = _blank_frame(seed * 31 + context_frames)
    draw = ImageDraw.Draw(image)
    # Draw stable response locations so spatial choice, not text recognition,
    # defines the public rule. The selected light is filled and ringed in white.
    for choice, (x, y) in enumerate(positions):
        radius = 7
        color = TARGET_COLORS[choice]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     outline=color, width=2)
        draw.rectangle((x - 7, 76, x + 7, 88), outline=(100, 110, 125), width=1)
    target_x, target_y = positions[target]
    draw.ellipse((target_x - 8, target_y - 8, target_x + 8, target_y + 8),
                 fill=TARGET_COLORS[target], outline=(255, 255, 255), width=2)

    for _ in range(difficulty.distractors):
        x = rng.integer(10, IMAGE_WIDTH - 10)
        y = rng.integer(12, IMAGE_HEIGHT - 20)
        radius = rng.integer(2, 6)
        color = tuple(rng.integer(35, 180) for _ in range(3))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     outline=color, width=1)
    for _ in range(difficulty.target_like_distractors):
        false_choice = (target + 1 + rng.integer(0, difficulty.choices - 1)) \
            % difficulty.choices
        x, y = positions[false_choice]
        radius = rng.integer(5, 8)
        # Bright, filled, response-aligned distractors force the policy to use
        # the target's white confirmation ring rather than mere salience.
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=TARGET_COLORS[false_choice],
                     outline=TARGET_COLORS[(false_choice + 2) % MAX_CHOICES], width=1)
    frames.append(np.asarray(image, dtype=np.uint8).copy())
    pcm.append(_audio_frame(seed * 47 + context_frames, None,
                            difficulty.audio_distractors))

    length = len(frames)
    actions = np.full(length, int(Action.WAIT), dtype=np.int64)
    actions[-1] = target
    unused = np.full(length, -1, dtype=np.int64)
    return PublicEpisode(np.stack(frames), np.stack(pcm), actions,
                         unused, unused.copy(), unused.copy(), length, seed,
                         group=difficulty.choices)


class ChoiceReactionDataset(Dataset):
    def __init__(self, samples: int, difficulties: tuple[ReactionDifficulty, ...], *,
                 start_seed: int = 0, heldout: bool = False):
        if not difficulties:
            raise ValueError("at least one reaction difficulty is required")
        self.samples = samples
        self.difficulties = difficulties
        self.start_seed = start_seed
        self.heldout = heldout

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> PublicEpisode:
        seed = self.start_seed + index
        difficulty = self.difficulties[seed % len(self.difficulties)]
        return generate_choice_reaction_episode(seed, difficulty, heldout=self.heldout)


class CognitiveMixtureDataset(Dataset):
    """Deterministically interleave reasoning and reaction episodes."""

    def __init__(self, reasoning: Dataset, reaction: Dataset):
        self.reasoning = reasoning
        self.reaction = reaction

    def __len__(self) -> int:
        return len(self.reasoning) + len(self.reaction)

    def __getitem__(self, index: int) -> PublicEpisode:
        if not 0 <= index < len(self):
            raise IndexError(index)
        total = len(self)
        reaction_count = len(self.reaction)
        reactions_before = index * reaction_count // total
        reactions_after = (index + 1) * reaction_count // total
        if reactions_after > reactions_before:
            return self.reaction[reactions_before]
        return self.reasoning[index - reactions_before]
