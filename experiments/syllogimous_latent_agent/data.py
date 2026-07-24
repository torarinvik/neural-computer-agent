from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from experiments.syllogimous_realtime.environment import (
    Action,
    PrivateQuestion,
    RELATIONS,
    XorShift64,
    generate_question,
)


IMAGE_WIDTH = 160
IMAGE_HEIGHT = 96
PCM_SAMPLES = 256
SAMPLE_RATE = 16_000
RELATION_NAMES = tuple(name for pair in RELATIONS for name in pair)
RELATION_TO_ID = {name: index for index, name in enumerate(RELATION_NAMES)}
TRAIN_PREFIXES = ("A", "B", "C", "Q", "X", "Y")


@lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    matplotlib = find_spec("matplotlib")
    if matplotlib is not None and matplotlib.submodule_search_locations:
        candidates.append(str(Path(next(iter(matplotlib.submodule_search_locations))) /
                              "mpl-data/fonts/ttf/DejaVuSansMono.ttf"))
    candidates.extend(("/System/Library/Fonts/Monaco.ttf",
                       "/Library/Fonts/Arial Unicode.ttf"))
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def render_public_card(text: str, index: int, total: int, style_seed: int,
                       is_final: bool = False) -> np.ndarray:
    """Render public information only; no answer or family enters the pixels."""
    rng = np.random.default_rng(style_seed)
    background = tuple(int(x) for x in rng.integers(5, 31, size=3))
    foreground = tuple(int(x) for x in rng.integers(215, 256, size=3))
    accent = tuple(int(x) for x in rng.integers(90, 211, size=3))
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4), radius=5,
                           outline=accent, width=2)
    header = "CONCLUSION" if is_final else f"PREMISE {index}/{total - 1}"
    draw.text((9, 8), header, font=_font(9), fill=accent)
    font_size = 13 if len(text) <= 21 else (11 if len(text) <= 23 else 9)
    bbox = draw.textbbox((0, 0), text, font=_font(font_size))
    x = max(7, (IMAGE_WIDTH - (bbox[2] - bbox[0])) // 2)
    y = 46 - (bbox[3] - bbox[1]) // 2
    draw.text((x, y), text, font=_font(font_size), fill=foreground)
    return np.asarray(image, dtype=np.uint8).copy()


def render_public_audio(index: int, samples: int = PCM_SAMPLES) -> np.ndarray:
    """A public navigation cue, deliberately carrying no proposition or answer."""
    t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    envelope = np.exp(-18.0 * t)
    return (0.08 * envelope * np.sin(2 * np.pi * (300 + 30 * (index % 4)) * t)).astype(np.float32)


def balanced_question(seed: int, premises: int, *, heldout: bool = False,
                      final: bool = False, entity_count: int = 64) -> PrivateQuestion:
    """Remove the base environment's conclusion-word shortcut.

    Endpoint order is randomized independently of truth. Consequently each
    relation word is a true conclusion in some episodes and false in others.
    """
    if entity_count == 64:
        base = generate_question(seed, premises=premises, heldout=heldout, final=final)
    else:
        if entity_count <= premises:
            raise ValueError("entity_count must exceed premises")
        if seed >= 100_000 and not final:
            raise ValueError("reserved evaluation seed requires final=True")
        rng = XorShift64(seed)
        prefix = "Z" if heldout else "Q"
        width = max(2, len(str(entity_count - 1)))
        vocabulary = tuple(f"{prefix}{index:0{width}d}" for index in range(entity_count))
        offset = rng.integer(0, len(vocabulary) - premises)
        symbols = list(vocabulary[offset:offset + premises + 1])
        forward, reverse = RELATIONS[rng.integer(0, len(RELATIONS))]
        statements = []
        for index in range(premises):
            if rng.coin():
                statements.append(f"{symbols[index]} IS {forward} {symbols[index + 1]}")
            else:
                statements.append(f"{symbols[index + 1]} IS {reverse} {symbols[index]}")
        answer = rng.coin()
        relation = forward if answer else reverse
        conclusion = f"{symbols[0]} IS {relation} {symbols[-1]}"
        for index in range(len(statements) - 1, 0, -1):
            other = rng.integer(0, index + 1)
            statements[index], statements[other] = statements[other], statements[index]
        base = PrivateQuestion(tuple(statements), conclusion, answer, forward, seed)
    first, tail = base.conclusion.split(" IS ", 1)
    _, last = tail.rsplit(" ", 1)
    reverse_by_forward = dict(RELATIONS)
    forward = base.family
    reverse = reverse_by_forward[forward]
    swap_endpoints = XorShift64(seed ^ 0x9E3779B97F4A7C15).coin()
    if swap_endpoints:
        subject, obj = last, first
        truthful, false_relation = reverse, forward
    else:
        subject, obj = first, last
        truthful, false_relation = forward, reverse
    relation = truthful if base.answer else false_relation
    conclusion = f"{subject} IS {relation} {obj}"
    return PrivateQuestion(base.premises, conclusion, base.answer, base.family, base.seed)


def visible_texts(question: PrivateQuestion, *, heldout: bool) -> tuple[str, ...]:
    texts = question.premises + (question.conclusion,)
    if heldout:
        return texts
    prefix = TRAIN_PREFIXES[question.seed % len(TRAIN_PREFIXES)]
    return tuple(text.replace("Q", prefix) for text in texts)


@dataclass(frozen=True)
class PublicEpisode:
    frames: np.ndarray
    pcm: np.ndarray
    actions: np.ndarray
    subjects: np.ndarray
    relations: np.ndarray
    objects: np.ndarray
    length: int
    seed: int
    group: int | None = None


def generate_public_episode(seed: int, premises: int, *, heldout: bool = False,
                            final: bool = False, entity_count: int = 64,
                            randomize_rendering: bool = False) -> PublicEpisode:
    question = balanced_question(seed, premises=premises, heldout=heldout, final=final,
                                 entity_count=entity_count)
    texts = visible_texts(question, heldout=heldout)
    frames = np.stack([
        render_public_card(
            text, i + 1, len(texts),
            ((seed * 0x9E3779B1 + i * 0x85EBCA77) & 0x7FFF_FFFF)
            if randomize_rendering else i,
            is_final=i == len(texts) - 1,
        ) for i, text in enumerate(texts)
    ])
    pcm = np.stack([render_public_audio(i) for i in range(len(texts))])
    actions = np.full(len(texts), int(Action.NEXT), dtype=np.int64)
    actions[-1] = int(Action.TRUE if question.answer else Action.FALSE)
    subjects, relations, objects = [], [], []
    for text in texts:
        subject, tail = text.split(" IS ", 1)
        relation, obj = tail.rsplit(" ", 1)
        subjects.append(int(subject[1:]))
        relations.append(RELATION_TO_ID[relation])
        objects.append(int(obj[1:]))
    return PublicEpisode(frames, pcm, actions,
                         np.asarray(subjects, dtype=np.int64),
                         np.asarray(relations, dtype=np.int64),
                         np.asarray(objects, dtype=np.int64), len(texts), seed)


class EpisodeDataset(Dataset):
    def __init__(self, samples: int, *, start_seed: int = 0,
                 premise_choices: Sequence[int] = (2, 3, 4, 5, 6),
                 heldout: bool = False, final: bool = False,
                 entity_count: int = 64, randomize_rendering: bool = False):
        self.samples = samples
        self.start_seed = start_seed
        self.premise_choices = tuple(int(x) for x in premise_choices)
        self.heldout = heldout
        self.final = final
        self.entity_count = entity_count
        self.randomize_rendering = randomize_rendering
        if not self.premise_choices or min(self.premise_choices) < 2:
            raise ValueError("premise choices must contain values >= 2")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> PublicEpisode:
        seed = self.start_seed + index
        premises = self.premise_choices[seed % len(self.premise_choices)]
        return generate_public_episode(seed, premises, heldout=self.heldout, final=self.final,
                                       entity_count=self.entity_count,
                                       randomize_rendering=self.randomize_rendering)


def collate_episodes(items: Sequence[PublicEpisode]) -> dict[str, torch.Tensor]:
    batch = len(items)
    steps = max(item.length for item in items)
    frames = torch.zeros(batch, steps, 3, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.float32)
    pcm = torch.zeros(batch, steps, PCM_SAMPLES, dtype=torch.float32)
    actions = torch.full((batch, steps), -100, dtype=torch.long)
    subjects = torch.full((batch, steps), -100, dtype=torch.long)
    relations = torch.full((batch, steps), -100, dtype=torch.long)
    objects = torch.full((batch, steps), -100, dtype=torch.long)
    mask = torch.zeros(batch, steps, dtype=torch.bool)
    seeds = torch.empty(batch, dtype=torch.long)
    groups = torch.full((batch,), -1, dtype=torch.long)
    for row, item in enumerate(items):
        length = item.length
        frames[row, :length] = torch.from_numpy(item.frames).permute(0, 3, 1, 2).float().div_(255.0)
        pcm[row, :length] = torch.from_numpy(item.pcm)
        actions[row, :length] = torch.from_numpy(item.actions)
        subjects[row, :length] = torch.from_numpy(item.subjects)
        relations[row, :length] = torch.from_numpy(item.relations)
        objects[row, :length] = torch.from_numpy(item.objects)
        mask[row, :length] = True
        seeds[row] = item.seed
        if item.group is not None:
            groups[row] = item.group
    return {"frames": frames, "pcm": pcm, "actions": actions, "subjects": subjects,
            "relations": relations, "objects": objects, "mask": mask, "seeds": seeds,
            "groups": groups}
