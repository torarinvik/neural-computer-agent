from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from experiments.syllogimous_latent_agent.data import (
    IMAGE_HEIGHT, IMAGE_WIDTH, PCM_SAMPLES, PublicEpisode, render_public_audio,
)
from experiments.syllogimous_realtime.environment import Action, XorShift64


@dataclass(frozen=True)
class SensoryLifetime:
    """A sequence of episodes whose hidden mapping changes every lifetime."""

    episodes: tuple[PublicEpisode, ...]
    associations: int
    delay: int
    seed: int
    audit_queries: tuple[PublicEpisode, ...] = ()


def _glyph(draw: ImageDraw.ImageDraw, code: int, center: tuple[int, int]) -> None:
    """Draw a compact public glyph without exposing its integer code as text."""
    x, y = center
    colors = ((239, 71, 111), (57, 189, 248), (93, 220, 132), (255, 194, 74),
              (180, 111, 255), (255, 125, 65), (75, 224, 216), (236, 93, 224))
    color = colors[code % len(colors)]
    shape = (code // len(colors)) % 4
    radius = 12
    box = (x - radius, y - radius, x + radius, y + radius)
    if shape == 0:
        draw.ellipse(box, fill=color, outline=(255, 255, 255), width=2)
    elif shape == 1:
        draw.rectangle(box, fill=color, outline=(255, 255, 255), width=2)
    elif shape == 2:
        draw.polygon(((x, y - radius), (x + radius, y + radius),
                      (x - radius, y + radius)), fill=color, outline=(255, 255, 255))
    else:
        draw.polygon(((x, y - radius), (x + radius, y), (x, y + radius),
                      (x - radius, y)), fill=color, outline=(255, 255, 255))
    stripes = 1 + (code // 32) % 3
    for offset in range(stripes):
        line_y = y - 5 + offset * 5
        draw.line((x - 7, line_y, x + 7, line_y), fill=(20, 20, 25), width=2)


def _frame(seed: int, glyph: int | None, *, answer: int | None = None,
           study: bool = False, context: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    background = tuple(int(value) for value in rng.integers(5, 24, size=3))
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4), radius=6,
                           outline=(80, 90, 110), width=2)
    if context is not None:
        draw.rectangle((5, 7, 35, 43), fill=(8, 8, 12), outline=(90, 100, 120), width=1)
        _glyph(draw, context, (20, 25))
    if glyph is not None:
        _glyph(draw, glyph, (IMAGE_WIDTH // 2, 39))
    response_x = tuple(int(round(18 + index * (IMAGE_WIDTH - 36) / 7)) for index in range(8))
    for index, x in enumerate(response_x):
        draw.ellipse((x - 4, 76, x + 4, 84), outline=(90, 100, 120), width=1)
        if study and answer == index:
            draw.ellipse((x - 7, 73, x + 7, 87), outline=(255, 255, 255), width=2)
            draw.ellipse((x - 3, 77, x + 3, 83), fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8).copy()


def _episode(frame: np.ndarray, seed: int, action: int, group: int) -> PublicEpisode:
    unused = np.asarray([-1], dtype=np.int64)
    return PublicEpisode(frame[None], render_public_audio(seed, PCM_SAMPLES)[None],
                         np.asarray([action], dtype=np.int64), unused, unused.copy(),
                         unused.copy(), 1, seed, group=group)


def generate_sensory_lifetime(seed: int, *, associations: int = 2, delay: int = 8,
                              choices: int = 8, heldout: bool = False,
                              contextual: bool = False,
                              audit_variants: int = 0) -> SensoryLifetime:
    """Create a deterministic study–delay–query lifetime from RGB/PCM only.

    A study card publicly pairs a novel glyph with a response location. Queries
    later show only that glyph. Assignments are permuted per lifetime, so model
    weights alone cannot predict held-out answers.
    """
    if not 1 <= associations <= choices <= 8:
        raise ValueError("require 1 <= associations <= choices <= 8")
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if audit_variants < 0:
        raise ValueError("audit_variants must be non-negative")
    salt = 0xD1B54A32D192ED03 if heldout else 0
    rng = XorShift64(seed ^ salt)
    glyph_pool = list(range(64))
    answers = list(range(choices))
    for values in (glyph_pool, answers):
        for index in range(len(values) - 1, 0, -1):
            other = rng.integer(0, index + 1)
            values[index], values[other] = values[other], values[index]
    glyphs = glyph_pool[:associations]
    mapping = dict(zip(glyphs, answers[:associations]))
    context = (seed * 17) % 64 if contextual else None
    episodes: list[PublicEpisode] = []
    for index, glyph in enumerate(glyphs):
        public_seed = seed * 1000 + index
        episodes.append(_episode(_frame(public_seed, glyph, answer=mapping[glyph], study=True,
                                             context=context),
                                 public_seed, int(Action.NEXT), associations))
    for index in range(delay):
        public_seed = seed * 1000 + associations + index
        episodes.append(_episode(_frame(public_seed, None, context=context), public_seed,
                                 int(Action.WAIT), associations))
    query_order = glyphs.copy()
    for index in range(len(query_order) - 1, 0, -1):
        other = rng.integer(0, index + 1)
        query_order[index], query_order[other] = query_order[other], query_order[index]
    for index, glyph in enumerate(query_order):
        public_seed = seed * 1000 + associations + delay + index
        episodes.append(_episode(_frame(public_seed, glyph, context=context), public_seed,
                                 mapping[glyph], associations))
    audit_queries = []
    for variant in range(audit_variants):
        for index, glyph in enumerate(query_order):
            # Same public glyph, context, and uniquely correct action; independently
            # rendered background and PCM. These views never enter commit decisions.
            public_seed = seed * 1_000_000 + 700_000 + variant * associations + index
            audit_queries.append(_episode(
                _frame(public_seed, glyph, context=context), public_seed,
                mapping[glyph], associations))
    return SensoryLifetime(tuple(episodes), associations, delay, seed,
                           tuple(audit_queries))
