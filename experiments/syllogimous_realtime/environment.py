from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RESERVED_SEED = 100_000
SYMBOLS_TRAIN = tuple(f"Q{i:02d}" for i in range(64))
SYMBOLS_HELDOUT = tuple(f"Z{i:02d}" for i in range(64))


class Action(IntEnum):
    WAIT = 0
    NEXT = 1
    FALSE = 2
    TRUE = 3
    PREVIOUS = 4


@dataclass(frozen=True)
class PrivateQuestion:
    premises: tuple[str, ...]
    conclusion: str
    answer: bool
    family: str
    seed: int


@dataclass(frozen=True)
class SensoryPacket:
    timestamp_ms: int
    frame: np.ndarray
    pcm: np.ndarray


@dataclass(frozen=True)
class StepResult:
    observation: SensoryPacket
    reward: float
    done: bool
    outcome: str | None


class XorShift64:
    def __init__(self, seed: int):
        self.state = (seed + 1) & ((1 << 64) - 1)
        for _ in range(12):
            self.next()

    def next(self) -> int:
        x = self.state
        x ^= (x << 13) & ((1 << 64) - 1)
        x ^= x >> 7
        x ^= (x << 17) & ((1 << 64) - 1)
        self.state = x & ((1 << 64) - 1)
        return self.state

    def integer(self, low: int, high: int) -> int:
        return low + self.next() % (high - low)

    def coin(self) -> bool:
        return bool(self.next() & 1)


RELATIONS = (
    ("BEFORE", "AFTER"),
    ("LEFT OF", "RIGHT OF"),
    ("ABOVE", "BELOW"),
    ("LESS THAN", "GREATER THAN"),
)


def generate_question(seed: int, premises: int = 6, heldout: bool = False,
                      final: bool = False) -> PrivateQuestion:
    if premises < 2:
        raise ValueError("at least two premises are required")
    if seed >= RESERVED_SEED and not final:
        raise ValueError("reserved evaluation seed requires final=True")
    rng = XorShift64(seed)
    vocabulary = SYMBOLS_HELDOUT if heldout else SYMBOLS_TRAIN
    offset = rng.integer(0, len(vocabulary) - premises - 1)
    symbols = list(vocabulary[offset:offset + premises + 1])
    forward, reverse = RELATIONS[rng.integer(0, len(RELATIONS))]
    statements: list[str] = []
    for index in range(premises):
        if rng.coin():
            statements.append(f"{symbols[index]} IS {forward} {symbols[index + 1]}")
        else:
            statements.append(f"{symbols[index + 1]} IS {reverse} {symbols[index]}")
    # Half of conclusions are valid; invalid conclusions reverse the entailed
    # endpoint relation rather than introducing a superficial lexical mismatch.
    answer = rng.coin()
    relation = forward if answer else reverse
    conclusion = f"{symbols[0]} IS {relation} {symbols[-1]}"
    # Fisher-Yates, retaining a chain that must be reconstructed from order-free premises.
    for index in range(len(statements) - 1, 0, -1):
        other = rng.integer(0, index + 1)
        statements[index], statements[other] = statements[other], statements[index]
    return PrivateQuestion(tuple(statements), conclusion, answer, forward, seed)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_card(text: str, index: int, total: int, remaining_fraction: float,
                width: int = 640, height: int = 400) -> np.ndarray:
    image = Image.new("RGB", (width, height), (10, 14, 18))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, width - 24, height - 24), outline=(90, 180, 210), width=3)
    draw.text((45, 48), f"CARD {index}/{total}", font=_font(22), fill=(140, 210, 230))
    draw.multiline_text((45, 150), text, font=_font(30), fill=(240, 244, 245),
                        spacing=12, align="left")
    bar_width = int((width - 90) * max(0.0, min(1.0, remaining_fraction)))
    draw.rectangle((45, height - 62, width - 45, height - 43), outline=(80, 80, 80))
    draw.rectangle((45, height - 62, 45 + bar_width, height - 43), fill=(60, 200, 110))
    return np.asarray(image, dtype=np.uint8)


def timing_pcm(timestamp_ms: int, event: str | None, samples: int = 533,
               sample_rate: int = 16_000) -> np.ndarray:
    """Raw real-time PCM: quiet clock bed plus distinct public feedback cues."""
    start = timestamp_ms * sample_rate // 1000
    indices = start + np.arange(samples)
    bed = 0.025 * np.sin(2 * np.pi * 220 * indices / sample_rate)
    frequency = {"next": 440, "previous": 330, "right": 880,
                 "wrong": 130, "timeout": 90}.get(event)
    if frequency is not None:
        bed += 0.22 * np.sin(2 * np.pi * frequency * indices / sample_rate)
    return bed.astype(np.float32)


class RealtimeEpisode:
    """Causal evaluator. PrivateQuestion never appears in SensoryPacket."""

    def __init__(self, question: PrivateQuestion, deadline_ms: int = 8_000,
                 clock_ns: Callable[[], int] | None = None):
        if deadline_ms <= 0:
            raise ValueError("deadline must be positive")
        import time
        self._question = question
        self._deadline_ms = deadline_ms
        self._clock_ns = clock_ns or time.monotonic_ns
        self._started_ns = self._clock_ns()
        self._card = 0
        self._done = False
        self._outcome: str | None = None
        self._last_event: str | None = None

    def elapsed_ms(self) -> int:
        return max(0, (self._clock_ns() - self._started_ns) // 1_000_000)

    def _packet(self) -> SensoryPacket:
        elapsed = self.elapsed_ms()
        texts = self._question.premises + (self._question.conclusion,)
        text = texts[self._card]
        frame = render_card(text, self._card + 1, len(texts),
                            1.0 - elapsed / self._deadline_ms)
        pcm = timing_pcm(elapsed, self._last_event)
        self._last_event = None
        return SensoryPacket(elapsed, frame, pcm)

    def step(self, action: Action) -> StepResult:
        if self._done:
            return StepResult(self._packet(), 0.0, True, self._outcome)
        elapsed = self.elapsed_ms()
        if elapsed >= self._deadline_ms:
            self._done, self._outcome, self._last_event = True, "timeout", "timeout"
            return StepResult(self._packet(), -1.0, True, self._outcome)
        conclusion_index = len(self._question.premises)
        if action == Action.NEXT and self._card < conclusion_index:
            self._card += 1
            self._last_event = "next"
        elif action == Action.PREVIOUS and self._card > 0:
            self._card -= 1
            self._last_event = "previous"
        elif action in (Action.TRUE, Action.FALSE) and self._card == conclusion_index:
            choice = action == Action.TRUE
            correct = choice == self._question.answer
            self._done = True
            self._outcome = "right" if correct else "wrong"
            self._last_event = self._outcome
            reward = (1.0 + 0.05 * (self._deadline_ms - elapsed) / self._deadline_ms
                      if correct else -1.0)
            return StepResult(self._packet(), reward, True, self._outcome)
        return StepResult(self._packet(), 0.0, False, None)
