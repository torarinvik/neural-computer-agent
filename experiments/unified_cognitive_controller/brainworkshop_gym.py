"""Headless Brain Workshop-style multimodal working-memory gym.

The upstream Brain Workshop application is a GPL-2.0 desktop program.  This
module is a clean-room, deterministic training surface inspired by its dual
N-back task: it does not import or vendor the GUI, audio assets, or source
code.  The agent receives only rendered vision/audio streams and returns an
opaque two-bit keypress intention.  Targets remain verifier-private.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from .amodal_interface import AmodalEvent, AmodalEventCollection

POSITION_MATCH: Final[int] = 1
AUDIO_MATCH: Final[int] = 2
NO_MATCH: Final[int] = 0
ALL_MATCHES: Final[int] = POSITION_MATCH | AUDIO_MATCH

_GRID_POSITIONS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)
)


@dataclass(frozen=True)
class BrainWorkshopConfig:
    """One deterministic task configuration.

    ``trial_ms`` is the nominal stimulus period.  The verifier receives the
    actual action latency separately so latency can be rewarded without
    leaking the answer or the generator state to the learner.
    """

    n_back: int = 1
    trials: int = 12
    position_vocab: int = 8
    audio_vocab: int = 8
    match_probability: float = 0.5
    trial_ms: int = 1_000
    audio_samples: int = 800
    sample_rate: int = 8_000
    vision_size: int = 72
    modalities: tuple[str, ...] = ("vision", "audio")

    def validate(self) -> BrainWorkshopConfig:
        if self.n_back < 1:
            raise ValueError("n_back must be positive")
        if self.trials <= self.n_back:
            raise ValueError("trials must exceed n_back")
        if self.position_vocab != 8 or self.audio_vocab != 8:
            raise ValueError("the initial gym uses eight-way stimuli")
        if not 0.0 < self.match_probability < 1.0:
            raise ValueError("match_probability must lie strictly between 0 and 1")
        if self.trial_ms < 1 or self.audio_samples < 8 or self.sample_rate < 100:
            raise ValueError("timing and audio dimensions are invalid")
        if self.vision_size < 24 or self.vision_size % 3:
            raise ValueError("vision_size must be divisible by three and >= 24")
        if not self.modalities or not set(self.modalities).issubset(
                {"vision", "audio"}):
            raise ValueError("modalities must contain vision and/or audio")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be unique")
        return self


@dataclass(frozen=True)
class BrainWorkshopStimulus:
    """Private generator representation for one trial."""

    position: int
    audio: int
    timestamp_ms: int


@dataclass(frozen=True)
class BrainWorkshopObservation:
    """Public sensory observation; no target or sequence metadata is present."""

    vision: torch.Tensor | None
    audio: torch.Tensor | None
    timestamp_ms: int


def _balanced_match_flags(
        count: int, probability: float, rng: random.Random) -> list[bool]:
    matches = round(count * probability)
    matches = max(1, min(count - 1, matches))
    flags = [index < matches for index in range(count)]
    rng.shuffle(flags)
    return flags


def _next_symbol(
        history: list[int], n_back: int, vocabulary: int, match: bool,
        rng: random.Random) -> int:
    if len(history) < n_back:
        return rng.randrange(vocabulary)
    reference = history[-n_back]
    if match:
        return reference
    choices = [value for value in range(vocabulary) if value != reference]
    return rng.choice(choices)


def render_position(
        position: int, *, size: int = 72, device: torch.device | str = "cpu"
        ) -> torch.Tensor:
    """Render one eight-position 3x3-grid stimulus as an RGB tensor."""
    if position < 0 or position >= len(_GRID_POSITIONS):
        raise ValueError("position must be in [0, 7]")
    frame = torch.full((3, size, size), 0.04, device=device)
    cell = size // 3
    row, column = _GRID_POSITIONS[position]
    margin = max(2, cell // 8)
    top, left = row * cell + margin, column * cell + margin
    bottom, right = (row + 1) * cell - margin, (column + 1) * cell - margin
    color = torch.tensor((0.20, 0.78, 0.94), device=device).view(3, 1, 1)
    frame[:, top:bottom, left:right] = color
    return frame


def render_audio(
        symbol: int, *, samples: int = 800, sample_rate: int = 8_000,
        device: torch.device | str = "cpu") -> torch.Tensor:
    """Render one deterministic audio token as a short waveform."""
    if symbol < 0 or symbol >= 8:
        raise ValueError("audio symbol must be in [0, 7]")
    time_axis = torch.arange(samples, device=device, dtype=torch.float32)
    frequency = 220.0 + 37.0 * symbol
    waveform = torch.sin(2.0 * math.pi * frequency * time_axis / sample_rate)
    fade = torch.linspace(0.0, 1.0, samples // 12, device=device)
    waveform[: fade.numel()] *= fade
    waveform[-fade.numel():] *= fade.flip(0)
    return waveform.unsqueeze(0)


@dataclass(frozen=True)
class BrainWorkshopEpisode:
    """A complete episode with private targets and public observations."""

    config: BrainWorkshopConfig
    seed: int
    stimuli: tuple[BrainWorkshopStimulus, ...]
    observations: tuple[BrainWorkshopObservation, ...]
    _targets: tuple[int, ...]

    @property
    def trials(self) -> int:
        return len(self.stimuli)

    def verifier_targets(self) -> tuple[int, ...]:
        """Return private targets only to an evaluation harness."""
        return self._targets

    def score_action(self, trial: int, action: int, latency_ms: float) -> float:
        """Score one opaque keypress bitmask with a small correct-speed bonus."""
        if trial < 0 or trial >= self.trials:
            raise IndexError("trial is outside the episode")
        if action < 0 or action > ALL_MATCHES:
            raise ValueError("action must be a two-bit keypress mask")
        if latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        correct = action == self._targets[trial]
        if not correct:
            return -1.0
        speed_bonus = 0.05 * max(
            0.0, 1.0 - min(float(latency_ms), self.config.trial_ms)
            / self.config.trial_ms)
        return 1.0 + speed_bonus


def generate_brainworkshop_episode(
        config: BrainWorkshopConfig | None = None, *, seed: int = 0,
        device: torch.device | str = "cpu") -> BrainWorkshopEpisode:
    """Generate a reproducible dual-stream N-back episode."""
    if config is None:
        config = BrainWorkshopConfig()
    config.validate()
    rng = random.Random(seed)
    position_flags = _balanced_match_flags(
        config.trials - config.n_back, config.match_probability, rng)
    audio_flags = _balanced_match_flags(
        config.trials - config.n_back, config.match_probability, rng)
    positions: list[int] = []
    audio: list[int] = []
    stimuli: list[BrainWorkshopStimulus] = []
    targets: list[int] = []
    observations: list[BrainWorkshopObservation] = []
    for trial in range(config.trials):
        offset = trial - config.n_back
        position_match = (
            bool(position_flags[offset]) if offset >= 0 else False)
        audio_match = bool(audio_flags[offset]) if offset >= 0 else False
        position = _next_symbol(
            positions, config.n_back, config.position_vocab, position_match, rng)
        audio_symbol = _next_symbol(
            audio, config.n_back, config.audio_vocab, audio_match, rng)
        positions.append(position)
        audio.append(audio_symbol)
        timestamp_ms = trial * config.trial_ms
        stimuli.append(BrainWorkshopStimulus(
            position, audio_symbol, timestamp_ms))
        target = (
            (POSITION_MATCH if position_match else 0)
            | (AUDIO_MATCH if audio_match else 0))
        targets.append(target)
        observations.append(BrainWorkshopObservation(
            vision=(
                render_position(position, size=config.vision_size, device=device)
                if "vision" in config.modalities else None),
            audio=(
                render_audio(
                    audio_symbol, samples=config.audio_samples,
                    sample_rate=config.sample_rate, device=device)
                if "audio" in config.modalities else None),
            timestamp_ms=timestamp_ms,
        ))
    return BrainWorkshopEpisode(
        config=config, seed=seed, stimuli=tuple(stimuli),
        observations=tuple(observations), _targets=tuple(targets))


class BrainWorkshopGym:
    """Minimal Gym-like loop with verifier-owned targets and scalar rewards."""

    def __init__(self, config: BrainWorkshopConfig | None = None,
                 *, seed: int = 0, device: torch.device | str = "cpu") -> None:
        if config is None:
            config = BrainWorkshopConfig()
        self.config = config.validate()
        self.seed = seed
        self.device = device
        self.episode: BrainWorkshopEpisode | None = None
        self.cursor = 0

    def reset(self, *, seed: int | None = None) -> BrainWorkshopObservation:
        if seed is not None:
            self.seed = seed
        self.episode = generate_brainworkshop_episode(
            self.config, seed=self.seed, device=self.device)
        self.cursor = 0
        return self.episode.observations[0]

    def step(
            self, action: int, *, latency_ms: float
            ) -> tuple[BrainWorkshopObservation | None, float, bool, dict[str, float]]:
        if self.episode is None:
            raise RuntimeError("reset must be called before step")
        reward = self.episode.score_action(self.cursor, action, latency_ms)
        self.cursor += 1
        done = self.cursor >= self.episode.trials
        observation = None if done else self.episode.observations[self.cursor]
        # Only scalar outcome and measured latency are returned to the learner.
        return observation, reward, done, {"latency_ms": float(latency_ms)}


class BrainWorkshopVisionEncoder(nn.Module):
    """Replaceable visual frontend for the gym's 3x3 position stream."""

    def __init__(self, event_width: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, event_width),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.network(frame)


class BrainWorkshopAudioEncoder(nn.Module):
    """Replaceable audio frontend for the gym's waveform stream."""

    def __init__(self, event_width: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=4),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(32 * 8, event_width),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.network(waveform)


class BrainWorkshopEventEncoders(nn.Module):
    """Encode both streams into independent amodal events for one bus update."""

    def __init__(self, event_width: int = 96) -> None:
        super().__init__()
        self.vision = BrainWorkshopVisionEncoder(event_width)
        self.audio = BrainWorkshopAudioEncoder(event_width)

    def encode(
            self, observation: BrainWorkshopObservation
            ) -> AmodalEventCollection:
        events: list[AmodalEvent] = []
        timestamp = torch.tensor([float(observation.timestamp_ms)])
        if observation.vision is not None:
            events.append(AmodalEvent(
                payload=self.vision(observation.vision.unsqueeze(0)),
                timestamp=timestamp,
            ))
        if observation.audio is not None:
            events.append(AmodalEvent(
                payload=self.audio(observation.audio.unsqueeze(0)),
                timestamp=timestamp,
            ))
        return AmodalEventCollection.from_events(events)


class BrainWorkshopKeypressDecoder(nn.Module):
    """Protocol adapter from one intention to four two-bit keypress actions."""

    def __init__(self, intention_width: int = 24) -> None:
        super().__init__()
        self.projection = nn.Linear(intention_width, 4)

    def forward(self, intention: torch.Tensor) -> torch.Tensor:
        if intention.ndim != 2:
            raise ValueError("intention must have shape [batch, width]")
        return self.projection(intention)

    @staticmethod
    def to_keypress_codes(action: int) -> tuple[int, ...]:
        if action < 0 or action > ALL_MATCHES:
            raise ValueError("action must be a two-bit keypress mask")
        return tuple(code for bit, code in ((POSITION_MATCH, 1), (AUDIO_MATCH, 2))
                     if action & bit)
