"""Clean-room rendered multimodal n-back environment.

The verifier owns symbol sequences and target comparisons. Public observations
contain only RGB frames and/or audio waveforms. This module does not import or
vendor Brain Workshop code, assets, state, or protocol implementation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from neural_computer import AmodalEvent, AmodalEventCollection

RENDERED_BRAINWORKSHOP_SCHEMA = "neural-computer.rendered-brainworkshop.v1"
RENDERED_ENCODER_SCHEMA = "neural-computer.rendered-brainworkshop-encoders.v1"
RENDERED_FRONTEND_ARTIFACT_SCHEMA = "neural-computer.rendered-frontend.v1"
SUPPORTED_RENDERED_STREAMS = ("vision", "audio")

_GRID_POSITIONS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)


@dataclass(frozen=True)
class RenderedBrainWorkshopConfig:
    n_back: int = 1
    steps: int = 24
    streams: tuple[str, ...] = ("vision", "audio")
    symbol_count: int = 8
    frame_size: int = 36
    audio_samples: int = 256
    sample_rate: int = 8_000
    neutral_true_negative_absent: bool = False
    match_rule: str = "n_back"
    target_symbol: int = 0

    def validate(self) -> RenderedBrainWorkshopConfig:
        if self.match_rule not in {"n_back", "current_symbol", "changed", "onset"}:
            raise ValueError(
                "rendered match_rule must be n_back, current_symbol, changed, or onset"
            )
        if self.match_rule == "current_symbol":
            if self.steps < 8:
                raise ValueError("current-symbol task needs at least eight steps")
            if not 0 <= self.target_symbol < self.symbol_count:
                raise ValueError("current-symbol target is outside the symbol set")
        elif self.match_rule == "changed":
            if self.steps < 8:
                raise ValueError("changed-symbol task needs at least eight steps")
        elif self.match_rule == "onset":
            if self.steps < 8:
                raise ValueError("onset task needs at least eight steps")
            if not 0 <= self.target_symbol < self.symbol_count:
                raise ValueError("onset target is outside the symbol set")
        elif self.n_back < 1 or self.steps <= self.n_back:
            raise ValueError("rendered n-back needs target-bearing positive dimensions")
        if not self.streams or len(set(self.streams)) != len(self.streams):
            raise ValueError("rendered streams must be non-empty and unique")
        if any(stream not in SUPPORTED_RENDERED_STREAMS for stream in self.streams):
            raise ValueError("unsupported rendered Neural Workshop stream")
        if not 2 <= self.symbol_count <= len(_GRID_POSITIONS):
            raise ValueError("rendered symbol count must lie between two and eight")
        if self.frame_size < 18 or self.frame_size % 3:
            raise ValueError("frame size must be divisible by three and at least 18")
        if self.audio_samples < 32 or self.sample_rate < 1_000:
            raise ValueError("rendered audio dimensions are too small")
        return self

    @property
    def action_count(self) -> int:
        return 1 << len(self.streams)


@dataclass(frozen=True)
class RenderedBrainWorkshopObservation:
    """Raw public device streams for one live trial."""

    vision: torch.Tensor | None
    audio: torch.Tensor | None


@dataclass(frozen=True)
class RenderedBrainWorkshopStep:
    reward: torch.Tensor
    eligible: torch.Tensor


def _balanced_flags(count: int, generator: torch.Generator) -> torch.Tensor:
    flags = (torch.arange(count) % 2).to(torch.bool)
    return flags[torch.randperm(count, generator=generator)]


def _generate_symbols(
    config: RenderedBrainWorkshopConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    symbols = torch.empty(config.steps, dtype=torch.long)
    symbols[: config.n_back] = torch.randint(
        config.symbol_count,
        (config.n_back,),
        generator=generator,
    )
    flags = _balanced_flags(config.steps - config.n_back, generator)
    for position in range(config.n_back, config.steps):
        reference = symbols[position - config.n_back]
        different = (
            reference
            + torch.randint(
                1,
                config.symbol_count,
                (),
                generator=generator,
            )
        ) % config.symbol_count
        symbols[position] = torch.where(
            flags[position - config.n_back], reference, different
        )
    return symbols, flags


def _generate_current_symbols(
    config: RenderedBrainWorkshopConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    flags = _balanced_flags(config.steps, generator)
    symbols = torch.randint(
        config.symbol_count, (config.steps,), generator=generator
    )
    target = int(config.target_symbol)
    for position in range(config.steps):
        if bool(flags[position]):
            symbols[position] = target
        elif int(symbols[position].item()) == target:
            symbols[position] = (target + 1) % config.symbol_count
    return symbols, flags


def _generate_onset_symbols(
    config: RenderedBrainWorkshopConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = int(config.target_symbol)
    off = (target + 1) % config.symbol_count
    stay = _balanced_flags(config.steps - 1, generator)
    symbols = torch.empty(config.steps, dtype=torch.long)
    symbols[0] = torch.randint(2, (), generator=generator)
    symbols[0] = target if int(symbols[0].item()) == 0 else off
    for position in range(1, config.steps):
        symbols[position] = (
            symbols[position - 1]
            if bool(stay[position - 1])
            else (off if int(symbols[position - 1].item()) == target else target)
        )
    onset = torch.zeros(config.steps, dtype=torch.bool)
    for position in range(1, config.steps):
        onset[position] = bool(symbols[position] == target) and bool(
            symbols[position - 1] != target
        )
    return symbols, onset


def render_position(symbol: int, *, size: int) -> torch.Tensor:
    """Render one position as a plain RGB frame with no target annotation."""

    if not 0 <= symbol < len(_GRID_POSITIONS):
        raise ValueError("position symbol is outside the rendered grid")
    frame = torch.zeros(3, size, size)
    cell = size // 3
    frame[:, cell - 1 : cell + 1, :] = 0.12
    frame[:, 2 * cell - 1 : 2 * cell + 1, :] = 0.12
    frame[:, :, cell - 1 : cell + 1] = 0.12
    frame[:, :, 2 * cell - 1 : 2 * cell + 1] = 0.12
    row, column = _GRID_POSITIONS[symbol]
    margin = max(2, cell // 5)
    top = row * cell + margin
    left = column * cell + margin
    bottom = (row + 1) * cell - margin
    right = (column + 1) * cell - margin
    color = torch.tensor((0.25, 0.70, 1.0)).view(3, 1, 1)
    frame[:, top:bottom, left:right] = color
    return frame


def render_audio(symbol: int, *, samples: int, sample_rate: int) -> torch.Tensor:
    """Render one symbol as a normalized waveform without label metadata."""

    if symbol < 0:
        raise ValueError("audio symbol must be non-negative")
    time = torch.arange(samples, dtype=torch.float32) / float(sample_rate)
    fundamental = 240.0 + 75.0 * symbol
    waveform = torch.sin(2.0 * math.pi * fundamental * time)
    waveform += 0.25 * torch.sin(2.0 * math.pi * fundamental * 2.0 * time + 0.3)
    envelope = torch.hann_window(samples, periodic=False)
    return (waveform * envelope).unsqueeze(0)


class RenderedBrainWorkshopVerifier:
    """Private multimodal n-back generator with one exact scalar outcome."""

    batch_size = 1

    def __init__(
        self,
        config: RenderedBrainWorkshopConfig,
        *,
        seed: int = 0,
    ) -> None:
        self.config = config.validate()
        self.action_count = self.config.action_count
        generator = torch.Generator().manual_seed(seed)
        self._symbols: dict[str, torch.Tensor] = {}
        self._matches: dict[str, torch.Tensor] = {}
        for stream in self.config.streams:
            if self.config.match_rule == "current_symbol":
                symbols, matches = _generate_current_symbols(self.config, generator)
            elif self.config.match_rule == "onset":
                symbols, matches = _generate_onset_symbols(self.config, generator)
            else:
                symbols, matches = _generate_symbols(self.config, generator)
            self._symbols[stream] = symbols
            self._matches[stream] = matches
        self._position = 0

    @property
    def done(self) -> bool:
        return self._position >= self.config.steps

    @property
    def position(self) -> int:
        return self._position

    @property
    def eligible_trials(self) -> int:
        if self.config.match_rule == "current_symbol":
            return self.config.steps
        if self.config.match_rule == "changed":
            return self.config.steps - 1
        if self.config.match_rule == "onset":
            return self.config.steps - 1
        return self.config.steps - self.config.n_back

    def observation(self) -> RenderedBrainWorkshopObservation:
        if self.done:
            raise RuntimeError("rendered verifier has no remaining observation")
        vision = (
            render_position(
                int(self._symbols["vision"][self._position]),
                size=self.config.frame_size,
            )
            if "vision" in self.config.streams
            else None
        )
        audio = (
            render_audio(
                int(self._symbols["audio"][self._position]),
                samples=self.config.audio_samples,
                sample_rate=self.config.sample_rate,
            )
            if "audio" in self.config.streams
            else None
        )
        return RenderedBrainWorkshopObservation(vision=vision, audio=audio)

    def score(self, action: torch.Tensor) -> RenderedBrainWorkshopStep:
        if self.done:
            raise RuntimeError("rendered verifier episode is complete")
        if action.shape != (1,) or action.dtype != torch.long:
            raise ValueError("rendered keypress action must be int64 with shape [1]")
        if bool(torch.any((action < 0) | (action >= self.action_count))):
            raise ValueError("rendered keypress action is outside the protocol")
        if self.config.match_rule == "current_symbol":
            eligible = torch.tensor([True])
            symbol = int(self._symbols[self.config.streams[0]][self._position])
            expected = int(symbol == int(self.config.target_symbol))
        elif self.config.match_rule == "changed":
            eligible = torch.tensor([self._position >= 1])
            expected = 0
            if bool(eligible.item()):
                stream = self.config.streams[0]
                expected = int(
                    int(self._symbols[stream][self._position])
                    != int(self._symbols[stream][self._position - 1])
                )
        elif self.config.match_rule == "onset":
            eligible = torch.tensor([self._position >= 1])
            expected = 0
            if bool(eligible.item()):
                stream = self.config.streams[0]
                current = int(self._symbols[stream][self._position])
                previous = int(self._symbols[stream][self._position - 1])
                expected = int(
                    current == int(self.config.target_symbol) and current != previous
                )
        else:
            eligible = torch.tensor([self._position >= self.config.n_back])
            expected = 0
            if bool(eligible.item()):
                offset = self._position - self.config.n_back
                for bit, stream in enumerate(self.config.streams):
                    if bool(self._matches[stream][offset]):
                        expected |= 1 << bit
        chosen = int(action.item())
        reward = torch.tensor([float(chosen == expected)])
        eligible = eligible & ~torch.tensor(
            [
                self.config.neutral_true_negative_absent
                and expected == 0
                and chosen == 0
            ]
        )
        reward = torch.where(eligible, reward, torch.zeros_like(reward))
        self._position += 1
        return RenderedBrainWorkshopStep(reward=reward, eligible=eligible)


class RenderedVisionEncoder(nn.Module):
    """Replaceable RGB frontend emitting one normalized learned event."""

    def __init__(self, event_width: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((12, 12))
        self.projection = nn.Linear(3 * 12 * 12, event_width, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.normalizer = nn.LayerNorm(event_width)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError("rendered RGB input must have shape [batch, 3, H, W]")
        pooled = self.pool(frame).flatten(1)
        return self.normalizer(self.projection(pooled))


class RenderedAudioEncoder(nn.Module):
    """Replaceable waveform frontend emitting one normalized learned event."""

    def __init__(self, event_width: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(256)
        self.projection = nn.Linear(256, event_width, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.normalizer = nn.LayerNorm(event_width)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("rendered audio input must have shape [batch, 1, samples]")
        pooled = self.pool(waveform).flatten(1)
        return self.normalizer(self.projection(pooled))


class RenderedBrainWorkshopEncoders(nn.Module):
    """Independent modality encoders with opaque learned source identities."""

    schema = RENDERED_ENCODER_SCHEMA

    def __init__(self, event_width: int, *, source_key_width: int = 4) -> None:
        super().__init__()
        if min(event_width, source_key_width) < 1:
            raise ValueError("rendered encoder dimensions must be positive")
        self.event_width = int(event_width)
        self.source_key_width = int(source_key_width)
        self.vision = RenderedVisionEncoder(event_width)
        self.audio = RenderedAudioEncoder(event_width)
        self.source_keys = nn.ParameterDict(
            {
                stream: nn.Parameter(torch.randn(source_key_width))
                for stream in SUPPORTED_RENDERED_STREAMS
            }
        )

    def digest(self) -> str:
        """Stable digest of the replaceable frontend, not of a task label."""

        hasher = hashlib.sha256()
        hasher.update(self.schema.encode())
        hasher.update(str(self.event_width).encode())
        hasher.update(str(self.source_key_width).encode())
        for name, parameter in sorted(self.named_parameters()):
            tensor = parameter.detach().cpu().contiguous()
            hasher.update(name.encode())
            hasher.update(str(tensor.dtype).encode())
            hasher.update(repr(tuple(tensor.shape)).encode())
            hasher.update(tensor.numpy().tobytes())
        return hasher.hexdigest()

    @classmethod
    def seeded(
        cls,
        event_width: int,
        *,
        source_key_width: int = 4,
        seed: int,
    ) -> RenderedBrainWorkshopEncoders:
        """Build one reproducible frozen frontend. This is an adapter, not a task."""

        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            encoders = cls(event_width, source_key_width=source_key_width)
        for parameter in encoders.parameters():
            parameter.requires_grad_(False)
        return encoders

    def payload(self, *, seed: int | None = None) -> dict[str, object]:
        state = {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}
        configuration = {
            "event_width": self.event_width,
            "source_key_width": self.source_key_width,
        }
        if seed is not None:
            configuration["seed"] = int(seed)
        return {
            "schema": RENDERED_FRONTEND_ARTIFACT_SCHEMA,
            "configuration": configuration,
            "state": state,
            "digest": self.digest(),
        }

    def save(self, path: Path, *, seed: int | None = None) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.payload(seed=seed), path)
        return self.digest()

    @classmethod
    def load(cls, path: Path) -> RenderedBrainWorkshopEncoders:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != RENDERED_FRONTEND_ARTIFACT_SCHEMA
            or not isinstance(payload.get("configuration"), dict)
            or not isinstance(payload.get("state"), dict)
            or not isinstance(payload.get("digest"), str)
        ):
            raise ValueError("rendered frontend artifact is malformed")
        configuration = payload["configuration"]
        encoders = cls(
            int(configuration["event_width"]),
            source_key_width=int(configuration["source_key_width"]),
        )
        encoders.load_state_dict(payload["state"])
        for parameter in encoders.parameters():
            parameter.requires_grad_(False)
        if encoders.digest() != payload["digest"]:
            raise ValueError("rendered frontend artifact digest mismatch")
        return encoders

    def encode(
        self,
        observation: RenderedBrainWorkshopObservation,
        *,
        now: float,
        reverse_order: bool = False,
    ) -> AmodalEventCollection:
        events: list[AmodalEvent] = []
        inputs = (
            ("vision", observation.vision, self.vision),
            ("audio", observation.audio, self.audio),
        )
        if reverse_order:
            inputs = tuple(reversed(inputs))
        for stream, raw, encoder in inputs:
            if raw is None:
                continue
            payload = encoder(raw.unsqueeze(0))
            events.append(
                AmodalEvent(
                    payload=payload,
                    source_key=self.source_keys[stream].unsqueeze(0),
                    timestamp=torch.tensor([now]),
                    confidence=torch.ones(1),
                )
            )
        return AmodalEventCollection.from_events(events, width=self.event_width)
