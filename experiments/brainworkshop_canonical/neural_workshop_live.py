"""Live pixel/action adapter for the external Neural Workshop environment.

The adapter is deliberately the only component that knows the environment's
transport protocol.  The cognitive machine receives one learned event tensor
per public stimulus frame, emits one of two opaque actions, and later receives
an exact-once scalar outcome bound to that action's receipt.  Grid settings,
phase scheduling, framebuffer bytes, and verifier evidence never enter the
controller.
"""

from __future__ import annotations

import importlib
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from neural_computer import (
    AmodalEvent,
    AmodalEventCollection,
    CognitiveTickRuntime,
    LiveActionReceipt,
    LiveInputBatch,
    LiveOutcomeEvent,
)

from .rendered_live import SourcePreservingTemporalMachine

NEURAL_WORKSHOP_LIVE_SCHEMA = "neural-computer.neural-workshop-live.v1"
_ACTION_INTERVENTIONS = {"normal", "passive", "random", "reversed"}
_REWARD_INTERVENTIONS = {"normal", "missing", "shuffled"}


@dataclass(frozen=True)
class NeuralWorkshopIntervention:
    """Discarded causal intervention; none of these fields reach the learner."""

    action: str = "normal"
    reward: str = "normal"
    reset_history_each_tick: bool = False
    seed: int = 0

    def validate(self) -> NeuralWorkshopIntervention:
        if self.action not in _ACTION_INTERVENTIONS:
            raise ValueError(f"unsupported action intervention: {self.action}")
        if self.reward not in _REWARD_INTERVENTIONS:
            raise ValueError(f"unsupported reward intervention: {self.reward}")
        return self


class NeuralWorkshopEnvironment(Protocol):
    """Small public transport surface used by the live adapter."""

    n_actions: int

    def observe(self) -> dict[str, Any]: ...

    def act(self, ports: object = None, logp: float | None = None) -> dict[str, Any]: ...

    def advance(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _validate_crop(crop: tuple[float, float, float, float], *, name: str) -> None:
    left, top, right, bottom = crop
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise ValueError(f"{name} must be ordered within [0, 1]")


@dataclass(frozen=True)
class NeuralWorkshopLiveConfig:
    grid_size: int = 3
    active_cells: int = 2
    n_back: int = 1
    trials: int = 60
    event_width: int = 16
    source_key_width: int = 4
    image_size: int = 36
    # Public framebuffer crop calibrated to the visible play field. It affects
    # only the replaceable visual adapter, never controller dimensions.
    crop: tuple[float, float, float, float] = (0.26, 0.17, 0.74, 0.82)
    # Separate public-header crop, calibrated so the visible N-Back numeral
    # changes the event. It never shares a source key with the play-field
    # encoder.
    instruction_crop: tuple[float, float, float, float] = (0.30, 0.00, 0.70, 0.05)
    instruction_image_size: int = 96
    instruction_pool_size: int = 32
    instruction_encoder_seed: int = 2_041
    schema: str = NEURAL_WORKSHOP_LIVE_SCHEMA

    def validate(self) -> NeuralWorkshopLiveConfig:
        if self.schema != NEURAL_WORKSHOP_LIVE_SCHEMA:
            raise ValueError(f"unsupported Neural Workshop schema: {self.schema}")
        if min(
            self.grid_size,
            self.active_cells,
            self.n_back,
            self.trials,
            self.event_width,
            self.source_key_width,
            self.image_size,
            self.instruction_image_size,
            self.instruction_pool_size,
        ) < 1:
            raise ValueError("Neural Workshop dimensions must be positive")
        if self.active_cells > self.grid_size * self.grid_size:
            raise ValueError("active cells cannot exceed the visible grid")
        _validate_crop(self.crop, name="frame crop")
        _validate_crop(self.instruction_crop, name="instruction crop")
        play = self.crop
        header = self.instruction_crop
        if not (header[3] <= play[1] or play[3] <= header[1]):
            raise ValueError("instruction crop must stay outside the play-field crop")
        return self


@dataclass(frozen=True)
class AuthenticatedNeuralWorkshopOutcome:
    runtime_receipt_id: int
    environment_receipt_id: int
    signed_scalar: float
    verifier_reward: float
    learner_reward: float
    evidence_digests: tuple[str, ...]
    frame_seq: int
    timestamp_ns: int


@dataclass(frozen=True)
class NeuralWorkshopLiveReport:
    grid_size: int
    active_cells: int
    n_back: int
    requested_trials: int
    logical_trials: int
    input_events: int
    emitted_actions: int
    unique_verifier_bits: int
    learner_outcome_bits: int
    positive_verifier_bits: int
    optimizer_updates: int
    program_file_updates: int
    replayed_examples: int
    controller_frozen: bool
    controller_digest_before: str | None
    controller_digest_after: str | None
    rewards: tuple[float, ...]
    verifier_rewards: tuple[float, ...]
    signed_scalars: tuple[float, ...]
    actions: tuple[int, ...]
    executed_actions: tuple[int, ...]
    propensities: tuple[float, ...]
    evidence_digests: tuple[tuple[str, ...], ...]
    event_payloads: tuple[tuple[float, ...], ...]
    instruction_payloads: tuple[tuple[float, ...], ...]
    ticks: int
    deadline_misses: int
    wall_seconds: float
    tick_seconds_p50: float | None
    tick_seconds_p99: float | None
    intervention: dict[str, object]
    schema: str = NEURAL_WORKSHOP_LIVE_SCHEMA

    @property
    def verifier_accuracy(self) -> float | None:
        return (
            None
            if not self.rewards
            else sum(self.verifier_rewards) / len(self.verifier_rewards)
        )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["verifier_accuracy"] = self.verifier_accuracy
        return payload


class NeuralWorkshopRGBAEncoder(nn.Module):
    """Frozen generic public-pixel projection into one amodal event.

    It assigns no cell IDs or task meanings.  A controller trained across
    independently projected renderers can consume this new visual frontend
    without resizing or receiving the raw RGBA format.
    """

    def __init__(
        self,
        config: NeuralWorkshopLiveConfig,
        *,
        seed: int,
        crop: tuple[float, float, float, float] | None = None,
        image_size: int | None = None,
        pool_size: int = 12,
    ) -> None:
        super().__init__()
        config.validate()
        if pool_size < 1:
            raise ValueError("pixel pool size must be positive")
        self.event_width = config.event_width
        self.source_key_width = config.source_key_width
        self.image_size = config.image_size if image_size is None else int(image_size)
        if self.image_size < 1:
            raise ValueError("encoder image size must be positive")
        self.crop = config.crop if crop is None else crop
        _validate_crop(self.crop, name="encoder crop")
        self.pool_size = int(pool_size)
        input_width = 3 * self.pool_size * self.pool_size
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.projection = nn.Linear(input_width, self.event_width, bias=False)
        self.normalization = nn.LayerNorm(self.event_width)
        # Match the independently randomized visual-encoder family used for
        # controller pretraining. The crop contents are still real Neural
        # Workshop pixels; only the replaceable frontend parameterization is
        # shared.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            nn.init.orthogonal_(self.projection.weight)
            self.source_key = nn.Parameter(torch.randn(self.source_key_width))
        self.requires_grad_(False)
        self.emitted_payloads: list[torch.Tensor] = []

    @staticmethod
    def _validate_observation(observation: dict[str, Any]) -> None:
        required = {"frame_seq", "timestamp_ns", "width", "height", "rgba", "done"}
        if not required.issubset(observation):
            raise ValueError("Neural Workshop observation is incomplete")
        width = observation["width"]
        height = observation["height"]
        rgba = observation["rgba"]
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or min(width, height) < 1
            or not isinstance(rgba, bytes)
            or len(rgba) != width * height * 4
        ):
            raise ValueError("Neural Workshop framebuffer is malformed")

    def encode(self, observation: dict[str, Any]) -> AmodalEventCollection:
        self._validate_observation(observation)
        width = int(observation["width"])
        height = int(observation["height"])
        image = Image.frombytes("RGBA", (width, height), observation["rgba"])
        left, top, right, bottom = self.crop
        box = (
            int(left * width),
            int(top * height),
            max(int(left * width) + 1, math.ceil(right * width)),
            max(int(top * height) + 1, math.ceil(bottom * height)),
        )
        resized = image.crop(box).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        pixels = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(self.image_size, self.image_size, 3)
        pixels = pixels.permute(2, 0, 1).to(torch.float32).unsqueeze(0) / 255.0
        # Remove the dominant screen background without assigning coordinates
        # or colors semantic meaning. This makes light and dark renderers share
        # the sparse-foreground geometry used during controller pretraining.
        background = pixels.flatten(2).median(dim=2).values[:, :, None, None]
        pixels = (pixels - background).abs()
        payload = self.normalization(self.projection(self.pool(pixels).flatten(1)))
        self.emitted_payloads.append(payload[0].detach().cpu().clone())
        timestamp = float(observation["timestamp_ns"]) / 1_000_000_000.0
        event = AmodalEvent(
            payload=payload,
            source_key=self.source_key.detach().reshape(1, -1),
            timestamp=torch.tensor([timestamp], dtype=payload.dtype),
            confidence=torch.ones(1, dtype=payload.dtype),
        )
        return AmodalEventCollection.from_events((event,), width=self.event_width)


class NeuralWorkshopInstructionEncoder(NeuralWorkshopRGBAEncoder):
    """Frozen public-header projection used only as bank-routing context.

    The visible mode line is ordinary rendered pixels. This frontend never
    receives ``n_back``, a task ID, or the correct action, and its events are
    not mixed into the play-field temporal history.
    """

    def __init__(
        self,
        config: NeuralWorkshopLiveConfig,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            config,
            seed=config.instruction_encoder_seed if seed is None else seed,
            crop=config.instruction_crop,
            image_size=config.instruction_image_size,
            pool_size=config.instruction_pool_size,
        )


def _event_at(collection: AmodalEventCollection, index: int) -> AmodalEvent:
    if collection.source_key is None:
        raise ValueError("merged Neural Workshop events require source keys")
    timestamp = None
    if collection.timestamp is not None:
        stamp = collection.timestamp[:, index]
        timestamp = stamp.reshape(collection.payload.shape[0])
    return AmodalEvent(
        payload=collection.payload[:, index],
        source_key=collection.source_key[:, index],
        timestamp=timestamp,
        confidence=collection.confidence[:, index],
    )


def encode_instruction_context(
    observation: dict[str, Any],
    encoder: NeuralWorkshopInstructionEncoder,
) -> torch.Tensor:
    """Encode one public frame's header into a bank-routing context."""

    events = encoder.encode(observation)
    payload = events.payload[0, 0].detach().cpu().to(torch.float32)
    if payload.numel() != encoder.event_width:
        raise ValueError("instruction event width does not match the encoder")
    if float(torch.linalg.vector_norm(payload)) <= 1e-8:
        raise ValueError("instruction event contains no address evidence")
    return F.normalize(payload, dim=0)


def observe_neural_workshop_instruction(
    directory: Path,
    config: NeuralWorkshopLiveConfig,
    *,
    seed: int,
    encoder: NeuralWorkshopInstructionEncoder | None = None,
) -> torch.Tensor:
    """Reset one public session and encode only the first visible header."""

    encoder = NeuralWorkshopInstructionEncoder(config) if encoder is None else encoder
    environment, _verifier = build_neural_workshop_environment(
        directory, config, seed=seed
    )
    try:
        return encode_instruction_context(environment.observe(), encoder)
    finally:
        environment.close()


class NeuralWorkshopLiveDevice:
    """Drive significant public frames and authenticate delayed outcomes."""

    batch_size = 1

    def __init__(
        self,
        environment: NeuralWorkshopEnvironment,
        encoder: NeuralWorkshopRGBAEncoder,
        verifier: Any,
        intervention: NeuralWorkshopIntervention | None = None,
        instruction_encoder: NeuralWorkshopInstructionEncoder | None = None,
    ) -> None:
        if environment.n_actions != 1:
            raise ValueError("first live rung requires Position-only one-port mode")
        if (
            instruction_encoder is not None
            and instruction_encoder.event_width != encoder.event_width
        ):
            raise ValueError("instruction encoder width must match the play-field encoder")
        if instruction_encoder is encoder:
            raise ValueError("instruction encoder must be a separate frontend")
        self.environment = environment
        self.encoder = encoder
        self.instruction_encoder = instruction_encoder
        self.event_width = encoder.event_width
        self.verifier = verifier
        self.intervention = (
            NeuralWorkshopIntervention() if intervention is None else intervention
        ).validate()
        self._random = random.Random(self.intervention.seed)
        self._observation = environment.observe()
        self._event_pending = not bool(self._observation["done"])
        self._advance_pending = False
        self._outcomes: list[LiveOutcomeEvent] = []
        self.authenticated_outcomes: list[AuthenticatedNeuralWorkshopOutcome] = []
        self.executed_actions: list[int] = []
        self._environment_receipts: dict[int, LiveActionReceipt] = {}

    @property
    def done(self) -> bool:
        return (
            bool(self._observation["done"])
            and not self._event_pending
            and not self._advance_pending
            and not self._outcomes
        )

    def _verify(self, outcome: dict[str, Any], observation: dict[str, Any]) -> bool:
        # Archive and receipt ledger are trusted verifier internals. They are
        # consumed here and never placed on the amodal event bus.
        archive = getattr(self.environment, "_archive", None)
        ledger = getattr(self.environment, "_receipt_ledger", None)
        return bool(
            self.verifier(
                outcome,
                observation["rgba"],
                observation["width"],
                observation["height"],
                archive=archive,
                receipt_ledger=ledger,
            )
        )

    def _close_action(self, feedback: dict[str, Any]) -> None:
        if len(self._environment_receipts) != 1:
            raise RuntimeError("Neural Workshop lost its single pending action")
        environment_id, receipt = next(iter(self._environment_receipts.items()))
        raw = feedback.get("outcome")
        if raw is None:
            event = LiveOutcomeEvent(
                receipt_id=receipt.receipt_id,
                reward=torch.zeros(1),
                present=torch.zeros(1, dtype=torch.bool),
                observed_at=float(feedback["timestamp_ns"]) / 1_000_000_000.0,
                confidence=torch.ones(1),
            )
        else:
            if int(raw.get("receipt_id", -1)) != environment_id:
                raise RuntimeError("outcome references the wrong environment receipt")
            if not self._verify(raw, feedback):
                raise RuntimeError("Neural Workshop public outcome failed authentication")
            signed = float(raw["scalar"])
            if not -1.0 <= signed <= 1.0:
                raise ValueError("verified scalar lies outside [-1, 1]")
            verifier_reward = (signed + 1.0) / 2.0
            learner_reward = verifier_reward
            present = True
            if self.intervention.reward == "shuffled":
                learner_reward = float(self._random.randrange(2))
            elif self.intervention.reward == "missing":
                learner_reward = 0.0
                present = False
            evidence = tuple(str(value) for value in raw["evidence_digests"])
            event = LiveOutcomeEvent(
                receipt_id=receipt.receipt_id,
                reward=torch.tensor([learner_reward], dtype=torch.float32),
                present=torch.tensor([present], dtype=torch.bool),
                observed_at=float(raw["timestamp_ns"]) / 1_000_000_000.0,
                confidence=torch.ones(1),
            )
            self.authenticated_outcomes.append(
                AuthenticatedNeuralWorkshopOutcome(
                    runtime_receipt_id=receipt.receipt_id,
                    environment_receipt_id=environment_id,
                    signed_scalar=signed,
                    verifier_reward=verifier_reward,
                    learner_reward=learner_reward,
                    evidence_digests=evidence,
                    frame_seq=int(raw["frame_seq"]),
                    timestamp_ns=int(raw["timestamp_ns"]),
                )
            )
        self._outcomes.append(event)
        self._environment_receipts.clear()

    def poll(self, now: float) -> LiveInputBatch:
        if self._advance_pending:
            feedback = self.environment.advance()
            self._close_action(feedback)
            self._observation = (
                feedback if bool(feedback["done"]) else self.environment.advance()
            )
            self._event_pending = not bool(self._observation["done"])
            self._advance_pending = False
        outcomes = tuple(self._outcomes)
        self._outcomes.clear()
        if self._event_pending:
            play = self.encoder.encode(self._observation)
            if self.instruction_encoder is None:
                events = play
            else:
                header = self.instruction_encoder.encode(self._observation)
                events = AmodalEventCollection.from_events(
                    (_event_at(play, 0), _event_at(header, 0)),
                    width=self.event_width,
                )
            self._event_pending = False
        else:
            events = AmodalEventCollection.empty(1, self.event_width)
        return LiveInputBatch(events=events, outcomes=outcomes, observed_at=now)

    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None:
        if self._advance_pending or self._environment_receipts:
            raise RuntimeError("Neural Workshop permits one action per stimulus")
        opaque = int(action.item())
        if opaque not in (0, 1):
            raise ValueError("Position-only decoder must emit action zero or one")
        executed = opaque
        execution_probability = float(receipt.propensity.item())
        if self.intervention.action == "passive":
            executed = 0
            execution_probability = 1.0
        elif self.intervention.action == "random":
            executed = self._random.randrange(2)
            execution_probability = 0.5
        elif self.intervention.action == "reversed":
            executed = 1 - opaque
        probability = max(execution_probability, torch.finfo(torch.float32).tiny)
        environment_receipt = self.environment.act(
            [] if executed == 0 else [0], logp=math.log(probability)
        )
        if not environment_receipt.get("ok"):
            raise RuntimeError("Neural Workshop rejected an in-window action")
        environment_id = int(environment_receipt["receipt_id"])
        self.executed_actions.append(executed)
        self._environment_receipts[environment_id] = receipt
        self._advance_pending = True


def _load_module(directory: Path) -> ModuleType:
    directory = Path(directory).resolve()
    module_path = directory / "nwenv.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    existing = sys.modules.get("nwenv")
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path != module_path:
            raise RuntimeError(f"another nwenv module is already loaded: {existing_path}")
        return existing
    # Neural Workshop intentionally owns its GUI/runtime dependencies while
    # this repository owns torch. Reuse compatible pure-Python/native wheels
    # from its adjacent virtualenv without copying or installing either repo.
    dependency_paths = sorted((directory / ".venv" / "lib").glob("python*/site-packages"))
    inserted = [str(directory), *(str(path) for path in dependency_paths)]
    for value in reversed(inserted):
        sys.path.insert(0, value)
    try:
        return importlib.import_module("nwenv")
    finally:
        for value in inserted:
            sys.path.remove(value)


def build_neural_workshop_environment(
    directory: Path,
    config: NeuralWorkshopLiveConfig,
    *,
    seed: int,
) -> tuple[NeuralWorkshopEnvironment, Any]:
    """Configure a manual Position N-Back session outside learner visibility."""

    config.validate()
    module = _load_module(directory)
    environment = module.NeuralWorkshopEnv(
        seed=seed, game_mode=10, num_trials=config.trials
    )
    bw = module.bw
    bw.cfg.GRID_SIZE = config.grid_size
    bw.cfg.ACTIVE_POSITION_CELLS = config.active_cells
    bw.cfg.POSITION_CELL_COUNT = config.active_cells
    bw.cfg.MANUAL = True
    bw.mode.manual = True
    bw.mode.back = config.n_back
    bw.cfg.USE_MUSIC = False
    bw.cfg.USE_APPLAUSE = False
    # The adapter's two-transition transport assumes no blank significant
    # phase. Headless fast mode normally has this property; fail closed if an
    # upstream configuration changes it.
    plan = bw.plan_current_trial_phases()
    if int(plan["blank_ticks"]) != 0:
        environment.close()
        raise ValueError("Neural Workshop live adapter requires zero blank ticks")
    observation = environment.reset(seed)
    if bw.mode.back != config.n_back:
        environment.close()
        raise RuntimeError("Neural Workshop changed the requested n-back level")
    if observation.get("done"):
        environment.close()
        raise RuntimeError("Neural Workshop ended during reset")
    return environment, module.verify_public_outcome


def _optional_digest(machine: SourcePreservingTemporalMachine, name: str) -> str | None:
    method = getattr(machine, name, None)
    return None if method is None else str(method())


def run_neural_workshop_live_lifetime(
    machine: SourcePreservingTemporalMachine,
    config: NeuralWorkshopLiveConfig,
    *,
    seed: int,
    environment: NeuralWorkshopEnvironment,
    verifier: Any,
    learn: bool = True,
    sample: bool = True,
    tick_seconds: float = 0.001,
    max_tick_seconds: float | None = None,
    intervention: NeuralWorkshopIntervention | None = None,
) -> NeuralWorkshopLiveReport:
    """Run one non-replayed live session with a frozen controller boundary."""

    config.validate()
    if tick_seconds <= 0.0:
        raise ValueError("tick duration must be positive")
    if machine.event_width != config.event_width or machine.action_count != 2:
        raise ValueError("machine is incompatible with the Neural Workshop adapter")
    machine.reset_history()
    machine.learning_enabled = bool(learn)
    machine.sample = bool(sample)
    intervention = (
        NeuralWorkshopIntervention() if intervention is None else intervention
    ).validate()
    machine.reset_history_each_tick = intervention.reset_history_each_tick
    encoder = NeuralWorkshopRGBAEncoder(config, seed=seed)
    instruction_encoder = NeuralWorkshopInstructionEncoder(config)
    bind = getattr(machine, "bind_executable_sources", None)
    if bind is not None:
        bind((encoder.source_key.detach().reshape(1, -1),))
    device = NeuralWorkshopLiveDevice(
        environment,
        encoder,
        verifier,
        intervention,
        instruction_encoder=instruction_encoder,
    )
    runtime = CognitiveTickRuntime(
        device, machine, {"keypress": device}, max_tick_seconds=max_tick_seconds
    )
    controller_before = _optional_digest(machine, "controller_digest")
    updates_before = machine.optimizer_updates
    program_updates_before = getattr(machine, "program_file_updates", 0)
    actions: list[int] = []
    propensities: list[float] = []
    results = []
    now = 0.0
    started = time.perf_counter()
    try:
        while not device.done or runtime.pending_receipts:
            result = runtime.tick(now)
            results.append(result)
            actions.extend(int(item.action.item()) for item in result.emitted_receipts)
            propensities.extend(
                float(item.propensity.item()) for item in result.emitted_receipts
            )
            if len(results) > config.trials + 2:
                raise RuntimeError("Neural Workshop live session failed to drain")
            now += tick_seconds
    finally:
        environment.close()
    wall_seconds = time.perf_counter() - started
    controller_after = _optional_digest(machine, "controller_digest")
    assert_frozen = getattr(machine, "assert_controller_frozen", None)
    if assert_frozen is not None:
        assert_frozen()
    authenticated = tuple(device.authenticated_outcomes)
    total_seconds = sorted(item.total_seconds for item in results)

    def percentile(fraction: float) -> float | None:
        if not total_seconds:
            return None
        index = min(len(total_seconds) - 1, int(fraction * len(total_seconds)))
        return total_seconds[index]

    logical_trials = int(
        getattr(getattr(environment, "accounting", None), "snapshot", dict)().get(
            "logical_trials", len(actions)
        )
    )
    return NeuralWorkshopLiveReport(
        grid_size=config.grid_size,
        active_cells=config.active_cells,
        n_back=config.n_back,
        requested_trials=config.trials,
        logical_trials=logical_trials,
        input_events=sum(item.input_event_count for item in results),
        emitted_actions=len(actions),
        unique_verifier_bits=len(authenticated),
        learner_outcome_bits=sum(
            item.event.present.sum().item()
            for result in results
            for item in result.resolved_outcomes
        ),
        positive_verifier_bits=sum(
            item.verifier_reward >= 0.5 for item in authenticated
        ),
        optimizer_updates=machine.optimizer_updates - updates_before,
        program_file_updates=(
            getattr(machine, "program_file_updates", 0) - program_updates_before
        ),
        replayed_examples=0,
        controller_frozen=(
            controller_before is not None and controller_before == controller_after
        ),
        controller_digest_before=controller_before,
        controller_digest_after=controller_after,
        rewards=tuple(item.learner_reward for item in authenticated),
        verifier_rewards=tuple(item.verifier_reward for item in authenticated),
        signed_scalars=tuple(item.signed_scalar for item in authenticated),
        actions=tuple(actions),
        executed_actions=tuple(device.executed_actions),
        propensities=tuple(propensities),
        evidence_digests=tuple(item.evidence_digests for item in authenticated),
        event_payloads=tuple(
            tuple(float(value) for value in payload)
            for payload in encoder.emitted_payloads
        ),
        instruction_payloads=tuple(
            tuple(float(value) for value in payload)
            for payload in instruction_encoder.emitted_payloads
        ),
        ticks=len(results),
        deadline_misses=sum(int(item.deadline_missed) for item in results),
        wall_seconds=wall_seconds,
        tick_seconds_p50=percentile(0.50),
        tick_seconds_p99=percentile(0.99),
        intervention=asdict(intervention),
    )


__all__ = [
    "NEURAL_WORKSHOP_LIVE_SCHEMA",
    "AuthenticatedNeuralWorkshopOutcome",
    "NeuralWorkshopEnvironment",
    "NeuralWorkshopInstructionEncoder",
    "NeuralWorkshopIntervention",
    "NeuralWorkshopLiveConfig",
    "NeuralWorkshopLiveDevice",
    "NeuralWorkshopLiveReport",
    "NeuralWorkshopRGBAEncoder",
    "build_neural_workshop_environment",
    "encode_instruction_context",
    "observe_neural_workshop_instruction",
    "run_neural_workshop_live_lifetime",
]
