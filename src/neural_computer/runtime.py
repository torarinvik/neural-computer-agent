"""Production runtime for the amodal neural computer.

This module contains transport, buffering, encoder registration, controller
composition, and decoder fan-out.  It has no knowledge of a concrete action
vocabulary or device protocol.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from .addressing import PersistentOpaqueContextRouteEvidence
from .controller import (
    EXECUTION_STATES,
    AmodalCognitiveController,
    ControllerOutput,
    ControllerState,
)
from .entry import (
    ExternalEntryBindingConsolidationReceipt,
    ExternalEntryBindingObservationReceipt,
    ExternalEntryBindingProposal,
    ExternalEntryBindingRepertoire,
    ExternalEntryObservationReceipt,
    ExternalEntryProposal,
    ExternalEntryRepertoire,
)
from .goal_memory import (
    ExternalGoalFragmentCandidate,
    ExternalGoalFragmentMemory,
    ExternalGoalFragmentObservationReceipt,
    ExternalGoalFragmentSet,
    ExternalGoalFragmentStager,
    ExternalGoalFragmentStagingAdmissionReceipt,
)
from .intention import (
    ExternalIntentionConsolidationReceipt,
    ExternalIntentionGenerationProposal,
    ExternalIntentionObservationReceipt,
    ExternalIntentionProposal,
    ExternalIntentionRepertoire,
    ExternalOutcomeIntentionGenerator,
    ExternalOutcomeIntentionGeneratorState,
)
from .intention_memory import (
    ExternalIntentionMemoryProposal,
    ExternalOutcomeIntentionMemory,
)
from .intention_router import (
    ExternalOutcomeIntentionRouter,
    ExternalRoutedIntentionMemoryState,
    ExternalRoutedIntentionProposal,
)
from .interface import (
    EVENT_SCHEMA,
    INTENTION_SCHEMA,
    AmodalEvent,
    AmodalEventCollection,
    ControllerFeedback,
    IntentEvent,
)
from .memory import MemoryBackend
from .plasticity import (
    EXTERNAL_OUTCOME_CREDIT_SCHEMA,
    EXTERNAL_OUTCOME_PROGRAM_ROUTER_SCHEMA,
    ExternalOutcomeProgramRouter,
    ExternalOutcomeProgramRouterState,
)
from .policies import EventWaitPolicy, EventWaitStatistics
from .program import ExternalProgramArtifact
from .register import (
    ExternalCapabilityRegisterMachine,
    ExternalExecutionSnapshot,
    ExternalRegisterState,
    ExternalSequenceProgramMemory,
)
from .representation import (
    DEFAULT_CONTROLLER_STATE_SPACE_ID,
    DEFAULT_EVENT_SPACE_ID,
    DEFAULT_INTENTION_SPACE_ID,
    REPRESENTATION_SPACE_SCHEMA,
    validate_representation_space_id,
)
from .world_model import (
    ExternalModelBasedPlanner,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
    ModelBasedPlanningResult,
)

RUNTIME_FORMAT = "neural-computer.amodal-runtime.v30"
POLICY_FREE_RUNTIME_SCHEMA = "neural-computer.policy-free-amodal-runtime.v1"


class OpaqueProtocolDecoder(nn.Module):
    """Independent backend from an intention vector to protocol outputs."""

    def __init__(
        self, intention_width: int, output_width: int, hidden: int = 0
    ) -> None:
        super().__init__()
        if min(intention_width, output_width) < 1 or hidden < 0:
            raise ValueError("decoder dimensions are invalid")
        self.intention_width = intention_width
        self.output_width = output_width
        self.network = (
            nn.Sequential(
                nn.Linear(intention_width, hidden),
                nn.GELU(),
                nn.Linear(hidden, output_width),
            )
            if hidden
            else nn.Linear(intention_width, output_width)
        )

    def forward(self, intention: IntentEvent | torch.Tensor) -> torch.Tensor:
        payload = intention.payload if isinstance(intention, IntentEvent) else intention
        if payload.ndim != 2 or payload.shape[1] < self.intention_width:
            raise ValueError("intention payload is too narrow for decoder")
        return self.network(payload[:, : self.intention_width])


class ConditionedOpaqueProtocolDecoder(nn.Module):
    """Shared decoder conditioned on an opaque learned program context.

    The decoder weights are shared across capabilities; context is learned
    external state and is not a protocol label, task ID, or privileged target.
    This preserves reusable decoding while allowing distinct learned programs
    to expose different action-relevant conventions.
    """

    schema = "neural-computer.conditioned-opaque-protocol-decoder.v1"

    def __init__(
        self,
        intention_width: int,
        context_width: int,
        output_width: int,
        *,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(intention_width, context_width, output_width, hidden) < 1:
            raise ValueError("conditioned decoder dimensions are invalid")
        self.intention_width = int(intention_width)
        self.context_width = int(context_width)
        self.output_width = int(output_width)
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.Linear(self.intention_width + self.context_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_width),
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "intention_width": self.intention_width,
            "context_width": self.context_width,
            "output_width": self.output_width,
            "hidden": self.hidden,
            "context": "opaque_learned_program_state_v1",
        }

    def forward(
        self,
        intention: IntentEvent | torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        payload = intention.payload if isinstance(intention, IntentEvent) else intention
        if (
            payload.ndim != 2
            or payload.shape[1] < self.intention_width
            or context.ndim != 2
            or context.shape != (payload.shape[0], self.context_width)
        ):
            raise ValueError("conditioned decoder inputs have incompatible shapes")
        return self.network(
            torch.cat((payload[:, : self.intention_width], context), dim=-1)
        )


class AmodalInputBus(nn.Module):
    """Transport-preserving input bus.

    Binding is performed by the controller over event tokens.  The bus only
    validates and packages the runtime-variable set, so metadata cannot be
    silently lost in a pre-controller averaging step.
    """

    def __init__(
        self,
        event_width: int,
        *,
        event_space_id: str = DEFAULT_EVENT_SPACE_ID,
    ) -> None:
        super().__init__()
        if event_width < 1:
            raise ValueError("event_width must be positive")
        self.event_width = event_width
        self.event_space_id = validate_representation_space_id(
            event_space_id, name="event_space_id"
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": "neural-computer.input-bus.v1",
            "event_width": self.event_width,
        }

    def forward(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent],
    ) -> AmodalEventCollection:
        collection = (
            events
            if isinstance(events, AmodalEventCollection)
            else AmodalEventCollection.from_events(events, width=self.event_width)
        )
        return collection.validate(width=self.event_width)


class AmodalOutputBus(nn.Module):
    """Fan one opaque intention out to any runtime-variable decoder set."""

    def __init__(
        self,
        decoders: Mapping[str, nn.Module] | None = None,
        *,
        intention_space_id: str = DEFAULT_INTENTION_SPACE_ID,
    ) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict(dict(decoders or {}))
        self.intention_space_id = validate_representation_space_id(
            intention_space_id, name="intention_space_id"
        )

    def register_decoder(self, name: str, decoder: nn.Module) -> None:
        if not name or name in self.decoders:
            raise ValueError("decoder name must be nonempty and unique")
        self.decoders[name] = decoder

    def forward(self, intention: IntentEvent) -> dict[str, torch.Tensor]:
        intention.validate()
        return {name: decoder(intention) for name, decoder in self.decoders.items()}


@dataclass(frozen=True)
class AmodalRuntimeOutput:
    controller: ControllerOutput
    intention: IntentEvent
    decoded: dict[str, torch.Tensor]
    execution_logits: torch.Tensor


@dataclass(frozen=True)
class RuntimeMigrationExample:
    """One paired source/target trajectory for runtime replacement checks."""

    source_events: AmodalEventCollection | Sequence[AmodalEvent]
    target_events: AmodalEventCollection | Sequence[AmodalEvent]
    source_state: ControllerState
    target_state: ControllerState
    feedback: ControllerFeedback
    elapsed: torch.Tensor | float = 1.0


@dataclass(frozen=True)
class RuntimeMigrationReceipt:
    """Verifier-gated, copy-on-write frontend/controller migration result."""

    accepted: bool
    source_event_space_id: str
    target_event_space_id: str
    source_state_space_id: str
    target_state_space_id: str
    source_intention_space_id: str
    target_intention_space_id: str
    example_count: int
    max_intention_difference: float
    max_execution_difference: float
    max_continuation_difference: float
    source_digest: str
    target_digest: str
    reason: str
    schema: str = "neural-computer.runtime-migration.v1"

    def validate(self) -> RuntimeMigrationReceipt:
        if self.schema != "neural-computer.runtime-migration.v1":
            raise ValueError("unsupported runtime migration schema")
        if self.example_count < 1:
            raise ValueError("runtime migration needs at least one example")
        for name, value in (
            ("max_intention_difference", self.max_intention_difference),
            ("max_execution_difference", self.max_execution_difference),
            ("max_continuation_difference", self.max_continuation_difference),
        ):
            if not torch.isfinite(torch.tensor(value)) or value < 0.0:
                raise ValueError(f"runtime migration {name} is invalid")
        for name, value in (
            ("source_event_space_id", self.source_event_space_id),
            ("target_event_space_id", self.target_event_space_id),
            ("source_state_space_id", self.source_state_space_id),
            ("target_state_space_id", self.target_state_space_id),
            ("source_intention_space_id", self.source_intention_space_id),
            ("target_intention_space_id", self.target_intention_space_id),
            ("source_digest", self.source_digest),
            ("target_digest", self.target_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"runtime migration {name} is missing")
        return self


@dataclass(frozen=True)
class AmodalExecutionResult:
    """One bounded execution decision from the single controller.

    ``output`` is the latest controller output.  A ``wait`` result is
    intentionally non-committing: the caller may append later events and
    continue from ``state``.  A ``think`` decision consumes an internal quiet
    controller tick and is therefore bounded by ``think_budget``.  If the
    policy still requests thought at the bound, the runtime returns a forced
    ``commit`` so an agent cannot deliberate indefinitely.
    """

    output: AmodalRuntimeOutput
    state: ControllerState
    initial_decision: str
    decision: str
    think_ticks: int
    forced_commit: bool
    trace: tuple[AmodalRuntimeOutput, ...]


@dataclass(frozen=True)
class AmodalEventWindow:
    timestamp: float
    collection: AmodalEventCollection
    complete: bool = True


@dataclass(frozen=True)
class AmodalEventWindowStatus:
    timestamp: float
    age: float
    present: tuple[bool, ...]
    complete: bool


class AmodalEventTimeline:
    """Group timestamped events without assuming arrival order."""

    def __init__(self, events: Sequence[AmodalEvent], tolerance: float = 0.0) -> None:
        if not events or tolerance < 0:
            raise ValueError("timeline requires events and nonnegative tolerance")
        validated = [event.validate() for event in events]
        if any(event.timestamp is None for event in validated):
            raise ValueError("timeline events require timestamps")
        self.events = validated
        self.tolerance = float(tolerance)
        self.timestamps = [self._timestamp(event) for event in validated]

    @staticmethod
    def _timestamp(event: AmodalEvent) -> float:
        value = event.timestamp.reshape(-1)
        if value.numel() == 0 or not torch.equal(value, value[:1].expand_as(value)):
            raise ValueError("batched timestamps must be uniform")
        return float(value[0])

    def windows(self) -> list[AmodalEventCollection]:
        pending = sorted(range(len(self.events)), key=self.timestamps.__getitem__)
        windows: list[AmodalEventCollection] = []
        while pending:
            anchor = self.timestamps[pending[0]]
            selected = [
                index
                for index in pending
                if abs(self.timestamps[index] - anchor) <= self.tolerance
            ]
            selected_set = set(selected)
            pending = [index for index in pending if index not in selected_set]
            windows.append(
                AmodalEventCollection.from_events(
                    [self.events[index] for index in selected]
                )
            )
        return windows


class AmodalEventWindowBuffer:
    """Bounded timestamp buffer for asynchronous event streams."""

    def __init__(
        self,
        stream_names: Sequence[str],
        *,
        tolerance: float = 0.0,
        max_wait: float | None = None,
        wait_policy: EventWaitPolicy | EventWaitStatistics | None = None,
    ) -> None:
        names = tuple(stream_names)
        if not names or any(not name for name in names):
            raise ValueError("stream_names must be nonempty")
        if len(set(names)) != len(names):
            raise ValueError("stream_names must be unique")
        if tolerance < 0 or (max_wait is not None and max_wait < 0):
            raise ValueError("tolerance and max_wait must be nonnegative")
        self.stream_names = names
        self.tolerance = float(tolerance)
        self.max_wait = None if max_wait is None else float(max_wait)
        self.wait_policy = wait_policy
        self._pending: dict[float, dict[str, AmodalEvent]] = {}
        self._arrival_count = 0

    @staticmethod
    def _timestamp(event: AmodalEvent) -> float:
        event.validate()
        if event.timestamp is None:
            raise ValueError("buffered events require timestamps")
        values = event.timestamp.reshape(-1)
        if values.numel() == 0 or not torch.equal(values, values[:1].expand_as(values)):
            raise ValueError("batched buffered timestamps must be uniform")
        return float(values[0])

    def _slot(self, timestamp: float) -> float:
        for existing in self._pending:
            if abs(existing - timestamp) <= self.tolerance:
                return existing
        return timestamp

    def push(self, arrivals: Mapping[str, AmodalEvent]) -> list[AmodalEventWindow]:
        if not arrivals:
            raise ValueError("at least one event arrival is required")
        self._arrival_count += len(arrivals)
        for name, event in arrivals.items():
            if name not in self.stream_names:
                raise KeyError(f"unknown buffered stream {name!r}")
            timestamp = self._slot(self._timestamp(event))
            bucket = self._pending.setdefault(timestamp, {})
            if name in bucket:
                raise ValueError("duplicate event for one stream/timestamp")
            bucket[name] = event

        current = max(self._pending)
        ready: list[AmodalEventWindow] = []
        for timestamp in sorted(self._pending):
            bucket = self._pending[timestamp]
            complete = all(name in bucket for name in self.stream_names)
            expired = self.max_wait is not None and current - timestamp >= self.max_wait
            learned_release = False
            if self.wait_policy is not None and not complete and not expired:
                age = torch.tensor([max(0.0, current - timestamp)])
                present_fraction = torch.tensor([len(bucket) / len(self.stream_names)])
                complete_value = torch.tensor([float(complete)])
                arrival_count = torch.tensor([float(self._arrival_count)])
                arrival_delta = torch.tensor([max(0.0, current - timestamp)])
                features = EventWaitPolicy.features(
                    age=age,
                    present_fraction=present_fraction,
                    complete=complete_value,
                    arrival_count=arrival_count,
                    arrival_delta=arrival_delta,
                )
                learned_release = bool(self.wait_policy(features).item() < 0.5)
            if complete or expired or learned_release:
                ready.append(self._release(timestamp, bucket, complete=complete))
                del self._pending[timestamp]
        return ready

    def _release(
        self,
        timestamp: float,
        bucket: Mapping[str, AmodalEvent],
        *,
        complete: bool,
    ) -> AmodalEventWindow:
        template = next(iter(bucket.values()))
        template.validate()
        batch, width = template.payload.shape
        payloads: list[torch.Tensor] = []
        present: list[bool] = []
        confidences: list[torch.Tensor] = []
        timestamps: list[torch.Tensor] = []
        durations: list[torch.Tensor] = []
        source_keys: list[torch.Tensor] = []
        source_event = next(
            (event for event in bucket.values() if event.source_key is not None), None
        )
        duration_present = any(event.duration is not None for event in bucket.values())
        has_source_keys = source_event is not None
        has_durations = duration_present
        source_width = (
            source_event.source_key.shape[-1] if source_event is not None else 0
        )
        for name in self.stream_names:
            event = bucket.get(name)
            if event is None:
                payloads.append(torch.zeros_like(template.payload))
                present.append(False)
                confidences.append(
                    torch.zeros(
                        batch,
                        device=template.payload.device,
                        dtype=template.payload.dtype,
                    )
                )
                timestamps.append(
                    torch.full(
                        (batch,),
                        timestamp,
                        device=template.payload.device,
                        dtype=template.payload.dtype,
                    )
                )
                durations.append(
                    torch.zeros(
                        batch,
                        device=template.payload.device,
                        dtype=template.payload.dtype,
                    )
                )
                if has_source_keys:
                    source_keys.append(
                        torch.zeros(
                            batch,
                            source_width,
                            device=template.payload.device,
                            dtype=template.payload.dtype,
                        )
                    )
                continue
            event.validate(width=width)
            payloads.append(event.payload)
            present.append(True)
            confidences.append(
                event.confidence.reshape(batch)
                if event.confidence is not None
                else torch.ones(
                    batch, device=event.payload.device, dtype=event.payload.dtype
                )
            )
            timestamps.append(
                event.timestamp.reshape(batch)
                if event.timestamp is not None
                else torch.full(
                    (batch,),
                    timestamp,
                    device=event.payload.device,
                    dtype=event.payload.dtype,
                )
            )
            durations.append(
                event.duration.reshape(batch)
                if event.duration is not None
                else torch.zeros(
                    batch, device=event.payload.device, dtype=event.payload.dtype
                )
            )
            if has_source_keys:
                if (
                    event.source_key is None
                    or event.source_key.shape[-1] != source_width
                ):
                    raise ValueError(
                        "source_key must be present consistently in a window"
                    )
                source_keys.append(event.source_key.reshape(batch, source_width))

        collection = AmodalEventCollection(
            payload=torch.stack(payloads, dim=1),
            present=torch.tensor(
                present, dtype=torch.bool, device=template.payload.device
            ).expand(batch, -1),
            confidence=torch.stack(confidences, dim=1),
            source_key=torch.stack(source_keys, dim=1) if has_source_keys else None,
            timestamp=torch.stack(timestamps, dim=1),
            duration=torch.stack(durations, dim=1) if has_durations else None,
        ).validate(width=width)
        return AmodalEventWindow(timestamp, collection, complete=complete)

    def pending_status(
        self, current_timestamp: float | None = None
    ) -> tuple[AmodalEventWindowStatus, ...]:
        if not self._pending:
            return ()
        now = (
            max(self._pending)
            if current_timestamp is None
            else float(current_timestamp)
        )
        return tuple(
            AmodalEventWindowStatus(
                timestamp=timestamp,
                age=max(0.0, now - timestamp),
                present=tuple(name in bucket for name in self.stream_names),
                complete=all(name in bucket for name in self.stream_names),
            )
            for timestamp, bucket in sorted(self._pending.items())
        )

    def release_pending(self, timestamp: float) -> AmodalEventWindow:
        timestamp = float(timestamp)
        if timestamp not in self._pending:
            raise KeyError(f"no pending timestamp {timestamp}")
        bucket = self._pending.pop(timestamp)
        complete = all(name in bucket for name in self.stream_names)
        return self._release(timestamp, bucket, complete=complete)


class AmodalControllerRuntime(nn.Module):
    """Compose N encoders, one clean controller, and M decoders."""

    def __init__(
        self,
        controller: AmodalCognitiveController,
        *,
        encoders: Mapping[str, nn.Module] | None = None,
        input_bus: AmodalInputBus | None = None,
        output_bus: AmodalOutputBus | None = None,
        memory: MemoryBackend | None = None,
        wait_policy: EventWaitPolicy | EventWaitStatistics | None = None,
        event_space_id: str = DEFAULT_EVENT_SPACE_ID,
        state_space_id: str = DEFAULT_CONTROLLER_STATE_SPACE_ID,
        intention_space_id: str = DEFAULT_INTENTION_SPACE_ID,
    ) -> None:
        super().__init__()
        if memory is not None and not isinstance(memory, MemoryBackend):
            raise TypeError("memory must implement the MemoryBackend contract")
        event_space_id = validate_representation_space_id(
            event_space_id, name="event_space_id"
        )
        state_space_id = validate_representation_space_id(
            state_space_id, name="state_space_id"
        )
        intention_space_id = validate_representation_space_id(
            intention_space_id, name="intention_space_id"
        )
        self.controller = controller
        self.encoders = nn.ModuleDict(dict(encoders or {}))
        self.input_bus = input_bus or AmodalInputBus(
            controller.width, event_space_id=event_space_id
        )
        if self.input_bus.event_space_id != event_space_id:
            raise ValueError("input bus event space does not match runtime")
        self.output_bus = output_bus or AmodalOutputBus(
            intention_space_id=intention_space_id
        )
        if self.output_bus.intention_space_id != intention_space_id:
            raise ValueError("output bus intention space does not match runtime")
        self.memory = memory
        self.wait_policy = wait_policy
        self.event_space_id = event_space_id
        self.state_space_id = state_space_id
        self.intention_space_id = intention_space_id

    @property
    def event_width(self) -> int:
        return self.controller.width

    @property
    def intention_width(self) -> int:
        return self.controller.intention_width

    def register_encoder(self, name: str, encoder: nn.Module) -> None:
        if not name or name in self.encoders:
            raise ValueError("encoder name must be nonempty and unique")
        self.encoders[name] = encoder

    def register_decoder(self, name: str, decoder: nn.Module) -> None:
        self.output_bus.register_decoder(name, decoder)

    def window_buffer(
        self,
        stream_names: Sequence[str],
        *,
        tolerance: float = 0.0,
        max_wait: float | None = None,
    ) -> AmodalEventWindowBuffer:
        """Create a transport buffer using this runtime's learned wait policy."""
        return AmodalEventWindowBuffer(
            stream_names,
            tolerance=tolerance,
            max_wait=max_wait,
            wait_policy=self.wait_policy,
        )

    def encode_streams(
        self,
        streams: Mapping[str, torch.Tensor | AmodalEvent],
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> AmodalEventCollection:
        if not streams:
            if batch_size is None:
                raise ValueError("empty streams require batch_size")
            return AmodalEventCollection.empty(
                batch_size, self.event_width, device=device, dtype=dtype
            )
        events: list[AmodalEvent] = []
        for name, source in streams.items():
            if isinstance(source, AmodalEvent):
                event = source
            else:
                if name not in self.encoders:
                    raise KeyError(f"no encoder registered for stream {name!r}")
                encoded = self.encoders[name](source)
                event = (
                    encoded
                    if isinstance(encoded, AmodalEvent)
                    else AmodalEvent(encoded)
                )
            events.append(event.validate(width=self.event_width))
        return self.input_bus(events)

    def step_events(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent],
        state: ControllerState,
        feedback: ControllerFeedback,
        *,
        elapsed: torch.Tensor | float = 1.0,
        disable_workspace: bool = False,
        memory_scope: torch.Tensor | None = None,
        sample_memory_writes: bool = False,
        memory_write_override: torch.Tensor | None = None,
        memory_write_uniform: torch.Tensor | None = None,
        memory_write_gradient: bool = True,
    ) -> tuple[AmodalRuntimeOutput, ControllerState]:
        collection = self.input_bus(events)
        controller_output, next_state = self.controller.step(
            collection,
            state,
            feedback,
            self.memory,
            elapsed=elapsed,
            disable_workspace=disable_workspace,
            memory_scope=memory_scope,
            sample_memory_writes=sample_memory_writes,
            memory_write_override=memory_write_override,
            memory_write_uniform=memory_write_uniform,
            memory_write_gradient=memory_write_gradient,
        )
        intention = controller_output.intention
        return (
            AmodalRuntimeOutput(
                controller=controller_output,
                intention=intention,
                decoded=self.output_bus(intention),
                execution_logits=controller_output.execution_logits,
            ),
            next_state,
        )

    @staticmethod
    def _quiet_feedback(feedback: ControllerFeedback) -> ControllerFeedback:
        """Remove repeated outcome credit from an internal thought tick."""
        return ControllerFeedback(
            action=torch.zeros_like(feedback.action),
            reward=torch.zeros_like(feedback.reward),
            propensity=torch.ones_like(feedback.propensity),
            has_feedback=torch.zeros_like(feedback.has_feedback),
        )

    @staticmethod
    def _uniform_execution_state(logits: torch.Tensor) -> str:
        """Return a batch-wide decision for the bounded convenience API."""
        if logits.ndim != 2 or logits.shape[1] != len(EXECUTION_STATES):
            raise ValueError("execution logits have the wrong shape")
        choices = logits.argmax(dim=-1)
        if choices.numel() == 0 or not torch.all(choices == choices[0]):
            raise ValueError(
                "automatic deliberation requires one trajectory or a batch with "
                "a uniform execution decision"
            )
        return EXECUTION_STATES[int(choices[0])]

    def deliberate(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent],
        state: ControllerState,
        feedback: ControllerFeedback,
        *,
        think_budget: int = 1,
        execution_mode: str | None = None,
        elapsed: torch.Tensor | float = 1.0,
        disable_workspace: bool = False,
        memory_scope: torch.Tensor | None = None,
    ) -> AmodalExecutionResult:
        """Run one bounded ``WAIT / THINK / COMMIT`` controller cycle.

        The default uses the controller's learned execution head.  An
        explicit ``execution_mode`` is useful for fixed-compute controls and
        causal audits, while the controller still produces the same opaque
        intention.  Automatic mode is deliberately single-trajectory (or
        uniform-batch) to avoid silently merging different per-example
        schedules; callers with mixed schedules should partition the batch.
        """
        if think_budget < 0:
            raise ValueError("think_budget must be nonnegative")
        if execution_mode is not None and execution_mode not in EXECUTION_STATES:
            raise ValueError(f"unknown execution mode {execution_mode!r}")

        current_events = events
        current_state = state
        current_feedback = feedback
        trace: list[AmodalRuntimeOutput] = []
        think_ticks = 0
        forced_commit = False
        initial_decision: str | None = None

        while True:
            output, next_state = self.step_events(
                current_events,
                current_state,
                current_feedback,
                elapsed=elapsed,
                disable_workspace=disable_workspace,
                memory_scope=memory_scope,
            )
            trace.append(output)
            decision = execution_mode or self._uniform_execution_state(
                output.execution_logits
            )
            if initial_decision is None:
                initial_decision = decision
            if decision != "think":
                return AmodalExecutionResult(
                    output=output,
                    state=next_state,
                    initial_decision=initial_decision,
                    decision=decision,
                    think_ticks=think_ticks,
                    forced_commit=False,
                    trace=tuple(trace),
                )
            if think_ticks >= think_budget:
                forced_commit = True
                return AmodalExecutionResult(
                    output=output,
                    state=next_state,
                    initial_decision=initial_decision,
                    decision="commit",
                    think_ticks=think_ticks,
                    forced_commit=forced_commit,
                    trace=tuple(trace),
                )
            think_ticks += 1
            current_events = AmodalEventCollection.empty(
                output.intention.payload.shape[0],
                self.event_width,
                device=output.intention.payload.device,
                dtype=output.intention.payload.dtype,
            )
            current_state = next_state
            current_feedback = self._quiet_feedback(feedback)
            # Explicit THINK means spend the whole requested budget.  Learned
            # mode can stop early when its next output says WAIT or COMMIT.
            if execution_mode is not None:
                execution_mode = "think" if think_ticks < think_budget else "commit"

    def step_streams(
        self,
        streams: Mapping[str, torch.Tensor | AmodalEvent],
        state: ControllerState,
        feedback: ControllerFeedback,
        *,
        elapsed: torch.Tensor | float = 1.0,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        disable_workspace: bool = False,
        memory_scope: torch.Tensor | None = None,
    ) -> tuple[AmodalRuntimeOutput, ControllerState]:
        events = self.encode_streams(
            streams,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return self.step_events(
            events,
            state,
            feedback,
            elapsed=elapsed,
            disable_workspace=disable_workspace,
            memory_scope=memory_scope,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ControllerState:
        return self.controller.initial_state(batch_size, device=device, dtype=dtype)

    @staticmethod
    def _controller_digest(runtime: AmodalControllerRuntime) -> str:
        digest = hashlib.sha256()
        digest.update(repr(runtime.configuration()).encode("utf-8"))
        for name, value in sorted(runtime.controller.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _migration_difference(first: torch.Tensor, second: torch.Tensor) -> float:
        if first.shape != second.shape:
            raise ValueError("runtime migration tensors have different shapes")
        return float((first - second).square().mean().detach())

    def migrate_controller_verified(
        self,
        candidate: AmodalControllerRuntime,
        examples: Sequence[RuntimeMigrationExample],
        *,
        prediction_tolerance: float = 1e-6,
        retention_probe: Callable[[AmodalControllerRuntime], bool] | None = None,
    ) -> RuntimeMigrationReceipt:
        """Approve a frontend/controller replacement without mutating either.

        Each example supplies source and replacement event representations plus
        their corresponding controller states. The candidate is accepted only
        when intentions, execution decisions, and continuation state remain
        within the held-out tolerance. This is an interface migration gate;
        it does not learn an alignment map or infer a decoder protocol.
        """

        if not isinstance(candidate, AmodalControllerRuntime):
            raise TypeError("runtime migration candidate is invalid")
        if prediction_tolerance < 0.0 or not torch.isfinite(
            torch.tensor(prediction_tolerance)
        ):
            raise ValueError("runtime migration tolerance is invalid")
        if not examples:
            raise ValueError("runtime migration needs held-out examples")
        if self.memory is not None or candidate.memory is not None:
            raise ValueError(
                "runtime migration requires memory-free probes; verify external memory separately"
            )
        if (
            self.event_width != candidate.event_width
            or self.intention_width != candidate.intention_width
            or self.controller.workspace_slots != candidate.controller.workspace_slots
            or self.controller.event_window_capacity
            != candidate.controller.event_window_capacity
            or self.controller.source_key_width != candidate.controller.source_key_width
            or self.controller.feedback_width != candidate.controller.feedback_width
        ):
            raise ValueError("runtime migration controller structure does not match")
        if (
            self.event_space_id == candidate.event_space_id
            and self.state_space_id == candidate.state_space_id
            and self.intention_space_id == candidate.intention_space_id
        ):
            raise ValueError(
                "runtime migration does not replace a representation space"
            )
        source_digest = self._controller_digest(self)
        target_digest = self._controller_digest(candidate)
        max_intention_difference = 0.0
        max_execution_difference = 0.0
        max_continuation_difference = 0.0
        for example in examples:
            source_collection = self.input_bus(example.source_events)
            target_collection = candidate.input_bus(example.target_events)
            batch = source_collection.payload.shape[0]
            if target_collection.payload.shape[0] != batch:
                raise ValueError("runtime migration example batch sizes differ")
            source_override = torch.zeros(
                batch,
                device=source_collection.payload.device,
                dtype=source_collection.payload.dtype,
            )
            target_override = torch.zeros(
                batch,
                device=target_collection.payload.device,
                dtype=target_collection.payload.dtype,
            )
            with torch.no_grad():
                source_output, source_next = self.controller.step(
                    source_collection,
                    example.source_state,
                    example.feedback,
                    memory=None,
                    elapsed=example.elapsed,
                    memory_write_override=source_override,
                    memory_write_gradient=False,
                )
                target_output, target_next = candidate.controller.step(
                    target_collection,
                    example.target_state,
                    example.feedback,
                    memory=None,
                    elapsed=example.elapsed,
                    memory_write_override=target_override,
                    memory_write_gradient=False,
                )
            max_intention_difference = max(
                max_intention_difference,
                self._migration_difference(
                    source_output.intention.payload,
                    target_output.intention.payload,
                ),
            )
            max_execution_difference = max(
                max_execution_difference,
                self._migration_difference(
                    source_output.execution_logits,
                    target_output.execution_logits,
                ),
            )
            for source_value, target_value in (
                (source_next.hidden, target_next.hidden),
                (source_next.workspace, target_next.workspace),
                (source_next.latest_event, target_next.latest_event),
                (source_next.workspace_usage, target_next.workspace_usage),
                (source_next.source_trust, target_next.source_trust),
            ):
                max_continuation_difference = max(
                    max_continuation_difference,
                    self._migration_difference(source_value, target_value),
                )
        accepted = (
            max(
                max_intention_difference,
                max_execution_difference,
                max_continuation_difference,
            )
            <= prediction_tolerance
        )
        if accepted and retention_probe is not None:
            if not callable(retention_probe):
                raise TypeError("runtime migration retention probe is invalid")
            accepted = bool(retention_probe(candidate))
        reason = (
            "candidate passed paired held-out controller and continuation checks"
            if accepted
            else "candidate changed held-out controller behavior"
        )
        return RuntimeMigrationReceipt(
            accepted=accepted,
            source_event_space_id=self.event_space_id,
            target_event_space_id=candidate.event_space_id,
            source_state_space_id=self.state_space_id,
            target_state_space_id=candidate.state_space_id,
            source_intention_space_id=self.intention_space_id,
            target_intention_space_id=candidate.intention_space_id,
            example_count=len(examples),
            max_intention_difference=max_intention_difference,
            max_execution_difference=max_execution_difference,
            max_continuation_difference=max_continuation_difference,
            source_digest=source_digest,
            target_digest=target_digest,
            reason=reason,
        ).validate()

    def configuration(self) -> dict[str, object]:
        return {
            "format": RUNTIME_FORMAT,
            "representation_space_schema": REPRESENTATION_SPACE_SCHEMA,
            "event_space_id": self.event_space_id,
            "state_space_id": self.state_space_id,
            "intention_space_id": self.intention_space_id,
            "controller": self.controller.configuration(),
            "input_bus": self.input_bus.configuration(),
            "encoder_names": tuple(self.encoders.keys()),
            "decoder_names": tuple(self.output_bus.decoders.keys()),
            "memory": None if self.memory is None else self.memory.configuration(),
            "wait_policy": (
                None if self.wait_policy is None else self.wait_policy.configuration()
            ),
        }

    def component_state_dicts(self) -> dict[str, object]:
        """Return separately loadable component weights.

        Encoder and decoder classes are intentionally not inferred from this
        payload.  A caller supplies the replacement component and loads only
        its own state dict, preventing a checkpoint from silently constructing
        a modality-specific branch inside the controller.
        """
        components: dict[str, object] = {
            "controller": self.controller.state_dict(),
            "input_bus": self.input_bus.state_dict(),
            "encoders": {
                name: encoder.state_dict() for name, encoder in self.encoders.items()
            },
            "decoders": {
                name: decoder.state_dict()
                for name, decoder in self.output_bus.decoders.items()
            },
        }
        if self.memory is not None:
            components["memory"] = self.memory.state_dict()
        if self.wait_policy is not None:
            components["wait_policy"] = self.wait_policy.state_dict()
        return components

    def checkpoint_payload(
        self, *, provenance: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        return {
            "format": RUNTIME_FORMAT,
            "event_schema": EVENT_SCHEMA,
            "intention_schema": INTENTION_SCHEMA,
            "configuration": self.configuration(),
            "provenance": dict(provenance or {}),
            "components": self.component_state_dicts(),
            "retention_ledger": (
                None
                if self.memory is None or not hasattr(self.memory, "retention")
                else self.memory.retention.payload()
            ),
        }

    def load_component_state_dicts(
        self,
        components: Mapping[str, object],
        *,
        allow_missing_execution: bool = False,
    ) -> None:
        expected = set(self.component_state_dicts())
        actual = set(components)
        if actual != expected:
            raise ValueError(
                f"component set mismatch: expected {sorted(expected)}, got {sorted(actual)}"
            )
        controller_result = self.controller.load_state_dict(
            components["controller"], strict=not allow_missing_execution
        )
        if allow_missing_execution and any(
            key == "memory_address.weight" for key in controller_result.missing_keys
        ):
            # v17 used independent event/state projections for reads and
            # writes. Seed the v18 shared event address from the old query
            # projection's event slice so legacy checkpoints retain a useful
            # starting point while the new path remains independently trained.
            old_query_weight = components["controller"].get("memory_query.weight")
            old_query_bias = components["controller"].get("memory_query.bias")
            if old_query_weight is not None and old_query_bias is not None:
                with torch.no_grad():
                    self.controller.memory_address.weight.copy_(
                        old_query_weight[:, : self.controller.width]
                    )
                    self.controller.memory_address.bias.copy_(old_query_bias)
        if allow_missing_execution and any(
            key == "memory_write_policy.0.weight"
            for key in controller_result.missing_keys
        ):
            # Legacy v18 had a linear value-context gate. Preserve its initial
            # write propensity as the v19 policy bias while leaving the new
            # nonlinear feature weights trainable.
            old_write_bias = components["controller"].get("memory_write.bias")
            if old_write_bias is not None:
                with torch.no_grad():
                    self.controller.memory_write_policy[-1].bias.copy_(old_write_bias)
        if allow_missing_execution:
            non_execution_unexpected = [
                key
                for key in controller_result.unexpected_keys
                if not key.startswith("execution_policy.")
                and not key.startswith("execution_transport_policy.")
                and not key.startswith("execution_timeout_policy.")
                and not key.startswith("event_pair_")
                and not key.startswith("event_feedback_relevance.")
                and not key.startswith("event_feedback_source_relevance.")
                and not key.startswith("source_credit_policy.")
                and not key.startswith("memory_address.")
                and not key.startswith("event_address_relevance.")
                and not key.startswith("memory_query.")
                and not key.startswith("memory_key.")
                and not key.startswith("memory_write_policy.")
                and not key.startswith("memory_value_feedback.")
                and not key.startswith("memory_value_stable.")
            ]
            missing = list(controller_result.missing_keys)
            non_execution_missing = [
                key
                for key in missing
                if not key.startswith("execution_policy.")
                and not key.startswith("execution_transport_policy.")
                and not key.startswith("execution_timeout_policy.")
                and not key.startswith("event_pair_")
                and not key.startswith("event_feedback_relevance.")
                and not key.startswith("event_feedback_source_relevance.")
                and not key.startswith("source_credit_policy.")
                and not key.startswith("memory_address.")
                and not key.startswith("event_address_relevance.")
                and not key.startswith("memory_write_policy.")
                and not key.startswith("memory_value_feedback.")
                and not key.startswith("memory_value_stable.")
            ]
            if non_execution_unexpected or non_execution_missing:
                raise ValueError(
                    "legacy checkpoint has incompatible controller parameters: "
                    f"missing={non_execution_missing}, unexpected={non_execution_unexpected}"
                )
        self.input_bus.load_state_dict(components["input_bus"])
        for name, encoder in self.encoders.items():
            encoder.load_state_dict(components["encoders"][name])
        for name, decoder in self.output_bus.decoders.items():
            decoder.load_state_dict(components["decoders"][name])
        if self.memory is not None:
            previous_memory = {
                name: value.detach().clone()
                for name, value in self.memory.state_dict().items()
            }
            try:
                self.memory.load_state_dict(components["memory"])
                self.memory.validate_state()
            except Exception:
                self.memory.load_state_dict(previous_memory)
                raise
        if self.wait_policy is not None:
            self.wait_policy.load_state_dict(components["wait_policy"])


class ExternalControllerStateAdapter(nn.Module):
    """Map the controller's learned state representation to model state.

    This adapter is deliberately outside the controller.  It consumes only
    the controller's opaque learned state representation and can therefore be
    frozen, replaced, or trained independently of the factual transition
    model.  It contains no action vocabulary, task label, or protocol branch.
    """

    schema = "neural-computer.external-controller-state-adapter.v1"

    def __init__(
        self,
        controller_feature_width: int,
        state_width: int,
        *,
        hidden_width: int = 0,
    ) -> None:
        super().__init__()
        if min(controller_feature_width, state_width) < 1 or hidden_width < 0:
            raise ValueError("controller state-adapter dimensions are invalid")
        self.controller_feature_width = int(controller_feature_width)
        self.state_width = int(state_width)
        self.hidden_width = int(hidden_width)
        if hidden_width:
            self.network = nn.Sequential(
                nn.Linear(self.controller_feature_width, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, self.state_width),
            )
        elif controller_feature_width == state_width:
            self.network = nn.Identity()
        else:
            self.network = nn.Linear(controller_feature_width, state_width)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "controller_feature_width": self.controller_feature_width,
            "state_width": self.state_width,
            "hidden_width": self.hidden_width,
            "input": "opaque_controller_state_representation_v1",
            "behavior": "replaceable_state_projection_not_policy_v1",
        }

    def forward(self, output: ControllerOutput | torch.Tensor) -> torch.Tensor:
        state = (
            output.state_representation
            if isinstance(output, ControllerOutput)
            else output
        )
        if state.ndim != 2 or state.shape[1] != self.controller_feature_width:
            raise ValueError("controller state representation has the wrong shape")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("controller state representation must be finite")
        projected = self.network(state)
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("adapted model state must be finite")
        return projected


class ExternalControllerTrajectoryQueryAdapter(nn.Module):
    """Build an opaque route query from controller state and event history.

    The factual planner still consumes the ordinary controller state. This
    replaceable memory-side adapter is only for addressing a growing external
    bank. It uses the final learned state plus masked mean/max statistics of
    the learned event-token window, preserving more regime identity than a
    single final-state projection without exposing raw modality formats.
    """

    schema = "neural-computer.external-controller-trajectory-query-adapter.v1"

    def __init__(
        self,
        controller_width: int,
        query_width: int | None = None,
        *,
        hidden_width: int = 0,
    ) -> None:
        super().__init__()
        if controller_width < 1 or hidden_width < 0:
            raise ValueError("trajectory-query adapter dimensions are invalid")
        self.controller_width = int(controller_width)
        self.controller_feature_width = self.controller_width * 3
        self.event_feature_width = self.controller_width
        self.input_width = self.controller_feature_width + 2 * self.event_feature_width
        self.query_width = self.input_width if query_width is None else int(query_width)
        if self.query_width < 1:
            raise ValueError("trajectory-query adapter query width must be positive")
        self.hidden_width = int(hidden_width)
        if hidden_width:
            self.network = nn.Sequential(
                nn.Linear(self.input_width, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, self.query_width),
            )
        elif self.query_width == self.input_width:
            self.network = nn.Identity()
        else:
            self.network = nn.Linear(self.input_width, self.query_width)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "controller_width": self.controller_width,
            "controller_feature_width": self.controller_feature_width,
            "event_feature_width": self.event_feature_width,
            "input_width": self.input_width,
            "query_width": self.query_width,
            "hidden_width": self.hidden_width,
            "input": "opaque_controller_state_plus_event_token_trajectory_v1",
            "statistics": "masked_mean_and_max_v1",
            "behavior": "replaceable_memory_address_query_not_reasoning_branch_v1",
        }

    def forward(
        self,
        output: ControllerOutput,
        state: ControllerState,
    ) -> torch.Tensor:
        representation = output.state_representation
        if (
            representation.ndim != 2
            or representation.shape[1] != self.controller_feature_width
        ):
            raise ValueError(
                "trajectory-query controller representation has the wrong shape"
            )
        payload = state.event_window.payload
        present = state.event_window.present
        if payload.ndim != 3 or payload.shape[2] != self.event_feature_width:
            raise ValueError("trajectory-query event tokens have the wrong shape")
        if present.shape != payload.shape[:2] or present.dtype != torch.bool:
            raise ValueError("trajectory-query event presence has the wrong shape")
        present_float = present.to(dtype=payload.dtype)
        denominator = present_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (payload * present_float.unsqueeze(-1)).sum(dim=1) / denominator
        max_values = payload.masked_fill(~present.unsqueeze(-1), -torch.inf).amax(dim=1)
        has_event = present.any(dim=1, keepdim=True)
        max_values = torch.where(has_event, max_values, torch.zeros_like(max_values))
        features = torch.cat((representation, mean, max_values), dim=-1)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("trajectory-query features must be finite")
        query = self.network(features)
        if not bool(torch.isfinite(query).all()):
            raise ValueError("trajectory-query output must be finite")
        return query


EXTERNAL_PROGRAM_RUNTIME_SCHEMA = "neural-computer.external-program-runtime.v6"
EXTERNAL_PROGRAM_RUNTIME_STATE_SCHEMA = (
    "neural-computer.external-program-runtime-state.v1"
)
_STANDALONE_PROGRAM_STATE_KEY = -1


def _digest_runtime_state_payload(value: object) -> str:
    """Hash nested tensor payloads without interpreting their learned values."""

    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            detached = item.detach().cpu().contiguous()
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExternalProgramRuntimeState:
    """Joint state for one controller and one external computation file."""

    controller: ControllerState
    program: ExternalRegisterState
    program_states: Mapping[int, ExternalRegisterState] = field(default_factory=dict)
    program_router: ExternalOutcomeProgramRouterState | None = None

    def payload(self) -> dict[str, object]:
        """Return a versioned tensor-only checkpoint for pause/resume."""

        payload: dict[str, object] = {
            "schema": EXTERNAL_PROGRAM_RUNTIME_STATE_SCHEMA,
            "controller": self.controller.payload(),
            "program": self.program.payload(),
            "program_states": tuple(
                {
                    "logical_id": int(logical_id),
                    "state": state.payload(),
                }
                for logical_id, state in sorted(self.program_states.items())
            ),
            "program_router": (
                None
                if self.program_router is None
                else {
                    "schema": EXTERNAL_OUTCOME_PROGRAM_ROUTER_SCHEMA,
                    "configuration": {
                        "schema": EXTERNAL_OUTCOME_PROGRAM_ROUTER_SCHEMA,
                        "feature_width": self.program_router.credit.policy.shape[1],
                        "program_capacity": self.program_router.credit.policy.shape[2],
                    },
                    "active_programs": self.program_router.active_programs,
                    "credit": {
                        "schema": EXTERNAL_OUTCOME_CREDIT_SCHEMA,
                        "configuration": {
                            "feature_width": self.program_router.credit.policy.shape[1],
                            "action_count": self.program_router.credit.policy.shape[2],
                        },
                        "policy": self.program_router.credit.policy.detach()
                        .cpu()
                        .clone(),
                        "eligibility": self.program_router.credit.eligibility.detach()
                        .cpu()
                        .clone(),
                        "baseline": self.program_router.credit.baseline.detach()
                        .cpu()
                        .clone(),
                        "decisions": self.program_router.credit.decisions.detach()
                        .cpu()
                        .clone(),
                        "feedbacks": self.program_router.credit.feedbacks.detach()
                        .cpu()
                        .clone(),
                    },
                }
            ),
        }
        payload["sha256"] = _digest_runtime_state_payload(payload)
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
        *,
        program_router: ExternalOutcomeProgramRouter | None = None,
    ) -> ExternalProgramRuntimeState:
        """Restore external runtime state without loading executable files."""

        if not isinstance(payload, dict):
            raise TypeError(
                "external program runtime state payload must be a dictionary"
            )
        if payload.get("schema") != EXTERNAL_PROGRAM_RUNTIME_STATE_SCHEMA:
            raise ValueError("unsupported external program runtime state schema")
        expected_digest = payload.get("sha256")
        unsigned = {key: value for key, value in payload.items() if key != "sha256"}
        if not isinstance(
            expected_digest, str
        ) or expected_digest != _digest_runtime_state_payload(unsigned):
            raise ValueError("external program runtime state checksum mismatch")
        controller = unsigned.get("controller")
        program = unsigned.get("program")
        records = unsigned.get("program_states")
        if not isinstance(controller, dict) or not isinstance(program, dict):
            raise TypeError("external program runtime state payload is incomplete")
        if not isinstance(records, (tuple, list)):
            raise TypeError(
                "external program runtime program states must be a sequence"
            )
        program_states: dict[int, ExternalRegisterState] = {}
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("external program runtime program state is malformed")
            logical_id = record.get("logical_id")
            state_payload = record.get("state")
            if not isinstance(logical_id, int) or logical_id < -1:
                raise ValueError("external program runtime logical ID is invalid")
            if logical_id in program_states:
                raise ValueError("external program runtime logical IDs must be unique")
            if not isinstance(state_payload, dict):
                raise TypeError("external program runtime program state is missing")
            program_states[logical_id] = ExternalRegisterState.from_payload(
                state_payload
            )
        router_payload = unsigned.get("program_router")
        if router_payload is None:
            restored_router = None
        elif program_router is None:
            raise ValueError(
                "restoring executable route state requires its route policy"
            )
        elif not isinstance(router_payload, dict):
            raise TypeError("external program runtime route state is malformed")
        else:
            restored_router = program_router.state_from_payload(router_payload)
        return cls(
            controller=ControllerState.from_payload(controller),
            program=ExternalRegisterState.from_payload(program),
            program_states=program_states,
            program_router=restored_router,
        )


@dataclass(frozen=True)
class ExternalProgramRuntimeOutput:
    """One INPUT -> PROCESS -> OUTPUT cycle with external computation.

    The controller remains the only amodal processor.  The register machine
    is a replaceable external computation file: it reads only the controller's
    learned state and opaque feedback, and its transient result is what the
    decoders receive.  ``controller.intention`` stays diagnostic and is never
    silently used as the device output.
    """

    controller: ControllerOutput
    execution: ExternalExecutionSnapshot
    intention: IntentEvent
    decoded: dict[str, torch.Tensor]
    selected_program_slot: int | None
    selected_program_logical_id: int | None = None
    selected_program_slots: torch.Tensor | None = None
    selected_program_logical_ids: torch.Tensor | None = None
    program_route_query: torch.Tensor | None = None
    program_route_probabilities: torch.Tensor | None = None
    program_route_propensities: torch.Tensor | None = None
    execution_snapshots: tuple[ExternalExecutionSnapshot, ...] = ()
    schema: str = EXTERNAL_PROGRAM_RUNTIME_SCHEMA


class ExternalProgramAmodalRuntime(nn.Module):
    """Run portable external programs through the canonical amodal boundary.

    This is the CPU-plus-files seam suggested by the architecture work.  A
    fixed controller handles learned events, working memory, and feedback;
    an external register interpreter executes a versioned opaque artifact;
    the intention bus fans the result out to any decoders. Program files can
    be replaced, routed, or grown without changing controller parameters.
    Each retained logical file owns an isolated recurrent execution state;
    switching files never leaks one capability's working state into another.
    """

    schema = EXTERNAL_PROGRAM_RUNTIME_SCHEMA

    def __init__(
        self,
        runtime: AmodalControllerRuntime,
        machine: ExternalCapabilityRegisterMachine,
        *,
        program: ExternalProgramArtifact | None = None,
        program_memory: ExternalSequenceProgramMemory | None = None,
        program_query_adapter: (
            ExternalControllerStateAdapter
            | ExternalControllerTrajectoryQueryAdapter
            | None
        ) = None,
        program_route_exploration: float = 0.0,
        program_router: ExternalOutcomeProgramRouter | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(runtime, AmodalControllerRuntime):
            raise TypeError("external program runtime requires an amodal runtime")
        if not isinstance(machine, ExternalCapabilityRegisterMachine):
            raise TypeError("external program runtime requires a register machine")
        if (program is None) == (program_memory is None):
            raise ValueError("external program runtime requires one program source")
        if not 0.0 <= float(program_route_exploration) <= 1.0:
            raise ValueError("program route exploration must lie in [0, 1]")
        controller_feature_width = runtime.controller.width * 3
        if machine.event_width != controller_feature_width:
            raise ValueError(
                "external program event width must match controller state width"
            )
        if machine.action_width != runtime.controller.feedback_width:
            raise ValueError(
                "external program action width must match controller feedback width"
            )
        if machine.intention_width != runtime.intention_width:
            raise ValueError(
                "external program intention width must match controller intention width"
            )
        if program is not None:
            program.validate_for(
                instruction_width=machine.instruction_width,
                interpreter_schema="neural-computer.external-register.v4",
                execution_schema="neural-computer.external-register-read-execute.v1",
            )
        if program_memory is not None:
            if program_memory.instruction_width != machine.instruction_width:
                raise ValueError(
                    "program memory instruction width does not match machine"
                )
            if len(program_memory.programs) < 1:
                raise ValueError("program memory must contain at least one artifact")
        if program_router is not None:
            if program_memory is None:
                raise ValueError("program router requires external program memory")
            if program_router.feature_width != machine.instruction_width:
                raise ValueError("program router feature width does not match machine")
            if program_router.initial_programs != program_memory.file_count:
                raise ValueError(
                    "program router initial programs must match program memory files"
                )
        if program_query_adapter is not None and not isinstance(
            program_query_adapter,
            (ExternalControllerStateAdapter, ExternalControllerTrajectoryQueryAdapter),
        ):
            raise TypeError("program query adapter has an incompatible type")
        if program_query_adapter is not None:
            adapter_width = (
                program_query_adapter.state_width
                if isinstance(program_query_adapter, ExternalControllerStateAdapter)
                else program_query_adapter.query_width
            )
            if (
                program_query_adapter.controller_feature_width
                != controller_feature_width
                or adapter_width != machine.instruction_width
            ):
                raise ValueError(
                    "program query adapter does not match controller or machine"
                )
        self.runtime = runtime
        self.machine = machine
        self.program = program
        self.program_memory = program_memory
        self.program_query_adapter = program_query_adapter or (
            ExternalControllerTrajectoryQueryAdapter(
                runtime.controller.width,
                machine.instruction_width,
            )
            if program_memory is not None
            else None
        )
        self.program_route_exploration = float(program_route_exploration)
        self.program_router = program_router

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "boundary": "controller_state_to_external_register_to_intention_bus_v1",
            "controller": self.runtime.configuration()["controller"],
            "runtime": self.runtime.configuration(),
            "machine": self.machine.configuration(),
            "program_source": (
                "portable_external_artifact_v1"
                if self.program is not None
                else "opaque_content_routed_external_program_memory_v1"
            ),
            "program": None if self.program is None else self.program.configuration(),
            "program_memory": (
                None
                if self.program_memory is None
                else self.program_memory.configuration()
            ),
            "program_query_adapter": (
                None
                if self.program_query_adapter is None
                else self.program_query_adapter.configuration()
            ),
            "program_route_exploration": self.program_route_exploration,
            "program_route_behavior": (
                "greedy_argmax_v1"
                if self.program_route_exploration == 0.0
                else "epsilon_mixture_sampled_with_propensity_v1"
            ),
            "program_router": (
                None
                if self.program_router is None
                else self.program_router.configuration()
            ),
            "controller_output": "diagnostic_only_v1",
            "decoder_input": "external_program_intention_v1",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalProgramRuntimeState:
        program_state = self.machine.initial_state(
            batch_size,
            device=device,
            dtype=dtype,
        )
        if self.program_memory is None:
            program_states = {_STANDALONE_PROGRAM_STATE_KEY: program_state}
        else:
            program_states = {
                logical_id: program_state
                for logical_id in self.program_memory.logical_slot_ids
            }
        program_router = (
            None
            if self.program_router is None
            else self.program_router.initial_state(
                batch_size,
                device=device,
                dtype=dtype,
            )
        )
        return ExternalProgramRuntimeState(
            controller=self.runtime.initial_state(
                batch_size,
                device=device,
                dtype=dtype,
            ),
            program=program_state,
            program_states=program_states,
            program_router=program_router,
        )

    def activate_program(
        self,
        state: ExternalProgramRuntimeState,
        *,
        initialization: str = "conservative",
    ) -> ExternalProgramRuntimeState:
        """Activate one newly admitted file in the external route policy."""

        if self.program_router is None or self.program_memory is None:
            raise RuntimeError("external program runtime has no learned route policy")
        if state.program_router is None:
            raise RuntimeError("external program route state is missing")
        if self.program_memory.file_count != state.program_router.active_programs + 1:
            raise ValueError(
                "activate_program requires exactly one newly admitted executable file"
            )
        next_router = self.program_router.append_program(
            state.program_router,
            initialization=initialization,
        )
        return ExternalProgramRuntimeState(
            controller=state.controller,
            program=state.program,
            program_states=state.program_states,
            program_router=next_router,
        )

    def _select_program(
        self,
        controller_output: ControllerOutput,
        controller_state: ControllerState,
        program_router_state: ExternalOutcomeProgramRouterState | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        ExternalOutcomeProgramRouterState | None,
    ]:
        if self.program is not None:
            batch_size = controller_output.state_representation.shape[0]
            return (
                torch.full(
                    (batch_size,),
                    _STANDALONE_PROGRAM_STATE_KEY,
                    dtype=torch.long,
                    device=controller_output.state_representation.device,
                ),
                torch.full(
                    (batch_size,),
                    -1,
                    dtype=torch.long,
                    device=controller_output.state_representation.device,
                ),
                None,
                None,
                None,
                program_router_state,
            )
        if self.program_memory is None or self.program_query_adapter is None:
            raise RuntimeError("external program runtime has no program source")
        if isinstance(
            self.program_query_adapter,
            ExternalControllerTrajectoryQueryAdapter,
        ):
            query = self.program_query_adapter(controller_output, controller_state)
        else:
            query = self.program_query_adapter(controller_output)
        if self.program_router is not None:
            if program_router_state is None:
                raise RuntimeError("external program route state is missing")
            if self.program_memory.file_count != program_router_state.active_programs:
                raise RuntimeError(
                    "external program memory and route policy are out of sync; "
                    "use activate_program or rebuild the route policy"
                )
            route_features = query.detach()
            behavior = self.program_router.behavior_probabilities(
                program_router_state,
                route_features,
                exploration=self.program_route_exploration,
            )
            if self.program_route_exploration:
                selected = torch.multinomial(behavior, 1).squeeze(-1)
                propensity = behavior.gather(
                    1,
                    selected.unsqueeze(-1),
                ).squeeze(-1)
            else:
                selected = behavior.argmax(dim=-1)
                propensity = torch.ones(
                    selected.shape,
                    device=selected.device,
                    dtype=behavior.dtype,
                )
            next_router_state = self.program_router.record_decision(
                program_router_state,
                route_features,
                selected,
                propensity,
            )
            logical_ids = torch.tensor(
                [self.program_memory.logical_slot_id(int(slot)) for slot in selected],
                dtype=torch.long,
                device=selected.device,
            )
            return (
                logical_ids,
                selected,
                query,
                behavior,
                propensity,
                next_router_state,
            )
        route_weights = self.program_memory.route_weights(query)
        if (
            type(self.program_memory).route_weights
            is ExternalSequenceProgramMemory.route_weights
        ):
            probabilities = self.program_memory.route_probabilities(query)
        else:
            # A replaceable memory-side route policy may override the route
            # weights without implementing the optional probability helper.
            # Preserve that policy and expose its normalized weights as the
            # only honest propensity surface available at this boundary.
            probabilities = route_weights.detach()
        probabilities = probabilities.clamp_min(0.0)
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        if self.program_route_exploration:
            behavior = (1.0 - self.program_route_exploration) * probabilities
            behavior = (
                behavior + self.program_route_exploration / probabilities.shape[1]
            )
            selected = torch.multinomial(behavior, 1).squeeze(-1)
            propensity = behavior.gather(1, selected.unsqueeze(-1)).squeeze(-1)
        else:
            behavior = probabilities
            selected = route_weights.argmax(dim=-1)
            propensity = torch.ones(
                selected.shape,
                device=selected.device,
                dtype=probabilities.dtype,
            )
        logical_ids = torch.tensor(
            [self.program_memory.logical_slot_id(int(slot)) for slot in selected],
            dtype=torch.long,
            device=selected.device,
        )
        return logical_ids, selected, query, behavior, propensity, program_router_state

    @staticmethod
    def _merge_register_states(
        snapshots: tuple[ExternalExecutionSnapshot, ...],
        masks: tuple[torch.Tensor, ...],
    ) -> ExternalRegisterState:
        """Merge row-partitioned observations without merging file state."""

        if not snapshots or len(snapshots) != len(masks):
            raise ValueError("cannot merge an empty or misaligned execution batch")
        merged = snapshots[0].observed
        for snapshot, mask in zip(snapshots[1:], masks[1:], strict=True):
            row = mask
            merged = ExternalRegisterState(
                register=torch.where(
                    row.unsqueeze(-1), snapshot.observed.register, merged.register
                ),
                context=torch.where(
                    row.unsqueeze(-1), snapshot.observed.context, merged.context
                ),
                initialized=torch.where(
                    row, snapshot.observed.initialized, merged.initialized
                ),
                event_window=(
                    torch.where(
                        row[:, None, None],
                        snapshot.observed.event_window,
                        merged.event_window,
                    )
                    if snapshot.observed.event_window is not None
                    and merged.event_window is not None
                    else None
                ),
                event_window_mask=(
                    torch.where(
                        row[:, None],
                        snapshot.observed.event_window_mask,
                        merged.event_window_mask,
                    )
                    if snapshot.observed.event_window_mask is not None
                    and merged.event_window_mask is not None
                    else None
                ),
            )
        return merged

    def step_events(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent],
        state: ExternalProgramRuntimeState,
        feedback: ControllerFeedback,
        *,
        elapsed: torch.Tensor | float = 1.0,
        disable_workspace: bool = False,
        memory_scope: torch.Tensor | None = None,
    ) -> tuple[ExternalProgramRuntimeOutput, ExternalProgramRuntimeState]:
        if not isinstance(state, ExternalProgramRuntimeState):
            raise TypeError("external program runtime state has the wrong type")
        collection = self.runtime.input_bus(events)
        controller_output, next_controller = self.runtime.controller.step(
            collection,
            state.controller,
            feedback,
            self.runtime.memory,
            elapsed=elapsed,
            disable_workspace=disable_workspace,
            memory_scope=memory_scope,
        )
        program_router_state = state.program_router
        if self.program_router is not None:
            if program_router_state is None:
                raise RuntimeError("external program route state is missing")
            feedback_present = feedback.has_feedback.reshape(-1).to(torch.bool)
            feedback_reward = feedback.reward.reshape(-1)
            if bool(
                ((feedback_reward < 0.0) | (feedback_reward > 1.0))
                .logical_and(feedback_present)
                .any()
            ):
                raise ValueError(
                    "program route feedback must lie in [0, 1] when present"
                )
            safe_reward = torch.where(
                feedback_present,
                feedback_reward,
                torch.zeros_like(feedback_reward),
            )
            program_router_state = self.program_router.apply_feedback(
                program_router_state,
                safe_reward,
                present=feedback_present,
                terminal=feedback_present,
            )
        (
            selected_logical_ids,
            selected_slots,
            program_route_query,
            program_route_probabilities,
            program_route_propensities,
            program_router_state,
        ) = self._select_program(
            controller_output,
            next_controller,
            program_router_state,
        )
        present = collection.present.any(dim=1)
        program_states = dict(state.program_states)
        snapshots: list[ExternalExecutionSnapshot] = []
        masks: list[torch.Tensor] = []
        for logical_id in torch.unique(selected_logical_ids, sorted=True).tolist():
            logical_id = int(logical_id)
            mask = present & (selected_logical_ids == logical_id)
            if logical_id == _STANDALONE_PROGRAM_STATE_KEY:
                if self.program is None:
                    raise RuntimeError("standalone program state has no artifact")
                artifact = self.program
            else:
                if self.program_memory is None:
                    raise RuntimeError("memory program state has no program memory")
                physical_slot = self.program_memory.physical_index_for_logical_id(
                    logical_id
                )
                artifact = self.program_memory.artifact(physical_slot)
            artifact.validate_for(
                instruction_width=self.machine.instruction_width,
                interpreter_schema="neural-computer.external-register.v4",
                execution_schema="neural-computer.external-register-read-execute.v1",
            )
            program_state = program_states.get(logical_id)
            if program_state is None:
                program_state = self.machine.initial_state(
                    present.shape[0],
                    device=present.device,
                    dtype=controller_output.state_representation.dtype,
                )
            snapshot = self.machine.read_execute_artifact_snapshot(
                event=controller_output.state_representation,
                action=feedback.action,
                outcome=feedback.reward,
                intention=controller_output.intention,
                state=program_state,
                artifact=artifact,
                present=mask,
            )
            snapshots.append(snapshot)
            masks.append(selected_logical_ids == logical_id)
            program_states[logical_id] = snapshot.observed
        execution_snapshots = tuple(snapshots)
        mask_tuple = tuple(masks)
        if len(execution_snapshots) == 1:
            snapshot = execution_snapshots[0]
        else:
            merged_observed = self._merge_register_states(
                execution_snapshots,
                mask_tuple,
            )
            merged_executed = execution_snapshots[0].executed
            for candidate, mask in zip(
                execution_snapshots[1:], mask_tuple[1:], strict=True
            ):
                merged_executed = torch.where(
                    mask.unsqueeze(-1), candidate.executed, merged_executed
                )
            trace: tuple[torch.Tensor, ...] = ()
            trace_lengths = {len(candidate.trace) for candidate in execution_snapshots}
            if len(trace_lengths) == 1:
                trace_values: list[torch.Tensor] = []
                for trace_index in range(len(execution_snapshots[0].trace)):
                    value = execution_snapshots[0].trace[trace_index]
                    for candidate, mask in zip(
                        execution_snapshots[1:], mask_tuple[1:], strict=True
                    ):
                        value = torch.where(
                            mask.unsqueeze(-1),
                            candidate.trace[trace_index],
                            value,
                        )
                    trace_values.append(value)
                trace = tuple(trace_values)
            snapshot = ExternalExecutionSnapshot(
                observed=merged_observed,
                executed=merged_executed,
                trace=trace,
            ).validate(
                batch_size=present.shape[0],
                register_width=self.machine.register_width,
                context_width=self.machine.context_width,
                event_width=self.machine.event_width,
                event_window_size=self.machine.event_window_size,
            )
        logical_id = int(selected_logical_ids[0])
        selected_slot = int(selected_slots[0])
        uniform_selection = bool(torch.all(selected_logical_ids == logical_id))
        intention = IntentEvent(
            payload=self.machine.to_intention(snapshot.executed).payload,
            confidence=controller_output.intention.confidence,
        ).validate(width=self.runtime.intention_width)
        output = ExternalProgramRuntimeOutput(
            controller=controller_output,
            execution=snapshot,
            intention=intention,
            decoded=self.runtime.output_bus(intention),
            selected_program_slot=(
                selected_slot if uniform_selection and selected_slot >= 0 else None
            ),
            selected_program_logical_id=(
                None
                if not uniform_selection or logical_id == _STANDALONE_PROGRAM_STATE_KEY
                else logical_id
            ),
            selected_program_slots=selected_slots.detach().clone(),
            selected_program_logical_ids=selected_logical_ids.detach().clone(),
            program_route_query=(
                None
                if program_route_query is None
                else program_route_query.detach().clone()
            ),
            program_route_probabilities=(
                None
                if program_route_probabilities is None
                else program_route_probabilities.detach().clone()
            ),
            program_route_propensities=(
                None
                if program_route_propensities is None
                else program_route_propensities.detach().clone()
            ),
            execution_snapshots=execution_snapshots,
        )
        if self.program_memory is not None:
            active_ids = set(self.program_memory.logical_slot_ids)
            program_states = {
                key: value for key, value in program_states.items() if key in active_ids
            }
        return output, ExternalProgramRuntimeState(
            controller=next_controller,
            program=snapshot.observed,
            program_states=program_states,
            program_router=program_router_state,
        )


@dataclass(frozen=True)
class PolicyFreeRuntimeOutput:
    """One model-derived intention produced by the policy-free runtime.

    ``controller.intention`` is retained inside the diagnostic controller
    output, but is intentionally not sent to the output bus.  The only
    intention exposed to decoders is the first step of the factual planner's
    verified model rollout.
    """

    controller: ControllerOutput
    planning: ModelBasedPlanningResult
    intention: IntentEvent
    decoded: dict[str, torch.Tensor]
    state: torch.Tensor
    goal_state: torch.Tensor | None
    selected_slot_id: int | None
    goal_fragments: ExternalGoalFragmentSet | None = None
    goal_fragment_indices: torch.Tensor | None = None
    proposal: ExternalIntentionProposal | None = None
    intention_generation: ExternalIntentionGenerationProposal | None = None
    intention_memory_generation: ExternalIntentionMemoryProposal | None = None
    intention_routing: ExternalRoutedIntentionProposal | None = None
    entry_proposal: ExternalEntryProposal | None = None
    binding_proposal: ExternalEntryBindingProposal | None = None
    schema: str = POLICY_FREE_RUNTIME_SCHEMA


class PolicyFreeAmodalRuntime:
    """Compose one amodal controller with factual model-based behavior.

    The controller updates working state and emits an opaque learned state
    representation.  An external goal/destination and candidate intention set
    are passed to a factual transition model; beam search derives behavior at
    inference time.  Thus a new regime can add facts or residuals without
    overwriting a stored action preference.  If the planner owns a model bank,
    it retrieves the best factual slot before any caller-owned adaptation.
    Optional candidate entry tensors are passed to the planner's external
    value model, keeping growing memory files outside the controller while
    allowing their signed factual deltas to change the searched intention.
    """

    schema = POLICY_FREE_RUNTIME_SCHEMA

    def __init__(
        self,
        runtime: AmodalControllerRuntime,
        planner: ExternalModelBasedPlanner,
        *,
        state_adapter: ExternalControllerStateAdapter | None = None,
        intention_repertoire: ExternalIntentionRepertoire | None = None,
        intention_generator: ExternalOutcomeIntentionGenerator | None = None,
        intention_memory: ExternalOutcomeIntentionMemory | None = None,
        intention_router: ExternalOutcomeIntentionRouter | None = None,
        route_query_adapter: ExternalControllerTrajectoryQueryAdapter | None = None,
        entry_repertoire: ExternalEntryRepertoire | None = None,
        entry_binding_repertoire: ExternalEntryBindingRepertoire | None = None,
        goal_memory: ExternalGoalFragmentMemory | None = None,
        goal_stager: ExternalGoalFragmentStager | None = None,
        goal_route_evidence: PersistentOpaqueContextRouteEvidence | None = None,
        include_exploration_seed: bool = False,
    ) -> None:
        if not isinstance(runtime, AmodalControllerRuntime):
            raise TypeError("policy-free runtime requires an amodal controller runtime")
        if not isinstance(planner, ExternalModelBasedPlanner):
            raise TypeError("policy-free runtime requires an external model planner")
        expected_input_width = runtime.controller.width * 3
        selected_adapter = state_adapter or ExternalControllerStateAdapter(
            expected_input_width,
            planner.model.state_width,
        )
        if selected_adapter.controller_feature_width != expected_input_width:
            raise ValueError(
                "policy-free state adapter input width does not match controller"
            )
        if selected_adapter.state_width != planner.model.state_width:
            raise ValueError(
                "policy-free state adapter output width does not match planner"
            )
        if planner.model.intention_width != runtime.intention_width:
            raise ValueError(
                "policy-free planner intention width does not match runtime"
            )
        if intention_repertoire is not None and (
            intention_repertoire.width != runtime.intention_width
        ):
            raise ValueError("intention repertoire width does not match runtime")
        if intention_generator is not None and (
            intention_generator.intention_width != runtime.intention_width
            or intention_generator.context_width != planner.model.state_width
        ):
            raise ValueError(
                "intention generator dimensions do not match policy-free runtime"
            )
        if intention_generator is not None and intention_memory is not None:
            raise ValueError(
                "policy-free runtime accepts either a generator or intention memory"
            )
        if intention_router is not None and (
            intention_generator is not None or intention_memory is not None
        ):
            raise ValueError(
                "policy-free runtime accepts one external intention-memory source"
            )
        if intention_memory is not None and (
            intention_memory.intention_width != runtime.intention_width
            or intention_memory.context_width != planner.model.state_width
        ):
            raise ValueError(
                "intention memory dimensions do not match policy-free runtime"
            )
        if intention_router is not None and (
            intention_router.intention_width != runtime.intention_width
        ):
            raise ValueError(
                "intention router dimensions do not match policy-free runtime"
            )
        if route_query_adapter is not None and not isinstance(
            route_query_adapter, ExternalControllerTrajectoryQueryAdapter
        ):
            raise TypeError("route query adapter has the wrong type")
        if route_query_adapter is not None and intention_router is None:
            raise ValueError("route query adapter requires an intention router")
        if (
            route_query_adapter is None
            and intention_router is not None
            and (intention_router.context_width != planner.model.state_width)
        ):
            raise ValueError(
                "intention router context width requires a route query adapter"
            )
        if (
            route_query_adapter is not None
            and intention_router is not None
            and (
                route_query_adapter.controller_width != runtime.controller.width
                or route_query_adapter.query_width != intention_router.context_width
            )
        ):
            raise ValueError("route query adapter does not match controller or router")
        if entry_repertoire is not None:
            entry_value_model = planner.entry_value_model
            if entry_value_model is None:
                raise ValueError(
                    "entry repertoire requires an external entry value model"
                )
            if entry_repertoire.width != entry_value_model.entry_width:
                raise ValueError("entry repertoire width does not match planner")
        if entry_binding_repertoire is not None:
            entry_value_model = planner.entry_value_model
            if entry_value_model is None:
                raise ValueError(
                    "entry binding repertoire requires an external entry value model"
                )
            if entry_binding_repertoire.intention_width != runtime.intention_width:
                raise ValueError("entry binding intention width does not match runtime")
            if entry_binding_repertoire.entry_width != entry_value_model.entry_width:
                raise ValueError("entry binding entry width does not match planner")
        if (
            goal_memory is not None
            and goal_memory.state_width != planner.model.state_width
        ):
            raise ValueError("goal-fragment memory width does not match planner state")
        if (
            goal_memory is not None
            and goal_memory.state_space_id != planner.state_space_id
        ):
            raise ValueError("goal-fragment memory space does not match planner")
        if (
            goal_stager is not None
            and goal_stager.state_width != planner.model.state_width
        ):
            raise ValueError("goal-fragment stager width does not match planner state")
        if (
            goal_stager is not None
            and goal_stager.state_space_id != planner.state_space_id
        ):
            raise ValueError("goal-fragment stager space does not match planner")
        if goal_route_evidence is not None and not isinstance(
            goal_route_evidence, PersistentOpaqueContextRouteEvidence
        ):
            raise TypeError("goal route evidence has the wrong type")
        if goal_route_evidence is not None and goal_memory is None:
            raise ValueError("goal route evidence requires goal-fragment memory")
        if goal_route_evidence is not None and (
            goal_route_evidence.width != planner.model.state_width
        ):
            raise ValueError("goal route evidence width does not match planner state")
        if goal_route_evidence is not None and (
            goal_route_evidence.slot_count > goal_memory.fragment_count
        ):
            raise ValueError(
                "goal route evidence cannot address fragments absent from memory"
            )
        if not isinstance(include_exploration_seed, bool):
            raise TypeError("policy-free exploration-seed flag must be boolean")
        self.runtime = runtime
        self.planner = planner
        self.state_adapter = selected_adapter
        self.intention_repertoire = intention_repertoire
        self.intention_generator = intention_generator
        self.intention_memory = intention_memory
        self.intention_router = intention_router
        self.route_query_adapter = route_query_adapter
        self.entry_repertoire = entry_repertoire
        self.entry_binding_repertoire = entry_binding_repertoire
        self.goal_memory = goal_memory
        self.goal_stager = goal_stager
        self.goal_route_evidence = goal_route_evidence
        self.include_exploration_seed = include_exploration_seed
        self._sync_goal_route_slots()

    @property
    def controller(self) -> AmodalCognitiveController:
        return self.runtime.controller

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "behavior": "factual_model_search_no_stored_policy_v1",
            "controller_intention": "diagnostic_only_not_decoded_v1",
            "goal_input": "opaque_external_destination_state_or_goal_set_v1",
            "candidate_intentions": (
                "external_verified_repertoire_plus_learned_intention_router_v1"
                if self.intention_repertoire is not None
                and self.intention_router is not None
                else "learned_opaque_intention_router_candidate_v1"
                if self.intention_router is not None
                else "external_verified_repertoire_plus_outcome_intention_memory_v1"
                if self.intention_repertoire is not None
                and self.intention_memory is not None
                else "outcome_intention_memory_candidates_v1"
                if self.intention_memory is not None
                else "external_verified_repertoire_plus_outcome_generator_v1"
                if self.intention_repertoire is not None
                and self.intention_generator is not None
                else "outcome_generated_opaque_intention_v1"
                if self.intention_generator is not None
                else "external_logical_addressed_intention_repertoire_v1"
                if self.intention_repertoire is not None
                else "runtime_variable_opaque_caller_set_v1"
            ),
            "retrieval": (
                "goal_conditioned_bank_search_before_adaptation_v1"
                if isinstance(self.planner.model, ExternalTransitionModelBank)
                else "caller_selected_factual_model_v1"
            ),
            "runtime": self.runtime.configuration(),
            "planner": self.planner.configuration(),
            "state_adapter": self.state_adapter.configuration(),
            "intention_repertoire": (
                None
                if self.intention_repertoire is None
                else self.intention_repertoire.configuration()
            ),
            "intention_generator": (
                None
                if self.intention_generator is None
                else self.intention_generator.configuration()
            ),
            "intention_memory": (
                None
                if self.intention_memory is None
                else self.intention_memory.configuration()
            ),
            "intention_router": (
                None
                if self.intention_router is None
                else self.intention_router.configuration()
            ),
            "route_query_adapter": (
                None
                if self.route_query_adapter is None
                else self.route_query_adapter.configuration()
            ),
            "entry_repertoire": (
                None
                if self.entry_repertoire is None
                else self.entry_repertoire.configuration()
            ),
            "entry_binding_repertoire": (
                None
                if self.entry_binding_repertoire is None
                else self.entry_binding_repertoire.configuration()
            ),
            "goal_memory": (
                None if self.goal_memory is None else self.goal_memory.configuration()
            ),
            "goal_stager": (
                None if self.goal_stager is None else self.goal_stager.configuration()
            ),
            "goal_route_evidence": (
                None
                if self.goal_route_evidence is None
                else self.goal_route_evidence.payload()
            ),
            "include_exploration_seed": self.include_exploration_seed,
        }

    def _sync_goal_route_slots(self) -> None:
        """Keep route-slot width aligned with append-only goal memory."""

        if self.goal_route_evidence is None:
            return
        if self.goal_memory is None:
            raise RuntimeError("goal route evidence has no goal-fragment memory")
        while self.goal_route_evidence.slot_count < self.goal_memory.fragment_count:
            self.goal_route_evidence.append_slot()

    def goal_memory_state_payload(self) -> dict[str, object]:
        """Return the independently reloadable opaque goal-memory state."""

        if self.goal_memory is None:
            raise RuntimeError("policy-free runtime has no goal-fragment memory")
        return self.goal_memory.state_payload()

    def load_goal_memory_state_payload(self, payload: Mapping[str, object]) -> None:
        """Replace goal memory after validating its ABI and checksum.

        The candidate is parsed before the live reference changes.  Route
        evidence is retained only when every existing opaque slot remains
        addressable in the replacement memory; no controller parameter or
        working state is loaded by this operation.
        """

        restored = ExternalGoalFragmentMemory.from_payload(payload)
        if restored.state_width != self.planner.model.state_width:
            raise ValueError("goal-memory state width does not match planner")
        if restored.state_space_id != self.planner.state_space_id:
            raise ValueError("goal-memory state space does not match planner")
        if (
            self.goal_route_evidence is not None
            and self.goal_route_evidence.slot_count > restored.fragment_count
        ):
            raise ValueError(
                "replacement goal memory cannot drop routed fragment slots"
            )
        self.goal_memory = restored
        self._sync_goal_route_slots()

    def goal_route_state_payload(self) -> dict[str, object]:
        """Return independently reloadable opaque goal-route evidence."""

        if self.goal_route_evidence is None:
            raise RuntimeError("policy-free runtime has no goal route evidence")
        return self.goal_route_evidence.payload()

    def load_goal_route_state_payload(self, payload: Mapping[str, object]) -> None:
        """Replace route evidence after validating width and slot alignment."""

        if self.goal_memory is None:
            raise RuntimeError("goal route evidence has no goal-fragment memory")
        restored = PersistentOpaqueContextRouteEvidence.from_payload(dict(payload))
        if restored.width != self.planner.model.state_width:
            raise ValueError("goal-route context width does not match planner")
        if restored.slot_count > self.goal_memory.fragment_count:
            raise ValueError("goal-route evidence addresses absent goal fragments")
        self.goal_route_evidence = restored
        self._sync_goal_route_slots()

    def observe_goal_fragment_route(
        self,
        contexts: torch.Tensor,
        fragment_indices: int | torch.Tensor,
        outcomes: torch.Tensor | float,
    ) -> None:
        """Record scalar held-out route outcomes in external memory.

        The route learner sees only the learned context, an opaque fragment
        index, and a deterministic verifier scalar.  It never updates the
        controller or stores the underlying trajectory.
        """

        if self.goal_route_evidence is None:
            raise RuntimeError("policy-free runtime has no goal route evidence")
        self._sync_goal_route_slots()
        if contexts.ndim != 2:
            raise ValueError("goal route contexts must have shape [batch, width]")
        batch_size = contexts.shape[0]
        if isinstance(fragment_indices, int):
            slots = torch.full(
                (batch_size,),
                fragment_indices,
                dtype=torch.long,
                device=contexts.device,
            )
        else:
            slots = torch.as_tensor(fragment_indices, device=contexts.device)
            if slots.ndim == 0:
                slots = slots.expand(batch_size)
            if slots.ndim != 1 or slots.shape[0] != batch_size:
                raise ValueError(
                    "goal route fragment indices must have shape [batch]"
                )
            slots = slots.to(dtype=torch.long)
        outcome_values = torch.as_tensor(outcomes, device=contexts.device)
        if outcome_values.ndim == 0:
            outcome_values = outcome_values.expand(batch_size)
        if outcome_values.ndim != 1 or outcome_values.shape[0] != batch_size:
            raise ValueError("goal route outcomes must have shape [batch]")
        self.goal_route_evidence.observe_batch(
            contexts.detach(),
            slots.detach(),
            outcome_values.detach().to(dtype=torch.float32),
        )

    def transition_observation(
        self,
        output: PolicyFreeRuntimeOutput,
        successor: PolicyFreeRuntimeOutput,
        *,
        confidence: torch.Tensor | None = None,
    ) -> ExternalTransitionObservation:
        """Build one opaque self-supervised transition row bundle.

        ``output.intention`` is the model-derived intention that was exposed
        to the decoder; ``successor.state`` is the next learned planner state
        observed after the environment returned another event.  The helper
        deliberately accepts no raw action, reward label, task name, or
        protocol metadata.  External transition banks can consume the result
        while the controller remains frozen.
        """

        if not isinstance(output, PolicyFreeRuntimeOutput) or not isinstance(
            successor, PolicyFreeRuntimeOutput
        ):
            raise TypeError("transition observations need policy-free outputs")
        if output.state.ndim != 2 or successor.state.ndim != 2:
            raise ValueError("policy-free transition states must be two-dimensional")
        if output.state.shape != successor.state.shape:
            raise ValueError("policy-free transition states must have equal shapes")
        intention = output.intention.payload
        if intention.ndim != 2 or intention.shape[0] != output.state.shape[0]:
            raise ValueError("policy-free intention batch does not match state batch")
        if intention.shape[1] != self.runtime.intention_width:
            raise ValueError("policy-free intention width does not match runtime")
        observation = ExternalTransitionObservation(
            state=output.state.detach().clone(),
            intention=intention.detach().clone(),
            next_state=successor.state.detach().clone(),
            confidence=(
                None if confidence is None else confidence.detach().clone()
            ),
        )
        return observation.validate(
            state_width=self.planner.model.state_width,
            intention_width=self.runtime.intention_width,
        )

    def learn_transition_once(
        self,
        observation: ExternalTransitionObservation,
        context: torch.Tensor,
        optimizer: torch.optim.Optimizer | Mapping[int, torch.optim.Optimizer] | None = None,
    ) -> float:
        """Consume one fresh transition observation without replay.

        The planner must own an affine or random-feature
        :class:`ExternalTransitionModelBank`, whose update is a sufficient-
        statistics write rather than an optimizer replay loop.  Each opaque
        context is ensured before the bank consumes the observation; old slots
        are therefore not updated merely because a new context arrived.
        Neural transition slots are intentionally rejected here because a
        one-pass gradient update would not establish retention.
        """

        if not isinstance(self.planner.model, ExternalTransitionModelBank):
            raise TypeError(
                "replay-free transition learning requires a transition-model bank"
            )
        bank = self.planner.model
        if not bank.replay_free_updates:
            raise ValueError(
                "replay-free transition learning requires an affine or "
                "random-feature bank"
            )
        observation.validate(
            state_width=bank.state_width,
            intention_width=bank.intention_width,
        )
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.ndim != 2 or context.shape[1] != bank.context_width:
            raise ValueError(
                "transition learning context must have shape [batch, context_width]"
            )
        if context.shape[0] == 1 and observation.state.shape[0] > 1:
            context = context.expand(observation.state.shape[0], -1)
        elif context.shape[0] != observation.state.shape[0]:
            raise ValueError("transition learning context batch does not match data")
        for row in context:
            bank.ensure_context(row)
        return bank.adaptation_step(observation, context, optimizer)

    def observe_goal_fragment(
        self,
        candidate: ExternalGoalFragmentCandidate,
        outcome: torch.Tensor | float,
        *,
        eligible: bool = True,
    ) -> ExternalGoalFragmentObservationReceipt:
        """Record one fresh destination outcome in external staging state."""

        if self.goal_stager is None:
            raise RuntimeError("policy-free runtime has no goal-fragment stager")
        return self.goal_stager.observe(candidate, outcome, eligible=eligible)

    def goal_fragment_candidate_from_controller_output(
        self,
        controller_output: ControllerOutput,
        *,
        mask: torch.Tensor | None = None,
    ) -> ExternalGoalFragmentCandidate:
        """Project an opaque controller state into planner goal space.

        This is the canonical candidate boundary for frozen-core learning.
        The caller supplies no protocol value or semantic coordinate; the
        replaceable external state adapter performs the only representation
        conversion before the candidate enters staging.
        """

        if not isinstance(controller_output, ControllerOutput):
            raise TypeError("goal-fragment candidate needs controller output")
        model_state = self.state_adapter(controller_output)
        if model_state.shape[0] != 1:
            raise ValueError(
                "goal-fragment candidate derivation requires one controller row"
            )
        return ExternalGoalFragmentCandidate.from_state(model_state, mask=mask)

    def observe_goal_fragment_controller_output(
        self,
        controller_output: ControllerOutput,
        outcome: torch.Tensor | float,
        *,
        mask: torch.Tensor | None = None,
        eligible: bool = True,
    ) -> ExternalGoalFragmentObservationReceipt:
        """Stage a planner-space candidate derived from one learned state."""

        candidate = self.goal_fragment_candidate_from_controller_output(
            controller_output,
            mask=mask,
        )
        return self.observe_goal_fragment(candidate, outcome, eligible=eligible)

    def observe_goal_fragment_state(
        self,
        state: torch.Tensor,
        outcome: torch.Tensor | float,
        *,
        mask: torch.Tensor | None = None,
        eligible: bool = True,
    ) -> ExternalGoalFragmentObservationReceipt:
        """Stage one learned terminal state without a caller-side wrapper."""

        if self.goal_stager is None:
            raise RuntimeError("policy-free runtime has no goal-fragment stager")
        return self.goal_stager.observe_state(
            state,
            outcome,
            mask=mask,
            eligible=eligible,
        )

    def admit_goal_fragment_verified(
        self,
        candidate_digest: str,
        retention_probe: Callable[[ExternalGoalFragmentMemory], bool],
        *,
        reason: str = "caller_owned_heldout_goal_fragment_probe",
    ) -> ExternalGoalFragmentStagingAdmissionReceipt:
        """Promote a stable staged destination through external memory."""

        if self.goal_stager is None:
            raise RuntimeError("policy-free runtime has no goal-fragment stager")
        if self.goal_memory is None:
            raise RuntimeError("policy-free runtime has no goal-fragment memory")
        receipt = self.goal_stager.admit_verified(
            self.goal_memory,
            candidate_digest,
            retention_probe,
            reason=reason,
        )
        if receipt.accepted:
            self._sync_goal_route_slots()
        return receipt

    def observe_intention(
        self,
        intention: torch.Tensor | IntentEvent,
        *,
        utility: torch.Tensor | float | None = None,
        propensity: torch.Tensor | float | None = None,
        timestamp: torch.Tensor | int | None = None,
    ) -> ExternalIntentionObservationReceipt:
        """Commit post-execution opaque experience to external memory."""

        if self.intention_repertoire is None:
            raise RuntimeError("policy-free runtime has no intention repertoire")
        payload = intention.payload if isinstance(intention, IntentEvent) else intention
        return self.intention_repertoire.observe(
            payload,
            utility=utility,
            propensity=propensity,
            timestamp=timestamp,
        )

    def consolidate_intention_verified(
        self,
        retired_ids: tuple[int, ...] | list[int],
        replacement_intention: torch.Tensor,
        retention_probe: Callable[[ExternalIntentionRepertoire], bool],
        *,
        reason: str = "caller_owned_heldout_retention_probe",
    ) -> ExternalIntentionConsolidationReceipt:
        """Run retention-safe intention maintenance outside the controller."""

        if self.intention_repertoire is None:
            raise RuntimeError("policy-free runtime has no intention repertoire")
        return self.intention_repertoire.consolidate_verified(
            retired_ids,
            replacement_intention,
            retention_probe,
            reason=reason,
        )

    def apply_intention_generation_feedback(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        proposal: ExternalIntentionGenerationProposal,
        outcome: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
        terminal: torch.Tensor | None = None,
        baseline_override: torch.Tensor | None = None,
    ) -> ExternalOutcomeIntentionGeneratorState:
        """Apply verifier feedback to caller-owned external generator state."""

        if self.intention_generator is None:
            raise RuntimeError("policy-free runtime has no intention generator")
        return self.intention_generator.apply_feedback(
            state,
            outcome,
            present=present,
            terminal=terminal,
            baseline_override=baseline_override,
        )

    def record_intention_generation_decision(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        proposal: ExternalIntentionGenerationProposal,
        *,
        present: torch.Tensor | None = None,
    ) -> ExternalOutcomeIntentionGeneratorState:
        """Record a generated proposal before its verifier outcome arrives."""

        if self.intention_generator is None:
            raise RuntimeError("policy-free runtime has no intention generator")
        return self.intention_generator.record_decision(
            state,
            proposal,
            present=present,
        )

    def record_intention_memory_decision(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        proposal: ExternalIntentionMemoryProposal,
        selected_cells: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
    ) -> ExternalOutcomeIntentionGeneratorState:
        """Record the selected external memory cell for outcome credit."""

        if self.intention_memory is None:
            raise RuntimeError("policy-free runtime has no intention memory")
        return self.intention_memory.record_decision(
            state,
            proposal,
            selected_cells,
            present=present,
        )

    def apply_intention_memory_feedback(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        proposal: ExternalIntentionMemoryProposal,
        selected_cells: torch.Tensor,
        outcome: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
        terminal: torch.Tensor | None = None,
    ) -> ExternalOutcomeIntentionGeneratorState:
        """Apply delayed scalar feedback to the selected external memory cell."""

        if self.intention_memory is None:
            raise RuntimeError("policy-free runtime has no intention memory")
        return self.intention_memory.apply_feedback(
            state,
            proposal,
            selected_cells,
            outcome,
            present=present,
            terminal=terminal,
        )

    def record_intention_routing_decision(
        self,
        state: ExternalRoutedIntentionMemoryState,
        proposal: ExternalRoutedIntentionProposal,
        *,
        present: torch.Tensor | None = None,
    ) -> ExternalRoutedIntentionMemoryState:
        """Record an opaque router's sampled external cell."""

        if self.intention_router is None:
            raise RuntimeError("policy-free runtime has no intention router")
        return self.intention_router.record_decision(
            state,
            proposal,
            present=present,
        )

    def apply_intention_routing_feedback(
        self,
        state: ExternalRoutedIntentionMemoryState,
        proposal: ExternalRoutedIntentionProposal,
        outcome: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
        terminal: torch.Tensor | None = None,
    ) -> ExternalRoutedIntentionMemoryState:
        """Apply delayed verifier feedback to routed cell and route state."""

        if self.intention_router is None:
            raise RuntimeError("policy-free runtime has no intention router")
        return self.intention_router.apply_feedback(
            state,
            proposal,
            outcome,
            present=present,
            terminal=terminal,
        )

    def observe_entry(
        self,
        entry: torch.Tensor,
        *,
        utility: torch.Tensor | float | None = None,
        propensity: torch.Tensor | float | None = None,
        timestamp: torch.Tensor | int | None = None,
    ) -> ExternalEntryObservationReceipt:
        """Commit post-search opaque entry experience to external memory."""

        if self.entry_repertoire is None:
            raise RuntimeError("policy-free runtime has no entry repertoire")
        return self.entry_repertoire.observe(
            entry,
            utility=utility,
            propensity=propensity,
            timestamp=timestamp,
        )

    def observe_entry_binding(
        self,
        intention: torch.Tensor,
        entry: torch.Tensor,
        *,
        utility: torch.Tensor | float | None = None,
        propensity: torch.Tensor | float | None = None,
        timestamp: torch.Tensor | int | None = None,
    ) -> ExternalEntryBindingObservationReceipt:
        """Commit an atomic intention-entry outcome to external memory."""

        if self.entry_binding_repertoire is None:
            raise RuntimeError("policy-free runtime has no entry binding repertoire")
        return self.entry_binding_repertoire.observe(
            intention,
            entry,
            utility=utility,
            propensity=propensity,
            timestamp=timestamp,
        )

    def consolidate_entry_binding_verified(
        self,
        retired_ids: tuple[int, ...] | list[int],
        replacement_intention: torch.Tensor,
        replacement_entry: torch.Tensor,
        retention_probe: Callable[[ExternalEntryBindingRepertoire], bool],
        *,
        reason: str = "caller_owned_heldout_retention_probe",
    ) -> ExternalEntryBindingConsolidationReceipt:
        """Run retention-safe external binding maintenance outside the controller."""

        if self.entry_binding_repertoire is None:
            raise RuntimeError("policy-free runtime has no entry binding repertoire")
        return self.entry_binding_repertoire.consolidate_verified(
            retired_ids,
            replacement_intention,
            replacement_entry,
            retention_probe,
            reason=reason,
        )

    def step_events(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent],
        state: ControllerState,
        feedback: ControllerFeedback,
        goal_state: torch.Tensor | None,
        candidate_intentions: torch.Tensor | None = None,
        *,
        horizon: int,
        beam_width: int | None = None,
        transition_context: torch.Tensor | None = None,
        intention_costs: torch.Tensor | None = None,
        candidate_entries: torch.Tensor | None = None,
        entry_value_weight: float = 0.0,
        step_cost_weight: float = 0.0,
        goal_progress_weight: float = 0.0,
        elapsed: torch.Tensor | float = 1.0,
        disable_workspace: bool = False,
        memory_scope: torch.Tensor | None = None,
        sample_memory_writes: bool = False,
        memory_write_override: torch.Tensor | None = None,
        memory_write_uniform: torch.Tensor | None = None,
        memory_write_gradient: bool = True,
        goal_fragments: ExternalGoalFragmentSet | None = None,
        goal_fragment_indices: Sequence[int] | torch.Tensor | None = None,
        goal_composition: str = "union",
        generator_state: ExternalOutcomeIntentionGeneratorState | None = None,
        intention_memory_state: ExternalOutcomeIntentionGeneratorState | None = None,
        intention_router_state: ExternalRoutedIntentionMemoryState | None = None,
        intention_context_mask: torch.Tensor | None = None,
    ) -> tuple[PolicyFreeRuntimeOutput, ControllerState]:
        if goal_fragments is not None and goal_fragment_indices is not None:
            raise ValueError(
                "policy-free runtime accepts goal fragments or goal memory indices, not both"
            )
        if goal_fragment_indices is not None and self.goal_memory is None:
            raise ValueError("goal memory indices supplied without goal memory")
        collection = self.runtime.input_bus(events)
        controller_output, next_state = self.runtime.controller.step(
            collection,
            state,
            feedback,
            self.runtime.memory,
            elapsed=elapsed,
            disable_workspace=disable_workspace,
            memory_scope=memory_scope,
            sample_memory_writes=sample_memory_writes,
            memory_write_override=memory_write_override,
            memory_write_uniform=memory_write_uniform,
            memory_write_gradient=memory_write_gradient,
        )
        model_state = self.state_adapter(controller_output)
        route_query = (
            model_state
            if self.route_query_adapter is None
            else self.route_query_adapter(controller_output, next_state)
        )
        selected_goal_indices: torch.Tensor | None = None
        if goal_fragment_indices is not None:
            if self.goal_memory is None:
                raise RuntimeError("goal memory disappeared during goal resolution")
            if isinstance(goal_fragment_indices, torch.Tensor) and (
                goal_fragment_indices.ndim == 2
            ):
                goal_fragments = self.goal_memory.propose_per_batch(
                    goal_fragment_indices,
                    batch_size=model_state.shape[0],
                    composition=goal_composition,
                    device=model_state.device,
                    dtype=model_state.dtype,
                )
                selected_goal_indices = goal_fragment_indices.detach().clone()
            else:
                goal_fragments = self.goal_memory.propose(
                    goal_fragment_indices,
                    batch_size=model_state.shape[0],
                    composition=goal_composition,
                    device=model_state.device,
                    dtype=model_state.dtype,
                )
                if isinstance(goal_fragment_indices, torch.Tensor):
                    shared_indices = goal_fragment_indices.detach().to(
                        device=model_state.device, dtype=torch.long
                    )
                else:
                    shared_indices = torch.tensor(
                        tuple(int(index) for index in goal_fragment_indices),
                        device=model_state.device,
                        dtype=torch.long,
                    )
                selected_goal_indices = shared_indices.unsqueeze(0).expand(
                    model_state.shape[0], -1
                ).clone()
        elif (
            goal_state is None
            and goal_fragments is None
            and self.goal_route_evidence is not None
        ):
            if self.goal_memory is None:
                raise RuntimeError("goal route evidence has no goal-fragment memory")
            self._sync_goal_route_slots()
            if self.goal_route_evidence.slot_count < 1:
                raise ValueError("goal route evidence has no goal fragments to select")
            selected_goal_indices = self.goal_route_evidence.preferred_slots(
                model_state.detach()
            ).unsqueeze(1)
            goal_fragments = self.goal_memory.propose_per_batch(
                selected_goal_indices,
                batch_size=model_state.shape[0],
                composition=goal_composition,
                device=model_state.device,
                dtype=model_state.dtype,
            )
        if (goal_state is None) == (goal_fragments is None):
            raise ValueError(
                "policy-free runtime requires exactly one goal state or fragment set"
            )
        proposal = None
        intention_generation = None
        intention_memory_generation = None
        entry_proposal = None
        binding_proposal = None
        if (
            sum(
                value is not None
                for value in (
                    generator_state,
                    intention_memory_state,
                    intention_router_state,
                )
            )
            > 1
        ):
            raise ValueError(
                "policy-free runtime accepts one intention-generation state"
            )
        if generator_state is not None:
            if self.intention_generator is None:
                raise ValueError(
                    "generator state supplied without an intention generator"
                )
            intention_generation = self.intention_generator.propose(
                generator_state,
                model_state,
                context_mask=intention_context_mask,
            )
            if self.entry_binding_repertoire is not None:
                raise ValueError(
                    "intention generation cannot be paired with atomic entry bindings"
                )
        if intention_memory_state is not None:
            if self.intention_memory is None:
                raise ValueError(
                    "intention memory state supplied without an intention memory"
                )
            intention_memory_generation = self.intention_memory.propose(
                intention_memory_state,
                model_state,
                context_mask=intention_context_mask,
            )
            if self.entry_binding_repertoire is not None:
                raise ValueError(
                    "intention memory cannot be paired with atomic entry bindings"
                )
        if intention_router_state is not None:
            if self.intention_router is None:
                raise ValueError(
                    "intention router state supplied without an intention router"
                )
            intention_routing = self.intention_router.propose(
                intention_router_state,
                route_query,
                context_mask=intention_context_mask,
            )
            if self.entry_binding_repertoire is not None:
                raise ValueError(
                    "intention routing cannot be paired with atomic entry bindings"
                )
        else:
            intention_routing = None
        generated_only = False
        if (
            self.entry_binding_repertoire is not None
            and candidate_intentions is None
            and candidate_entries is None
        ):
            binding_proposal = self.entry_binding_repertoire.propose(
                device=model_state.device,
                dtype=model_state.dtype,
            )
            candidate_intentions = binding_proposal.intentions
            candidate_entries = binding_proposal.entries
        elif self.entry_binding_repertoire is not None and (
            candidate_intentions is None or candidate_entries is None
        ):
            raise ValueError(
                "entry binding repertoire requires both candidate intentions and entries"
            )
        if candidate_intentions is None:
            if (
                self.intention_repertoire is None
                and intention_generation is None
                and intention_memory_generation is None
                and intention_routing is None
            ):
                raise ValueError(
                    "candidate intentions require a repertoire or intention generator"
                )
            if self.intention_repertoire is not None:
                proposal = self.intention_repertoire.propose(
                    controller_output.intention.payload,
                    include_seed=(
                        self.include_exploration_seed
                        or self.intention_repertoire.record_count == 0
                    ),
                )
                candidate_intentions = proposal.intentions
            elif intention_memory_generation is not None:
                candidate_intentions = intention_memory_generation.intentions
                generated_only = True
            elif intention_routing is not None:
                candidate_intentions = intention_routing.selected_intentions.unsqueeze(
                    1
                )
                generated_only = True
            else:
                candidate_intentions = intention_generation.intentions.unsqueeze(1)
                generated_only = True
        generated_candidates = (
            None
            if intention_generation is None
            else intention_generation.intentions.unsqueeze(1)
        )
        if intention_memory_generation is not None:
            generated_candidates = intention_memory_generation.intentions
        if intention_routing is not None:
            generated_candidates = intention_routing.selected_intentions.unsqueeze(1)
        if generated_candidates is not None and not generated_only:
            if candidate_intentions.ndim == 2:
                candidate_intentions = candidate_intentions.unsqueeze(0).expand(
                    model_state.shape[0], -1, -1
                )
            candidate_intentions = torch.cat(
                (candidate_intentions, generated_candidates),
                dim=1,
            )
        if candidate_entries is None and self.entry_repertoire is not None:
            entry_proposal = self.entry_repertoire.propose(
                device=model_state.device,
                dtype=model_state.dtype,
            )
            candidate_entries = entry_proposal.entries
        if isinstance(self.planner.model, ExternalTransitionModelBank):
            if transition_context is not None:
                raise ValueError(
                    "bank-backed policy-free planning selects context internally"
                )
            selection = self.planner.select_bank_model(
                self.planner.model,
                model_state,
                goal_state,
                candidate_intentions,
                horizon=horizon,
                beam_width=beam_width,
                intention_costs=intention_costs,
                candidate_entries=candidate_entries,
                entry_value_weight=entry_value_weight,
                step_cost_weight=step_cost_weight,
                goal_fragments=goal_fragments,
            )
            planning = selection.planning
            selected_slot_id = selection.selected_slot_id
        else:
            planning = self.planner.plan(
                model_state,
                goal_state,
                candidate_intentions,
                horizon=horizon,
                beam_width=beam_width,
                transition_context=transition_context,
                intention_costs=intention_costs,
                candidate_entries=candidate_entries,
                entry_value_weight=entry_value_weight,
                step_cost_weight=step_cost_weight,
                goal_progress_weight=goal_progress_weight,
                goal_fragments=goal_fragments,
            )
            selected_slot_id = None
        planned_intention = IntentEvent(
            payload=planning.intentions[:, 0, :],
            confidence=controller_output.intention.confidence,
        ).validate(width=self.runtime.intention_width)
        return (
            PolicyFreeRuntimeOutput(
                controller=controller_output,
                planning=planning,
                intention=planned_intention,
                decoded=self.runtime.output_bus(planned_intention),
                state=model_state,
                goal_state=(
                    None if goal_state is None else goal_state.detach().clone()
                ),
                goal_fragments=goal_fragments,
                goal_fragment_indices=(
                    None
                    if selected_goal_indices is None
                    else selected_goal_indices.detach().clone()
                ),
                selected_slot_id=selected_slot_id,
                proposal=proposal,
                intention_generation=intention_generation,
                intention_memory_generation=intention_memory_generation,
                intention_routing=intention_routing,
                entry_proposal=entry_proposal,
                binding_proposal=binding_proposal,
            ),
            next_state,
        )
