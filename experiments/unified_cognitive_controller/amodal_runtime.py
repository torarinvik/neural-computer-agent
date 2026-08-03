"""Behavior-preserving extracted runtime for the amodal migration rung."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .amodal_interface import (
    INTENTION_SCHEMA,
    NEURAL_IR_SCHEMA,
    AmodalEvent,
    AmodalEventCollection,
    IntentEvent,
)
from .environment import ACTIONS
from .model import (
    ControllerCoreOutput,
    ControllerOutput,
    ControllerState,
    UnifiedCognitiveController,
    VisionEventEncoder,
)

EXTRACTED_CHECKPOINT_FORMAT = "neural-computer.extracted-amodal-runtime.v1"
CANONICAL_INTENTION_MIGRATION = "neural-computer.action-residual-to-intention.v1"


class ActionIntentDecoder(nn.Module):
    """External decoder reproducing the inherited opaque action protocol.

    Migration-v1 intentions append ``ACTIONS`` compatibility coordinates to
    the inherited intention. This decoder is the only component that knows how
    those coordinates affect the old protocol.
    """

    def __init__(self, projection: nn.Linear, *, intention_width: int) -> None:
        super().__init__()
        if projection.in_features != intention_width:
            raise ValueError("decoder projection and intention width disagree")
        if projection.out_features != ACTIONS:
            raise ValueError("decoder projection and action count disagree")
        self.projection = projection
        self.intention_width = intention_width

    @property
    def payload_width(self) -> int:
        return self.intention_width + ACTIONS

    def forward(self, intention: IntentEvent | torch.Tensor) -> torch.Tensor:
        payload = intention.payload if isinstance(intention, IntentEvent) else intention
        if isinstance(intention, IntentEvent):
            intention.validate(width=self.payload_width)
        elif payload.ndim != 2 or payload.shape[1] != self.payload_width:
            raise ValueError(f"intention must have shape [batch, {self.payload_width}]")
        return (
            self.projection(payload[:, : self.intention_width])
            + payload[:, self.intention_width :]
        )


class OpaqueProtocolDecoder(nn.Module):
    """Independent backend from base intention to opaque protocol commands."""

    def __init__(
        self, intention_width: int, commands: int = ACTIONS, hidden: int = 0
    ) -> None:
        super().__init__()
        if intention_width < 1 or commands < 2 or hidden < 0:
            raise ValueError("protocol decoder dimensions are invalid")
        self.intention_width = intention_width
        self.commands = commands
        self.network = (
            nn.Sequential(
                nn.Linear(intention_width, hidden),
                nn.GELU(),
                nn.Linear(hidden, commands),
            )
            if hidden
            else nn.Linear(intention_width, commands)
        )

    def forward(self, intention: IntentEvent | torch.Tensor) -> torch.Tensor:
        payload = intention.payload if isinstance(intention, IntentEvent) else intention
        if payload.ndim != 2 or payload.shape[1] < self.intention_width:
            raise ValueError("intention payload is too narrow for protocol decoder")
        return self.network(payload[:, : self.intention_width])


class FactorizedOpaqueProtocolDecoder(nn.Module):
    """Decode an opaque bitmask as independent protocol decisions.

    The bits have no semantic names here. This is useful when a verifier's
    action protocol is a product of independent binary choices: the decoder
    still emits one categorical distribution over masks, but its logit is the
    sum of the learned per-bit log-probabilities. This improves credit
    assignment without exposing verifier targets to the controller.
    """

    def __init__(self, intention_width: int, bits: int) -> None:
        super().__init__()
        if intention_width < 1 or bits < 1:
            raise ValueError("factorized decoder dimensions are invalid")
        self.intention_width = intention_width
        self.bits = bits
        self.commands = 1 << bits
        self.network = nn.Linear(intention_width, bits * 2)

    def forward(self, intention: IntentEvent | torch.Tensor) -> torch.Tensor:
        log_probs = self.binary_log_probs(intention)
        masks = torch.arange(
            self.commands, device=log_probs.device, dtype=torch.long)
        choices = torch.stack([
            (masks >> bit) & 1 for bit in range(self.bits)
        ], dim=-1)
        batch_indices = choices.unsqueeze(0).expand(log_probs.shape[0], -1, -1)
        bit_indices = batch_indices.unsqueeze(-1)
        selected = log_probs.unsqueeze(1).expand(
            -1, self.commands, -1, -1).gather(-1, bit_indices).squeeze(-1)
        return selected.sum(dim=-1)

    def binary_log_probs(
            self, intention: IntentEvent | torch.Tensor) -> torch.Tensor:
        """Return independent bit log-probabilities as [batch, bits, 2]."""
        payload = intention.payload if isinstance(intention, IntentEvent) else intention
        if payload.ndim != 2 or payload.shape[1] < self.intention_width:
            raise ValueError("intention payload is too narrow for protocol decoder")
        binary_logits = self.network(payload[:, :self.intention_width]).view(
            payload.shape[0], self.bits, 2)
        return torch.log_softmax(binary_logits, dim=-1)


class AmodalOutputBus(nn.Module):
    """Fan one intention out to a runtime-variable set of output backends."""

    def __init__(self, decoders: Mapping[str, nn.Module] | None = None) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict(dict(decoders or {}))

    def register_decoder(self, name: str, decoder: nn.Module) -> None:
        if not name or name in self.decoders:
            raise ValueError("decoder name must be nonempty and unique")
        self.decoders[name] = decoder

    def forward(self, intention: IntentEvent) -> dict[str, torch.Tensor]:
        return {name: decoder(intention) for name, decoder in self.decoders.items()}


class AmodalInputBus(nn.Module):
    """Combine a runtime-variable event set with generic learned attention."""

    def __init__(self, event_width: int, residual_hidden: int = 0,
                 second_moment: bool = False,
                 normalize_events: bool = False,
                 event_rms_target: float = 1.0) -> None:
        super().__init__()
        if event_width < 1 or residual_hidden < 0:
            raise ValueError("input-bus dimensions are invalid")
        if event_rms_target <= 0:
            raise ValueError("event RMS target must be positive")
        self.event_width = event_width
        self.second_moment = bool(second_moment)
        self.normalize_events = bool(normalize_events)
        self.event_rms_target = float(event_rms_target)
        self.relevance = nn.Linear(event_width, 1)
        # Uniform attention is a behavior-preserving starting point. Learning
        # may later prefer useful events using only verified behavioral reward.
        nn.init.zeros_(self.relevance.weight)
        nn.init.zeros_(self.relevance.bias)
        residual_input_width = event_width * 2
        if self.second_moment:
            residual_input_width += event_width * event_width
        self.residual = (
            nn.Sequential(
                nn.Linear(residual_input_width, residual_hidden),
                nn.GELU(),
                nn.Linear(residual_hidden, event_width),
            )
            if residual_hidden
            else None
        )
        if self.residual is not None:
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self, collection: AmodalEventCollection | Sequence[AmodalEvent]
    ) -> AmodalEvent:
        if not isinstance(collection, AmodalEventCollection):
            collection = AmodalEventCollection.from_events(collection)
        collection.validate(width=self.event_width)
        confidence = collection.confidence.to(collection.payload.dtype)
        payloads = collection.payload
        if self.normalize_events:
            # Independent frontends may have arbitrary coordinate scales. A
            # generic RMS normalization prevents one stream from drowning out
            # another at the amodal boundary, without assigning semantics to
            # either payload or changing the default behavior.
            rms = payloads.square().mean(dim=-1, keepdim=True).sqrt()
            payloads = payloads / rms.clamp_min(1e-6) * self.event_rms_target
        scores = self.relevance(payloads).squeeze(-1)
        scores = scores + torch.log(confidence.clamp_min(1e-8))
        scores = scores.masked_fill(~collection.present, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        payload = torch.einsum("be,bew->bw", weights, payloads)
        if self.residual is not None:
            centered = payloads - payload.unsqueeze(1)
            variance = torch.einsum("be,bew->bw", weights, centered.square())
            diversity = variance.mean(dim=-1, keepdim=True)
            # This exact-zero gate makes N=1 and identical duplicates permanent
            # compatibility invariants even after the generic set residual learns.
            diversity_gate = diversity / (diversity + 1e-4)
            residual_features = [payload, variance]
            if self.second_moment:
                second_moment = torch.einsum(
                    "be,bew,bex->bwx", weights, payloads, payloads)
                residual_features.append(second_moment.flatten(1))
            payload = payload + diversity_gate * self.residual(
                torch.cat(residual_features, dim=-1)
            )
        timestamp = (
            torch.einsum("be,be->b", weights, collection.timestamp)
            if collection.timestamp is not None
            else None
        )
        combined_confidence = torch.einsum("be,be->b", weights, confidence)
        return AmodalEvent(
            payload=payload,
            timestamp=timestamp,
            confidence=combined_confidence,
        ).validate(width=self.event_width)


class AmodalEventTimeline:
    """Align out-of-order encoded events using their generic timestamps.

    This is transport plumbing, not a task-specific rule. A future learned
    temporal policy may choose the tolerance, but the baseline must never merge
    events whose timestamps are outside the declared window.
    """

    def __init__(self, events: Sequence[AmodalEvent], tolerance: float = 0.0):
        if not events or tolerance < 0:
            raise ValueError("timeline requires events and nonnegative tolerance")
        validated = [event.validate() for event in events]
        if any(event.timestamp is None for event in validated):
            raise ValueError("timeline events require timestamps")
        batch = validated[0].payload.shape[0]
        width = validated[0].payload.shape[1]
        timestamps = []
        for event in validated:
            event.validate(width=width)
            value = event.timestamp.reshape(batch, -1)
            if value.shape[1] != 1:
                raise ValueError("timeline timestamps must have one value per batch")
            if not torch.equal(value, value[:1].expand_as(value)):
                raise ValueError("batched timeline timestamps must be uniform")
            timestamps.append(float(value[0, 0]))
        self.events = validated
        self.timestamps = timestamps
        self.tolerance = float(tolerance)

    def windows(self) -> list[AmodalEventCollection]:
        """Return stable timestamp windows, independent of arrival order."""
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


@dataclass(frozen=True)
class AmodalEventWindow:
    """One timestamp-aligned collection released by a streaming buffer."""

    timestamp: float
    collection: AmodalEventCollection
    complete: bool = True


@dataclass(frozen=True)
class AmodalEventWindowStatus:
    """Opaque transport metadata exposed to a learned wait policy."""

    timestamp: float
    age: float
    present: tuple[bool, ...]
    complete: bool


class AmodalEventWindowBuffer:
    """Buffer streams until a complete or bounded-expired timestamp window.

    ``stream_names`` are runtime transport handles, not semantic modality
    labels. The handles never enter event payloads. The buffer only uses each
    event's generic timestamp and can therefore impose a bounded wait on any
    combination of encoders without knowing what they represent. With
    ``max_wait`` set, the value is measured in the same opaque timestamp units
    and expired windows carry an explicit presence mask rather than fabricated
    sensory content.
    """

    def __init__(
        self,
        stream_names: Sequence[str],
        *,
        tolerance: float = 0.0,
        max_wait: float | None = None,
    ) -> None:
        names = tuple(stream_names)
        if not names or any(not name for name in names):
            raise ValueError("stream_names must be nonempty")
        if len(set(names)) != len(names):
            raise ValueError("stream_names must be unique")
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        if max_wait is not None and max_wait < 0:
            raise ValueError("max_wait must be nonnegative")
        self.stream_names = names
        self.tolerance = float(tolerance)
        self.max_wait = None if max_wait is None else float(max_wait)
        self._pending: dict[float, dict[str, AmodalEvent]] = {}

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
        """Add any newly arrived events and release complete windows in order."""
        if not arrivals:
            raise ValueError("at least one event arrival is required")
        for name, event in arrivals.items():
            if name not in self.stream_names:
                raise KeyError(f"unknown buffered stream {name!r}")
            timestamp = self._timestamp(event)
            slot = self._slot(timestamp)
            bucket = self._pending.setdefault(slot, {})
            if name in bucket:
                raise ValueError("duplicate event for one stream/timestamp")
            bucket[name] = event
        ready: list[AmodalEventWindow] = []
        current_timestamp = max(self._pending)
        for timestamp in sorted(tuple(self._pending)):
            bucket = self._pending[timestamp]
            complete = all(name in bucket for name in self.stream_names)
            expired = (
                self.max_wait is not None
                and current_timestamp - timestamp >= self.max_wait
            )
            if not complete and not expired:
                continue
            ready.append(self._release(timestamp, bucket, complete=complete))
            del self._pending[timestamp]
        ready.sort(key=lambda window: window.timestamp)
        return ready

    def _release(
        self,
        timestamp: float,
        bucket: Mapping[str, AmodalEvent],
        *,
        complete: bool,
    ) -> AmodalEventWindow:
        if complete:
            events = [bucket[name] for name in self.stream_names]
            collection = AmodalEventCollection.from_events(events)
            return AmodalEventWindow(timestamp, collection, complete=True)
        template = next(iter(bucket.values()))
        batch, width = template.payload.shape
        payloads = []
        present = []
        confidence = []
        timestamps = []
        for name in self.stream_names:
            event = bucket.get(name)
            if event is None:
                payloads.append(torch.zeros_like(template.payload))
                present.append(False)
                confidence.append(
                    torch.zeros(
                        batch,
                        dtype=template.payload.dtype,
                        device=template.payload.device,
                    )
                )
                timestamps.append(torch.full(
                    (batch,), timestamp, device=template.payload.device
                ))
            else:
                event.validate(width=width)
                payloads.append(event.payload)
                present.append(True)
                confidence.append(
                    event.confidence.reshape(batch)
                    if event.confidence is not None
                    else torch.ones(
                        batch,
                        dtype=event.payload.dtype,
                        device=event.payload.device,
                    )
                )
                timestamps.append(
                    event.timestamp.reshape(batch)
                    if event.timestamp is not None
                    else torch.full(
                        (batch,), timestamp, device=event.payload.device
                    )
                )
        collection = AmodalEventCollection(
            payload=torch.stack(payloads, dim=1),
            present=torch.tensor(
                [present], dtype=torch.bool, device=template.payload.device
            ).expand(batch, -1),
            confidence=torch.stack(confidence, dim=1),
            timestamp=torch.stack(timestamps, dim=1),
        ).validate(width=width)
        return AmodalEventWindow(timestamp, collection, complete=False)

    def pending_status(
        self, current_timestamp: float | None = None
    ) -> tuple[AmodalEventWindowStatus, ...]:
        """Expose generic presence/age metadata without exposing event content."""
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
                present=tuple(name in self._pending[timestamp] for name in self.stream_names),
                complete=all(
                    name in self._pending[timestamp] for name in self.stream_names
                ),
            )
            for timestamp in sorted(self._pending)
        )

    def release_pending(self, timestamp: float) -> AmodalEventWindow:
        """Release one pending window at a caller-selected transport deadline."""
        timestamp = float(timestamp)
        if timestamp not in self._pending:
            raise KeyError(f"no pending timestamp {timestamp}")
        bucket = self._pending.pop(timestamp)
        complete = all(name in bucket for name in self.stream_names)
        return self._release(timestamp, bucket, complete=complete)

    @property
    def pending_timestamps(self) -> tuple[float, ...]:
        """Return incomplete timestamp windows in chronological order."""
        return tuple(sorted(self._pending))


@dataclass(frozen=True)
class AmodalRuntimeOutput:
    """One controller update plus every registered decoder result.

    ``core`` and ``intention`` remain available for memory and audit code,
    while ``decoded`` is a runtime-sized mapping.  Decoder names are transport
    handles only; they are never appended to the controller event payload.
    """

    core: ControllerCoreOutput
    intention: IntentEvent
    decoded: dict[str, torch.Tensor]


class AmodalControllerRuntime(nn.Module):
    """The first complete ``N encoders -> one controller -> M decoders`` rung.

    Encoder names select frontends at the process boundary only.  Each
    frontend must emit either an :class:`AmodalEvent` or a ``[batch, width]``
    tensor; the controller sees only the resulting opaque event collection.
    The input bus is permutation-invariant for simultaneous events by default,
    and the output bus can be expanded without changing controller shapes.

    This wrapper is deliberately additive: the legacy and one-event
    ``ExtractedAmodalRuntime`` paths remain unchanged and bit-identical.
    """

    def __init__(
        self,
        controller: UnifiedCognitiveController,
        *,
        encoders: Mapping[str, nn.Module] | None = None,
        input_bus: AmodalInputBus | None = None,
        output_bus: AmodalOutputBus | None = None,
        intention_basis: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if controller.vision is not None or controller.actuator is not None:
            raise ValueError(
                "amodal controller runtime requires extracted controller adapters"
            )
        if controller.action_adapter_into_intention and intention_basis is None:
            raise ValueError(
                "an external intention basis is required for action-to-intention adapters"
            )
        self.controller = controller
        self.encoders = nn.ModuleDict(dict(encoders or {}))
        self.input_bus = input_bus or AmodalInputBus(controller.width)
        self.output_bus = output_bus or AmodalOutputBus()
        self.register_buffer(
            "intention_basis",
            intention_basis.detach().clone()
            if intention_basis is not None
            else torch.empty(0),
            persistent=False,
        )

    @property
    def event_width(self) -> int:
        return self.controller.width

    @property
    def intention_width(self) -> int:
        return self.controller.intention_width

    def register_encoder(self, name: str, encoder: nn.Module) -> None:
        """Attach another frontend without changing the controller."""
        if not name or name in self.encoders:
            raise ValueError("encoder name must be nonempty and unique")
        self.encoders[name] = encoder

    def register_decoder(self, name: str, decoder: nn.Module) -> None:
        """Attach another backend without changing the controller."""
        self.output_bus.register_decoder(name, decoder)

    def encode_streams(
        self,
        streams: Mapping[str, torch.Tensor | AmodalEvent],
    ) -> list[AmodalEvent]:
        """Lower any nonempty set of named raw streams into opaque events.

        The mapping key is used only to choose an encoder.  It is not embedded
        in the event and therefore cannot become a hidden semantic task label.
        Pre-encoded events are accepted for sensors whose frontend lives
        outside this process.
        """
        if not streams:
            raise ValueError("at least one input stream is required")
        events: list[AmodalEvent] = []
        batch: int | None = None
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
                    else AmodalEvent(payload=encoded)
                )
            event.validate(width=self.event_width)
            if batch is None:
                batch = event.payload.shape[0]
            elif batch != event.payload.shape[0]:
                raise ValueError("all input streams must share the batch size")
            events.append(event)
        return events

    def combine_events(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent],
    ) -> AmodalEvent:
        """Combine simultaneous events at the opaque neural-IR boundary."""
        return self.input_bus(events)

    def step_events(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent],
        state: ControllerState,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        has_feedback: torch.Tensor,
        retrieved_memory: torch.Tensor | None = None,
        *,
        disable_workspace: bool = False,
    ) -> tuple[AmodalRuntimeOutput, ControllerState]:
        event = self.combine_events(events)
        basis = self.intention_basis if self.intention_basis.numel() else None
        core, next_state = self.controller.step_event(
            event,
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
            intention_basis=basis,
        )
        intention = core.intent_event
        return (
            AmodalRuntimeOutput(
                core=core,
                intention=intention,
                decoded=self.output_bus(intention),
            ),
            next_state,
        )

    def step_streams(
        self,
        streams: Mapping[str, torch.Tensor | AmodalEvent],
        state: ControllerState,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        has_feedback: torch.Tensor,
        retrieved_memory: torch.Tensor | None = None,
        *,
        disable_workspace: bool = False,
    ) -> tuple[AmodalRuntimeOutput, ControllerState]:
        """Encode and process any number of simultaneous streams."""
        return self.step_events(
            self.encode_streams(streams),
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ControllerState:
        return self.controller.initial_state(batch_size, device=device, dtype=dtype)


class ExtractedAmodalRuntime(nn.Module):
    """External encoder, one controller core, and external decoder.

    The three children own disjoint parameters and serialize independently.
    This migration runtime accepts one event per step. Variable-N input is a
    later rung; variable-M output is provided by the external output bus.
    """

    def __init__(
        self,
        encoder: VisionEventEncoder,
        controller: UnifiedCognitiveController,
        decoder: ActionIntentDecoder,
    ) -> None:
        super().__init__()
        if controller.vision is not None or controller.actuator is not None:
            raise ValueError("extracted controller must not own input/output adapters")
        if encoder.network[-1].normalized_shape != (controller.width,):
            raise ValueError("encoder and controller event widths disagree")
        if decoder.intention_width != controller.intention_width:
            raise ValueError("controller and decoder intention widths disagree")
        self.encoder = encoder
        self.controller = controller
        self.decoder = decoder

    @classmethod
    def from_legacy(
        cls, model: UnifiedCognitiveController, *, copy_model: bool = True
    ) -> ExtractedAmodalRuntime:
        """Extract adapters from a legacy model without changing its weights."""
        controller = copy.deepcopy(model) if copy_model else model
        encoder = controller.vision
        actuator = controller.actuator
        if encoder is None or actuator is None:
            raise ValueError("model is already missing its legacy adapters")
        controller.vision = None
        controller.actuator = None
        decoder = ActionIntentDecoder(
            actuator, intention_width=controller.intention_width
        )
        return cls(encoder, controller, decoder)

    def encode(self, frame: torch.Tensor) -> AmodalEvent:
        return AmodalEvent(payload=self.encoder(frame)).validate(
            width=self.controller.width
        )

    @property
    def compatibility_suffix_active(self) -> bool:
        """Whether the controller can emit a nonzero migration suffix."""
        return bool(
            self.controller.action_adapter is not None
            and not self.controller.action_adapter_emits_intention
            and not self.controller.action_adapter_into_intention
        )

    def decode(self, intention: IntentEvent) -> torch.Tensor:
        return self.decoder(intention)

    def step_intention_event(
        self,
        event: AmodalEvent,
        state: ControllerState,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        has_feedback: torch.Tensor,
        retrieved_memory: torch.Tensor | None = None,
        *,
        disable_workspace: bool = False,
    ) -> tuple[ControllerCoreOutput, ControllerState]:
        """Run the controller without selecting or formatting an output."""
        return self.controller.step_event(
            event,
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
            intention_basis=self.decoder.projection.weight,
        )

    def step_intention(
        self,
        frame: torch.Tensor,
        state: ControllerState,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        has_feedback: torch.Tensor,
        retrieved_memory: torch.Tensor | None = None,
        *,
        disable_workspace: bool = False,
    ) -> tuple[ControllerCoreOutput, ControllerState]:
        return self.step_intention_event(
            self.encode(frame),
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
        )

    def step_event(
        self,
        event: AmodalEvent,
        state: ControllerState,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        has_feedback: torch.Tensor,
        retrieved_memory: torch.Tensor | None = None,
        *,
        disable_workspace: bool = False,
    ) -> tuple[ControllerOutput, ControllerState]:
        core_output, next_state = self.step_intention_event(
            event,
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
        )
        return self._decode_core_output(core_output), next_state

    def step(
        self,
        frame: torch.Tensor,
        state: ControllerState,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        has_feedback: torch.Tensor,
        retrieved_memory: torch.Tensor | None = None,
        *,
        disable_workspace: bool = False,
    ) -> tuple[ControllerOutput, ControllerState]:
        return self.step_event(
            self.encode(frame),
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ControllerState:
        return self.controller.initial_state(batch_size, device=device, dtype=dtype)

    def _decode_core_output(self, output: ControllerCoreOutput) -> ControllerOutput:
        return ControllerOutput(
            logits=self.decode(output.intent_event),
            intention=output.legacy_intention,
            memory_key=output.memory_key,
            memory_value=output.memory_value,
            memory_write_strength=output.memory_write_strength,
            workspace_read=output.workspace_read,
            skill_adapter_openings=output.skill_adapter_openings,
            skill_adapter_residual_norms=output.skill_adapter_residual_norms,
        )

    def component_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return independently loadable encoder/core/decoder weights."""
        return {
            "encoder_state_dict": self.encoder.state_dict(),
            "controller_state_dict": self.controller.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
        }

    def legacy_state_dict(self) -> dict[str, torch.Tensor]:
        """Reassemble the exact key layout expected by legacy checkpoints."""
        state = dict(self.controller.state_dict())
        state.update(
            {
                f"vision.{name}": value
                for name, value in self.encoder.state_dict().items()
            }
        )
        state.update(
            {
                f"actuator.{name.removeprefix('projection.')}": value
                for name, value in self.decoder.state_dict().items()
            }
        )
        return state

    def extracted_payload(
        self,
        *,
        model_configuration: Mapping[str, Any],
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "format": EXTRACTED_CHECKPOINT_FORMAT,
            "event_schema": NEURAL_IR_SCHEMA,
            "intention_schema": INTENTION_SCHEMA,
            "model_configuration": dict(model_configuration),
            "provenance": dict(provenance or {}),
            **self.component_state_dicts(),
        }


def runtime_from_legacy_payload(
    payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
) -> ExtractedAmodalRuntime:
    """Load an ordinary historical checkpoint through the extracted runtime."""
    configuration = payload.get("model_configuration")
    state = payload.get("state_dict")
    if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
        raise TypeError("legacy payload lacks model configuration or state dict")
    model = UnifiedCognitiveController(**dict(configuration)).to(device)
    _load_controller_state_compatibly(model, state)
    return ExtractedAmodalRuntime.from_legacy(model, copy_model=False)


def runtime_from_extracted_payload(
    payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
) -> ExtractedAmodalRuntime:
    """Load independently serialized components into an extracted runtime."""
    if payload.get("format") != EXTRACTED_CHECKPOINT_FORMAT:
        raise ValueError("unsupported extracted checkpoint format")
    if payload.get("event_schema") != NEURAL_IR_SCHEMA:
        raise ValueError("unsupported extracted event schema")
    if payload.get("intention_schema") != INTENTION_SCHEMA:
        raise ValueError("unsupported extracted intention schema")
    configuration = payload.get("model_configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError("extracted payload lacks model configuration")
    blank = UnifiedCognitiveController(**dict(configuration)).to(device)
    runtime = ExtractedAmodalRuntime.from_legacy(blank, copy_model=False)
    runtime.encoder.load_state_dict(payload["encoder_state_dict"])
    _load_controller_state_compatibly(runtime.controller, payload["controller_state_dict"])
    runtime.decoder.load_state_dict(payload["decoder_state_dict"])
    return runtime


def _load_controller_state_compatibly(
    model: UnifiedCognitiveController,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Load checkpoints across zero-initialized optional controller additions.

    The critic-scale parameters were added after the promoted amodal
    checkpoint was created. They are zero-initialized and have no effect until
    explicitly trained, so an older checkpoint may omit only those keys. Any
    other missing or unexpected key remains a hard error; this is deliberately
    not a general ``strict=False`` escape hatch.
    """
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {
        name
        for name in incompatible.missing_keys
        if name.startswith("skill_adapter_critic_scales.")
    }
    unexpected = set(incompatible.unexpected_keys)
    remaining_missing = set(incompatible.missing_keys) - allowed_missing
    if remaining_missing or unexpected:
        details = []
        if remaining_missing:
            details.append(f"missing keys: {sorted(remaining_missing)}")
        if unexpected:
            details.append(f"unexpected keys: {sorted(unexpected)}")
        raise RuntimeError("incompatible controller checkpoint (" + "; ".join(details) + ")")


def canonicalize_action_adapter_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fold a legacy two-logit residual into the base intention coordinates.

    The conversion is algebraic and uses no examples, labels, rewards, or
    optimizer updates. For actuator matrix ``W`` it computes the minimum-norm
    right inverse ``P = (W W^T)^-1 W`` and folds ``residual @ P`` into the last
    action-adapter layer. The external decoder then recovers the same residual
    through ``(residual @ P) @ W^T``.
    """
    configuration = payload.get("model_configuration")
    source_state = payload.get("state_dict")
    if not isinstance(configuration, Mapping) or not isinstance(source_state, Mapping):
        raise TypeError("checkpoint lacks model configuration or state dict")
    source_configuration = dict(configuration)
    if not int(source_configuration.get("action_adapter_width", 0)):
        raise ValueError("checkpoint has no action adapter to canonicalize")
    if source_configuration.get("action_adapter_emits_intention", False):
        raise ValueError("action adapter already emits an intention residual")
    if source_configuration.get("action_adapter_into_intention", False):
        raise ValueError(
            "runtime-projected action adapter must be normalized separately"
        )

    actuator_weight = source_state["actuator.weight"]
    old_weight = source_state["action_adapter.2.weight"]
    old_bias = source_state["action_adapter.2.bias"]
    if (
        actuator_weight.ndim != 2
        or old_weight.shape[0] != ACTIONS
        or old_bias.shape != (ACTIONS,)
    ):
        raise ValueError("checkpoint has an unsupported legacy adapter shape")
    calculation_dtype = (
        torch.float64
        if actuator_weight.dtype in (torch.float16, torch.float32)
        else actuator_weight.dtype
    )
    decoder = actuator_weight.to(calculation_dtype)
    gram = decoder @ decoder.T
    if int(torch.linalg.matrix_rank(gram)) != ACTIONS:
        raise ValueError("actuator has no full-row-rank intention right inverse")
    right_inverse = torch.linalg.solve(gram, decoder)

    state = {name: value.detach().clone() for name, value in source_state.items()}
    state["action_adapter.2.weight"] = (
        right_inverse.T @ old_weight.to(calculation_dtype)
    ).to(old_weight.dtype)
    state["action_adapter.2.bias"] = (
        old_bias.to(calculation_dtype) @ right_inverse
    ).to(old_bias.dtype)
    migrated_configuration = dict(source_configuration)
    migrated_configuration["action_adapter_emits_intention"] = True
    migrated_configuration["action_adapter_into_intention"] = False

    # Strict construction is part of conversion: a malformed state never
    # becomes a candidate checkpoint.
    candidate = UnifiedCognitiveController(**migrated_configuration)
    candidate.load_state_dict(state)
    return {
        "schema": payload.get("schema", "unified-cognitive-controller-v1"),
        "model_configuration": migrated_configuration,
        "state_dict": candidate.state_dict(),
        "migration": {
            "format": CANONICAL_INTENTION_MIGRATION,
            "source_action_adapter_output_width": ACTIONS,
            "target_action_adapter_output_width": candidate.intention_width,
            "uses_examples": False,
            "uses_optimizer_updates": False,
        },
    }


def canonicalize_action_adapter_checkpoint(
    source: Path, destination: Path, *, device: torch.device | str = "cpu"
) -> None:
    payload = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("legacy checkpoint must contain a mapping")
    canonical = canonicalize_action_adapter_payload(payload)
    canonical["migration"]["source_checkpoint"] = str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(canonical, destination)


def convert_legacy_checkpoint(
    source: Path, destination: Path, *, device: torch.device | str = "cpu"
) -> None:
    """Convert one checkpoint without training or changing any tensor value."""
    payload = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("legacy checkpoint must contain a mapping")
    runtime = runtime_from_legacy_payload(payload, device=device)
    extracted = runtime.extracted_payload(
        model_configuration=payload["model_configuration"],
        provenance={"legacy_checkpoint": str(source)},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extracted, destination)
