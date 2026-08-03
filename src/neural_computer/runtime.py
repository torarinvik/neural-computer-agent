"""Production runtime for the amodal neural computer.

This module contains transport, buffering, encoder registration, controller
composition, and decoder fan-out.  It has no knowledge of a concrete action
vocabulary or device protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .controller import (
    AmodalCognitiveController,
    ControllerOutput,
    ControllerState,
)
from .interface import (
    EVENT_SCHEMA,
    INTENTION_SCHEMA,
    AmodalEvent,
    AmodalEventCollection,
    ControllerFeedback,
    IntentEvent,
)
from .memory import ContentAddressedMemory
from .policies import EventWaitPolicy


RUNTIME_FORMAT = "neural-computer.amodal-runtime.v1"


class OpaqueProtocolDecoder(nn.Module):
    """Independent backend from an intention vector to protocol outputs."""

    def __init__(self, intention_width: int, output_width: int, hidden: int = 0) -> None:
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


class AmodalInputBus(nn.Module):
    """Transport-preserving input bus.

    Binding is performed by the controller over event tokens.  The bus only
    validates and packages the runtime-variable set, so metadata cannot be
    silently lost in a pre-controller averaging step.
    """

    def __init__(self, event_width: int) -> None:
        super().__init__()
        if event_width < 1:
            raise ValueError("event_width must be positive")
        self.event_width = event_width

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

    def __init__(self, decoders: Mapping[str, nn.Module] | None = None) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict(dict(decoders or {}))

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
        wait_policy: EventWaitPolicy | None = None,
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
        for timestamp in sorted(tuple(self._pending)):
            bucket = self._pending[timestamp]
            complete = all(name in bucket for name in self.stream_names)
            expired = self.max_wait is not None and current - timestamp >= self.max_wait
            learned_release = False
            if self.wait_policy is not None and not complete and not expired:
                age = torch.tensor([max(0.0, current - timestamp)])
                present_fraction = torch.tensor(
                    [len(bucket) / len(self.stream_names)]
                )
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
        source_width = source_event.source_key.shape[-1] if source_event is not None else 0
        for name in self.stream_names:
            event = bucket.get(name)
            if event is None:
                payloads.append(torch.zeros_like(template.payload))
                present.append(False)
                confidences.append(torch.zeros(batch, device=template.payload.device, dtype=template.payload.dtype))
                timestamps.append(torch.full((batch,), timestamp, device=template.payload.device, dtype=template.payload.dtype))
                durations.append(torch.zeros(batch, device=template.payload.device, dtype=template.payload.dtype))
                if has_source_keys:
                    source_keys.append(torch.zeros(batch, source_width, device=template.payload.device, dtype=template.payload.dtype))
                continue
            event.validate(width=width)
            payloads.append(event.payload)
            present.append(True)
            confidences.append(
                event.confidence.reshape(batch)
                if event.confidence is not None
                else torch.ones(batch, device=event.payload.device, dtype=event.payload.dtype)
            )
            timestamps.append(
                event.timestamp.reshape(batch)
                if event.timestamp is not None
                else torch.full((batch,), timestamp, device=event.payload.device, dtype=event.payload.dtype)
            )
            durations.append(
                event.duration.reshape(batch)
                if event.duration is not None
                else torch.zeros(batch, device=event.payload.device, dtype=event.payload.dtype)
            )
            if has_source_keys:
                if event.source_key is None or event.source_key.shape[-1] != source_width:
                    raise ValueError("source_key must be present consistently in a window")
                source_keys.append(event.source_key.reshape(batch, source_width))

        collection = AmodalEventCollection(
            payload=torch.stack(payloads, dim=1),
            present=torch.tensor(present, dtype=torch.bool, device=template.payload.device).expand(batch, -1),
            confidence=torch.stack(confidences, dim=1),
            source_key=torch.stack(source_keys, dim=1) if has_source_keys else None,
            timestamp=torch.stack(timestamps, dim=1),
            duration=torch.stack(durations, dim=1) if has_durations else None,
        ).validate(width=width)
        return AmodalEventWindow(timestamp, collection, complete=complete)

    def pending_status(self, current_timestamp: float | None = None) -> tuple[AmodalEventWindowStatus, ...]:
        if not self._pending:
            return ()
        now = max(self._pending) if current_timestamp is None else float(current_timestamp)
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
        memory: ContentAddressedMemory | None = None,
        wait_policy: EventWaitPolicy | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.encoders = nn.ModuleDict(dict(encoders or {}))
        self.input_bus = input_bus or AmodalInputBus(controller.width)
        self.output_bus = output_bus or AmodalOutputBus()
        self.memory = memory
        self.wait_policy = wait_policy

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
                event = encoded if isinstance(encoded, AmodalEvent) else AmodalEvent(encoded)
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
    ) -> tuple[AmodalRuntimeOutput, ControllerState]:
        collection = self.input_bus(events)
        controller_output, next_state = self.controller.step(
            collection,
            state,
            feedback,
            self.memory,
            elapsed=elapsed,
            disable_workspace=disable_workspace,
        )
        intention = controller_output.intention
        return (
            AmodalRuntimeOutput(
                controller=controller_output,
                intention=intention,
                decoded=self.output_bus(intention),
            ),
            next_state,
        )

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
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ControllerState:
        return self.controller.initial_state(batch_size, device=device, dtype=dtype)

    def configuration(self) -> dict[str, object]:
        return {
            "format": RUNTIME_FORMAT,
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

    def checkpoint_payload(self, *, provenance: Mapping[str, object] | None = None) -> dict[str, object]:
        return {
            "format": RUNTIME_FORMAT,
            "event_schema": EVENT_SCHEMA,
            "intention_schema": INTENTION_SCHEMA,
            "configuration": self.configuration(),
            "provenance": dict(provenance or {}),
            "components": self.component_state_dicts(),
        }

    def load_component_state_dicts(self, components: Mapping[str, object]) -> None:
        expected = set(self.component_state_dicts())
        actual = set(components)
        if actual != expected:
            raise ValueError(f"component set mismatch: expected {sorted(expected)}, got {sorted(actual)}")
        self.controller.load_state_dict(components["controller"])
        self.input_bus.load_state_dict(components["input_bus"])
        for name, encoder in self.encoders.items():
            encoder.load_state_dict(components["encoders"][name])
        for name, decoder in self.output_bus.decoders.items():
            decoder.load_state_dict(components["decoders"][name])
        if self.memory is not None:
            self.memory.load_state_dict(components["memory"])
        if self.wait_policy is not None:
            self.wait_policy.load_state_dict(components["wait_policy"])
