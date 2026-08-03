"""Stable opaque neural-IR transport types.

The production package deliberately contains no device protocol, modality name,
task label, or action-count constant.  Payload coordinates are learned; this
module validates only transport shape and metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


EVENT_SCHEMA = "neural-computer.amodal-event.v2"
INTENTION_SCHEMA = "neural-computer.intent-event.v2"
EVENT_WINDOW_SCHEMA = "neural-computer.event-window.v1"
MEMORY_SCHEMA = "neural-computer.content-addressed-memory.v1"


def _validate_batch(value: torch.Tensor | None, batch: int, name: str) -> None:
    if value is not None and value.shape[0] != batch:
        raise ValueError(f"{name} batch does not match payload")


@dataclass(frozen=True)
class AmodalEvent:
    """One opaque learned event emitted by an external frontend."""

    payload: torch.Tensor
    source_key: torch.Tensor | None = None
    timestamp: torch.Tensor | None = None
    duration: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    schema: str = EVENT_SCHEMA

    def validate(self, *, width: int | None = None) -> AmodalEvent:
        if self.schema != EVENT_SCHEMA:
            raise ValueError(f"unsupported event schema: {self.schema}")
        if self.payload.ndim != 2:
            raise ValueError("event payload must have shape [batch, width]")
        if width is not None and self.payload.shape[1] != width:
            raise ValueError(
                f"event width {self.payload.shape[1]} does not match {width}"
            )
        batch = self.payload.shape[0]
        _validate_batch(self.source_key, batch, "source_key")
        _validate_batch(self.timestamp, batch, "timestamp")
        _validate_batch(self.duration, batch, "duration")
        _validate_batch(self.confidence, batch, "confidence")
        if self.duration is not None and torch.any(self.duration < 0):
            raise ValueError("event duration cannot be negative")
        if self.confidence is not None and torch.any(self.confidence < 0):
            raise ValueError("event confidence cannot be negative")
        return self


@dataclass(frozen=True)
class AmodalEventCollection:
    """A padded, runtime-variable event set with metadata kept per event.

    ``present`` is allowed to be all false.  That represents a valid quiet
    controller tick and is distinct from fabricated zero sensory evidence.
    ``source_key`` and ``duration`` stay on the event axis instead of being
    discarded by an early reduction.
    """

    payload: torch.Tensor
    present: torch.Tensor
    confidence: torch.Tensor
    source_key: torch.Tensor | None = None
    timestamp: torch.Tensor | None = None
    duration: torch.Tensor | None = None
    schema: str = EVENT_SCHEMA

    @classmethod
    def from_events(
        cls,
        events: Sequence[AmodalEvent],
        *,
        batch_size: int | None = None,
        width: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> AmodalEventCollection:
        if not events:
            if batch_size is None or width is None:
                raise ValueError(
                    "an empty collection requires batch_size and width"
                )
            payload = torch.zeros(batch_size, 0, width, device=device, dtype=dtype)
            return cls(
                payload=payload,
                present=torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
                confidence=torch.zeros(batch_size, 0, device=device, dtype=dtype),
            ).validate(width=width)

        validated = [event.validate(width=width) for event in events]
        event_width = validated[0].payload.shape[1]
        if any(event.payload.shape[1] != event_width for event in validated):
            raise ValueError("all event payloads must have the same width")
        payload = torch.stack([event.payload for event in validated], dim=1)
        batch = payload.shape[0]
        present = torch.ones(batch, len(events), dtype=torch.bool, device=payload.device)
        confidence = torch.stack(
            [
                (
                    event.confidence.reshape(batch)
                    if event.confidence is not None
                    else torch.ones(batch, dtype=payload.dtype, device=payload.device)
                )
                for event in validated
            ],
            dim=1,
        )

        def stack_optional(name: str) -> torch.Tensor | None:
            values = [getattr(event, name) for event in validated]
            if not any(value is not None for value in values):
                return None
            if not all(value is not None for value in values):
                raise ValueError(f"{name} must be supplied for every event or none")
            return torch.stack([value.reshape(batch, -1) for value in values], dim=1)

        source_key = stack_optional("source_key")
        timestamp = stack_optional("timestamp")
        duration = stack_optional("duration")
        if timestamp is not None and timestamp.shape[-1] != 1:
            raise ValueError("timestamps must have one value per event")
        if duration is not None and duration.shape[-1] != 1:
            raise ValueError("durations must have one value per event")
        return cls(
            payload=payload,
            present=present,
            confidence=confidence,
            source_key=source_key,
            timestamp=None if timestamp is None else timestamp.squeeze(-1),
            duration=None if duration is None else duration.squeeze(-1),
        ).validate(width=event_width)

    @classmethod
    def empty(
        cls,
        batch_size: int,
        width: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> AmodalEventCollection:
        """Create a quiet tick without fabricating an event token."""
        return cls.from_events(
            (), batch_size=batch_size, width=width, device=device, dtype=dtype
        )

    def validate(self, *, width: int | None = None) -> AmodalEventCollection:
        if self.schema != EVENT_SCHEMA:
            raise ValueError(f"unsupported event schema: {self.schema}")
        if self.payload.ndim != 3:
            raise ValueError("collection payload must have shape [batch, events, width]")
        batch, events, payload_width = self.payload.shape
        if width is not None and payload_width != width:
            raise ValueError(f"event width {payload_width} does not match {width}")
        if self.present.shape != (batch, events) or self.present.dtype != torch.bool:
            raise ValueError("presence mask must be boolean with shape [batch, events]")
        if self.confidence.shape != (batch, events):
            raise ValueError("confidence must have shape [batch, events]")
        if torch.any(self.confidence < 0):
            raise ValueError("event confidence cannot be negative")
        for name, value in (
            ("timestamp", self.timestamp),
            ("duration", self.duration),
        ):
            if value is not None and value.shape != (batch, events):
                raise ValueError(f"{name} must have shape [batch, events]")
        if self.source_key is not None:
            if self.source_key.ndim != 3 or self.source_key.shape[:2] != (batch, events):
                raise ValueError("source_key must have shape [batch, events, key_width]")
        if self.duration is not None and torch.any(self.duration < 0):
            raise ValueError("event duration cannot be negative")
        return self


@dataclass(frozen=True)
class EventTokenWindow:
    """Persistent event tokens retained across controller updates.

    The window is a bounded transport cache, not a semantic slot table.  The
    controller may attend to every present token; eviction is explicit FIFO
    transport policy when the configured capacity is exceeded.
    """

    payload: torch.Tensor
    present: torch.Tensor
    confidence: torch.Tensor
    timestamp: torch.Tensor
    timestamp_present: torch.Tensor
    duration: torch.Tensor
    age: torch.Tensor
    source_key: torch.Tensor | None = None
    schema: str = EVENT_WINDOW_SCHEMA

    @classmethod
    def empty(
        cls,
        batch_size: int,
        capacity: int,
        width: int,
        *,
        source_key_width: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> EventTokenWindow:
        return cls(
            payload=torch.zeros(batch_size, capacity, width, device=device, dtype=dtype),
            present=torch.zeros(batch_size, capacity, dtype=torch.bool, device=device),
            confidence=torch.zeros(batch_size, capacity, device=device, dtype=dtype),
            timestamp=torch.zeros(batch_size, capacity, device=device, dtype=dtype),
            timestamp_present=torch.zeros(
                batch_size, capacity, dtype=torch.bool, device=device
            ),
            duration=torch.zeros(batch_size, capacity, device=device, dtype=dtype),
            age=torch.zeros(batch_size, capacity, device=device, dtype=dtype),
            source_key=(
                torch.zeros(
                    batch_size,
                    capacity,
                    source_key_width,
                    device=device,
                    dtype=dtype,
                )
                if source_key_width
                else None
            ),
        ).validate(width=width, source_key_width=source_key_width)

    def validate(
        self,
        *,
        width: int | None = None,
        source_key_width: int = 0,
    ) -> EventTokenWindow:
        if self.schema != EVENT_WINDOW_SCHEMA:
            raise ValueError(f"unsupported event-window schema: {self.schema}")
        if self.payload.ndim != 3:
            raise ValueError("event-window payload must have shape [batch, tokens, width]")
        batch, tokens, payload_width = self.payload.shape
        if width is not None and payload_width != width:
            raise ValueError(f"event width {payload_width} does not match {width}")
        if self.present.shape != (batch, tokens) or self.present.dtype != torch.bool:
            raise ValueError("event-window presence must be boolean [batch, tokens]")
        for name, value in (
            ("confidence", self.confidence),
            ("timestamp", self.timestamp),
            ("timestamp_present", self.timestamp_present),
            ("duration", self.duration),
            ("age", self.age),
        ):
            if value.shape != (batch, tokens):
                raise ValueError(f"event-window {name} must have shape [batch, tokens]")
        if self.timestamp_present.dtype != torch.bool:
            raise ValueError("event-window timestamp_present must be boolean")
        if torch.any(self.confidence < 0) or torch.any(self.duration < 0):
            raise ValueError("event-window confidence and duration cannot be negative")
        if torch.any(self.age < 0):
            raise ValueError("event-window age cannot be negative")
        if source_key_width:
            if self.source_key is None or self.source_key.shape != (
                batch,
                tokens,
                source_key_width,
            ):
                raise ValueError("event-window source_key has the wrong shape")
        elif self.source_key is not None:
            raise ValueError("event-window source_key is disabled for this controller")
        return self


@dataclass(frozen=True)
class ControllerFeedback:
    """Opaque action/outcome feedback visible to the controller.

    ``action`` is a learned or externally encoded vector.  No discrete device
    action IDs or protocol cardinality enter the production controller.
    """

    action: torch.Tensor
    reward: torch.Tensor
    propensity: torch.Tensor
    has_feedback: torch.Tensor

    def validate(self, *, batch: int, action_width: int) -> ControllerFeedback:
        if self.action.ndim != 2 or self.action.shape != (batch, action_width):
            raise ValueError(f"action must have shape [{batch}, {action_width}]")
        for name, value in (
            ("reward", self.reward),
            ("propensity", self.propensity),
            ("has_feedback", self.has_feedback),
        ):
            if value.shape not in ((batch,), (batch, 1)):
                raise ValueError(f"{name} must have shape [{batch}] or [{batch}, 1]")
        if torch.any(self.propensity <= 0) or torch.any(self.propensity > 1):
            raise ValueError("feedback propensity must be in (0, 1]")
        if torch.any(self.has_feedback < 0) or torch.any(self.has_feedback > 1):
            raise ValueError("has_feedback must be in [0, 1]")
        return self


@dataclass(frozen=True)
class IntentEvent:
    """One opaque intention emitted by the cognitive controller."""

    payload: torch.Tensor
    timestamp: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    target_key: torch.Tensor | None = None
    schema: str = INTENTION_SCHEMA

    def validate(self, *, width: int | None = None) -> IntentEvent:
        if self.schema != INTENTION_SCHEMA:
            raise ValueError(f"unsupported intention schema: {self.schema}")
        if self.payload.ndim != 2:
            raise ValueError("intention payload must have shape [batch, width]")
        if width is not None and self.payload.shape[1] != width:
            raise ValueError(
                f"intention width {self.payload.shape[1]} does not match {width}"
            )
        batch = self.payload.shape[0]
        _validate_batch(self.timestamp, batch, "timestamp")
        _validate_batch(self.confidence, batch, "confidence")
        _validate_batch(self.target_key, batch, "target_key")
        return self
