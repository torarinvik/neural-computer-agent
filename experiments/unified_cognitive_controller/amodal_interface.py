"""Versioned transport objects for the extracted neural-IR boundary.

Only tensor shape and transport metadata are engineered here. Payload
coordinates deliberately have no assigned semantic meaning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

NEURAL_IR_SCHEMA = "neural-computer.amodal-event.v1"
INTENTION_SCHEMA = "neural-computer.intent-event.migration-v1"


@dataclass(frozen=True)
class AmodalEvent:
    """One opaque learned event emitted by an external input frontend."""

    payload: torch.Tensor
    source_key: torch.Tensor | None = None
    timestamp: torch.Tensor | None = None
    duration: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    schema: str = NEURAL_IR_SCHEMA

    def validate(self, *, width: int | None = None) -> AmodalEvent:
        if self.schema != NEURAL_IR_SCHEMA:
            raise ValueError(f"unsupported event schema: {self.schema}")
        if self.payload.ndim != 2:
            raise ValueError("event payload must have shape [batch, width]")
        if width is not None and self.payload.shape[1] != width:
            raise ValueError(
                f"event width {self.payload.shape[1]} does not match {width}"
            )
        batch = self.payload.shape[0]
        for name in ("source_key", "timestamp", "duration", "confidence"):
            value = getattr(self, name)
            if value is not None and value.shape[0] != batch:
                raise ValueError(f"{name} batch does not match event payload")
        return self


@dataclass(frozen=True)
class AmodalEventCollection:
    """Runtime-variable set of opaque events for one controller update.

    The event axis has no fixed modality meaning. ``present`` permits different
    examples in a batch to carry different cardinalities without padding being
    interpreted as sensory evidence.
    """

    payload: torch.Tensor
    present: torch.Tensor
    confidence: torch.Tensor
    timestamp: torch.Tensor | None = None
    schema: str = NEURAL_IR_SCHEMA

    @classmethod
    def from_events(cls, events: Sequence[AmodalEvent]) -> AmodalEventCollection:
        if not events:
            raise ValueError("an event collection requires at least one event")
        validated = [event.validate() for event in events]
        payload = torch.stack([event.payload for event in validated], dim=1)
        batch = payload.shape[0]
        present = torch.ones(
            batch, len(events), dtype=torch.bool, device=payload.device
        )
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
        timestamps = [event.timestamp for event in validated]
        timestamp = (
            torch.stack([value.reshape(batch) for value in timestamps], dim=1)
            if all(value is not None for value in timestamps)
            else None
        )
        return cls(payload, present, confidence, timestamp).validate()

    def validate(self, *, width: int | None = None) -> AmodalEventCollection:
        if self.schema != NEURAL_IR_SCHEMA:
            raise ValueError(f"unsupported event schema: {self.schema}")
        if self.payload.ndim != 3:
            raise ValueError(
                "collection payload must have shape [batch, events, width]"
            )
        batch, events, payload_width = self.payload.shape
        if width is not None and payload_width != width:
            raise ValueError(f"event width {payload_width} does not match {width}")
        if self.present.shape != (batch, events):
            raise ValueError("presence mask must have shape [batch, events]")
        if self.present.dtype != torch.bool:
            raise ValueError("presence mask must be boolean")
        if not torch.all(self.present.any(dim=1)):
            raise ValueError("every example requires at least one present event")
        if self.confidence.shape != (batch, events):
            raise ValueError("confidence must have shape [batch, events]")
        if torch.any(self.confidence < 0):
            raise ValueError("confidence cannot be negative")
        if self.timestamp is not None and self.timestamp.shape != (batch, events):
            raise ValueError("timestamp must have shape [batch, events]")
        return self


@dataclass(frozen=True)
class IntentEvent:
    """One opaque learned intention emitted by the cognitive controller.

    During the bit-identical migration, ``payload`` contains the inherited
    intention followed by a two-coordinate compatibility extension carrying a
    legacy action residual. The external decoder owns the interpretation. This
    extension is explicitly migration debt, not evidence that the old action
    branch was already amodal.
    """

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
        for name in ("timestamp", "confidence", "target_key"):
            value = getattr(self, name)
            if value is not None and value.shape[0] != batch:
                raise ValueError(f"{name} batch does not match intention payload")
        return self
