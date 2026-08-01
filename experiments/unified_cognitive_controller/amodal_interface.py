"""Versioned transport objects for the extracted neural-IR boundary.

Only tensor shape and transport metadata are engineered here. Payload
coordinates deliberately have no assigned semantic meaning.
"""

from __future__ import annotations

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
