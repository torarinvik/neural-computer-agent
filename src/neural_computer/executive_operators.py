"""Allow-listed generic operators for durable external-executive artifacts."""

from __future__ import annotations

import torch

from .executive import ExternalExecutiveOperator, TypedWorkspaceValue
from .interface import AmodalEventCollection

SINGLETON_EVENT_VALUE_INTERFACE = "neural-computer.singleton-event-value.v1"
VALUE_EQUALITY_EVIDENCE_INTERFACE = "neural-computer.value-equality-evidence.v1"
EVIDENCE_BINARY_INTENTION_INTERFACE = "neural-computer.evidence-binary-intention.v1"


class ExternalSingletonEventValueOperator(ExternalExecutiveOperator):
    """Expose one event only when exactly one stream is present per row."""

    def __init__(self, handle: int, *, width: int) -> None:
        if width < 1:
            raise ValueError("singleton event value width must be positive")
        self.width = int(width)
        super().__init__(
            handle,
            ("events",),
            "value",
            interface_version=SINGLETON_EVENT_VALUE_INTERFACE,
        )

    def configuration(self) -> dict[str, object]:
        return {**super().configuration(), "width": self.width}

    def execute(self, arguments: tuple[TypedWorkspaceValue, ...]) -> TypedWorkspaceValue:
        events = arguments[0].payload
        assert isinstance(events, AmodalEventCollection)
        batch, event_count, width = events.payload.shape
        if width != self.width:
            raise ValueError("singleton event value input width is incompatible")
        present_count = events.present.sum(dim=1)
        present = present_count == 1
        if event_count == 0:
            values = torch.zeros(batch, width, device=events.payload.device, dtype=events.payload.dtype)
            confidence = torch.zeros(batch, device=events.payload.device, dtype=events.payload.dtype)
        else:
            selected = events.present.to(torch.long).argmax(dim=1)
            rows = torch.arange(batch, device=events.payload.device)
            values = events.payload[rows, selected]
            confidence = events.confidence[rows, selected]
        return TypedWorkspaceValue.from_tensor(
            "value", values, present=present, confidence=confidence
        )


class ExternalValueEqualityEvidenceOperator(ExternalExecutiveOperator):
    """Generic equality relation over two opaque learned value tensors."""

    def __init__(self, handle: int) -> None:
        super().__init__(
            handle,
            ("value", "value"),
            "evidence",
            interface_version=VALUE_EQUALITY_EVIDENCE_INTERFACE,
        )

    def execute(self, arguments: tuple[TypedWorkspaceValue, ...]) -> TypedWorkspaceValue:
        left, right = arguments
        assert isinstance(left.payload, torch.Tensor)
        assert isinstance(right.payload, torch.Tensor)
        if left.payload.shape != right.payload.shape:
            raise ValueError("value equality inputs must have identical shapes")
        equal = torch.isclose(left.payload, right.payload).all(dim=1, keepdim=True)
        score = torch.where(
            equal,
            torch.ones_like(equal, dtype=left.payload.dtype),
            -torch.ones_like(equal, dtype=left.payload.dtype),
        )
        return TypedWorkspaceValue.from_tensor(
            "evidence",
            score,
            present=left.present & right.present,
            confidence=torch.minimum(left.confidence, right.confidence),
        )


class ExternalEvidenceBinaryIntentionOperator(ExternalExecutiveOperator):
    """Map signed evidence to two opaque intention alternatives."""

    def __init__(self, handle: int) -> None:
        super().__init__(
            handle,
            ("evidence",),
            "intention",
            interface_version=EVIDENCE_BINARY_INTENTION_INTERFACE,
        )

    def execute(self, arguments: tuple[TypedWorkspaceValue, ...]) -> TypedWorkspaceValue:
        evidence = arguments[0]
        assert isinstance(evidence.payload, torch.Tensor)
        return TypedWorkspaceValue.from_tensor(
            "intention",
            torch.cat((-evidence.payload, evidence.payload), dim=1),
            present=evidence.present,
            confidence=evidence.confidence,
        )


__all__ = [
    "EVIDENCE_BINARY_INTENTION_INTERFACE",
    "SINGLETON_EVENT_VALUE_INTERFACE",
    "VALUE_EQUALITY_EVIDENCE_INTERFACE",
    "ExternalEvidenceBinaryIntentionOperator",
    "ExternalSingletonEventValueOperator",
    "ExternalValueEqualityEvidenceOperator",
]
