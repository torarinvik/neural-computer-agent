"""Generic persistent-memory operators for the external executive."""

from __future__ import annotations

import torch

from .executive import (
    ExternalExecutiveOperator,
    ExternalExecutiveOperatorState,
    TypedWorkspaceValue,
)

EXTERNAL_VALUE_DELAY_INTERFACE = "neural-computer.external-value-delay.v1"


class ExternalValueDelayOperator(ExternalExecutiveOperator):
    """Return a value from a fixed relative displacement and retain the present.

    ``delay`` is an opaque external-program binding, not a task ID or controller
    input. A delay of one returns the value from the preceding CALL. Missing
    history remains explicitly absent. State is per executive, not stored in
    this operator object, so one reusable operator can safely serve many agents.
    """

    def __init__(self, handle: int, *, width: int, delay: int) -> None:
        if width < 1 or delay < 1:
            raise ValueError("external value delay dimensions must be positive")
        self.width = int(width)
        self.delay = int(delay)
        super().__init__(
            handle,
            ("value",),
            "value",
            interface_version=EXTERNAL_VALUE_DELAY_INTERFACE,
        )

    def configuration(self) -> dict[str, object]:
        return {
            **super().configuration(),
            "width": self.width,
            "delay": self.delay,
            "binding_semantics": "positive_relative_call_displacement",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> ExternalExecutiveOperatorState:
        return ExternalExecutiveOperatorState.from_mapping(
            self.interface_version,
            {
                "confidence": torch.zeros(batch_size, self.delay, device=device, dtype=dtype),
                "present": torch.zeros(batch_size, self.delay, device=device, dtype=torch.bool),
                "values": torch.zeros(
                    batch_size, self.delay, self.width, device=device, dtype=dtype
                ),
            },
        )

    def validate_state(
        self,
        state: ExternalExecutiveOperatorState,
        *,
        batch_size: int,
    ) -> ExternalExecutiveOperatorState:
        state.validate()
        if state.interface_version != self.interface_version:
            raise ValueError("external value delay state interface is incompatible")
        if tuple(name for name, _ in state.tensors) != (
            "confidence",
            "present",
            "values",
        ):
            raise ValueError("external value delay state fields are incompatible")
        confidence = state.tensor("confidence")
        present = state.tensor("present")
        values = state.tensor("values")
        if values.shape != (batch_size, self.delay, self.width):
            raise ValueError("external value delay values have an incompatible shape")
        if present.shape != (batch_size, self.delay) or present.dtype != torch.bool:
            raise ValueError("external value delay presence has an incompatible shape")
        if confidence.shape != (batch_size, self.delay):
            raise ValueError("external value delay confidence has an incompatible shape")
        if not values.is_floating_point() or not confidence.is_floating_point():
            raise TypeError("external value delay state must use floating values")
        if values.device != present.device or values.device != confidence.device:
            raise ValueError("external value delay state tensors must share a device")
        if bool(torch.any(confidence < 0.0)):
            raise ValueError("external value delay confidence cannot be negative")
        return state

    def execute(
        self,
        arguments: tuple[TypedWorkspaceValue, ...],
    ) -> TypedWorkspaceValue:
        raise RuntimeError("external value delay requires explicit operator state")

    def execute_with_state(
        self,
        arguments: tuple[TypedWorkspaceValue, ...],
        state: ExternalExecutiveOperatorState,
    ) -> tuple[TypedWorkspaceValue, ExternalExecutiveOperatorState]:
        current = arguments[0]
        assert isinstance(current.payload, torch.Tensor)
        if current.payload.shape[1] != self.width:
            raise ValueError("external value delay input width is incompatible")
        values = state.tensor("values")
        present = state.tensor("present")
        confidence = state.tensor("confidence")
        delayed = TypedWorkspaceValue.from_tensor(
            "value",
            values[:, 0],
            present=present[:, 0],
            confidence=confidence[:, 0],
        )
        next_state = ExternalExecutiveOperatorState.from_mapping(
            self.interface_version,
            {
                "confidence": torch.cat((confidence[:, 1:], current.confidence[:, None]), dim=1),
                "present": torch.cat((present[:, 1:], current.present[:, None]), dim=1),
                "values": torch.cat((values[:, 1:], current.payload[:, None, :]), dim=1),
            },
        )
        return delayed, next_state


__all__ = ["EXTERNAL_VALUE_DELAY_INTERFACE", "ExternalValueDelayOperator"]
