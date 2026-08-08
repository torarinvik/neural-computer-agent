"""Replaceable learned bridges from controller state to standardized events."""

from __future__ import annotations

import torch
from torch import nn


EVENT_BRIDGE_SCHEMA = "neural-computer.event-bridge.v1"


class AmodalEventBridge(nn.Module):
    """Adapt learned frontend/state context without exposing raw modalities.

    The bridge is external to the controller and starts as an identity map when
    frontend and output widths match. Its residual is zero-initialized, so a
    candidate can be added without perturbing inherited behavior. A caller may
    train and discard it transactionally with a new external capability.
    """

    def __init__(
        self,
        frontend_width: int,
        controller_state_width: int,
        event_width: int,
        *,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(frontend_width, controller_state_width, event_width, hidden) < 1:
            raise ValueError("event bridge dimensions must be positive")
        self.frontend_width = int(frontend_width)
        self.controller_state_width = int(controller_state_width)
        self.event_width = int(event_width)
        self.hidden = int(hidden)
        if frontend_width == event_width:
            self.base = None
        else:
            self.base = nn.Linear(frontend_width, event_width)
        self.residual = nn.Sequential(
            nn.Linear(frontend_width + controller_state_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, event_width),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "schema": EVENT_BRIDGE_SCHEMA,
            "frontend_width": self.frontend_width,
            "controller_state_width": self.controller_state_width,
            "event_width": self.event_width,
            "hidden": self.hidden,
            "base_identity": self.base is None,
            "residual": "zero_initialized_controller_state_conditioned_v1",
        }

    def forward(
        self,
        frontend_event: torch.Tensor,
        controller_state: torch.Tensor,
    ) -> torch.Tensor:
        expected = (frontend_event.shape[0], self.frontend_width)
        if frontend_event.ndim != 2 or frontend_event.shape != expected:
            raise ValueError("frontend event has the wrong shape for event bridge")
        state_expected = (frontend_event.shape[0], self.controller_state_width)
        if controller_state.ndim != 2 or controller_state.shape != state_expected:
            raise ValueError("controller state has the wrong shape for event bridge")
        if not bool(torch.isfinite(frontend_event).all()):
            raise ValueError("frontend event must contain only finite values")
        if not bool(torch.isfinite(controller_state).all()):
            raise ValueError("controller state must contain only finite values")
        base = frontend_event if self.base is None else self.base(frontend_event)
        residual = self.residual(torch.cat((frontend_event, controller_state), dim=-1))
        return base + residual


__all__ = ["EVENT_BRIDGE_SCHEMA", "AmodalEventBridge"]
