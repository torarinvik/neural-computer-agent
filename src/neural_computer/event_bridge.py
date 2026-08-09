"""Replaceable learned bridges from controller state to standardized events."""

from __future__ import annotations

import torch
from torch import nn


EVENT_BRIDGE_SCHEMA = "neural-computer.event-bridge.v1"
CONDITIONED_EVENT_BRIDGE_SCHEMA = "neural-computer.conditioned-event-bridge.v1"


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


class CapabilityConditionedEventBridge(nn.Module):
    """Shared bridge parameters selected by an opaque capability context.

    The context is a learned vector supplied by the external capability
    system. It has no assigned semantic coordinates and is stored separately
    from the bridge weights, so one reusable interface can adapt to many
    capability/program states without copying a protocol decoder.
    """

    def __init__(
        self,
        frontend_width: int,
        controller_state_width: int,
        event_width: int,
        context_width: int,
        *,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(
            frontend_width,
            controller_state_width,
            event_width,
            context_width,
            hidden,
        ) < 1:
            raise ValueError("conditioned event bridge dimensions must be positive")
        self.frontend_width = int(frontend_width)
        self.controller_state_width = int(controller_state_width)
        self.event_width = int(event_width)
        self.context_width = int(context_width)
        self.hidden = int(hidden)
        self.base = None if frontend_width == event_width else nn.Linear(
            frontend_width, event_width
        )
        self.residual = nn.Sequential(
            nn.Linear(frontend_width + controller_state_width + context_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, event_width),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer("context", torch.zeros(context_width))

    def set_context(self, context: torch.Tensor) -> None:
        if context.ndim != 1 or context.shape[0] != self.context_width:
            raise ValueError("capability context has the wrong shape")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("capability context must contain only finite values")
        self.context.copy_(context.detach().to(self.context))

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "schema": CONDITIONED_EVENT_BRIDGE_SCHEMA,
            "frontend_width": self.frontend_width,
            "controller_state_width": self.controller_state_width,
            "event_width": self.event_width,
            "context_width": self.context_width,
            "hidden": self.hidden,
            "base_identity": self.base is None,
            "residual": "zero_initialized_capability_conditioned_v1",
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
        context = self.context.to(device=frontend_event.device, dtype=frontend_event.dtype)
        context = context.unsqueeze(0).expand(frontend_event.shape[0], -1)
        base = frontend_event if self.base is None else self.base(frontend_event)
        residual = self.residual(
            torch.cat((frontend_event, controller_state, context), dim=-1)
        )
        return base + residual


__all__ = [
    "CONDITIONED_EVENT_BRIDGE_SCHEMA",
    "EVENT_BRIDGE_SCHEMA",
    "AmodalEventBridge",
    "CapabilityConditionedEventBridge",
]
