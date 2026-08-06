"""Replaceable memory-side capability programs.

The shared controller remains frozen while a capability owns its recurrent
external state and its learned intention residual.  Output decoding stays
outside this module so a capability can be connected to any compatible
decoder on the intention bus.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .episodic import EpisodicContextEncoder, EpisodicIntentAdapter
from .interface import IntentEvent

EXTERNAL_CAPABILITY_SCHEMA = "neural-computer.external-capability.v1"


@dataclass(frozen=True)
class ExternalCapabilityState:
    """External recurrent state owned by one capability instance."""

    context: torch.Tensor

    def validate(self, *, batch_size: int, hidden: int) -> ExternalCapabilityState:
        if self.context.ndim != 2 or self.context.shape != (batch_size, hidden):
            raise ValueError("capability context state has the wrong shape")
        if not bool(torch.isfinite(self.context).all()):
            raise ValueError("capability context state must be finite")
        return self


class ExternalCapabilityProgram(nn.Module):
    """A generic recurrent memory-side program for one frozen controller.

    The program consumes standardized learned events, opaque action vectors,
    scalar outcomes, and the controller's opaque intention.  It returns an
    adapted intention and keeps its recurrent state outside the controller.
    It never receives raw modality data, task identifiers, correct actions, or
    protocol-specific fields.  A caller may attach any compatible decoder to
    the returned intention through the ordinary output bus.
    """

    def __init__(
        self,
        event_width: int,
        action_width: int,
        intention_width: int,
        *,
        context_hidden: int = 64,
        context_width: int = 32,
        adapter_hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(
            event_width,
            action_width,
            intention_width,
            context_hidden,
            context_width,
            adapter_hidden,
        ) < 1:
            raise ValueError("external capability dimensions must be positive")
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.intention_width = int(intention_width)
        self.context_hidden = int(context_hidden)
        self.context_width = int(context_width)
        self.adapter_hidden = int(adapter_hidden)
        self.context_encoder = EpisodicContextEncoder(
            self.event_width,
            self.action_width,
            hidden=self.context_hidden,
            context_width=self.context_width,
        )
        self.intent_adapter = EpisodicIntentAdapter(
            self.context_width,
            self.intention_width,
            hidden=self.adapter_hidden,
        )

    def configuration(self) -> dict[str, int | str]:
        """Return the versioned capability interface contract."""

        return {
            "schema": EXTERNAL_CAPABILITY_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "context_hidden": self.context_hidden,
            "context_width": self.context_width,
            "adapter_hidden": self.adapter_hidden,
            "state": "external_recurrent_context_v1",
            "output": "opaque_intention_residual_v1",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalCapabilityState:
        if batch_size < 1:
            raise ValueError("capability batch size must be positive")
        return ExternalCapabilityState(
            context=torch.zeros(
                batch_size,
                self.context_hidden,
                device=device,
                dtype=dtype,
            )
        )

    def step(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityState]:
        """Advance external state and adapt one opaque controller intention."""

        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for capability")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for capability")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for capability")
        intention.validate(width=self.intention_width)
        if intention.payload.shape[0] != event.shape[0]:
            raise ValueError("intention batch does not match capability event")
        state.validate(batch_size=event.shape[0], hidden=self.context_hidden)
        context, next_context = self.context_encoder.step(
            event,
            action,
            outcome,
            state.context,
            present,
        )
        adapted = self.intent_adapter(intention, context.context)
        return adapted, ExternalCapabilityState(next_context)
