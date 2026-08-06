"""Replaceable memory-side capability programs.

The shared controller remains frozen while a capability owns its recurrent
external state and its learned intention residual.  Output decoding stays
outside this module so a capability can be connected to any compatible
decoder on the intention bus.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from .episodic import EpisodicContextEncoder, EpisodicIntentAdapter
from .interface import IntentEvent

EXTERNAL_CAPABILITY_SCHEMA = "neural-computer.external-capability.v1"
EXTERNAL_CAPABILITY_PIPELINE_SCHEMA = "neural-computer.external-capability-pipeline.v1"
EXTERNAL_CAPABILITY_COMPOSITION_SCHEMA = (
    "neural-computer.external-capability-composition.v1"
)


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


@dataclass(frozen=True)
class ExternalCapabilityPipelineState:
    """Independent recurrent states for a memory-side capability pipeline."""

    programs: tuple[ExternalCapabilityState, ...]

    def validate(
        self,
        *,
        batch_size: int,
        hidden_sizes: tuple[int, ...],
    ) -> ExternalCapabilityPipelineState:
        if len(self.programs) != len(hidden_sizes):
            raise ValueError("pipeline state does not match program count")
        for state, hidden in zip(self.programs, hidden_sizes, strict=True):
            state.validate(batch_size=batch_size, hidden=hidden)
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
        if (
            min(
                event_width,
                action_width,
                intention_width,
                context_hidden,
                context_width,
                adapter_hidden,
            )
            < 1
        ):
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


class ExternalCapabilityPipeline(nn.Module):
    """Compose zero or more replaceable capability programs in memory.

    The pipeline is an orchestration boundary, not a controller branch. Each
    program receives the same standardized event, opaque feedback, and scalar
    outcome, while the adapted intention from one program becomes the opaque
    input to the next. Every program retains its own recurrent state outside
    the controller, so the chain can grow, shrink, persist, or be rehydrated
    without resizing the controller or merging program memories.
    """

    def __init__(
        self,
        programs: Iterable[ExternalCapabilityProgram] = (),
        *,
        event_width: int | None = None,
        action_width: int | None = None,
        intention_width: int | None = None,
        hide_downstream_events: bool = False,
    ) -> None:
        super().__init__()
        members = tuple(programs)
        if members:
            dimensions = {
                "event_width": members[0].event_width,
                "action_width": members[0].action_width,
                "intention_width": members[0].intention_width,
            }
            for program in members[1:]:
                if any(
                    getattr(program, name) != value
                    for name, value in dimensions.items()
                ):
                    raise ValueError(
                        "pipeline programs must share interface dimensions"
                    )
        else:
            if None in (event_width, action_width, intention_width):
                raise ValueError(
                    "empty pipelines require event, action, and intention widths"
                )
            dimensions = {
                "event_width": int(event_width),
                "action_width": int(action_width),
                "intention_width": int(intention_width),
            }
        if min(dimensions.values()) < 1:
            raise ValueError("pipeline interface dimensions must be positive")
        self.event_width = dimensions["event_width"]
        self.action_width = dimensions["action_width"]
        self.intention_width = dimensions["intention_width"]
        self.hide_downstream_events = bool(hide_downstream_events)
        self.programs = nn.ModuleList(members)

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        return tuple(program.context_hidden for program in self.programs)

    def configuration(self) -> dict[str, object]:
        """Return the versioned, order-sensitive composition contract."""

        return {
            "schema": EXTERNAL_CAPABILITY_PIPELINE_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "program_count": len(self.programs),
            "program_schemas": tuple(
                program.configuration()["schema"] for program in self.programs
            ),
            "event_visibility": (
                "head_only" if self.hide_downstream_events else "all_programs"
            ),
            "state": "independent_external_recurrent_contexts_v1",
            "composition": "adapted_intention_serial_chain_v1",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalCapabilityPipelineState:
        return ExternalCapabilityPipelineState(
            tuple(
                program.initial_state(batch_size, device=device, dtype=dtype)
                for program in self.programs
            )
        )

    def step(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityPipelineState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityPipelineState]:
        """Run one event through the chain while preserving state isolation."""

        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for pipeline")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for pipeline")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for pipeline")
        intention.validate(width=self.intention_width)
        state.validate(
            batch_size=event.shape[0],
            hidden_sizes=self.hidden_sizes,
        )
        current = intention
        next_states: list[ExternalCapabilityState] = []
        for index, (program, program_state) in enumerate(
            zip(
                self.programs,
                state.programs,
                strict=True,
            )
        ):
            program_event = event
            program_present = present
            if self.hide_downstream_events and index > 0:
                program_event = torch.zeros_like(event)
                program_present = torch.zeros(
                    event.shape[0],
                    dtype=torch.bool,
                    device=event.device,
                )
            current, next_state = program.step(
                event=program_event,
                action=action,
                outcome=outcome,
                intention=current,
                state=program_state,
                present=program_present,
            )
            next_states.append(next_state)
        return current, ExternalCapabilityPipelineState(tuple(next_states))


class ExternalCapabilityComposition(nn.Module):
    """Bind independently learned external programs into a learned sequence.

    Every slot is evaluated from the current opaque intention and an external
    router chooses the slot for each composition step. The controller remains
    unaware of slot identity; program states, router weights, and binding
    decisions all live outside it. Soft routing keeps the boundary
    differentiable while scalar outcome training discovers which learned
    event cues should open each slot.
    """

    def __init__(
        self,
        programs: Iterable[ExternalCapabilityProgram] = (),
        *,
        event_width: int | None = None,
        action_width: int | None = None,
        intention_width: int | None = None,
        composition_steps: int = 2,
        router_hidden: int = 64,
    ) -> None:
        super().__init__()
        members = tuple(programs)
        if members:
            dimensions = {
                "event_width": members[0].event_width,
                "action_width": members[0].action_width,
                "intention_width": members[0].intention_width,
            }
            for program in members[1:]:
                if any(
                    getattr(program, name) != value
                    for name, value in dimensions.items()
                ):
                    raise ValueError(
                        "composition programs must share interface dimensions"
                    )
        else:
            if None in (event_width, action_width, intention_width):
                raise ValueError(
                    "empty compositions require event, action, and intention widths"
                )
            dimensions = {
                "event_width": int(event_width),
                "action_width": int(action_width),
                "intention_width": int(intention_width),
            }
        if len(members) < 2:
            raise ValueError("compositions require at least two programs")
        if composition_steps < 1 or router_hidden < 1:
            raise ValueError("composition steps and router hidden must be positive")
        if min(dimensions.values()) < 1:
            raise ValueError("composition interface dimensions must be positive")
        self.event_width = dimensions["event_width"]
        self.action_width = dimensions["action_width"]
        self.intention_width = dimensions["intention_width"]
        self.composition_steps = int(composition_steps)
        self.router_hidden = int(router_hidden)
        self.programs = nn.ModuleList(members)
        router_input = (
            self.event_width
            + self.action_width
            + 1
            + self.intention_width
        )
        self.router = nn.Sequential(
            nn.Linear(router_input, self.router_hidden),
            nn.GELU(),
            nn.Linear(
                self.router_hidden,
                self.composition_steps * len(self.programs),
            ),
        )

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        return tuple(program.context_hidden for program in self.programs)

    def configuration(self) -> dict[str, object]:
        """Return the versioned learned-binding contract."""

        return {
            "schema": EXTERNAL_CAPABILITY_COMPOSITION_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "program_count": len(self.programs),
            "composition_steps": self.composition_steps,
            "router_hidden": self.router_hidden,
            "program_schemas": tuple(
                program.configuration()["schema"] for program in self.programs
            ),
            "state": "independent_external_recurrent_contexts_v1",
            "routing": "learned_event_conditioned_soft_slot_binding_v1",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalCapabilityPipelineState:
        return ExternalCapabilityPipelineState(
            tuple(
                program.initial_state(batch_size, device=device, dtype=dtype)
                for program in self.programs
            )
        )

    def step(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityPipelineState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityPipelineState]:
        """Apply a learned slot sequence while keeping state external."""

        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for composition")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for composition")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for composition")
        intention.validate(width=self.intention_width)
        state.validate(
            batch_size=event.shape[0],
            hidden_sizes=self.hidden_sizes,
        )
        current = intention
        next_states = list(state.programs)
        for step_index in range(self.composition_steps):
            router_input = torch.cat(
                (event, action, outcome.unsqueeze(1), current.payload),
                dim=-1,
            )
            route_logits = self.router(router_input).reshape(
                event.shape[0], self.composition_steps, len(self.programs)
            )[:, step_index]
            weights = torch.softmax(route_logits, dim=-1)
            candidates: list[torch.Tensor] = []
            step_states: list[ExternalCapabilityState] = []
            for program, program_state in zip(
                self.programs,
                next_states,
                strict=True,
            ):
                adapted, next_state = program.step(
                    event=event,
                    action=action,
                    outcome=outcome,
                    intention=current,
                    state=program_state,
                    present=present,
                )
                candidates.append(adapted.payload)
                step_states.append(next_state)
            current = IntentEvent(
                torch.stack(candidates, dim=1)
                .mul(weights.unsqueeze(-1))
                .sum(dim=1)
            )
            next_states = step_states
        return current, ExternalCapabilityPipelineState(tuple(next_states))
