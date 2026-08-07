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
EXTERNAL_CAPABILITY_SHARED_RESIDUAL_SCHEMA = (
    "neural-computer.external-capability-shared-residual.v1"
)
EXTERNAL_CAPABILITY_SLOT_BINDING_SCHEMA = (
    "neural-computer.external-capability-slot-binding.v1"
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


class ExternalCapabilitySharedResidualBank(nn.Module):
    """Share a frozen context basis while growing isolated residual slots.

    The context encoder is one replaceable memory-side base.  Each residual
    adapter has its own externally owned recurrent state and can be trained or
    replaced independently, so adding a capability never updates an earlier
    residual.  The bank is deliberately unaware of task names, raw protocols,
    and correct actions; an external opaque binding chooses ``slot_index``.

    This is a compression candidate, not an unconditional consolidation
    operation.  Callers must freeze ``shared_context_encoder`` before adding
    new slots and retain each alias only after fresh behavior verification.
    """

    def __init__(
        self,
        event_width: int,
        action_width: int,
        intention_width: int,
        *,
        slot_count: int = 1,
        context_hidden: int = 64,
        context_width: int = 32,
        adapter_hidden: int = 64,
    ) -> None:
        super().__init__()
        if slot_count < 1:
            raise ValueError("shared residual bank needs at least one slot")
        if min(
            event_width,
            action_width,
            intention_width,
            context_hidden,
            context_width,
            adapter_hidden,
        ) < 1:
            raise ValueError("shared residual dimensions must be positive")
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.intention_width = int(intention_width)
        self.context_hidden = int(context_hidden)
        self.context_width = int(context_width)
        self.adapter_hidden = int(adapter_hidden)
        self.shared_context_encoder = EpisodicContextEncoder(
            self.event_width,
            self.action_width,
            hidden=self.context_hidden,
            context_width=self.context_width,
        )
        self.residual_slots = nn.ModuleList(
            self._new_residual() for _ in range(slot_count)
        )

    def _new_residual(self) -> EpisodicIntentAdapter:
        return EpisodicIntentAdapter(
            self.context_width,
            self.intention_width,
            hidden=self.adapter_hidden,
        )

    @property
    def slot_count(self) -> int:
        return len(self.residual_slots)

    def configuration(self) -> dict[str, int | str]:
        """Return the versioned shared-base/residual contract."""

        return {
            "schema": EXTERNAL_CAPABILITY_SHARED_RESIDUAL_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "context_hidden": self.context_hidden,
            "context_width": self.context_width,
            "adapter_hidden": self.adapter_hidden,
            "slot_count": self.slot_count,
            "state": "independent_external_recurrent_contexts_v1",
            "base": "one_shared_context_encoder_v1",
            "residual": "independent_intention_adapters_v1",
        }

    def freeze_shared_base(self) -> None:
        """Make the shared representation immutable for later slot growth."""

        for parameter in self.shared_context_encoder.parameters():
            parameter.requires_grad_(False)

    def add_slot(self) -> int:
        """Append a zero-initialized residual without changing old weights."""

        residual = self._new_residual()
        reference = next(self.shared_context_encoder.parameters())
        residual.to(device=reference.device, dtype=reference.dtype)
        self.residual_slots.append(residual)
        return self.slot_count - 1

    def freeze_slot(self, slot_index: int) -> None:
        """Protect one residual from later capability-specific updates."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("shared residual slot is out of range")
        for parameter in self.residual_slots[slot_index].parameters():
            parameter.requires_grad_(False)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalCapabilityPipelineState:
        if batch_size < 1:
            raise ValueError("shared residual batch size must be positive")
        return ExternalCapabilityPipelineState(
            tuple(
                ExternalCapabilityState(
                    self.shared_context_encoder.initial_state(
                        batch_size,
                        device=device,
                        dtype=dtype,
                    )
                )
                for _ in self.residual_slots
            )
        )

    def step(
        self,
        *,
        slot_index: int,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityPipelineState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityPipelineState]:
        """Execute one opaque residual binding and advance only its state."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("shared residual slot is out of range")
        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for shared residual bank")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for shared residual bank")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for shared residual bank")
        intention.validate(width=self.intention_width)
        if intention.payload.shape[0] != event.shape[0]:
            raise ValueError("intention batch does not match shared residual event")
        state.validate(
            batch_size=event.shape[0],
            hidden_sizes=(self.context_hidden,) * self.slot_count,
        )
        adapted, next_context = self.step_slot(
            slot_index=slot_index,
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state.programs[slot_index],
            present=present,
        )
        next_states = list(state.programs)
        next_states[slot_index] = ExternalCapabilityState(next_context)
        return adapted, ExternalCapabilityPipelineState(tuple(next_states))

    def step_slot(
        self,
        slot_index: int,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        *,
        intention: IntentEvent,
        state: ExternalCapabilityState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, torch.Tensor]:
        """Execute one slot using only that slot's externally owned state."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("shared residual slot is out of range")
        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for shared residual bank")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for shared residual bank")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for shared residual bank")
        intention.validate(width=self.intention_width)
        if intention.payload.shape[0] != event.shape[0]:
            raise ValueError("intention batch does not match shared residual event")
        state.validate(batch_size=event.shape[0], hidden=self.context_hidden)
        context, next_context = self.shared_context_encoder.step(
            event,
            action,
            outcome,
            state.context,
            present,
        )
        adapted = self.residual_slots[slot_index](intention, context.context)
        return adapted, next_context


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
            "binding": "optional_opaque_external_slot_mask_v1",
            "execution": "masked_sparse_active_slots_v1",
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
        slot_mask: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityPipelineState]:
        """Apply a learned slot sequence while keeping state external.

        ``slot_mask`` is an opaque memory-side binding.  It can restrict the
        slots eligible for this alias without exposing a task identifier to
        the controller or changing the learned event representation.
        """

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
        if slot_mask is not None:
            if slot_mask.ndim != 2 or slot_mask.shape != (
                event.shape[0],
                len(self.programs),
            ):
                raise ValueError("slot mask has the wrong shape for composition")
            if slot_mask.dtype is not torch.bool:
                raise TypeError("slot mask must be boolean")
            if not bool(slot_mask.any(dim=-1).all()):
                raise ValueError("slot mask must allow at least one slot per row")
            slot_mask = slot_mask.to(device=event.device)
        current = intention
        next_states = list(state.programs)
        if slot_mask is None:
            active_indices = tuple(range(len(self.programs)))
        else:
            active_indices = tuple(
                index
                for index in range(len(self.programs))
                if bool(slot_mask[:, index].any())
            )
        for step_index in range(self.composition_steps):
            router_input = torch.cat(
                (event, action, outcome.unsqueeze(1), current.payload),
                dim=-1,
            )
            route_logits = self.router(router_input).reshape(
                event.shape[0], self.composition_steps, len(self.programs)
            )[:, step_index]
            if slot_mask is not None:
                route_logits = route_logits.masked_fill(
                    ~slot_mask,
                    torch.finfo(route_logits.dtype).min,
                )
            weights = torch.softmax(route_logits, dim=-1)
            candidates: list[torch.Tensor] = []
            for index in active_indices:
                program = self.programs[index]
                program_state = next_states[index]
                adapted, next_state = program.step(
                    event=event,
                    action=action,
                    outcome=outcome,
                    intention=current,
                    state=program_state,
                    present=present,
                )
                candidates.append(adapted.payload)
                if slot_mask is not None:
                    enabled = slot_mask[:, index].unsqueeze(-1)
                    next_state = ExternalCapabilityState(
                        torch.where(
                            enabled,
                            next_state.context,
                            program_state.context,
                        )
                    )
                next_states[index] = next_state
            active_weights = weights[:, list(active_indices)]
            current = IntentEvent(
                torch.stack(candidates, dim=1)
                .mul(active_weights.unsqueeze(-1))
                .sum(dim=1)
            )
        return current, ExternalCapabilityPipelineState(tuple(next_states))
