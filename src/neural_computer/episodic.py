"""Task-agnostic episodic context and causal credit primitives.

The deployed controller never receives task identifiers, correct actions, or
unattempted-action labels through this boundary.  A memory-side learner may
encode a short trajectory of learned events, opaque actions, scalar outcomes,
and presence into a context key.  A trainer may then use paired
common-random-number outcomes to assign credit to individual trajectory
positions.  The resulting context encoder is replaceable external state; it
is not a controller branch or a symbolic task solver.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .interface import IntentEvent


@dataclass(frozen=True)
class EpisodicContextOutput:
    """Encoded episode context and per-event credit scores."""

    context: torch.Tensor
    credit_logits: torch.Tensor
    credit_weights: torch.Tensor
    sequence: torch.Tensor


@dataclass(frozen=True)
class EpisodicBindingRoute:
    """Opaque slot scores produced by an external episodic binding router."""

    context: torch.Tensor
    scores: torch.Tensor
    selected_slot: torch.Tensor


@dataclass(frozen=True)
class OnlineEpisodicRelationState:
    """External fixed-window state for online relation retrieval."""

    events: torch.Tensor
    actions: torch.Tensor
    outcomes: torch.Tensor
    present: torch.Tensor


EXTERNAL_WORKING_MEMORY_CELL_SCHEMA = (
    "neural-computer.external-working-memory-cell.v1"
)


def _validate_episode_inputs(
    events: torch.Tensor,
    actions: torch.Tensor,
    outcomes: torch.Tensor,
    present: torch.Tensor | None,
) -> torch.Tensor:
    if events.ndim != 3:
        raise ValueError("events must have shape [batch, time, event_width]")
    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, time, action_width]")
    if outcomes.ndim != 2:
        raise ValueError("outcomes must have shape [batch, time]")
    if events.shape[:2] != actions.shape[:2] or events.shape[:2] != outcomes.shape:
        raise ValueError("episode inputs must share batch and time dimensions")
    if events.shape[1] < 1 or actions.shape[2] < 1 or events.shape[2] < 1:
        raise ValueError("episode widths and time must be positive")
    if present is None:
        present = torch.ones(
            events.shape[:2], dtype=torch.bool, device=events.device
        )
    if present.shape != events.shape[:2]:
        raise ValueError("present must have shape [batch, time]")
    if present.device != events.device:
        raise ValueError("episode tensors must share a device")
    for name, value in (
        ("events", events),
        ("actions", actions),
        ("outcomes", outcomes),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    return present.to(dtype=torch.bool)


def credit_weights_from_logits(
    credit_logits: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    """Normalize credit scores over present events only."""
    if credit_logits.ndim != 2 or present.shape != credit_logits.shape:
        raise ValueError("credit logits and presence must share shape [batch, time]")
    weights = torch.softmax(
        credit_logits.masked_fill(
            ~present, torch.finfo(credit_logits.dtype).min
        ),
        dim=-1,
    )
    return torch.where(present, weights, torch.zeros_like(weights))


class EpisodicCreditHead(nn.Module):
    """Replaceable event-credit state for one external capability."""

    def __init__(self, hidden: int, context_width: int) -> None:
        super().__init__()
        if min(hidden, context_width) < 1:
            raise ValueError("credit-head dimensions must be positive")
        self.hidden = int(hidden)
        self.context_width = int(context_width)
        self.network = nn.Sequential(
            nn.Linear(hidden + context_width + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
        outcomes: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, hidden]")
        if context.ndim != 2 or context.shape != (
            sequence.shape[0],
            self.context_width,
        ):
            raise ValueError("context has the wrong shape")
        if outcomes.shape != sequence.shape[:2] or present.shape != outcomes.shape:
            raise ValueError("outcomes and presence must align with sequence")
        if sequence.shape[-1] != self.hidden:
            raise ValueError("sequence has the wrong hidden width")
        context_rows = context.unsqueeze(1).expand(-1, sequence.shape[1], -1)
        features = torch.cat(
            (
                sequence,
                context_rows,
                outcomes.unsqueeze(-1).to(dtype=sequence.dtype),
                present.unsqueeze(-1).to(dtype=sequence.dtype),
            ),
            dim=-1,
        )
        logits = self.network(features).squeeze(-1)
        return torch.where(present, logits, torch.zeros_like(logits))


class EpisodicContextEncoder(nn.Module):
    """Encode an opaque trajectory into a replaceable memory-side context key.

    The encoder consumes only learned event payloads, opaque action vectors,
    scalar outcomes, and a presence mask.  Its recurrent summary is
    permutation-sensitive over time, while its credit head scores which
    positions in the same episode should receive outcome credit.  The final
    credit layer is neutral at initialization so constructing this component
    cannot prefer a time position before evidence is observed.
    """

    def __init__(
        self,
        event_width: int,
        action_width: int,
        *,
        hidden: int = 64,
        context_width: int = 32,
    ) -> None:
        super().__init__()
        if min(event_width, action_width, hidden, context_width) < 1:
            raise ValueError("episodic context dimensions must be positive")
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.hidden = int(hidden)
        self.context_width = int(context_width)
        token_width = self.event_width + self.action_width + 2
        self.token_encoder = nn.Sequential(
            nn.Linear(token_width, hidden),
            nn.GELU(),
        )
        self.recurrent = nn.GRU(hidden, hidden, batch_first=True)
        self.context_projection = nn.Linear(hidden, context_width)
        self.credit_policy = EpisodicCreditHead(hidden, context_width)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create the external recurrent state for one online episode."""

        if batch_size < 1:
            raise ValueError("episodic context batch size must be positive")
        return torch.zeros(batch_size, self.hidden, device=device, dtype=dtype)

    def step(
        self,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        state: torch.Tensor,
        present: torch.Tensor | None = None,
    ) -> tuple[EpisodicContextOutput, torch.Tensor]:
        """Advance the context encoder by one learner-visible event.

        This online counterpart to :meth:`forward` consumes only the current
        learned event, the previous opaque action/outcome record, and
        presence. The recurrent state remains external to the controller and
        can be grown, checkpointed, or replaced independently.
        """

        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event must have shape [batch, event_width]")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action must have shape [batch, action_width]")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome must have shape [batch]")
        if state.ndim != 2 or state.shape != (event.shape[0], self.hidden):
            raise ValueError("state must have shape [batch, hidden]")
        if present is None:
            present = torch.ones(
                event.shape[0], dtype=torch.bool, device=event.device
            )
        if present.shape != outcome.shape:
            raise ValueError("present must have shape [batch]")
        for name, value in (
            ("event", event),
            ("action", action),
            ("outcome", outcome),
            ("state", state),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")
        present_float = present.to(dtype=event.dtype)
        token = torch.cat(
            (
                event,
                action,
                outcome.unsqueeze(-1).to(dtype=event.dtype),
                present_float.unsqueeze(-1),
            ),
            dim=-1,
        )
        encoded = self.token_encoder(token).unsqueeze(1)
        encoded = encoded * present_float[:, None, None]
        sequence, hidden = self.recurrent(encoded, state.unsqueeze(0))
        context = F.normalize(self.context_projection(hidden[-1]), dim=-1)
        credit_logits = self.credit_policy(
            sequence,
            context,
            outcome.unsqueeze(1),
            present.unsqueeze(1),
        )
        credit_weights = credit_weights_from_logits(
            credit_logits, present.unsqueeze(1)
        )
        return (
            EpisodicContextOutput(
                context=context,
                credit_logits=credit_logits,
                credit_weights=credit_weights,
                sequence=sequence,
            ),
            hidden[-1],
        )

    def forward(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
        outcomes: torch.Tensor,
        present: torch.Tensor | None = None,
    ) -> EpisodicContextOutput:
        present = _validate_episode_inputs(events, actions, outcomes, present)
        if events.shape[2] != self.event_width:
            raise ValueError("events have the wrong width")
        if actions.shape[2] != self.action_width:
            raise ValueError("actions have the wrong width")
        present_float = present.to(dtype=events.dtype)
        token = torch.cat(
            (
                events,
                actions,
                outcomes.unsqueeze(-1).to(dtype=events.dtype),
                present_float.unsqueeze(-1),
            ),
            dim=-1,
        )
        token = self.token_encoder(token)
        token = token * present_float.unsqueeze(-1)
        sequence, _ = self.recurrent(token)
        lengths = present.sum(dim=1).clamp_min(1).to(torch.long) - 1
        final = sequence.gather(
            1, lengths[:, None, None].expand(-1, 1, self.hidden)
        ).squeeze(1)
        context = F.normalize(self.context_projection(final), dim=-1)
        credit_logits = self.credit_policy(
            sequence,
            context,
            outcomes,
            present,
        )
        weights = credit_weights_from_logits(credit_logits, present)
        return EpisodicContextOutput(context, credit_logits, weights, sequence)


class EpisodicBindingRouter(nn.Module):
    """Discover and route opaque memory bindings from scalar utility.

    This component owns only external, replaceable state.  It converts a
    learned event trajectory into a normalized context, stores opaque context
    snapshots as provisioned slot keys, and adapts the encoder with the
    utility of the slot that was actually attempted.  No task identifier,
    semantic key, correct unattempted slot, or controller parameter enters the
    boundary.

    Slot keys are intentionally fixed after provisioning.  The trainable
    state is the episodic encoder, so a learned route must generalize from
    fresh trajectories to the original opaque keys.  Callers can freeze that
    encoder after independent promotion while retaining the keys as growing
    external memory.
    """

    schema = "neural-computer.episodic-binding-router.v1"

    def __init__(
        self,
        event_width: int,
        action_width: int,
        *,
        hidden: int = 64,
        context_width: int = 32,
        max_slots: int | None = None,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        if max_slots is not None and (
            not isinstance(max_slots, int)
            or isinstance(max_slots, bool)
            or max_slots < 1
        ):
            raise ValueError("episodic binding router max slots is invalid")
        if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
            raise ValueError("episodic binding router temperature is invalid")
        self.encoder = EpisodicContextEncoder(
            event_width,
            action_width,
            hidden=hidden,
            context_width=context_width,
        )
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.hidden = int(hidden)
        self.context_width = int(context_width)
        self.max_slots = max_slots
        self.temperature = float(temperature)
        self.slot_keys = nn.ParameterList()

    @property
    def slot_count(self) -> int:
        return len(self.slot_keys)

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "hidden": self.hidden,
            "context_width": self.context_width,
            "slot_count": self.slot_count,
            "max_slots": self.max_slots if self.max_slots is not None else 0,
            "temperature": self.temperature,
            "state": "external_encoder_plus_opaque_fixed_slot_keys_v1",
            "updates": "attempted_slot_scalar_utility_without_replay_v1",
        }

    def encode(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
        outcomes: torch.Tensor,
        present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a trajectory into the context consumed by slot routing."""

        return self.encoder(events, actions, outcomes, present).context

    @torch.no_grad()
    def add_slot(self, context_key: torch.Tensor) -> int:
        """Provision one opaque slot from an observed context snapshot."""

        if self.max_slots is not None and self.slot_count >= self.max_slots:
            raise RuntimeError(
                "episodic binding router slot capacity is exhausted"
            )
        if context_key.ndim != 1 or context_key.shape[0] != self.context_width:
            raise ValueError("episodic binding router slot key has the wrong shape")
        if not bool(torch.isfinite(context_key).all()):
            raise ValueError("episodic binding router slot key must be finite")
        if not bool(context_key.square().sum().gt(1e-12)):
            raise ValueError("episodic binding router slot key cannot be zero")
        reference = next(self.encoder.parameters())
        key = F.normalize(context_key.detach(), dim=0).to(reference)
        self.slot_keys.append(nn.Parameter(key, requires_grad=False))
        return self.slot_count - 1

    def route_scores(
        self,
        context: torch.Tensor,
        *,
        slot_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return cosine scores, optionally under a physical slot permutation."""

        if context.ndim != 2 or context.shape[1] != self.context_width:
            raise ValueError("episodic binding router context has the wrong shape")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("episodic binding router context must be finite")
        if self.slot_count < 1:
            raise RuntimeError("episodic binding router has no slots")
        keys = torch.stack(tuple(self.slot_keys), dim=0).to(context)
        if slot_order is not None:
            if (
                slot_order.ndim != 1
                or slot_order.shape[0] != self.slot_count
                or slot_order.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            ):
                raise ValueError("episodic binding router slot order is invalid")
            if sorted(slot_order.detach().cpu().tolist()) != list(
                range(self.slot_count)
            ):
                raise ValueError(
                    "episodic binding router slot order is not a permutation"
                )
            keys = keys[slot_order.to(device=keys.device)]
        return F.normalize(context, dim=-1) @ keys.transpose(0, 1)

    def route(
        self,
        context: torch.Tensor,
        *,
        slot_order: torch.Tensor | None = None,
    ) -> EpisodicBindingRoute:
        scores = self.route_scores(context, slot_order=slot_order)
        return EpisodicBindingRoute(
            context=context,
            scores=scores,
            selected_slot=scores.argmax(dim=-1),
        )

    def trainable_parameters(self):
        """Return only mutable context-encoder parameters."""

        return (
            parameter
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        )

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        """Freeze learned routing while retaining external slot keys."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def adaptation_step(
        self,
        context: torch.Tensor,
        selected_slot: int,
        verifier_utility: float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        baseline: float = 0.5,
        temperature: float | None = None,
    ) -> float:
        """Apply one REINFORCE-style update from an attempted slot outcome."""

        if context.ndim != 2 or context.shape[0] != 1:
            raise ValueError("episodic binding adaptation needs one context")
        if not 0 <= selected_slot < self.slot_count:
            raise IndexError("episodic binding router slot is out of range")
        if not torch.isfinite(torch.tensor(verifier_utility)) or not (
            0.0 <= verifier_utility <= 1.0
        ):
            raise ValueError("episodic binding utility must lie in [0, 1]")
        if not torch.isfinite(torch.tensor(baseline)) or not (
            0.0 <= baseline <= 1.0
        ):
            raise ValueError("episodic binding baseline must lie in [0, 1]")
        selected_temperature = (
            self.temperature if temperature is None else float(temperature)
        )
        if not torch.isfinite(torch.tensor(selected_temperature)) or (
            selected_temperature <= 0.0
        ):
            raise ValueError("episodic binding adaptation temperature is invalid")
        parameters = tuple(self.trainable_parameters())
        if not parameters:
            raise RuntimeError("episodic binding router encoder is frozen")
        scores = self.route_scores(context)
        loss = -(verifier_utility - baseline) * torch.log_softmax(
            scores / selected_temperature, dim=-1
        )[0, selected_slot]
        selected_optimizer = optimizer
        if selected_optimizer is None:
            selected_optimizer = torch.optim.Adam(parameters, lr=0.01)
        selected_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        selected_optimizer.step()
        return float(loss.detach())


class OnlineEpisodicRelationReader(nn.Module):
    """Read recent learned events with content-and-age attention.

    This memory-side reader keeps a bounded external event window. It scores
    prior rows from the current learned event and a learned age representation,
    then exposes the retrieved event/action/outcome relation to a replaceable
    intention adapter. No task identity, target, or protocol field enters the
    reader, and a newly constructed reader is independent of the controller.
    """

    schema = "neural-computer.online-episodic-relation-reader.v1"

    def __init__(
        self,
        event_width: int,
        action_width: int,
        *,
        memory_capacity: int = 8,
        context_width: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(
            event_width,
            action_width,
            memory_capacity,
            context_width,
            hidden,
        ) < 1:
            raise ValueError("online relation reader dimensions must be positive")
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.memory_capacity = int(memory_capacity)
        self.context_width = int(context_width)
        self.hidden = int(hidden)
        self.query = nn.Linear(self.event_width, self.hidden)
        self.key = nn.Linear(self.event_width, self.hidden)
        self.age_embedding = nn.Parameter(
            torch.randn(self.memory_capacity, self.hidden) * 0.02
        )
        self.score = nn.Sequential(
            nn.Linear(self.hidden * 3 + 1, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        relation_width = (
            self.event_width * 4
            + self.action_width
            + 2
        )
        self.relation = nn.Sequential(
            nn.Linear(relation_width, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.context_width),
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "memory_capacity": self.memory_capacity,
            "context_width": self.context_width,
            "hidden": self.hidden,
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> OnlineEpisodicRelationState:
        if batch_size < 1:
            raise ValueError("relation reader batch size must be positive")
        return OnlineEpisodicRelationState(
            events=torch.zeros(
                batch_size,
                self.memory_capacity,
                self.event_width,
                device=device,
                dtype=dtype,
            ),
            actions=torch.zeros(
                batch_size,
                self.memory_capacity,
                self.action_width,
                device=device,
                dtype=dtype,
            ),
            outcomes=torch.zeros(
                batch_size,
                self.memory_capacity,
                device=device,
                dtype=dtype,
            ),
            present=torch.zeros(
                batch_size,
                self.memory_capacity,
                device=device,
                dtype=torch.bool,
            ),
        )

    def step(
        self,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        state: OnlineEpisodicRelationState,
        present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, OnlineEpisodicRelationState]:
        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event must have shape [batch, event_width]")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action must have shape [batch, action_width]")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome must have shape [batch]")
        expected_state = (
            event.shape[0],
            self.memory_capacity,
        )
        if state.events.shape[:2] != expected_state:
            raise ValueError("relation reader event state has the wrong shape")
        if state.events.shape[2] != self.event_width:
            raise ValueError("relation reader event state has the wrong width")
        if state.actions.shape != (
            event.shape[0],
            self.memory_capacity,
            self.action_width,
        ):
            raise ValueError("relation reader action state has the wrong shape")
        if state.outcomes.shape != expected_state or state.present.shape != expected_state:
            raise ValueError("relation reader state fields must share the window shape")
        if present is None:
            present = torch.ones(
                event.shape[0], dtype=torch.bool, device=event.device
            )
        if present.shape != outcome.shape:
            raise ValueError("present must have shape [batch]")
        for name, value in (
            ("event", event),
            ("action", action),
            ("outcome", outcome),
            ("state.events", state.events),
            ("state.actions", state.actions),
            ("state.outcomes", state.outcomes),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")

        query = self.query(event).unsqueeze(1)
        keys = self.key(state.events)
        age = torch.linspace(
            -1.0,
            0.0,
            self.memory_capacity,
            device=event.device,
            dtype=event.dtype,
        ).reshape(1, self.memory_capacity, 1)
        age_embedding = self.age_embedding.to(device=event.device, dtype=event.dtype)
        score_input = torch.cat(
            (
                query.expand(-1, self.memory_capacity, -1),
                keys,
                age_embedding.unsqueeze(0).expand(event.shape[0], -1, -1),
                age.expand(event.shape[0], -1, -1),
            ),
            dim=-1,
        )
        scores = self.score(score_input).squeeze(-1)
        scores = scores.masked_fill(
            ~state.present,
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(state.present, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        retrieved_event = torch.einsum("bc,bce->be", weights, state.events)
        retrieved_action = torch.einsum("bc,bca->ba", weights, state.actions)
        retrieved_outcome = torch.einsum("bc,bc->b", weights, state.outcomes)
        relation_input = torch.cat(
            (
                event,
                retrieved_event,
                event * retrieved_event,
                (event - retrieved_event).abs(),
                retrieved_action,
                retrieved_outcome.unsqueeze(-1),
                weights.max(dim=-1).values.unsqueeze(-1),
            ),
            dim=-1,
        )
        context = self.relation(relation_input)
        next_state = OnlineEpisodicRelationState(
            events=torch.cat((state.events[:, 1:], event.unsqueeze(1)), dim=1),
            actions=torch.cat((state.actions[:, 1:], action.unsqueeze(1)), dim=1),
            outcomes=torch.cat((state.outcomes[:, 1:], outcome.unsqueeze(1)), dim=1),
            present=torch.cat((state.present[:, 1:], present.unsqueeze(1)), dim=1),
        )
        return context, next_state


class AdaptiveOnlineEpisodicRelationReader(OnlineEpisodicRelationReader):
    """Read a generic window by scoring each candidate relation separately.

    The original reader first pools the whole window and then transforms the
    pooled event.  That is efficient for a pre-sized horizon but can blur the
    correct relation when a generic capability is provisioned with extra
    history.  This variant keeps the same opaque external state contract while
    evaluating each present event/action/outcome row independently before
    mixing the resulting relation contexts.  It receives no task horizon.
    """

    schema = "neural-computer.adaptive-online-episodic-relation-reader.v1"

    def __init__(
        self,
        event_width: int,
        action_width: int,
        *,
        memory_capacity: int = 8,
        context_width: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__(
            event_width,
            action_width,
            memory_capacity=memory_capacity,
            context_width=context_width,
            hidden=hidden,
        )
        candidate_width = self.event_width * 4 + self.action_width + 3
        self.candidate_relation = nn.Sequential(
            nn.Linear(candidate_width, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.context_width),
        )

    def configuration(self) -> dict[str, int | str]:
        config = super().configuration()
        config["schema"] = self.schema
        return config

    def expand_capacity(
        self,
        memory_capacity: int,
        *,
        preserve_weights: bool = True,
    ) -> AdaptiveOnlineEpisodicRelationReader:
        """Return a larger reader, optionally preserving learned weights.

        Resetting is intended only for an unmastered candidate whose fresh
        scalar outcomes have already demonstrated that its current reader is
        inadequate. Previously protected capabilities remain in separate
        external slots and are never reset by this operation.
        """

        if memory_capacity <= self.memory_capacity:
            raise ValueError("expanded relation capacity must be larger")
        expanded = type(self)(
            self.event_width,
            self.action_width,
            memory_capacity=memory_capacity,
            context_width=self.context_width,
            hidden=self.hidden,
        )
        if not preserve_weights:
            return expanded
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name == "age_embedding":
                    old_age = parameter.detach().T.unsqueeze(0)
                    expanded_age = F.interpolate(
                        old_age,
                        size=memory_capacity,
                        mode="linear",
                        align_corners=True,
                    ).squeeze(0).T
                    expanded.age_embedding.copy_(expanded_age)
                else:
                    expanded_parameter = dict(expanded.named_parameters())[name]
                    expanded_parameter.copy_(parameter.detach())
        return expanded

    def step(
        self,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        state: OnlineEpisodicRelationState,
        present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, OnlineEpisodicRelationState]:
        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event must have shape [batch, event_width]")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action must have shape [batch, action_width]")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome must have shape [batch]")
        expected_state = (event.shape[0], self.memory_capacity)
        if state.events.shape[:2] != expected_state:
            raise ValueError("relation reader event state has the wrong shape")
        if state.events.shape[2] != self.event_width:
            raise ValueError("relation reader event state has the wrong width")
        if state.actions.shape != (
            event.shape[0],
            self.memory_capacity,
            self.action_width,
        ):
            raise ValueError("relation reader action state has the wrong shape")
        if state.outcomes.shape != expected_state or state.present.shape != expected_state:
            raise ValueError("relation reader state fields must share the window shape")
        if present is None:
            present = torch.ones(
                event.shape[0], dtype=torch.bool, device=event.device
            )
        if present.shape != outcome.shape:
            raise ValueError("present must have shape [batch]")
        for name, value in (
            ("event", event),
            ("action", action),
            ("outcome", outcome),
            ("state.events", state.events),
            ("state.actions", state.actions),
            ("state.outcomes", state.outcomes),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")

        query = self.query(event).unsqueeze(1)
        keys = self.key(state.events)
        age = torch.linspace(
            -1.0,
            0.0,
            self.memory_capacity,
            device=event.device,
            dtype=event.dtype,
        ).reshape(1, self.memory_capacity, 1)
        age_embedding = self.age_embedding.to(device=event.device, dtype=event.dtype)
        score_input = torch.cat(
            (
                query.expand(-1, self.memory_capacity, -1),
                keys,
                age_embedding.unsqueeze(0).expand(event.shape[0], -1, -1),
                age.expand(event.shape[0], -1, -1),
            ),
            dim=-1,
        )
        scores = self.score(score_input).squeeze(-1)
        scores = scores.masked_fill(
            ~state.present,
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(state.present, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        event_rows = event.unsqueeze(1).expand(-1, self.memory_capacity, -1)
        relation_input = torch.cat(
            (
                event_rows,
                state.events,
                event_rows * state.events,
                (event_rows - state.events).abs(),
                state.actions,
                state.outcomes.unsqueeze(-1),
                age.expand(event.shape[0], -1, -1),
                state.present.unsqueeze(-1).to(dtype=event.dtype),
            ),
            dim=-1,
        )
        candidate_contexts = self.candidate_relation(relation_input)
        context = torch.einsum("bc,bch->bh", weights, candidate_contexts)
        next_state = OnlineEpisodicRelationState(
            events=torch.cat((state.events[:, 1:], event.unsqueeze(1)), dim=1),
            actions=torch.cat((state.actions[:, 1:], action.unsqueeze(1)), dim=1),
            outcomes=torch.cat((state.outcomes[:, 1:], outcome.unsqueeze(1)), dim=1),
            present=torch.cat((state.present[:, 1:], present.unsqueeze(1)), dim=1),
        )
        return context, next_state


class ExternalWorkingMemoryCell(nn.Module):
    """Versioned external working memory with an explicit causal boundary.

    The cell owns the learned relation codec while its event/action/outcome
    window is runtime state.  :meth:`step` reads the current event against the
    old state and only then appends the current row.  This makes it impossible
    for a memory audit to claim action-production capability by reading a value
    after the same value was written.

    The payload contains only learned event tensors, opaque actions, scalar
    outcomes, and presence.  It is therefore a memory file, not a controller
    checkpoint or a modality-specific branch.  Capacity can grow by padding
    the oldest side of the window while preserving the newest logical rows.
    """

    schema = EXTERNAL_WORKING_MEMORY_CELL_SCHEMA

    def __init__(
        self,
        event_width: int,
        action_width: int,
        *,
        memory_capacity: int = 8,
        context_width: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(
            event_width,
            action_width,
            memory_capacity,
            context_width,
            hidden,
        ) < 1:
            raise ValueError("working-memory cell dimensions must be positive")
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.memory_capacity = int(memory_capacity)
        self.context_width = int(context_width)
        self.hidden = int(hidden)
        self.reader = AdaptiveOnlineEpisodicRelationReader(
            event_width,
            action_width,
            memory_capacity=memory_capacity,
            context_width=context_width,
            hidden=hidden,
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "memory_capacity": self.memory_capacity,
            "context_width": self.context_width,
            "hidden": self.hidden,
            "reader_schema": self.reader.schema,
            "read_order": "read_old_state_then_append_current_row_v1",
            "state": "external_tensor_window_v1",
            "write_fields": "learned_event_opaque_action_scalar_outcome_presence_v1",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> OnlineEpisodicRelationState:
        return self.reader.initial_state(
            batch_size,
            device=device,
            dtype=dtype,
        )

    def step(
        self,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        state: OnlineEpisodicRelationState,
        present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, OnlineEpisodicRelationState]:
        """Read against ``state`` and append the current row afterward."""

        return self.reader.step(event, action, outcome, state, present)

    def read(
        self,
        event: torch.Tensor,
        state: OnlineEpisodicRelationState,
    ) -> torch.Tensor:
        """Read current context without changing external state."""

        action = torch.zeros(
            event.shape[0],
            self.action_width,
            device=event.device,
            dtype=event.dtype,
        )
        outcome = torch.zeros(event.shape[0], device=event.device, dtype=event.dtype)
        present = torch.zeros(event.shape[0], dtype=torch.bool, device=event.device)
        context, _ = self.reader.step(event, action, outcome, state, present)
        return context

    def grow_state(
        self,
        state: OnlineEpisodicRelationState,
        memory_capacity: int,
    ) -> OnlineEpisodicRelationState:
        """Grow state without discarding the newest logical rows."""

        if memory_capacity <= self.memory_capacity:
            raise ValueError("working-memory growth must increase capacity")
        if state.events.shape[1] != self.memory_capacity:
            raise ValueError("working-memory state capacity does not match cell")
        batch_size = state.events.shape[0]
        device = state.events.device
        dtype = state.events.dtype
        events = torch.zeros(
            batch_size,
            memory_capacity,
            self.event_width,
            device=device,
            dtype=dtype,
        )
        actions = torch.zeros(
            batch_size,
            memory_capacity,
            self.action_width,
            device=device,
            dtype=dtype,
        )
        outcomes = torch.zeros(
            batch_size,
            memory_capacity,
            device=device,
            dtype=dtype,
        )
        present = torch.zeros(
            batch_size,
            memory_capacity,
            dtype=torch.bool,
            device=device,
        )
        old_start = memory_capacity - self.memory_capacity
        events[:, old_start:] = state.events
        actions[:, old_start:] = state.actions
        outcomes[:, old_start:] = state.outcomes
        present[:, old_start:] = state.present
        return OnlineEpisodicRelationState(events, actions, outcomes, present)

    def grow(self, memory_capacity: int) -> ExternalWorkingMemoryCell:
        """Return a larger cell while preserving learned relation weights."""

        if memory_capacity <= self.memory_capacity:
            raise ValueError("working-memory growth must increase capacity")
        expanded = ExternalWorkingMemoryCell(
            self.event_width,
            self.action_width,
            memory_capacity=memory_capacity,
            context_width=self.context_width,
            hidden=self.hidden,
        )
        with torch.no_grad():
            for name, parameter in self.reader.named_parameters():
                if name == "age_embedding":
                    old_age = parameter.detach().T.unsqueeze(0)
                    expanded_age = F.interpolate(
                        old_age,
                        size=memory_capacity,
                        mode="linear",
                        align_corners=True,
                    ).squeeze(0).T
                    expanded.reader.age_embedding.copy_(expanded_age)
                else:
                    dict(expanded.reader.named_parameters())[name].copy_(
                        parameter.detach()
                    )
        return expanded

    def state_payload(self, state: OnlineEpisodicRelationState) -> dict[str, object]:
        """Serialize only external working-memory state tensors."""

        self._validate_state(state)
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "events": state.events.detach().cpu().clone(),
            "actions": state.actions.detach().cpu().clone(),
            "outcomes": state.outcomes.detach().cpu().clone(),
            "present": state.present.detach().cpu().clone(),
        }

    def state_from_payload(
        self,
        payload: Mapping[str, object],
    ) -> OnlineEpisodicRelationState:
        if payload.get("schema") != self.schema:
            raise ValueError("unsupported working-memory cell state schema")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("working-memory cell configuration is invalid")
        expected = self.configuration()
        for name in (
            "schema",
            "event_width",
            "action_width",
            "memory_capacity",
            "context_width",
            "hidden",
        ):
            if configuration.get(name) != expected[name]:
                raise ValueError("working-memory cell configuration does not match")
        tensors = {
            name: payload.get(name)
            for name in ("events", "actions", "outcomes", "present")
        }
        if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("working-memory state payload must contain tensors")
        state = OnlineEpisodicRelationState(
            tensors["events"],
            tensors["actions"],
            tensors["outcomes"],
            tensors["present"],
        )
        self._validate_state(state)
        return state

    def _validate_state(self, state: OnlineEpisodicRelationState) -> None:
        if state.events.ndim != 3 or state.events.shape[1:] != (
            self.memory_capacity,
            self.event_width,
        ):
            raise ValueError("working-memory state events have the wrong shape")
        if state.actions.shape != (
            state.events.shape[0],
            self.memory_capacity,
            self.action_width,
        ):
            raise ValueError("working-memory state actions have the wrong shape")
        if state.outcomes.shape != (
            state.events.shape[0],
            self.memory_capacity,
        ) or state.present.shape != state.outcomes.shape:
            raise ValueError("working-memory state rows have the wrong shape")
        if state.present.dtype is not torch.bool:
            raise TypeError("working-memory state presence must be boolean")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (state.events, state.actions, state.outcomes)
        ):
            raise ValueError("working-memory state must be finite")


class EpisodicIntentAdapter(nn.Module):
    """Apply replaceable episodic state to an opaque controller intention.

    The output residual is zero-initialized, so adding an adapter preserves the
    inherited intention path until the external capability is trained. It is
    memory-side growth state; the shared controller never receives task IDs or
    protocol-specific fields.
    """

    def __init__(
        self,
        context_width: int,
        intention_width: int,
        *,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        if min(context_width, intention_width, hidden) < 1:
            raise ValueError("episodic intent adapter dimensions must be positive")
        self.context_width = int(context_width)
        self.intention_width = int(intention_width)
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.Linear(self.context_width + self.intention_width, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.intention_width),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": "neural-computer.episodic-intent-adapter.v1",
            "context_width": self.context_width,
            "intention_width": self.intention_width,
            "hidden": self.hidden,
        }

    def forward(
        self,
        intention: IntentEvent,
        context: torch.Tensor,
    ) -> IntentEvent:
        intention.validate(width=self.intention_width)
        if context.ndim != 2 or context.shape != (
            intention.payload.shape[0],
            self.context_width,
        ):
            raise ValueError("context has the wrong shape for intention adapter")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("context must contain only finite values")
        residual = self.network(torch.cat((intention.payload, context), dim=-1))
        return IntentEvent(
            payload=intention.payload + residual,
            timestamp=intention.timestamp,
            confidence=intention.confidence,
            target_key=intention.target_key,
        ).validate(width=self.intention_width)


def episodic_context_contrastive_loss(
    left_context: torch.Tensor,
    right_context: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Match two augmented views of each episode without task labels."""
    if left_context.ndim != 2 or right_context.shape != left_context.shape:
        raise ValueError("context pairs must have shape [batch, context_width]")
    if left_context.shape[0] < 2:
        raise ValueError("contrastive context matching needs at least two pairs")
    if not 0.0 < temperature:
        raise ValueError("temperature must be positive")
    if not bool(torch.isfinite(left_context).all()) or not bool(
        torch.isfinite(right_context).all()
    ):
        raise ValueError("contexts must contain only finite values")
    left = F.normalize(left_context, dim=-1)
    right = F.normalize(right_context, dim=-1)
    logits = left @ right.transpose(0, 1) / temperature
    labels = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def paired_event_credit_loss(
    credit_logits: torch.Tensor,
    utilities: torch.Tensor,
    *,
    present: torch.Tensor | None = None,
    positive_arm: int = 0,
    negative_arm: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train event credit from paired scalar counterfactual outcomes.

    ``utilities`` contains common-random verifier outcomes for a write/read
    intervention at every episode position.  The returned detached advantage
    is the causal target for each event; no correct action or task identifier
    is needed.  Per-episode centering removes a shared baseline and keeps the
    loss focused on which positions changed the result.
    """
    if credit_logits.ndim != 2 or utilities.ndim != 3:
        raise ValueError("credit logits/utilities have invalid ranks")
    if utilities.shape[:2] != credit_logits.shape or utilities.shape[2] < 2:
        raise ValueError("utilities must have shape [batch, time, at least two arms]")
    arms = utilities.shape[2]
    if not 0 <= positive_arm < arms or not 0 <= negative_arm < arms:
        raise ValueError("counterfactual arm is out of range")
    if positive_arm == negative_arm:
        raise ValueError("counterfactual arms must be distinct")
    if not bool(torch.isfinite(credit_logits).all()) or not bool(
        torch.isfinite(utilities).all()
    ):
        raise ValueError("credit inputs must be finite")
    if present is None:
        present = torch.ones_like(credit_logits, dtype=torch.bool)
    if present.shape != credit_logits.shape:
        raise ValueError("present must align with credit logits")
    advantage = (
        utilities[..., positive_arm] - utilities[..., negative_arm]
    ).detach()
    present_float = present.to(dtype=advantage.dtype)
    count = present_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (advantage * present_float).sum(dim=1, keepdim=True) / count
    centered = (advantage - mean) * present_float
    scale = centered.abs().sum(dim=1, keepdim=True).div(count).clamp_min(1e-6)
    target = (centered / scale).clamp(-4.0, 4.0)
    if not bool(present.any()):
        return credit_logits.sum() * 0.0, advantage
    loss = F.smooth_l1_loss(credit_logits[present], target[present])
    return loss, advantage
