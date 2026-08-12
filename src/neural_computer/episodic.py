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

import copy
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
    known: torch.Tensor


@dataclass(frozen=True)
class EpisodicBindingContext:
    """Learned route context plus an immutable generic episode signature."""

    context: torch.Tensor
    signature: torch.Tensor


@dataclass(frozen=True)
class EpisodicBindingLookup:
    """Opaque archive lookup result for one learned binding signature."""

    binding_id: int | None
    similarity: float
    active_slot: int | None


@dataclass(frozen=True)
class EpisodicBindingArchiveStatus:
    """Auditable external archive state without semantic task labels."""

    record_count: int
    active_slots: tuple[int | None, ...]
    attempts: tuple[int, ...]
    successes: tuple[float, ...]
    posterior: tuple[float, ...]
    stable_prefix_minimum: tuple[float, ...]
    protected: tuple[bool, ...]
    last_seen: tuple[int, ...]
    version: int


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


class EpisodicBindingArchive:
    """Growable external files for bindings behind a bounded active cache.

    The archive stores normalized learned context/signature keys and only
    generic scalar verifier telemetry.  Replacing an active router slot clears
    its cache residency, never the archived record.  A later episode can find
    the old record by its immutable trajectory signature and reactivate it
    without replaying the old training stream or changing the controller.

    This is deliberately a memory-side lifecycle primitive, not a reasoning
    branch.  It does not assign meaning to coordinates, expose task labels, or
    manufacture outcomes for unattempted bindings.
    """

    schema = "neural-computer.episodic-binding-archive.v1"

    def __init__(
        self,
        context_width: int,
        signature_width: int,
        *,
        active_slots: int,
        matching_threshold: float = 0.85,
        prior_strength: float = 1.0,
        mastery_threshold: float = 0.8,
        min_mastery_observations: int = 8,
    ) -> None:
        if min(context_width, signature_width, active_slots) < 1:
            raise ValueError("episodic binding archive dimensions must be positive")
        if not torch.isfinite(torch.tensor(matching_threshold)) or not (
            -1.0 <= matching_threshold <= 1.0
        ):
            raise ValueError("episodic binding archive matching threshold is invalid")
        if prior_strength <= 0.0:
            raise ValueError("episodic binding archive prior must be positive")
        if not 0.0 <= mastery_threshold <= 1.0:
            raise ValueError("episodic binding archive mastery threshold is invalid")
        if min_mastery_observations < 1:
            raise ValueError("episodic binding archive mastery observations are invalid")
        self.context_width = int(context_width)
        self.signature_width = int(signature_width)
        self.active_slots = int(active_slots)
        self.matching_threshold = float(matching_threshold)
        self.prior_strength = float(prior_strength)
        self.mastery_threshold = float(mastery_threshold)
        self.min_mastery_observations = int(min_mastery_observations)
        self._context_keys: list[tuple[float, ...]] = []
        self._signature_keys: list[tuple[float, ...]] = []
        self._attempts: list[int] = []
        self._successes: list[float] = []
        self._stable_prefix_minimum: list[float] = []
        self._last_seen: list[int] = []
        self._active_slots: list[int | None] = [None] * self.active_slots
        self._version = 0

    @property
    def record_count(self) -> int:
        return len(self._context_keys)

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "signature_width": self.signature_width,
            "active_slots": self.active_slots,
            "record_count": self.record_count,
            "matching_threshold": self.matching_threshold,
            "prior_strength": self.prior_strength,
            "mastery_threshold": self.mastery_threshold,
            "min_mastery_observations": self.min_mastery_observations,
            "state": "immutable_binding_records_plus_bounded_active_cache_v1",
        }

    @staticmethod
    def _normalize_key(
        value: torch.Tensor,
        *,
        width: int,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.ndim != 1 or value.shape[0] != width:
            raise ValueError(f"{name} must have shape [{width}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
        if not bool(value.square().sum().gt(1e-12)):
            raise ValueError(f"{name} cannot be zero")
        return F.normalize(value.detach().to(device="cpu", dtype=torch.float32), dim=0)

    def _validate_binding_id(self, binding_id: int) -> None:
        if not isinstance(binding_id, int) or not 0 <= binding_id < self.record_count:
            raise IndexError("episodic binding archive record is out of range")

    def _validate_slot(self, slot: int) -> None:
        if not isinstance(slot, int) or not 0 <= slot < self.active_slots:
            raise IndexError("episodic binding archive active slot is out of range")

    def lookup(
        self,
        signature_key: torch.Tensor,
        *,
        threshold: float | None = None,
    ) -> EpisodicBindingLookup:
        """Find the nearest immutable record, if it clears the novelty gate."""

        signature = self._normalize_key(
            signature_key,
            width=self.signature_width,
            name="episodic binding archive signature",
        )
        selected_threshold = (
            self.matching_threshold if threshold is None else float(threshold)
        )
        if not torch.isfinite(torch.tensor(selected_threshold)) or not (
            -1.0 <= selected_threshold <= 1.0
        ):
            raise ValueError("episodic binding archive lookup threshold is invalid")
        if not self._signature_keys:
            return EpisodicBindingLookup(None, -1.0, None)
        keys = torch.tensor(self._signature_keys, dtype=signature.dtype)
        scores = keys @ signature
        best_score, best_index = scores.max(dim=0)
        value = float(best_score)
        binding_id = int(best_index)
        if value < selected_threshold:
            return EpisodicBindingLookup(None, value, None)
        active_slot = next(
            (
                slot
                for slot, active_binding in enumerate(self._active_slots)
                if active_binding == binding_id
            ),
            None,
        )
        return EpisodicBindingLookup(binding_id, value, active_slot)

    def register(
        self,
        context_key: torch.Tensor,
        signature_key: torch.Tensor,
    ) -> int:
        """Append a novel external record or return its existing opaque id."""

        context = self._normalize_key(
            context_key,
            width=self.context_width,
            name="episodic binding archive context",
        )
        signature = self._normalize_key(
            signature_key,
            width=self.signature_width,
            name="episodic binding archive signature",
        )
        existing = self.lookup(signature)
        if existing.binding_id is not None:
            return existing.binding_id
        self._context_keys.append(tuple(float(value) for value in context.tolist()))
        self._signature_keys.append(tuple(float(value) for value in signature.tolist()))
        self._attempts.append(0)
        self._successes.append(0.0)
        self._stable_prefix_minimum.append(1.0)
        self._last_seen.append(-1)
        self._version += 1
        return self.record_count - 1

    def context_key(self, binding_id: int) -> torch.Tensor:
        self._validate_binding_id(binding_id)
        return torch.tensor(self._context_keys[binding_id], dtype=torch.float32)

    def signature_key(self, binding_id: int) -> torch.Tensor:
        self._validate_binding_id(binding_id)
        return torch.tensor(self._signature_keys[binding_id], dtype=torch.float32)

    def active_binding(self, slot: int) -> int | None:
        self._validate_slot(slot)
        return self._active_slots[slot]

    def binding_slot(self, binding_id: int) -> int | None:
        self._validate_binding_id(binding_id)
        return next(
            (
                slot
                for slot, active_binding in enumerate(self._active_slots)
                if active_binding == binding_id
            ),
            None,
        )

    @property
    def active_binding_ids(self) -> tuple[int | None, ...]:
        return tuple(self._active_slots)

    def activate(self, binding_id: int, slot: int) -> None:
        """Move one archived record into one active cache slot."""

        self._validate_binding_id(binding_id)
        self._validate_slot(slot)
        for index, active_binding in enumerate(self._active_slots):
            if active_binding == binding_id:
                self._active_slots[index] = None
        self._active_slots[slot] = binding_id
        self._version += 1

    def deactivate(self, slot: int) -> int | None:
        """Remove one active residency while retaining its archive record."""

        self._validate_slot(slot)
        binding_id = self._active_slots[slot]
        self._active_slots[slot] = None
        self._version += 1
        return binding_id

    def observe(
        self,
        binding_id: int,
        outcome: float | torch.Tensor,
        *,
        step: int,
    ) -> None:
        """Update only generic scalar evidence for an attempted record."""

        self._validate_binding_id(binding_id)
        if not isinstance(step, int) or step < 0:
            raise ValueError("episodic binding archive step must be non-negative")
        if isinstance(outcome, torch.Tensor):
            if outcome.numel() != 1:
                raise ValueError("episodic binding archive outcome must be scalar")
            value = float(outcome.detach().cpu().item())
        else:
            value = float(outcome)
        if not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0:
            raise ValueError("episodic binding archive outcome must lie in [0, 1]")
        self._attempts[binding_id] += 1
        self._successes[binding_id] += value
        mean = self._successes[binding_id] / self._attempts[binding_id]
        self._stable_prefix_minimum[binding_id] = min(
            self._stable_prefix_minimum[binding_id], mean
        )
        self._last_seen[binding_id] = step
        self._version += 1

    def posterior(self, binding_id: int) -> float:
        """Return a Beta-smoothed scalar reliability estimate."""

        self._validate_binding_id(binding_id)
        return (
            self._successes[binding_id] + self.prior_strength
        ) / (self._attempts[binding_id] + 2.0 * self.prior_strength)

    def is_protected(self, binding_id: int) -> bool:
        """Return whether the stable-prefix verifier gate protects a record."""

        self._validate_binding_id(binding_id)
        return (
            self._attempts[binding_id] >= self.min_mastery_observations
            and self._stable_prefix_minimum[binding_id] >= self.mastery_threshold
        )

    def telemetry(
        self,
        binding_id: int,
        *,
        step: int,
        age_horizon: int = 1,
    ) -> tuple[float, float]:
        """Return generic reliability/recency features for a memory policy."""

        self._validate_binding_id(binding_id)
        if not isinstance(step, int) or step < 0 or age_horizon < 1:
            raise ValueError("episodic binding archive telemetry horizon is invalid")
        last_seen = self._last_seen[binding_id]
        age = step + 1 if last_seen < 0 else max(0, step - last_seen)
        return self.posterior(binding_id), min(1.0, age / age_horizon)

    def status(self) -> EpisodicBindingArchiveStatus:
        return EpisodicBindingArchiveStatus(
            record_count=self.record_count,
            active_slots=self.active_binding_ids,
            attempts=tuple(self._attempts),
            successes=tuple(self._successes),
            posterior=tuple(self.posterior(index) for index in range(self.record_count)),
            stable_prefix_minimum=tuple(self._stable_prefix_minimum),
            protected=tuple(
                self.is_protected(index) for index in range(self.record_count)
            ),
            last_seen=tuple(self._last_seen),
            version=self._version,
        )

    def payload(self) -> dict[str, object]:
        """Serialize the archive as a versioned, JSON-compatible memory file."""

        status = self.status()
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "context_keys": [list(key) for key in self._context_keys],
            "signature_keys": [list(key) for key in self._signature_keys],
            "attempts": list(status.attempts),
            "successes": list(status.successes),
            "stable_prefix_minimum": list(status.stable_prefix_minimum),
            "last_seen": list(status.last_seen),
            "active_slots": list(status.active_slots),
            "version": status.version,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> EpisodicBindingArchive:
        """Restore a validated archive snapshot without controller state."""

        if payload.get("schema") != cls.schema:
            raise ValueError("episodic binding archive schema is incompatible")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("episodic binding archive configuration is invalid")
        archive = cls(
            int(configuration["context_width"]),
            int(configuration["signature_width"]),
            active_slots=int(configuration["active_slots"]),
            matching_threshold=float(configuration["matching_threshold"]),
            prior_strength=float(configuration["prior_strength"]),
            mastery_threshold=float(configuration["mastery_threshold"]),
            min_mastery_observations=int(configuration["min_mastery_observations"]),
        )
        context_keys = payload.get("context_keys")
        signature_keys = payload.get("signature_keys")
        attempts = payload.get("attempts")
        successes = payload.get("successes")
        prefixes = payload.get("stable_prefix_minimum")
        last_seen = payload.get("last_seen")
        active_slots = payload.get("active_slots")
        rows = (context_keys, signature_keys, attempts, successes, prefixes, last_seen)
        if not all(isinstance(rows_value, list) for rows_value in rows):
            raise TypeError("episodic binding archive rows must be lists")
        count = len(context_keys)
        if any(len(rows_value) != count for rows_value in rows):
            raise ValueError("episodic binding archive rows have different lengths")
        if not isinstance(active_slots, list) or len(active_slots) != archive.active_slots:
            raise ValueError("episodic binding archive active slots are invalid")
        for context, signature, attempt, success, prefix, seen in zip(
            context_keys, signature_keys, attempts, successes, prefixes, last_seen
        ):
            if not isinstance(context, list) or not isinstance(signature, list):
                raise TypeError("episodic binding archive keys must be lists")
            context_tensor = archive._normalize_key(
                torch.tensor(context),
                width=archive.context_width,
                name="episodic binding archive context",
            )
            signature_tensor = archive._normalize_key(
                torch.tensor(signature),
                width=archive.signature_width,
                name="episodic binding archive signature",
            )
            if not isinstance(attempt, int) or attempt < 0:
                raise ValueError("episodic binding archive attempts are invalid")
            if not isinstance(success, (int, float)) or not 0.0 <= float(success) <= attempt:
                raise ValueError("episodic binding archive successes are invalid")
            if not isinstance(prefix, (int, float)) or not 0.0 <= float(prefix) <= 1.0:
                raise ValueError("episodic binding archive stable prefixes are invalid")
            if not isinstance(seen, int) or seen < -1:
                raise ValueError("episodic binding archive last-seen values are invalid")
            archive._context_keys.append(tuple(float(value) for value in context_tensor.tolist()))
            archive._signature_keys.append(tuple(float(value) for value in signature_tensor.tolist()))
            archive._attempts.append(attempt)
            archive._successes.append(float(success))
            archive._stable_prefix_minimum.append(float(prefix))
            archive._last_seen.append(seen)
        for slot, binding_id in enumerate(active_slots):
            if binding_id is not None and (
                not isinstance(binding_id, int)
                or not 0 <= binding_id < archive.record_count
            ):
                raise ValueError("episodic binding archive active id is invalid")
            archive._active_slots[slot] = binding_id
        archive._version = int(payload.get("version", 0))
        if archive._version < 0:
            raise ValueError("episodic binding archive version is invalid")
        return archive


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

    schema = "neural-computer.episodic-binding-router.v3"

    def __init__(
        self,
        event_width: int,
        action_width: int,
        *,
        hidden: int = 64,
        context_width: int = 32,
        max_slots: int | None = None,
        temperature: float = 0.2,
        route_threshold: float = 0.75,
        signature_weight: float = 0.5,
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
        if not torch.isfinite(torch.tensor(route_threshold)) or not (
            -1.0 <= route_threshold <= 1.0
        ):
            raise ValueError("episodic binding router route threshold is invalid")
        if not torch.isfinite(torch.tensor(signature_weight)) or not (
            0.0 <= signature_weight <= 1.0
        ):
            raise ValueError("episodic binding router signature weight is invalid")
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
        self.route_threshold = float(route_threshold)
        self.signature_weight = float(signature_weight)
        self.signature_width = 2 * self.event_width + self.action_width + 2
        self.slot_keys = nn.ParameterList()
        self.slot_signatures = nn.ParameterList()
        self.register_buffer("slot_frozen", torch.empty(0, dtype=torch.bool))
        self.register_buffer(
            "slot_signature_active", torch.empty(0, dtype=torch.bool)
        )

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
            "route_threshold": self.route_threshold,
            "signature_width": self.signature_width,
            "signature_weight": self.signature_weight,
            "frozen_slots": int(self.slot_frozen.sum().item()),
            "active_signatures": int(self.slot_signature_active.sum().item()),
            "state": "external_encoder_plus_immutable_episode_signature_v3",
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

    def binding_signature(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
        outcomes: torch.Tensor,
        present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a generic, non-semantic signature for binding discovery.

        The signature preserves first-observed and aggregate learned event
        content plus aggregate opaque action/outcome context.  It is not
        trainable route state and carries no task or protocol field.  Keeping
        this path separate from the learned route embedding gives novelty
        detection a stable reference when the learned encoder has only seen a
        subset of future bindings.
        """

        present = _validate_episode_inputs(events, actions, outcomes, present)
        present_float = present.to(dtype=events.dtype)
        count = present_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        first_index = present.to(torch.long).argmax(dim=1)
        first_event = events[
            torch.arange(events.shape[0], device=events.device), first_index
        ]
        mean_event = (events * present_float.unsqueeze(-1)).sum(dim=1) / count
        mean_action = (actions * present_float.unsqueeze(-1)).sum(dim=1) / count
        mean_outcome = (outcomes * present_float).sum(dim=1) / count.squeeze(-1)
        signature = torch.cat(
            (
                first_event,
                mean_event,
                mean_action,
                mean_outcome.unsqueeze(-1),
                present_float.mean(dim=1, keepdim=True),
            ),
            dim=-1,
        )
        return F.normalize(signature, dim=-1)

    def encode_binding(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
        outcomes: torch.Tensor,
        present: torch.Tensor | None = None,
    ) -> EpisodicBindingContext:
        """Encode one episode for learned routing and novelty discovery."""

        return EpisodicBindingContext(
            context=self.encode(events, actions, outcomes, present),
            signature=self.binding_signature(
                events, actions, outcomes, present
            ),
        )

    def _validate_signature(self, signature: torch.Tensor) -> None:
        if signature.ndim != 2 or signature.shape[1] != self.signature_width:
            raise ValueError(
                "episodic binding signature has the wrong shape"
            )
        if not bool(torch.isfinite(signature).all()):
            raise ValueError("episodic binding signature must be finite")

    def _validate_signature_key(self, signature_key: torch.Tensor) -> None:
        if (
            signature_key.ndim != 1
            or signature_key.shape[0] != self.signature_width
        ):
            raise ValueError(
                "episodic binding signature key has the wrong shape"
            )
        if not bool(torch.isfinite(signature_key).all()):
            raise ValueError("episodic binding signature key must be finite")
        if not bool(signature_key.square().sum().gt(1e-12)):
            raise ValueError("episodic binding signature key cannot be zero")

    @torch.no_grad()
    def add_slot(
        self,
        context_key: torch.Tensor,
        signature_key: torch.Tensor | None = None,
    ) -> int:
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
        if signature_key is None:
            signature = torch.zeros(
                self.signature_width,
                device=reference.device,
                dtype=reference.dtype,
            )
            signature_active = False
        else:
            self._validate_signature_key(signature_key)
            signature = F.normalize(signature_key.detach(), dim=0).to(reference)
            signature_active = True
        self.slot_signatures.append(nn.Parameter(signature, requires_grad=False))
        self.slot_frozen = torch.cat(
            (
                self.slot_frozen,
                torch.zeros(1, dtype=torch.bool, device=key.device),
            )
        )
        self.slot_signature_active = torch.cat(
            (
                self.slot_signature_active,
                torch.tensor(
                    [signature_active],
                    dtype=torch.bool,
                    device=key.device,
                ),
            )
        )
        return self.slot_count - 1

    @torch.no_grad()
    def freeze_slot(self, slot_index: int) -> None:
        """Mark one externally verified binding as protected from replacement."""

        if not 0 <= slot_index < self.slot_count:
            raise IndexError("episodic binding router slot is out of range")
        self.slot_frozen[slot_index] = True

    @torch.no_grad()
    def slot_replacement_candidate(
        self,
        slot_index: int,
        context_key: torch.Tensor,
        signature_key: torch.Tensor | None = None,
    ) -> EpisodicBindingRouter:
        """Build a copy-on-write candidate for one physical binding slot."""

        if not 0 <= slot_index < self.slot_count:
            raise IndexError("episodic binding router slot is out of range")
        if context_key.ndim != 1 or context_key.shape[0] != self.context_width:
            raise ValueError("episodic binding replacement key has the wrong shape")
        if not bool(torch.isfinite(context_key).all()):
            raise ValueError("episodic binding replacement key must be finite")
        if not bool(context_key.square().sum().gt(1e-12)):
            raise ValueError("episodic binding replacement key cannot be zero")
        candidate = copy.deepcopy(self)
        reference = next(candidate.encoder.parameters())
        key = F.normalize(context_key.detach(), dim=0).to(reference)
        candidate.slot_keys[slot_index] = nn.Parameter(key, requires_grad=False)
        if signature_key is not None:
            candidate._validate_signature_key(signature_key)
            signature = F.normalize(signature_key.detach(), dim=0).to(reference)
            candidate.slot_signatures[slot_index] = nn.Parameter(
                signature,
                requires_grad=False,
            )
            candidate.slot_signature_active[slot_index] = True
        candidate.slot_frozen[slot_index] = False
        return candidate

    @torch.no_grad()
    def replace_slot_from_candidate(
        self,
        candidate: EpisodicBindingRouter,
        slot_index: int,
        *,
        retention_probe=None,
    ) -> bool:
        """Commit a replacement only after an independent retention probe."""

        if not isinstance(candidate, EpisodicBindingRouter):
            raise TypeError("episodic binding replacement candidate is invalid")
        if not 0 <= slot_index < self.slot_count:
            raise IndexError("episodic binding router slot is out of range")
        if candidate.slot_count != self.slot_count:
            raise ValueError("episodic binding candidate slot count changed")
        if any(
            not torch.equal(value, candidate.encoder.state_dict()[name])
            for name, value in self.encoder.state_dict().items()
        ):
            raise ValueError("episodic binding candidate changed the encoder")
        expected = self.configuration().copy()
        observed = candidate.configuration().copy()
        expected.pop("frozen_slots", None)
        observed.pop("frozen_slots", None)
        if expected != observed:
            raise ValueError("episodic binding candidate configuration changed")
        for index in range(self.slot_count):
            if index == slot_index:
                continue
            if not torch.equal(self.slot_keys[index], candidate.slot_keys[index]):
                raise ValueError("episodic binding candidate changed a sibling key")
            if bool(self.slot_frozen[index]) != bool(candidate.slot_frozen[index]):
                raise ValueError("episodic binding candidate changed a sibling state")
            if not torch.equal(
                self.slot_signatures[index], candidate.slot_signatures[index]
            ):
                raise ValueError(
                    "episodic binding candidate changed a sibling signature"
                )
            if bool(self.slot_signature_active[index]) != bool(
                candidate.slot_signature_active[index]
            ):
                raise ValueError(
                    "episodic binding candidate changed a sibling signature state"
                )
        accepted = retention_probe is None or bool(retention_probe(candidate))
        if not accepted:
            return False
        self.slot_keys[slot_index].data.copy_(candidate.slot_keys[slot_index].data)
        self.slot_signatures[slot_index].data.copy_(
            candidate.slot_signatures[slot_index].data
        )
        self.slot_frozen[slot_index] = candidate.slot_frozen[slot_index]
        self.slot_signature_active[slot_index] = candidate.slot_signature_active[
            slot_index
        ]
        return True

    def route_scores(
        self,
        context: torch.Tensor,
        *,
        slot_order: torch.Tensor | None = None,
        signature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return cosine scores, optionally under a physical slot permutation."""

        if context.ndim != 2 or context.shape[1] != self.context_width:
            raise ValueError("episodic binding router context has the wrong shape")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("episodic binding router context must be finite")
        if self.slot_count < 1:
            raise RuntimeError("episodic binding router has no slots")
        keys = torch.stack(tuple(self.slot_keys), dim=0).to(context)
        learned_scores = F.normalize(context, dim=-1) @ keys.transpose(0, 1)
        order = None
        if slot_order is not None:
            if (
                slot_order.ndim != 1
                or slot_order.shape[0] != self.slot_count
                or slot_order.dtype
                not in (torch.int8, torch.int16, torch.int32, torch.int64)
            ):
                raise ValueError("episodic binding router slot order is invalid")
            if sorted(slot_order.detach().cpu().tolist()) != list(
                range(self.slot_count)
            ):
                raise ValueError(
                    "episodic binding router slot order is not a permutation"
                )
            order = slot_order.to(device=keys.device)
        if signature is None or not bool(self.slot_signature_active.any()):
            return learned_scores if order is None else learned_scores[:, order]
        self._validate_signature(signature)
        if signature.shape[0] != context.shape[0]:
            raise ValueError("episodic binding signature batch does not match")
        signatures = torch.stack(tuple(self.slot_signatures), dim=0).to(context)
        signature_scores = F.normalize(signature, dim=-1) @ signatures.transpose(
            0, 1
        )
        active = self.slot_signature_active.to(context.device)
        combined = torch.where(
            active.unsqueeze(0),
            (1.0 - self.signature_weight) * learned_scores
            + self.signature_weight * signature_scores,
            learned_scores,
        )
        return combined if order is None else combined[:, order]

    def route(
        self,
        context: torch.Tensor,
        *,
        slot_order: torch.Tensor | None = None,
        route_threshold: float | None = None,
        signature: torch.Tensor | None = None,
    ) -> EpisodicBindingRoute:
        scores = self.route_scores(
            context,
            slot_order=slot_order,
            signature=signature,
        )
        selected_threshold = (
            self.route_threshold
            if route_threshold is None
            else float(route_threshold)
        )
        if not torch.isfinite(torch.tensor(selected_threshold)) or not (
            -1.0 <= selected_threshold <= 1.0
        ):
            raise ValueError("episodic binding route threshold is invalid")
        if signature is not None and bool(self.slot_signature_active.any()):
            self._validate_signature(signature)
            signatures = torch.stack(tuple(self.slot_signatures), dim=0).to(
                context
            )
            signature_scores = F.normalize(signature, dim=-1) @ signatures.T
            active = self.slot_signature_active.to(context.device)
            known_score = signature_scores.masked_fill(
                ~active.unsqueeze(0),
                torch.finfo(signature_scores.dtype).min,
            ).max(dim=-1).values
        else:
            known_score = scores.max(dim=-1).values
        return EpisodicBindingRoute(
            context=context,
            scores=scores,
            selected_slot=scores.argmax(dim=-1),
            known=known_score >= selected_threshold,
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
        signature: torch.Tensor | None = None,
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
        scores = self.route_scores(context, signature=signature)
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
