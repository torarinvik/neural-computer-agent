"""External learned transition models and protocol-agnostic search.

The controller remains outside this module.  A transition model stores learned
facts about how an opaque state changes after an opaque intention; a planner
derives a candidate intention sequence at inference time from the current
state and an opaque goal state.  No device action IDs, task labels, modality
formats, or privileged simulator fields cross this boundary.

This is deliberately a small boundary rather than a claim of unrestricted
world-model learning.  Its purpose is to make the important continual-learning
property testable: new experience updates factual transition knowledge while
behavior is recomputed instead of being stored as a task-specific policy.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .growth import compress_growth_artifact, decompress_growth_artifact
from .memory import (
    AppendOnlyContentAddressedMemory,
    MemoryQuery,
    MemoryWriteReceipt,
)

if TYPE_CHECKING:
    from .online_transition import ExternalTransitionModelFamilySelection

EXTERNAL_TRANSITION_OBSERVATION_SCHEMA = (
    "neural-computer.external-transition-observation.v1"
)
EXTERNAL_TRANSITION_MODEL_SCHEMA = "neural-computer.external-transition-model.v1"
EXTERNAL_TRANSITION_MEMORY_SCHEMA = "neural-computer.external-transition-memory.v1"
EXTERNAL_CONTEXT_ADDRESS_RESOLVER_SCHEMA = (
    "neural-computer.external-context-address-resolver.v1"
)
EXTERNAL_ONLINE_CONTEXT_RESOLVER_SCHEMA = (
    "neural-computer.external-online-context-resolver.v1"
)
EXTERNAL_GOAL_EVALUATOR_SCHEMA = "neural-computer.external-goal-evaluator.v1"
EXTERNAL_TRANSITION_EVIDENCE_EVALUATOR_SCHEMA = (
    "neural-computer.external-transition-evidence-evaluator.v1"
)
EXTERNAL_TRANSITION_EVIDENCE_CALIBRATOR_SCHEMA = (
    "neural-computer.external-transition-evidence-calibrator.v1"
)
EXTERNAL_CONTEXTUAL_EVIDENCE_CALIBRATOR_SCHEMA = (
    "neural-computer.contextual-evidence-calibrator.v1"
)
EXTERNAL_TRANSITION_CONTEXT_ENCODER_SCHEMA = (
    "neural-computer.external-transition-context-encoder.v1"
)
EXTERNAL_ONLINE_TRANSITION_CONTEXT_ROUTER_SCHEMA = (
    "neural-computer.external-online-transition-context-router.v1"
)
EXTERNAL_TRANSITION_MODEL_BANK_SCHEMA = (
    "neural-computer.external-transition-model-bank.v1"
)
EXTERNAL_TRANSITION_MODEL_GROWTH_SCHEMA = (
    "neural-computer.external-transition-model-growth.v1"
)
EXTERNAL_TRANSITION_MODEL_EVICTION_SCHEMA = (
    "neural-computer.external-transition-model-eviction.v1"
)
EXTERNAL_TRANSITION_MODEL_SLOT_ADDRESS_SCHEMA = (
    "neural-computer.external-transition-model-slot-address.v1"
)
EXTERNAL_TRANSITION_MODEL_LIFETIME_POLICY_SCHEMA = (
    "neural-computer.external-transition-model-lifetime-policy.v1"
)
EXTERNAL_TRANSITION_MODEL_LIFETIME_TELEMETRY_SCHEMA = (
    "neural-computer.external-transition-model-lifetime-telemetry.v1"
)
EXTERNAL_TRANSITION_MODEL_CANDIDATE_SCHEMA = (
    "neural-computer.external-transition-model-candidate.v1"
)
EXTERNAL_TRANSITION_MODEL_CONSOLIDATION_SCHEMA = (
    "neural-computer.external-transition-model-consolidation.v1"
)
EXTERNAL_TRANSITION_MODEL_COMPRESSION_SCHEMA = (
    "neural-computer.external-transition-model-compression.v1"
)
EXTERNAL_TRANSITION_MODEL_COMPRESSION_SELECTION_SCHEMA = (
    "neural-computer.external-transition-model-compression-selection.v1"
)
EXTERNAL_TRANSITION_MODEL_FAMILY_SELECTION_SCHEMA = (
    "neural-computer.external-transition-model-family-selection.v1"
)
EXTERNAL_MODEL_PLANNER_SCHEMA = "neural-computer.external-model-planner.v1"
EXTERNAL_GOAL_CONDITIONED_MODEL_SELECTION_SCHEMA = (
    "neural-computer.external-goal-conditioned-model-selection.v1"
)

EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY = "nonlinear_mlp_v1"
EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY = "affine_sufficient_statistics_v1"
EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY = "mixed_verified_v1"
EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY = (
    "random_feature_sufficient_statistics_v1"
)


def _validate_tensor(
    value: torch.Tensor,
    *,
    name: str,
    ndim: int,
    width: int | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if width is not None and value.shape[-1] != width:
        raise ValueError(f"{name} has the wrong width")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class ExternalTransitionObservation:
    """One opaque, self-supervised transition observation.

    ``state`` and ``next_state`` are learned event representations.  The
    intention is learned output content, not a protocol action identifier.
    ``confidence`` is a generic verifier/front-end reliability scalar and does
    not carry privileged task information.
    """

    state: torch.Tensor
    intention: torch.Tensor
    next_state: torch.Tensor
    confidence: torch.Tensor | None = None
    schema: str = EXTERNAL_TRANSITION_OBSERVATION_SCHEMA

    def validate(
        self,
        *,
        state_width: int,
        intention_width: int,
    ) -> ExternalTransitionObservation:
        if self.schema != EXTERNAL_TRANSITION_OBSERVATION_SCHEMA:
            raise ValueError("unsupported transition-observation schema")
        _validate_tensor(
            self.state,
            name="state",
            ndim=2,
            width=state_width,
        )
        _validate_tensor(
            self.intention,
            name="intention",
            ndim=2,
            width=intention_width,
        )
        _validate_tensor(
            self.next_state,
            name="next_state",
            ndim=2,
            width=state_width,
        )
        if self.state.shape[0] != self.intention.shape[0] or (
            self.state.shape[0] != self.next_state.shape[0]
        ):
            raise ValueError("transition observation batch dimensions differ")
        if self.confidence is not None:
            if self.confidence.shape not in (
                (self.state.shape[0],),
                (self.state.shape[0], 1),
            ):
                raise ValueError("transition confidence must match the batch")
            if not bool(torch.isfinite(self.confidence).all()):
                raise ValueError("transition confidence must be finite")
            if bool(torch.any(self.confidence < 0)):
                raise ValueError("transition confidence cannot be negative")
        return self


class ExternalTransitionModel(nn.Module):
    """A replaceable external model of opaque state transitions.

    Parameters belong to the external memory/model component, never to the
    cognitive controller.  Training is intentionally caller-owned: a runtime
    may update this module from verified observations while the controller is
    frozen, and a checkpoint may persist the module independently.
    """

    schema = EXTERNAL_TRANSITION_MODEL_SCHEMA

    def __init__(
        self,
        state_width: int,
        intention_width: int,
        *,
        hidden_width: int = 64,
    ) -> None:
        super().__init__()
        if min(state_width, intention_width, hidden_width) < 1:
            raise ValueError("transition-model dimensions must be positive")
        self.state_width = int(state_width)
        self.intention_width = int(intention_width)
        self.hidden_width = int(hidden_width)
        self.network = nn.Sequential(
            nn.Linear(self.state_width + self.intention_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, self.state_width),
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "hidden_width": self.hidden_width,
            "representation": "opaque_learned_state_and_intention_v1",
            "training": "caller_owned_self_supervised_transition_outcomes_v1",
            "behavior": "derived_by_external_search_not_stored_policy_v1",
        }

    def _validate_inputs(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
    ) -> None:
        _validate_tensor(
            state,
            name="state",
            ndim=2,
            width=self.state_width,
        )
        _validate_tensor(
            intention,
            name="intention",
            ndim=2,
            width=self.intention_width,
        )
        if state.shape[0] != intention.shape[0]:
            raise ValueError("state and intention batches differ")

    def forward(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(state, intention)
        return self.network(torch.cat((state, intention), dim=-1))

    def loss(self, observation: ExternalTransitionObservation) -> torch.Tensor:
        """Return confidence-weighted self-supervised prediction loss."""

        observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )
        prediction = self(observation.state, observation.intention)
        errors = (prediction - observation.next_state).square().mean(dim=-1)
        if observation.confidence is None:
            return errors.mean()
        confidence = observation.confidence.reshape(-1).to(
            device=errors.device,
            dtype=errors.dtype,
        )
        return (errors * confidence).sum() / confidence.sum().clamp_min(1e-12)

    def state_payload(self) -> dict[str, Any]:
        """Return a detached, versioned payload for external persistence."""

        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "sha256": self.digest(),
        }

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        for name, value in sorted(self.state_dict().items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExternalTransitionModel:
        """Restore a model and reject schema, shape, or checksum drift."""

        if not isinstance(payload, Mapping):
            raise TypeError("transition-model payload must be a mapping")
        if payload.get("schema") != EXTERNAL_TRANSITION_MODEL_SCHEMA:
            raise ValueError("unsupported transition-model schema")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("transition-model configuration is missing")
        required = ("state_width", "intention_width", "hidden_width")
        if any(not isinstance(configuration.get(name), int) for name in required):
            raise TypeError("transition-model dimensions are missing")
        model = cls(
            int(configuration["state_width"]),
            int(configuration["intention_width"]),
            hidden_width=int(configuration["hidden_width"]),
        )
        state = payload.get("state")
        if not isinstance(state, Mapping):
            raise TypeError("transition-model state is missing")
        current = model.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("transition-model state names do not match")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"transition-model state {name!r} is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"transition-model state {name!r} is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"transition-model state {name!r} is not finite")
            normalized[name] = value.detach().clone()
        model.load_state_dict(normalized, strict=True)
        expected_digest = payload.get("sha256")
        if expected_digest != model.digest():
            raise ValueError("transition-model checksum mismatch")
        return model


@dataclass(frozen=True)
class ExternalTransitionModelLifetimeTelemetry:
    """Bank-owned generic lifetime state ordered by stable logical address.

    This state is operational metadata for external memory management, not a
    semantic label.  It can grow, update, and persist independently while the
    controller and transition-model weights remain frozen.
    """

    slot_ids: tuple[int, ...]
    usage: torch.Tensor
    age: torch.Tensor
    prediction_error: torch.Tensor
    logical_clock: int
    schema: str = EXTERNAL_TRANSITION_MODEL_LIFETIME_TELEMETRY_SCHEMA

    def validate(self) -> ExternalTransitionModelLifetimeTelemetry:
        if self.schema != EXTERNAL_TRANSITION_MODEL_LIFETIME_TELEMETRY_SCHEMA:
            raise ValueError("unsupported transition lifetime telemetry schema")
        if self.logical_clock < 0:
            raise ValueError("transition lifetime logical clock cannot be negative")
        if len(set(self.slot_ids)) != len(self.slot_ids) or any(
            slot_id < 0 for slot_id in self.slot_ids
        ):
            raise ValueError("transition lifetime telemetry slot IDs are invalid")
        count = len(self.slot_ids)
        for name, value in (
            ("usage", self.usage),
            ("age", self.age),
            ("prediction error", self.prediction_error),
        ):
            if value.ndim != 1 or value.shape[0] != count:
                raise ValueError(f"transition lifetime telemetry {name} is misaligned")
            if not bool(torch.isfinite(value).all()) or bool(torch.any(value < 0)):
                raise ValueError(f"transition lifetime telemetry {name} is invalid")
        return self


class ExternalTransitionModelBank(nn.Module):
    """Append-only external transition-model slots keyed by opaque context.

    Each slot is a replaceable factual model rather than a policy.  A new slot
    may copy an existing model as a transfer prior, but subsequent optimizer
    updates are selected by the caller and affect only the addressed slot.
    The controller never sees this bank's parameters or context semantics.
    """

    schema = EXTERNAL_TRANSITION_MODEL_BANK_SCHEMA

    def __init__(
        self,
        state_width: int,
        intention_width: int,
        context_width: int,
        *,
        hidden_width: int = 64,
        matching_tolerance: float = 1e-4,
        capacity: int | None = None,
        model_family: str = EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
        affine_ridge: float = 1e-5,
        adaptation_learning_rate: float = 1e-2,
        random_feature_width: int = 128,
        random_feature_seed: int = 0,
    ) -> None:
        super().__init__()
        if min(state_width, intention_width, context_width, hidden_width) < 1:
            raise ValueError("transition-model bank dimensions must be positive")
        if matching_tolerance < 0.0:
            raise ValueError("transition-model context tolerance cannot be negative")
        if capacity is not None and capacity < 1:
            raise ValueError("transition-model bank capacity must be positive")
        if model_family not in {
            EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
            EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY,
            EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        }:
            raise ValueError("unsupported external transition-model family")
        if affine_ridge <= 0.0:
            raise ValueError("affine transition ridge must be positive")
        if adaptation_learning_rate <= 0.0 or not math.isfinite(adaptation_learning_rate):
            raise ValueError("transition-model adaptation learning rate must be positive")
        if random_feature_width < 1:
            raise ValueError("random-feature width must be positive")
        self.state_width = int(state_width)
        self.intention_width = int(intention_width)
        self.context_width = int(context_width)
        self.hidden_width = int(hidden_width)
        self.matching_tolerance = float(matching_tolerance)
        self.capacity = None if capacity is None else int(capacity)
        self.model_family = str(model_family)
        self.affine_ridge = float(affine_ridge)
        self.adaptation_learning_rate = float(adaptation_learning_rate)
        self.random_feature_width = int(random_feature_width)
        self.random_feature_seed = int(random_feature_seed)
        self.models = nn.ModuleList()
        self._contexts: list[torch.Tensor] = []
        self._model_families: list[str] = []
        self._slot_ids: list[int] = []
        self._next_slot_id = 0
        self._lifetime_clock = 0
        self._lifetime_usage: dict[int, int] = {}
        self._lifetime_last_access: dict[int, int] = {}
        self._lifetime_prediction_error: dict[int, float] = {}

    @property
    def replay_free_updates(self) -> bool:
        """Whether this bank family consumes each observation once in-place."""

        return self.model_family == EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY

    def _new_model(self, model_family: str | None = None) -> nn.Module:
        selected_family = self.model_family if model_family is None else model_family
        if selected_family == EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY:
            selected_family = EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY
        if selected_family not in {
            EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
            EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        }:
            raise ValueError("unsupported external transition-model family")
        if selected_family == EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY:
            return ExternalTransitionModel(
                self.state_width,
                self.intention_width,
                hidden_width=self.hidden_width,
            )
        # Keep this import local: online_transition depends on the observation
        # type defined in this module, while the bank only needs the optional
        # fast-path implementation when it instantiates one.
        from .online_transition import (
            ExternalAffineTransitionStatistics,
            ExternalRandomFeatureTransitionStatistics,
        )

        if selected_family == EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY:
            return ExternalRandomFeatureTransitionStatistics(
                self.state_width,
                self.intention_width,
                feature_width=self.random_feature_width,
                ridge=self.affine_ridge,
                seed=self.random_feature_seed,
            )

        return ExternalAffineTransitionStatistics(
            self.state_width,
            self.intention_width,
            ridge=self.affine_ridge,
        )

    def new_model(self, model_family: str) -> nn.Module:
        """Create an uncommitted candidate for verifier-gated family selection."""

        return self._new_model(model_family)

    def model_family_at(self, index: int) -> str:
        """Return the opaque implementation family assigned to one slot."""

        if not 0 <= index < self.context_count:
            raise IndexError("transition-model context is out of range")
        return self._model_families[index]

    @property
    def slot_ids(self) -> tuple[int, ...]:
        """Return stable logical addresses in current physical-slot order."""

        return tuple(self._slot_ids)

    def slot_id_at(self, index: int) -> int:
        """Return the stable logical address for a physical slot."""

        if not 0 <= index < self.context_count:
            raise IndexError("transition-model context index is out of range")
        return self._slot_ids[index]

    def physical_index_for_slot_id(self, slot_id: int) -> int:
        """Resolve a stable logical address to its current physical index."""

        if not isinstance(slot_id, int) or isinstance(slot_id, bool) or slot_id < 0:
            raise ValueError("transition-model slot ID must be a non-negative integer")
        try:
            return self._slot_ids.index(slot_id)
        except ValueError as error:
            raise KeyError(f"unknown transition-model slot ID: {slot_id}") from error

    def select_model_family_verified(
        self,
        candidates: Mapping[str, nn.Module],
        heldout_observation: ExternalTransitionObservation,
        *,
        prediction_tolerance: float = 0.05,
        retention_probe: Callable[[nn.Module], bool] | None = None,
    ) -> ExternalTransitionModelFamilySelection:
        """Select an opaque model family without mutating the live bank."""

        from .online_transition import select_verified_transition_model_family

        heldout_observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )
        return select_verified_transition_model_family(
            candidates,
            heldout_observation,
            prediction_tolerance=prediction_tolerance,
            retention_probe=retention_probe,
        )

    @property
    def context_count(self) -> int:
        return len(self._contexts)

    @property
    def physical_model_count(self) -> int:
        """Return the number of unique parameter objects behind all contexts."""

        return len({id(model) for model in self.models})

    @property
    def contexts(self) -> torch.Tensor:
        """Return a detached snapshot of the opaque slot keys."""

        if not self._contexts:
            return torch.empty((0, self.context_width), dtype=torch.float32)
        return torch.stack(self._contexts).detach().clone()

    def context_at(self, index: int) -> torch.Tensor:
        """Return one detached opaque slot key without exposing semantics."""

        if not 0 <= index < self.context_count:
            raise IndexError("transition-model context index is out of range")
        return self._contexts[index].detach().clone()

    def _validate_context(self, context: torch.Tensor) -> torch.Tensor:
        _validate_tensor(
            context,
            name="transition-model context",
            ndim=2,
            width=self.context_width,
        )
        norms = torch.linalg.vector_norm(context, dim=-1)
        if bool(torch.any(norms <= 1e-12)):
            raise ValueError("transition-model contexts must be non-zero")
        return torch.nn.functional.normalize(context.detach().to("cpu"), dim=-1)

    def _nearest_context(self, normalized: torch.Tensor) -> int | None:
        if not self._contexts:
            return None
        distances = torch.linalg.vector_norm(
            torch.stack(self._contexts) - normalized,
            dim=-1,
        )
        nearest = int(distances.argmin())
        if float(distances[nearest]) <= self.matching_tolerance:
            return nearest
        return None

    def ensure_context(
        self,
        context: torch.Tensor,
        *,
        initialize_from: int | None = None,
        model_family: str | None = None,
    ) -> int:
        """Return a slot or append one, optionally copying a transfer prior."""

        normalized = self._validate_context(
            context if context.ndim == 2 else context.unsqueeze(0)
        )[0]
        nearest = self._nearest_context(normalized)
        if nearest is not None:
            return nearest
        if initialize_from is not None and not 0 <= initialize_from < self.context_count:
            raise IndexError("transition-model transfer slot is out of range")
        if self.capacity is not None and self.context_count >= self.capacity:
            raise MemoryError("transition-model bank capacity is full")
        selected_family = self.model_family if model_family is None else model_family
        if selected_family == EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY:
            selected_family = EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY
        if selected_family not in {
            EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
            EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        }:
            raise ValueError("unsupported transition-model slot family")
        model = self._new_model(selected_family)
        if (
            initialize_from is not None
            and selected_family == EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY
        ):
            if self._model_families[initialize_from] != selected_family:
                raise ValueError("transition-model transfer families differ")
            model.load_state_dict(self.models[initialize_from].state_dict())
        self._contexts.append(normalized.clone())
        self.models.append(model)
        self._model_families.append(selected_family)
        self._slot_ids.append(self._next_slot_id)
        self._initialize_lifetime_slot(self._next_slot_id)
        self._next_slot_id += 1
        return self.context_count - 1

    def ensure_context_id(
        self,
        context: torch.Tensor,
        *,
        initialize_from: int | None = None,
        model_family: str | None = None,
    ) -> int:
        """Return a stable logical address, creating the slot if necessary."""

        index = self.ensure_context(
            context,
            initialize_from=initialize_from,
            model_family=model_family,
        )
        return self.slot_id_at(index)

    def _initialize_lifetime_slot(self, slot_id: int) -> None:
        self._lifetime_usage.setdefault(slot_id, 0)
        self._lifetime_last_access.setdefault(slot_id, self._lifetime_clock)

    def record_lifetime_observation(
        self,
        slot_id: int,
        prediction_error: float | None = None,
        *,
        elapsed_steps: int = 1,
    ) -> None:
        """Update generic lifetime telemetry from one committed observation."""

        self.record_lifetime_observations(
            (slot_id,),
            None if prediction_error is None else (prediction_error,),
            elapsed_steps=elapsed_steps,
        )

    def record_lifetime_observations(
        self,
        slot_ids: Sequence[int],
        prediction_errors: Sequence[float | None] | None = None,
        *,
        elapsed_steps: int = 1,
    ) -> None:
        """Update several accessed slots at one logical time step.

        A committed observation bundle is simultaneous for lifetime purposes:
        usage counts each row, but all rows receive the same logical timestamp.
        This prevents arbitrary minibatch ordering from becoming a learned age
        signal.
        """

        if not slot_ids:
            raise ValueError("transition lifetime observation batch cannot be empty")
        if (
            not isinstance(elapsed_steps, int)
            or isinstance(elapsed_steps, bool)
            or elapsed_steps < 1
        ):
            raise ValueError("transition lifetime elapsed steps must be positive")
        if prediction_errors is not None and len(prediction_errors) != len(slot_ids):
            raise ValueError("transition lifetime errors are misaligned")
        normalized_errors = (
            [None] * len(slot_ids)
            if prediction_errors is None
            else list(prediction_errors)
        )
        for slot_id, prediction_error in zip(
            slot_ids, normalized_errors, strict=True
        ):
            self.physical_index_for_slot_id(slot_id)
            if prediction_error is not None and (
                not math.isfinite(prediction_error) or prediction_error < 0.0
            ):
                raise ValueError("transition lifetime prediction error is invalid")
            self._initialize_lifetime_slot(slot_id)
        self._lifetime_clock += elapsed_steps
        for slot_id, prediction_error in zip(
            slot_ids, normalized_errors, strict=True
        ):
            self._lifetime_usage[slot_id] += 1
            self._lifetime_last_access[slot_id] = self._lifetime_clock
            if prediction_error is not None:
                previous = self._lifetime_prediction_error.get(slot_id)
                self._lifetime_prediction_error[slot_id] = float(
                    prediction_error
                    if previous is None
                    else 0.75 * previous + 0.25 * prediction_error
                )

    def lifetime_telemetry(self) -> ExternalTransitionModelLifetimeTelemetry:
        """Return bank-owned telemetry in current physical order."""

        usage = torch.tensor(
            [self._lifetime_usage.get(slot_id, 0) for slot_id in self._slot_ids],
            dtype=torch.float32,
        )
        age = torch.tensor(
            [
                self._lifetime_clock - self._lifetime_last_access.get(
                    slot_id, self._lifetime_clock
                )
                for slot_id in self._slot_ids
            ],
            dtype=torch.float32,
        )
        prediction_error = torch.tensor(
            [
                self._lifetime_prediction_error.get(slot_id, 0.0)
                for slot_id in self._slot_ids
            ],
            dtype=torch.float32,
        )
        return ExternalTransitionModelLifetimeTelemetry(
            slot_ids=self.slot_ids,
            usage=usage,
            age=age,
            prediction_error=prediction_error,
            logical_clock=self._lifetime_clock,
        ).validate()

    @staticmethod
    def _lifetime_telemetry_digest(payload: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        for key in (
            "schema",
            "logical_clock",
            "slot_ids",
            "usage",
            "last_access",
            "prediction_error",
        ):
            digest.update(key.encode("utf-8"))
            digest.update(repr(payload.get(key)).encode("utf-8"))
        return digest.hexdigest()

    def _lifetime_telemetry_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": EXTERNAL_TRANSITION_MODEL_LIFETIME_TELEMETRY_SCHEMA,
            "logical_clock": self._lifetime_clock,
            "slot_ids": list(self._slot_ids),
            "usage": [self._lifetime_usage.get(slot_id, 0) for slot_id in self._slot_ids],
            "last_access": [
                self._lifetime_last_access.get(slot_id, self._lifetime_clock)
                for slot_id in self._slot_ids
            ],
            "prediction_error": [
                self._lifetime_prediction_error.get(slot_id, 0.0)
                for slot_id in self._slot_ids
            ],
        }
        payload["sha256"] = self._lifetime_telemetry_digest(payload)
        return payload

    def _restore_lifetime_telemetry(self, payload: Mapping[str, Any] | None) -> None:
        self._lifetime_clock = 0
        self._lifetime_usage = {}
        self._lifetime_last_access = {}
        self._lifetime_prediction_error = {}
        if payload is None:
            for slot_id in self._slot_ids:
                self._initialize_lifetime_slot(slot_id)
            return
        if payload.get("schema") != EXTERNAL_TRANSITION_MODEL_LIFETIME_TELEMETRY_SCHEMA:
            raise ValueError("unsupported transition lifetime telemetry payload")
        if payload.get("sha256") != self._lifetime_telemetry_digest(payload):
            raise ValueError("transition lifetime telemetry checksum mismatch")
        slot_ids = payload.get("slot_ids")
        usage = payload.get("usage")
        last_access = payload.get("last_access")
        prediction_error = payload.get("prediction_error")
        if not all(isinstance(value, list) for value in (slot_ids, usage, last_access, prediction_error)):
            raise TypeError("transition lifetime telemetry lists are invalid")
        if slot_ids != list(self._slot_ids) or not (
            len(usage) == len(last_access) == len(prediction_error) == len(self._slot_ids)
        ):
            raise ValueError("transition lifetime telemetry slots are misaligned")
        logical_clock = payload.get("logical_clock")
        if not isinstance(logical_clock, int) or logical_clock < 0:
            raise ValueError("transition lifetime logical clock is invalid")
        self._lifetime_clock = logical_clock
        for slot_id, count, access, error in zip(
            self._slot_ids, usage, last_access, prediction_error, strict=True
        ):
            if (
                not isinstance(count, int)
                or count < 0
                or not isinstance(access, int)
                or access < 0
                or access > logical_clock
                or not isinstance(error, (float, int))
                or not math.isfinite(float(error))
                or float(error) < 0.0
            ):
                raise ValueError("transition lifetime telemetry value is invalid")
            self._lifetime_usage[slot_id] = count
            self._lifetime_last_access[slot_id] = access
            self._lifetime_prediction_error[slot_id] = float(error)

    def _context_indices(self, context: torch.Tensor) -> list[int]:
        normalized = self._validate_context(context)
        indices: list[int] = []
        for row in normalized:
            index = self._nearest_context(row)
            if index is None:
                raise KeyError(
                    "unknown transition-model context; call ensure_context first"
                )
            indices.append(index)
        return indices

    def configuration(self) -> dict[str, int | float | str | None]:
        configuration: dict[str, int | float | str | None] = {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "context_width": self.context_width,
            "hidden_width": self.hidden_width,
            "model_family": self.model_family,
            "model_families": list(self._model_families),
            "affine_ridge": self.affine_ridge,
            "adaptation_learning_rate": self.adaptation_learning_rate,
            "random_feature_width": self.random_feature_width,
            "random_feature_seed": self.random_feature_seed,
            "matching_tolerance": self.matching_tolerance,
            "growth": "append_only_isolated_model_slots_v1",
            "slot_addressing": EXTERNAL_TRANSITION_MODEL_SLOT_ADDRESS_SCHEMA,
            "behavior": "derived_by_external_model_search_v1",
            "updates": "caller_or_bank_owned_optimizer_v1",
        }
        if self.capacity is not None:
            configuration["capacity"] = self.capacity
        return configuration

    @staticmethod
    def _payload_model_families(
        configuration: Mapping[str, Any],
        count: int,
    ) -> list[str]:
        families = configuration.get("model_families")
        if families is None:
            family = str(
                configuration.get(
                    "model_family", EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY
                )
            )
            if family == EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY:
                family = EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY
            families = [family] * count
        if not isinstance(families, list) or len(families) != count:
            raise ValueError("transition-model payload family list is invalid")
        allowed = {
            EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
            EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        }
        normalized = [str(family) for family in families]
        if any(family not in allowed for family in normalized):
            raise ValueError("transition-model payload contains an unknown family")
        return normalized

    def _validate_batch(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        context: torch.Tensor,
    ) -> list[int]:
        _validate_tensor(state, name="bank state", ndim=2, width=self.state_width)
        _validate_tensor(
            intention,
            name="bank intention",
            ndim=2,
            width=self.intention_width,
        )
        if state.shape[0] != intention.shape[0]:
            raise ValueError("bank state and intention batches differ")
        if context.shape[0] != state.shape[0]:
            raise ValueError("bank context batch differs")
        return self._context_indices(context)

    def grow_verified(
        self,
        destination_capacity: int,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
    ) -> ExternalTransitionModelGrowthReceipt:
        """Grow a bounded bank only after a caller-owned retention proof.

        Growth changes capacity metadata only; it must not rewrite contexts or
        model weights. The probe is run before and after the transaction so a
        caller can verify held-out behavior for every retained slot. A failed
        post-growth probe rolls back the capacity metadata.
        """

        if self.capacity is None:
            raise ValueError("verified growth requires an explicit bank capacity")
        if not isinstance(destination_capacity, int):
            raise TypeError("destination capacity must be an integer")
        if destination_capacity <= self.capacity:
            raise ValueError("destination capacity must exceed current capacity")
        if not callable(retention_probe):
            raise TypeError("retention probe must be callable")
        source_capacity = self.capacity
        content_before = self.content_digest()
        if not bool(retention_probe(self)):
            return ExternalTransitionModelGrowthReceipt(
                accepted=False,
                source_capacity=source_capacity,
                destination_capacity=source_capacity,
                context_count=self.context_count,
                content_digest_before=content_before,
                content_digest_after=content_before,
                reason="pre-growth retention probe failed",
            )
        self.capacity = destination_capacity
        content_after = self.content_digest()
        if content_after != content_before or not bool(retention_probe(self)):
            self.capacity = source_capacity
            return ExternalTransitionModelGrowthReceipt(
                accepted=False,
                source_capacity=source_capacity,
                destination_capacity=source_capacity,
                context_count=self.context_count,
                content_digest_before=content_before,
                content_digest_after=self.content_digest(),
                reason="post-growth retention or content-integrity probe failed",
            )
        return ExternalTransitionModelGrowthReceipt(
            accepted=True,
            source_capacity=source_capacity,
            destination_capacity=destination_capacity,
            context_count=self.context_count,
            content_digest_before=content_before,
            content_digest_after=content_after,
            reason="retention-verified capacity growth committed",
        )

    def evict_verified(
        self,
        index: int,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
    ) -> ExternalTransitionModelEvictionReceipt:
        """Remove one slot only after a disposable post-eviction proof.

        Only the tail slot may be evicted until stable slot IDs/remapping are
        introduced; removing a middle index would renumber opaque addresses.
        The live bank is untouched while the candidate is constructed and
        tested. The probe owns the definition of retained behavior; callers
        should include every still-required opaque context and its held-out
        verifier floor. Aliased model objects are reconstructed through the
        normal payload boundary, so removing the tail cannot accidentally
        remove another context's shared parameters.
        """

        if not 0 <= index < self.context_count:
            raise IndexError("transition-model eviction slot is out of range")
        if not callable(retention_probe):
            raise TypeError("transition-model eviction retention probe is invalid")
        before = self.content_digest()
        physical_before = self.physical_model_count
        evicted_slot_id = self.slot_id_at(index)
        if index != self.context_count - 1:
            return ExternalTransitionModelEvictionReceipt(
                accepted=False,
                evicted_index=index,
                source_context_count=self.context_count,
                destination_context_count=self.context_count,
                physical_models_before=physical_before,
                physical_models_after=physical_before,
                content_digest_before=before,
                content_digest_after=before,
                reason=(
                    "non-tail eviction rejected to preserve opaque slot "
                    "indices"
                ),
            ).validate()
        if not bool(retention_probe(self)):
            return ExternalTransitionModelEvictionReceipt(
                accepted=False,
                evicted_index=index,
                source_context_count=self.context_count,
                destination_context_count=self.context_count,
                physical_models_before=physical_before,
                physical_models_after=physical_before,
                content_digest_before=before,
                content_digest_after=before,
                reason="pre-eviction retention probe failed",
            ).validate()

        candidate = ExternalTransitionModelBank.from_payload(self.payload())
        del candidate._contexts[index]
        del candidate.models[index]
        del candidate._model_families[index]
        del candidate._slot_ids[index]
        candidate._lifetime_usage.pop(evicted_slot_id, None)
        candidate._lifetime_last_access.pop(evicted_slot_id, None)
        candidate._lifetime_prediction_error.pop(evicted_slot_id, None)
        after = candidate.content_digest()
        if not bool(retention_probe(candidate)):
            return ExternalTransitionModelEvictionReceipt(
                accepted=False,
                evicted_index=index,
                source_context_count=self.context_count,
                destination_context_count=candidate.context_count,
                physical_models_before=physical_before,
                physical_models_after=candidate.physical_model_count,
                content_digest_before=before,
                content_digest_after=before,
                reason="post-eviction retention probe failed",
            ).validate()

        self._contexts = candidate._contexts
        self.models = candidate.models
        self._model_families = candidate._model_families
        self._slot_ids = candidate._slot_ids
        self._next_slot_id = candidate._next_slot_id
        self._lifetime_clock = candidate._lifetime_clock
        self._lifetime_usage = candidate._lifetime_usage
        self._lifetime_last_access = candidate._lifetime_last_access
        self._lifetime_prediction_error = candidate._lifetime_prediction_error
        return ExternalTransitionModelEvictionReceipt(
            accepted=True,
            evicted_index=index,
            evicted_slot_id=evicted_slot_id,
            source_context_count=self.context_count + 1,
            destination_context_count=self.context_count,
            physical_models_before=physical_before,
            physical_models_after=self.physical_model_count,
            content_digest_before=before,
            content_digest_after=after,
            reason="retention-verified model-slot eviction committed",
        ).validate()

    def evict_verified_id(
        self,
        slot_id: int,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
    ) -> ExternalTransitionModelEvictionReceipt:
        """Evict any logical slot without renumbering surviving addresses.

        The operation is copy-on-write and verifier-gated.  Physical indices
        may change after a middle removal, but stable slot IDs of survivors do
        not.  Callers that persist references must use this method's logical
        address API rather than retaining a physical index.
        """

        index = self.physical_index_for_slot_id(slot_id)
        if not callable(retention_probe):
            raise TypeError("transition-model eviction retention probe is invalid")
        before = self.content_digest()
        physical_before = self.physical_model_count
        source_count = self.context_count
        if not bool(retention_probe(self)):
            return ExternalTransitionModelEvictionReceipt(
                accepted=False,
                evicted_index=index,
                evicted_slot_id=slot_id,
                source_context_count=source_count,
                destination_context_count=source_count,
                physical_models_before=physical_before,
                physical_models_after=physical_before,
                content_digest_before=before,
                content_digest_after=before,
                reason="pre-eviction retention probe failed",
            ).validate()

        candidate = ExternalTransitionModelBank.from_payload(self.payload())
        del candidate._contexts[index]
        del candidate.models[index]
        del candidate._model_families[index]
        del candidate._slot_ids[index]
        candidate._lifetime_usage.pop(slot_id, None)
        candidate._lifetime_last_access.pop(slot_id, None)
        candidate._lifetime_prediction_error.pop(slot_id, None)
        after = candidate.content_digest()
        if not bool(retention_probe(candidate)):
            return ExternalTransitionModelEvictionReceipt(
                accepted=False,
                evicted_index=index,
                evicted_slot_id=slot_id,
                source_context_count=source_count,
                destination_context_count=candidate.context_count,
                physical_models_before=physical_before,
                physical_models_after=candidate.physical_model_count,
                content_digest_before=before,
                content_digest_after=before,
                reason="post-eviction retention probe failed",
            ).validate()

        self._contexts = candidate._contexts
        self.models = candidate.models
        self._model_families = candidate._model_families
        self._slot_ids = candidate._slot_ids
        self._next_slot_id = candidate._next_slot_id
        self._lifetime_clock = candidate._lifetime_clock
        self._lifetime_usage = candidate._lifetime_usage
        self._lifetime_last_access = candidate._lifetime_last_access
        self._lifetime_prediction_error = candidate._lifetime_prediction_error
        return ExternalTransitionModelEvictionReceipt(
            accepted=True,
            evicted_index=index,
            evicted_slot_id=slot_id,
            source_context_count=source_count,
            destination_context_count=self.context_count,
            physical_models_before=physical_before,
            physical_models_after=self.physical_model_count,
            content_digest_before=before,
            content_digest_after=after,
            reason="retention-verified logical model-slot eviction committed",
        ).validate()

    def forward(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        indices = self._validate_batch(state, intention, context)
        values = [
            self.models[index](state[row : row + 1], intention[row : row + 1]).squeeze(0)
            for row, index in enumerate(indices)
        ]
        return torch.stack(values)

    def predict_with_context(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self(state, intention, context)

    def loss(
        self,
        observation: ExternalTransitionObservation,
        context: torch.Tensor,
    ) -> torch.Tensor:
        observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )
        indices = self._validate_batch(
            observation.state,
            observation.intention,
            context,
        )
        predictions = [
            self.models[index](
                observation.state[row : row + 1],
                observation.intention[row : row + 1],
            ).squeeze(0)
            for row, index in enumerate(indices)
        ]
        prediction = torch.stack(predictions)
        errors = (prediction - observation.next_state).square().mean(dim=-1)
        if observation.confidence is None:
            return errors.mean()
        confidence = observation.confidence.reshape(-1).to(
            device=errors.device,
            dtype=errors.dtype,
        )
        return (errors * confidence).sum() / confidence.sum().clamp_min(1e-12)

    @staticmethod
    def _observation_rows(
        observation: ExternalTransitionObservation,
        rows: Sequence[int],
    ) -> ExternalTransitionObservation:
        row_index = torch.tensor(
            list(rows),
            dtype=torch.long,
            device=observation.state.device,
        )
        return ExternalTransitionObservation(
            state=observation.state.index_select(0, row_index),
            intention=observation.intention.index_select(0, row_index),
            next_state=observation.next_state.index_select(0, row_index),
            confidence=(
                None
                if observation.confidence is None
                else observation.confidence.reshape(-1).index_select(0, row_index)
            ),
        )

    def adaptation_step(
        self,
        observation: ExternalTransitionObservation,
        context: torch.Tensor,
        optimizer: torch.optim.Optimizer | Mapping[int, torch.optim.Optimizer] | None,
    ) -> float:
        """Update only parameters selected by the supplied context batch."""

        loss = self.loss(observation, context)
        indices = self._context_indices(context)
        lifetime_slot_ids: list[int] = []
        lifetime_errors: list[float] = []
        with torch.no_grad():
            for row, index in enumerate(indices):
                prediction = self.models[index](
                    observation.state[row : row + 1],
                    observation.intention[row : row + 1],
                )
                error = float(
                    (prediction - observation.next_state[row : row + 1])
                    .square()
                    .mean()
                )
                lifetime_slot_ids.append(self.slot_id_at(index))
                lifetime_errors.append(error)
        self.record_lifetime_observations(lifetime_slot_ids, lifetime_errors)
        for index in sorted(set(indices)):
            rows = [row for row, selected in enumerate(indices) if selected == index]
            subset = self._observation_rows(observation, rows)
            model = self.models[index]
            if hasattr(model, "observe"):
                with torch.no_grad():
                    model.observe(subset)
                continue
            selected_optimizer = (
                optimizer.get(index)
                if isinstance(optimizer, Mapping)
                else optimizer
            )
            if selected_optimizer is None:
                selected_optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=self.adaptation_learning_rate,
                )
            selected_optimizer.zero_grad()
            model_loss = model.loss(subset)
            model_loss.backward()
            selected_optimizer.step()
        return float(loss.detach())

    def consolidate_verified(
        self,
        first: int,
        second: int,
        heldout: Sequence[ExternalTransitionObservation],
        *,
        prediction_tolerance: float = 1e-6,
        retention_probe: Callable[[ExternalTransitionModelBank], bool] | None = None,
    ) -> ExternalTransitionModelConsolidationReceipt:
        """Share equivalent slot parameters only after held-out verification.

        Context keys and indices remain intact. The second context is made an
        alias of the first model object, reducing physical parameter storage
        without forcing callers to rewrite opaque addresses. Distinct factual
        functions are rejected before mutation.
        """

        if not 0 <= first < self.context_count or not 0 <= second < self.context_count:
            raise IndexError("transition-model consolidation slot is out of range")
        if first == second:
            raise ValueError("transition-model consolidation slots must differ")
        if self.model_family_at(first) != self.model_family_at(second):
            raise ValueError("transition-model consolidation families must match")
        if prediction_tolerance < 0.0:
            raise ValueError("transition-model consolidation tolerance cannot be negative")
        if not heldout:
            raise ValueError("transition-model consolidation needs held-out evidence")
        for observation in heldout:
            observation.validate(
                state_width=self.state_width,
                intention_width=self.intention_width,
            )
        before_content = self.content_digest()
        if retention_probe is not None and not callable(retention_probe):
            raise TypeError("transition-model consolidation retention probe is invalid")
        if retention_probe is not None and not bool(retention_probe(self)):
            return ExternalTransitionModelConsolidationReceipt(
                accepted=False,
                first=first,
                second=second,
                context_count=self.context_count,
                physical_models_before=self.physical_model_count,
                physical_models_after=self.physical_model_count,
                max_heldout_difference=float("inf"),
                content_digest_before=before_content,
                content_digest_after=before_content,
                reason="pre-consolidation retention probe failed",
            ).validate()

        max_difference = 0.0
        for observation in heldout:
            first_prediction = self.models[first](
                observation.state,
                observation.intention,
            )
            second_prediction = self.models[second](
                observation.state,
                observation.intention,
            )
            max_difference = max(
                max_difference,
                float(
                    (first_prediction - second_prediction).square().mean().detach()
                ),
            )
        if max_difference > prediction_tolerance:
            return ExternalTransitionModelConsolidationReceipt(
                accepted=False,
                first=first,
                second=second,
                context_count=self.context_count,
                physical_models_before=self.physical_model_count,
                physical_models_after=self.physical_model_count,
                max_heldout_difference=max_difference,
                content_digest_before=before_content,
                content_digest_after=before_content,
                reason="held-out transition functions are not equivalent",
            ).validate()

        physical_before = self.physical_model_count
        original_second = self.models[second]
        self.models[second] = self.models[first]
        after_content = self.content_digest()
        if retention_probe is not None and not bool(retention_probe(self)):
            self.models[second] = original_second
            return ExternalTransitionModelConsolidationReceipt(
                accepted=False,
                first=first,
                second=second,
                context_count=self.context_count,
                physical_models_before=physical_before,
                physical_models_after=self.physical_model_count,
                max_heldout_difference=max_difference,
                content_digest_before=before_content,
                content_digest_after=self.content_digest(),
                reason="post-consolidation retention probe failed",
            ).validate()
        return ExternalTransitionModelConsolidationReceipt(
            accepted=True,
            first=first,
            second=second,
            context_count=self.context_count,
            physical_models_before=physical_before,
            physical_models_after=self.physical_model_count,
            max_heldout_difference=max_difference,
            content_digest_before=before_content,
            content_digest_after=after_content,
            reason="equivalent transition models now share parameters",
        ).validate()

    @staticmethod
    def _tensor_map_digest(artifact: Mapping[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(artifact.items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(repr(tuple(value.shape)).encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _tensor_map_bytes(artifact: Mapping[str, torch.Tensor]) -> int:
        return sum(
            value.numel() * value.element_size() for value in artifact.values()
        )

    def _compressed_payload_digest(self, payload: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        digest.update(EXTERNAL_TRANSITION_MODEL_COMPRESSION_SCHEMA.encode("utf-8"))
        digest.update(str(payload["codec"]).encode("utf-8"))
        digest.update(repr(payload["configuration"]).encode("utf-8"))
        for context in payload["contexts"]:
            digest.update(
                torch.tensor(context, dtype=torch.float32).numpy().tobytes()
            )
        digest.update(repr(payload.get("slot_ids")).encode("utf-8"))
        digest.update(repr(payload.get("next_slot_id")).encode("utf-8"))
        digest.update(repr(payload["model_aliases"]).encode("utf-8"))
        for model_payload in payload["models"]:
            digest.update(
                self._tensor_map_digest(model_payload["state"]).encode("utf-8")
            )
        return digest.hexdigest()

    def _legacy_compressed_payload_digest(self, payload: Mapping[str, Any]) -> str:
        """Checksum the pre-stable-address compressed payload format."""

        digest = hashlib.sha256()
        digest.update(EXTERNAL_TRANSITION_MODEL_COMPRESSION_SCHEMA.encode("utf-8"))
        digest.update(str(payload["codec"]).encode("utf-8"))
        digest.update(repr(payload["configuration"]).encode("utf-8"))
        for context in payload["contexts"]:
            digest.update(
                torch.tensor(context, dtype=torch.float32).numpy().tobytes()
            )
        digest.update(repr(payload["model_aliases"]).encode("utf-8"))
        for model_payload in payload["models"]:
            digest.update(
                self._tensor_map_digest(model_payload["state"]).encode("utf-8")
            )
        return digest.hexdigest()

    def compressed_payload(
        self,
        *,
        dtype: torch.dtype | str = torch.float16,
    ) -> dict[str, object]:
        """Create a storage-compressed, caller-owned bank checkpoint."""

        models: list[dict[str, object]] = []
        for model in self.models:
            state = compress_growth_artifact(model.state_dict(), dtype=dtype)
            models.append(
                {
                    "state": state,
                    "sha256": self._tensor_map_digest(state),
                }
            )
        payload: dict[str, object] = {
            "schema": EXTERNAL_TRANSITION_MODEL_COMPRESSION_SCHEMA,
            "configuration": self.configuration(),
            "codec": str(dtype),
            "contexts": [context.tolist() for context in self._contexts],
            "slot_ids": list(self._slot_ids),
            "next_slot_id": self._next_slot_id,
            "model_aliases": self.model_aliases(),
            "models": models,
            "lifetime_telemetry": self._lifetime_telemetry_payload(),
        }
        payload["sha256"] = self._compressed_payload_digest(payload)
        return payload

    @classmethod
    def from_compressed_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalTransitionModelBank:
        """Restore and decompress a storage checkpoint into runtime models."""

        if not isinstance(payload, Mapping) or payload.get("schema") != (
            EXTERNAL_TRANSITION_MODEL_COMPRESSION_SCHEMA
        ):
            raise ValueError("unsupported compressed transition-model payload")
        configuration = payload.get("configuration")
        contexts = payload.get("contexts")
        models = payload.get("models")
        if not isinstance(configuration, Mapping):
            raise TypeError("compressed transition-model configuration is missing")
        if not isinstance(contexts, list) or not isinstance(models, list):
            raise TypeError("compressed transition-model payload lists are invalid")
        aliases_payload = payload.get("model_aliases")
        aliases = (
            list(range(len(models)))
            if aliases_payload is None
            else [int(alias) for alias in aliases_payload]
        )
        if len(contexts) != len(models) or len(aliases) != len(models):
            raise ValueError("compressed transition-model payload lengths differ")
        slot_ids_payload = payload.get("slot_ids")
        if slot_ids_payload is None:
            slot_ids = list(range(len(contexts)))
            next_slot_id = len(slot_ids)
        else:
            if not isinstance(slot_ids_payload, list):
                raise TypeError("compressed transition-model slot IDs are invalid")
            slot_ids = [int(slot_id) for slot_id in slot_ids_payload]
            next_slot_id = int(payload.get("next_slot_id", len(slot_ids)))
        if len(slot_ids) != len(contexts) or len(set(slot_ids)) != len(slot_ids):
            raise ValueError("compressed transition-model slot IDs are invalid")
        if any(slot_id < 0 for slot_id in slot_ids) or next_slot_id <= max(
            slot_ids, default=-1
        ):
            raise ValueError("compressed transition-model slot ID sequence is invalid")
        bank = cls(
            int(configuration["state_width"]),
            int(configuration["intention_width"]),
            int(configuration["context_width"]),
            hidden_width=int(configuration["hidden_width"]),
            matching_tolerance=float(configuration["matching_tolerance"]),
            model_family=str(
                configuration.get(
                    "model_family", EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY
                )
            ),
            affine_ridge=float(configuration.get("affine_ridge", 1e-5)),
            adaptation_learning_rate=float(
                configuration.get("adaptation_learning_rate", 1e-2)
            ),
            random_feature_width=int(configuration.get("random_feature_width", 128)),
            random_feature_seed=int(configuration.get("random_feature_seed", 0)),
            capacity=(
                None
                if configuration.get("capacity") is None
                else int(configuration["capacity"])
            ),
        )
        if bank.capacity is not None and len(contexts) > bank.capacity:
            raise ValueError("compressed transition-model payload exceeds capacity")
        model_families = bank._payload_model_families(configuration, len(models))
        for index, (values, model_payload) in enumerate(
            zip(contexts, models, strict=True)
        ):
            if not isinstance(model_payload, Mapping):
                raise TypeError("compressed transition-model slot is invalid")
            context = torch.tensor(values, dtype=torch.float32)
            _validate_tensor(
                context,
                name="compressed transition-model context",
                ndim=1,
                width=bank.context_width,
            )
            if not torch.allclose(
                torch.linalg.vector_norm(context),
                torch.ones((), dtype=context.dtype),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError("compressed transition-model context is not normalized")
            bank._contexts.append(context.clone())
            bank.models.append(bank._new_model(model_families[index]))
            bank._model_families.append(model_families[index])
            bank._slot_ids.append(slot_ids[index])
            state = model_payload.get("state")
            if not isinstance(state, Mapping):
                raise TypeError("compressed transition-model state is missing")
            if model_payload.get("sha256") != bank._tensor_map_digest(state):
                raise ValueError("compressed transition-model slot checksum mismatch")
            decompressed = decompress_growth_artifact(state)
            current = bank.models[index].state_dict()
            if tuple(decompressed) != tuple(current):
                raise ValueError("compressed transition-model state names differ")
            normalized: dict[str, torch.Tensor] = {}
            for name, expected in current.items():
                value = decompressed[name]
                if value.shape != expected.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError("compressed transition-model state is incompatible")
                normalized[name] = value.to(dtype=expected.dtype)
            bank.models[index].load_state_dict(normalized, strict=True)
        if any(alias < 0 or alias > index for index, alias in enumerate(aliases)):
            raise ValueError("compressed transition-model aliases are invalid")
        for index, alias in enumerate(aliases):
            if alias != index:
                if model_families[index] != model_families[alias]:
                    raise ValueError("compressed transition-model aliases cross families")
                bank.models[index] = bank.models[alias]
                bank._model_families[index] = bank._model_families[alias]
        bank._next_slot_id = next_slot_id
        bank._restore_lifetime_telemetry(payload.get("lifetime_telemetry"))
        if (
            payload.get("sha256") != bank._compressed_payload_digest(payload)
            and (
                payload.get("slot_ids") is not None
                or payload.get("sha256")
                != bank._legacy_compressed_payload_digest(payload)
            )
        ):
            raise ValueError("compressed transition-model payload checksum mismatch")
        return bank


    def compress_verified(
        self,
        *,
        dtype: torch.dtype | str = torch.float16,
        retention_probe: Callable[[ExternalTransitionModelBank], bool] | None = None,
    ) -> ExternalTransitionModelCompressionReceipt:
        """Accept a compressed storage candidate only after behavior probing."""

        payload = self.compressed_payload(dtype=dtype)
        candidate = self.from_compressed_payload(payload)
        if retention_probe is not None and not callable(retention_probe):
            raise TypeError("transition-model compression retention probe is invalid")
        accepted = retention_probe is None or bool(retention_probe(candidate))
        source_bytes = sum(
            self._tensor_map_bytes(model.state_dict()) for model in self.models
        )
        compressed_bytes = sum(
            self._tensor_map_bytes(model_payload["state"])
            for model_payload in payload["models"]
        )
        return ExternalTransitionModelCompressionReceipt(
            accepted=accepted,
            codec=str(dtype),
            source_bytes=source_bytes,
            compressed_bytes=compressed_bytes,
            context_count=self.context_count,
            physical_models=self.physical_model_count,
            candidate_digest=candidate.digest(),
            reason=(
                "compressed candidate passed retention probe"
                if accepted
                else "compressed candidate failed retention probe"
            ),
        ).validate()

    def select_compression_verified(
        self,
        codecs: Sequence[torch.dtype | str],
        *,
        retention_probe: Callable[[ExternalTransitionModelBank], bool] | None = None,
    ) -> ExternalTransitionModelCompressionSelection:
        """Select the smallest codec whose independent candidate is retained."""

        if not codecs:
            raise ValueError("compression codec candidates must be nonempty")
        if len({str(codec) for codec in codecs}) != len(codecs):
            raise ValueError("compression codec candidates must be unique")
        receipts = tuple(
            self.compress_verified(dtype=codec, retention_probe=retention_probe)
            for codec in codecs
        )
        accepted = [receipt for receipt in receipts if receipt.accepted]
        if not accepted:
            return ExternalTransitionModelCompressionSelection(
                accepted=False,
                selected_codec=None,
                receipts=receipts,
                reason="no compression candidate passed retention",
            ).validate()
        selected = min(accepted, key=lambda receipt: receipt.compressed_bytes)
        return ExternalTransitionModelCompressionSelection(
            accepted=True,
            selected_codec=selected.codec,
            receipts=receipts,
            reason="smallest retained compression candidate selected",
        ).validate()

    def content_digest(self) -> str:
        """Digest stable slot addresses, keys, and model weights."""

        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        aliases = self.model_aliases()
        if aliases != list(range(len(aliases))):
            digest.update(repr(aliases).encode("utf-8"))
        for slot_id, context in zip(self._slot_ids, self._contexts, strict=True):
            digest.update(str(slot_id).encode("utf-8"))
            detached = context.detach().cpu().contiguous()
            digest.update(detached.numpy().tobytes())
        for index, model in enumerate(self.models):
            digest.update(str(index).encode("utf-8"))
            digest.update(self._model_families[index].encode("utf-8"))
            digest.update(model.digest().encode("utf-8"))
        return digest.hexdigest()

    def _legacy_content_digest(self) -> str:
        """Checksum the pre-stable-address logical content representation."""

        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        aliases = self.model_aliases()
        if aliases != list(range(len(aliases))):
            digest.update(repr(aliases).encode("utf-8"))
        for context in self._contexts:
            digest.update(context.detach().cpu().contiguous().numpy().tobytes())
        for index, model in enumerate(self.models):
            digest.update(str(index).encode("utf-8"))
            digest.update(self._model_families[index].encode("utf-8"))
            digest.update(model.digest().encode("utf-8"))
        return digest.hexdigest()

    def model_aliases(self) -> list[int]:
        """Return the canonical model index for each opaque context key."""

        canonical: dict[int, int] = {}
        aliases: list[int] = []
        for index, model in enumerate(self.models):
            identity = id(model)
            aliases.append(canonical.setdefault(identity, index))
        return aliases

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr(self.configuration()).encode("utf-8"))
        digest.update(self.content_digest().encode("utf-8"))
        return digest.hexdigest()

    def _legacy_digest(self) -> str:
        configuration = self.configuration()
        configuration.pop("slot_addressing", None)
        digest = hashlib.sha256()
        digest.update(repr(configuration).encode("utf-8"))
        digest.update(self._legacy_content_digest().encode("utf-8"))
        return digest.hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "contexts": [context.tolist() for context in self._contexts],
            "slot_ids": list(self._slot_ids),
            "next_slot_id": self._next_slot_id,
            "model_aliases": self.model_aliases(),
            "models": [
                {
                    "state": {
                        name: value.detach().cpu().tolist()
                        for name, value in model.state_dict().items()
                    },
                    "sha256": model.digest(),
                }
                for model in self.models
            ],
            "lifetime_telemetry": self._lifetime_telemetry_payload(),
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalTransitionModelBank:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported transition-model bank payload")
        configuration = payload.get("configuration")
        contexts = payload.get("contexts")
        models = payload.get("models")
        model_aliases = payload.get("model_aliases")
        if not isinstance(configuration, Mapping):
            raise TypeError("transition-model bank configuration is missing")
        if not isinstance(contexts, list) or not isinstance(models, list):
            raise TypeError("transition-model bank payload lists are invalid")
        if len(contexts) != len(models):
            raise ValueError("transition-model bank payload lengths differ")
        slot_ids_payload = payload.get("slot_ids")
        if slot_ids_payload is None:
            slot_ids = list(range(len(contexts)))
            next_slot_id = len(slot_ids)
        else:
            if not isinstance(slot_ids_payload, list):
                raise TypeError("transition-model bank slot IDs are invalid")
            slot_ids = [int(slot_id) for slot_id in slot_ids_payload]
            next_slot_id = int(payload.get("next_slot_id", len(slot_ids)))
        if len(slot_ids) != len(contexts) or len(set(slot_ids)) != len(slot_ids):
            raise ValueError("transition-model bank slot IDs are invalid")
        if any(slot_id < 0 for slot_id in slot_ids) or next_slot_id <= max(
            slot_ids, default=-1
        ):
            raise ValueError("transition-model bank slot ID sequence is invalid")
        if model_aliases is None:
            aliases = list(range(len(models)))
        elif isinstance(model_aliases, list):
            aliases = [int(alias) for alias in model_aliases]
        else:
            raise TypeError("transition-model bank model aliases are invalid")
        if len(aliases) != len(models) or any(
            alias < 0 or alias > index
            for index, alias in enumerate(aliases)
        ):
            raise ValueError("transition-model bank model aliases are invalid")
        bank = cls(
            int(configuration["state_width"]),
            int(configuration["intention_width"]),
            int(configuration["context_width"]),
            hidden_width=int(configuration["hidden_width"]),
            matching_tolerance=float(configuration["matching_tolerance"]),
            model_family=str(
                configuration.get(
                    "model_family", EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY
                )
            ),
            affine_ridge=float(configuration.get("affine_ridge", 1e-5)),
            adaptation_learning_rate=float(
                configuration.get("adaptation_learning_rate", 1e-2)
            ),
            random_feature_width=int(configuration.get("random_feature_width", 128)),
            random_feature_seed=int(configuration.get("random_feature_seed", 0)),
            capacity=(
                None
                if configuration.get("capacity") is None
                else int(configuration["capacity"])
            ),
        )
        if bank.capacity is not None and len(contexts) > bank.capacity:
            raise ValueError("transition-model bank payload exceeds capacity")
        model_families = bank._payload_model_families(configuration, len(models))
        for values, model_payload in zip(contexts, models, strict=True):
            context = torch.tensor(values, dtype=torch.float32)
            _validate_tensor(
                context,
                name="transition-model payload context",
                ndim=1,
                width=bank.context_width,
            )
            if not torch.allclose(
                torch.linalg.vector_norm(context),
                torch.ones((), dtype=context.dtype),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError("transition-model payload context is not normalized")
            if bool(torch.linalg.vector_norm(context) <= 1e-12):
                raise ValueError("transition-model payload context is zero")
            index = bank.context_count
            bank._contexts.append(context.clone())
            family = model_families[index]
            bank.models.append(bank._new_model(family))
            bank._model_families.append(family)
            bank._slot_ids.append(slot_ids[index])
            if not isinstance(model_payload, Mapping):
                raise TypeError("transition-model bank slot is invalid")
            state_payload = model_payload.get("state")
            if not isinstance(state_payload, Mapping):
                raise TypeError("transition-model bank slot state is missing")
            current = bank.models[index].state_dict()
            if tuple(state_payload) != tuple(current):
                raise ValueError("transition-model bank slot state names differ")
            normalized: dict[str, torch.Tensor] = {}
            for name, expected in current.items():
                value = torch.tensor(state_payload[name], dtype=expected.dtype)
                if value.shape != expected.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError("transition-model bank slot state is incompatible")
                normalized[name] = value
            bank.models[index].load_state_dict(normalized, strict=True)
            if model_payload.get("sha256") != bank.models[index].digest():
                raise ValueError("transition-model bank slot checksum mismatch")
        for index, alias in enumerate(aliases):
            if alias != index:
                if model_families[index] != model_families[alias]:
                    raise ValueError("transition-model aliases cross families")
                bank.models[index] = bank.models[alias]
                bank._model_families[index] = bank._model_families[alias]
        bank._next_slot_id = next_slot_id
        bank._restore_lifetime_telemetry(payload.get("lifetime_telemetry"))
        if (
            payload.get("sha256") != bank.digest()
            and (
                payload.get("slot_ids") is not None
                or payload.get("sha256") != bank._legacy_digest()
            )
        ):
            raise ValueError("transition-model bank checksum mismatch")
        return bank


@dataclass(frozen=True)
class ExternalTransitionModelLifetimeProposal:
    """A learned, verifier-independent proposal for one logical slot."""

    selected_slot_id: int | None
    scores: torch.Tensor
    eligible_slot_ids: tuple[int, ...]
    reason: str
    schema: str = EXTERNAL_TRANSITION_MODEL_LIFETIME_POLICY_SCHEMA

    def validate(self) -> ExternalTransitionModelLifetimeProposal:
        if self.schema != EXTERNAL_TRANSITION_MODEL_LIFETIME_POLICY_SCHEMA:
            raise ValueError("unsupported transition-model lifetime schema")
        if self.scores.ndim != 1 or self.scores.shape[0] != len(
            self.eligible_slot_ids
        ):
            raise ValueError("transition-model lifetime scores are misaligned")
        if bool(torch.isnan(self.scores).any()):
            raise ValueError("transition-model lifetime scores contain NaN")
        if len(set(self.eligible_slot_ids)) != len(self.eligible_slot_ids):
            raise ValueError("transition-model lifetime slot IDs are duplicated")
        if any(slot_id < 0 for slot_id in self.eligible_slot_ids):
            raise ValueError("transition-model lifetime slot ID is invalid")
        if self.selected_slot_id is not None and self.selected_slot_id not in (
            self.eligible_slot_ids
        ):
            raise ValueError("transition-model lifetime selected slot is ineligible")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("transition-model lifetime proposal reason is missing")
        return self


class ExternalTransitionModelLifetimePolicy(nn.Module):
    """Learn a generic slot-lifetime score outside the frozen controller.

    The scorer sees only opaque context keys and generic usage/age/error
    telemetry. It is permutation-equivariant over slots and emits a proposal;
    a verifier-owned retention probe remains the sole authority that can
    commit eviction. A verifier outcome can update this policy once without
    retaining or replaying the transition evidence that produced it.
    """

    schema = EXTERNAL_TRANSITION_MODEL_LIFETIME_POLICY_SCHEMA

    def __init__(
        self,
        context_width: int,
        *,
        hidden_width: int = 32,
        learning_rate: float = 1e-2,
    ) -> None:
        super().__init__()
        if context_width < 1 or hidden_width < 1:
            raise ValueError("transition lifetime policy widths must be positive")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("transition lifetime policy learning rate must be positive")
        self.context_width = int(context_width)
        self.hidden_width = int(hidden_width)
        self.learning_rate = float(learning_rate)
        self.feature_width = self.context_width + 3
        self.network = nn.Sequential(
            nn.Linear(self.feature_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, 1),
        )

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "hidden_width": self.hidden_width,
            "learning_rate": self.learning_rate,
            "inputs": "opaque_context_usage_age_prediction_error_v1",
            "output": "larger_score_means_evict_v1",
            "updates": "single_verified_outcome_v1",
        }

    def _features(
        self,
        contexts: torch.Tensor,
        usage: torch.Tensor,
        age: torch.Tensor,
        prediction_error: torch.Tensor,
    ) -> torch.Tensor:
        _validate_tensor(
            contexts,
            name="transition lifetime contexts",
            ndim=2,
            width=self.context_width,
        )
        values = []
        for name, value in (
            ("transition lifetime usage", usage),
            ("transition lifetime age", age),
            ("transition lifetime prediction error", prediction_error),
        ):
            if value.ndim != 1 or value.shape[0] != contexts.shape[0]:
                raise ValueError(f"{name} must have shape [slot_count]")
            if not bool(torch.isfinite(value).all()) or bool(torch.any(value < 0)):
                raise ValueError(f"{name} must be finite and non-negative")
            values.append(torch.log1p(value).to(contexts))
        return torch.cat((contexts, torch.stack(values, dim=-1)), dim=-1)

    def forward(
        self,
        contexts: torch.Tensor,
        usage: torch.Tensor,
        age: torch.Tensor,
        prediction_error: torch.Tensor,
    ) -> torch.Tensor:
        features = self._features(contexts, usage, age, prediction_error)
        return self.network(features).squeeze(-1)

    @torch.no_grad()
    def propose(
        self,
        contexts: torch.Tensor,
        slot_ids: Sequence[int],
        usage: torch.Tensor,
        age: torch.Tensor,
        prediction_error: torch.Tensor,
        protected: torch.Tensor,
    ) -> ExternalTransitionModelLifetimeProposal:
        if len(slot_ids) != contexts.shape[0]:
            raise ValueError("transition lifetime slot IDs do not match contexts")
        if protected.ndim != 1 or protected.shape[0] != contexts.shape[0]:
            raise ValueError("transition lifetime protection mask is misaligned")
        if protected.dtype != torch.bool:
            raise TypeError("transition lifetime protection mask must be boolean")
        scores = self(contexts, usage, age, prediction_error)
        eligible = tuple(
            int(slot_id)
            for slot_id, is_protected in zip(slot_ids, protected.tolist(), strict=True)
            if not is_protected
        )
        eligible_indices = [
            index
            for index, is_protected in enumerate(protected.tolist())
            if not is_protected
        ]
        if not eligible_indices:
            return ExternalTransitionModelLifetimeProposal(
                selected_slot_id=None,
                scores=scores.new_empty((0,)),
                eligible_slot_ids=eligible,
                reason="all external transition slots are protected",
            ).validate()
        selected_index = max(
            eligible_indices,
            key=lambda index: (float(scores[index]), -index),
        )
        return ExternalTransitionModelLifetimeProposal(
            selected_slot_id=int(slot_ids[selected_index]),
            scores=scores[eligible_indices].detach().clone(),
            eligible_slot_ids=eligible,
            reason="learned lifetime score selected an unprotected logical slot",
        ).validate()

    @torch.no_grad()
    def propose_from_query(
        self,
        bank: ExternalTransitionModelBank,
        query: torch.Tensor,
        protected: torch.Tensor,
        *,
        relevance_weight: float = 1.0,
    ) -> ExternalTransitionModelLifetimeProposal:
        """Bias eviction away from slots aligned with a learned query.

        The query and bank contexts already inhabit a learned opaque space.
        Shared cosine addressing supplies the relevance term without adding a
        fresh random adapter to the critical path.  Larger adjusted scores
        still mean eviction.
        """

        if not math.isfinite(relevance_weight) or relevance_weight < 0.0:
            raise ValueError("transition lifetime relevance weight is invalid")
        if query.ndim == 1:
            query = query.unsqueeze(0)
        _validate_tensor(
            query,
            name="transition lifetime relevance query",
            ndim=2,
            width=bank.context_width,
        )
        if query.shape[0] != 1:
            raise ValueError("transition lifetime relevance query must have one row")
        contexts = bank.contexts.to(query)
        if not bank.context_count:
            raise ValueError("transition lifetime query needs at least one slot")
        if protected.ndim != 1 or protected.shape[0] != bank.context_count:
            raise ValueError("transition lifetime query protection mask is misaligned")
        if protected.dtype != torch.bool:
            raise TypeError("transition lifetime query protection mask must be boolean")
        telemetry = bank.lifetime_telemetry()
        lifetime_scores = self(
            contexts,
            telemetry.usage.to(contexts),
            telemetry.age.to(contexts),
            telemetry.prediction_error.to(contexts),
        )
        query_key = torch.nn.functional.normalize(query, dim=-1)
        memory_keys = torch.nn.functional.normalize(contexts, dim=-1)
        relevance = memory_keys @ query_key.squeeze(0)
        scores = lifetime_scores - relevance_weight * relevance
        eligible_indices = [
            index
            for index, is_protected in enumerate(protected.tolist())
            if not is_protected
        ]
        if not eligible_indices:
            return ExternalTransitionModelLifetimeProposal(
                selected_slot_id=None,
                scores=scores.new_empty((0,)),
                eligible_slot_ids=(),
                reason="all query-conditioned lifetime slots are protected",
            ).validate()
        selected_index = max(
            eligible_indices,
            key=lambda index: (float(scores[index]), -index),
        )
        eligible = tuple(bank.slot_id_at(index) for index in eligible_indices)
        return ExternalTransitionModelLifetimeProposal(
            selected_slot_id=bank.slot_id_at(selected_index),
            scores=scores[eligible_indices].detach().clone(),
            eligible_slot_ids=eligible,
            reason="query-conditioned relevance adjusted lifetime eviction score",
        ).validate()

    def adaptation_step(
        self,
        contexts: torch.Tensor,
        slot_ids: Sequence[int],
        usage: torch.Tensor,
        age: torch.Tensor,
        prediction_error: torch.Tensor,
        selected_slot_id: int,
        verifier_accepted: bool,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        """Consume exactly one verifier bit as external policy learning."""

        if selected_slot_id not in slot_ids:
            raise KeyError("transition lifetime selected slot is unknown")
        selected_index = list(slot_ids).index(selected_slot_id)
        logits = self(contexts, usage, age, prediction_error)
        target = torch.tensor(
            [float(verifier_accepted)],
            device=logits.device,
            dtype=logits.dtype,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[selected_index : selected_index + 1],
            target,
        )
        selected_optimizer = optimizer
        if selected_optimizer is None:
            selected_optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
            )
        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        return float(loss.detach())

    def evict_verified(
        self,
        bank: ExternalTransitionModelBank,
        usage: torch.Tensor,
        age: torch.Tensor,
        prediction_error: torch.Tensor,
        protected: torch.Tensor,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[
        ExternalTransitionModelLifetimeProposal,
        ExternalTransitionModelEvictionReceipt | None,
    ]:
        """Propose, verifier-check, optionally learn from, and commit eviction."""

        proposal = self.propose(
            bank.contexts,
            bank.slot_ids,
            usage,
            age,
            prediction_error,
            protected,
        )
        if proposal.selected_slot_id is None:
            return proposal, None
        contexts_before = bank.contexts
        slot_ids_before = bank.slot_ids
        receipt = bank.evict_verified_id(
            proposal.selected_slot_id,
            retention_probe,
        )
        self.adaptation_step(
            contexts_before,
            slot_ids_before,
            usage,
            age,
            prediction_error,
            proposal.selected_slot_id,
            receipt.accepted,
            optimizer,
        )
        return proposal, receipt

    def evict_from_bank_verified(
        self,
        bank: ExternalTransitionModelBank,
        protected: torch.Tensor,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
        optimizer: torch.optim.Optimizer | None = None,
        *,
        update: bool = True,
    ) -> tuple[
        ExternalTransitionModelLifetimeProposal,
        ExternalTransitionModelEvictionReceipt | None,
    ]:
        """Manage a bank using its own persisted lifetime telemetry."""

        telemetry = bank.lifetime_telemetry()
        if update:
            return self.evict_verified(
                bank,
                telemetry.usage,
                telemetry.age,
                telemetry.prediction_error,
                protected,
                retention_probe,
                optimizer,
            )
        proposal = self.propose(
            bank.contexts,
            bank.slot_ids,
            telemetry.usage,
            telemetry.age,
            telemetry.prediction_error,
            protected,
        )
        if proposal.selected_slot_id is None:
            return proposal, None
        return proposal, bank.evict_verified_id(
            proposal.selected_slot_id,
            retention_probe,
        )

    def evict_from_bank_query_verified(
        self,
        bank: ExternalTransitionModelBank,
        query: torch.Tensor,
        protected: torch.Tensor,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
        *,
        relevance_weight: float = 1.0,
        optimizer: torch.optim.Optimizer | None = None,
        update: bool = True,
    ) -> tuple[
        ExternalTransitionModelLifetimeProposal,
        ExternalTransitionModelEvictionReceipt | None,
    ]:
        """Run a query-conditioned, verifier-gated external eviction."""

        telemetry = bank.lifetime_telemetry()
        contexts_before = bank.contexts
        slot_ids_before = bank.slot_ids
        proposal = self.propose_from_query(
            bank,
            query,
            protected,
            relevance_weight=relevance_weight,
        )
        if proposal.selected_slot_id is None:
            return proposal, None
        receipt = bank.evict_verified_id(
            proposal.selected_slot_id,
            retention_probe,
        )
        if update:
            self.adaptation_step(
                contexts_before,
                slot_ids_before,
                telemetry.usage,
                telemetry.age,
                telemetry.prediction_error,
                proposal.selected_slot_id,
                receipt.accepted,
                optimizer,
            )
        return proposal, receipt

    def state_payload(self) -> dict[str, object]:
        """Persist the external policy without any controller parameters."""

        state = {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
        }
        digest = hashlib.sha256()
        digest.update(repr(self.configuration()).encode("utf-8"))
        for name, value in state.items():
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(repr(tuple(value.shape)).encode("utf-8"))
            digest.update(value.contiguous().numpy().tobytes())
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": state,
            "sha256": digest.hexdigest(),
        }

    def digest(self) -> str:
        payload = self.state_payload()
        return str(payload["sha256"])

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalTransitionModelLifetimePolicy:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported transition lifetime policy payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("transition lifetime policy payload is incomplete")
        policy = cls(
            int(configuration["context_width"]),
            hidden_width=int(configuration["hidden_width"]),
            learning_rate=float(configuration["learning_rate"]),
        )
        current = policy.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("transition lifetime policy state names differ")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("transition lifetime policy state is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("transition lifetime policy state is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("transition lifetime policy state is not finite")
            normalized[name] = value.detach().clone()
        policy.load_state_dict(normalized, strict=True)
        if payload.get("sha256") != policy.digest():
            raise ValueError("transition lifetime policy checksum mismatch")
        return policy



@dataclass(frozen=True)
class ExternalTransitionModelGrowthReceipt:
    """Auditable result of verifier-gated external model-bank growth."""

    accepted: bool
    source_capacity: int
    destination_capacity: int
    context_count: int
    content_digest_before: str
    content_digest_after: str
    reason: str
    schema: str = EXTERNAL_TRANSITION_MODEL_GROWTH_SCHEMA

    def validate(self) -> ExternalTransitionModelGrowthReceipt:
        if self.schema != EXTERNAL_TRANSITION_MODEL_GROWTH_SCHEMA:
            raise ValueError("unsupported transition-model growth schema")
        if min(self.source_capacity, self.destination_capacity, self.context_count) < 0:
            raise ValueError("transition-model growth receipt counts are invalid")
        if self.accepted and self.destination_capacity <= self.source_capacity:
            raise ValueError("accepted transition-model growth did not grow capacity")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("transition-model growth receipt reason is missing")
        return self


@dataclass(frozen=True)
class ExternalTransitionModelEvictionReceipt:
    """Auditable result of verifier-gated logical model-slot eviction."""

    accepted: bool
    evicted_index: int
    source_context_count: int
    destination_context_count: int
    physical_models_before: int
    physical_models_after: int
    content_digest_before: str
    content_digest_after: str
    reason: str
    evicted_slot_id: int | None = None
    schema: str = EXTERNAL_TRANSITION_MODEL_EVICTION_SCHEMA

    def validate(self) -> ExternalTransitionModelEvictionReceipt:
        if self.schema != EXTERNAL_TRANSITION_MODEL_EVICTION_SCHEMA:
            raise ValueError("unsupported transition-model eviction schema")
        if min(
            self.evicted_index,
            self.source_context_count,
            self.destination_context_count,
            self.physical_models_before,
            self.physical_models_after,
        ) < 0:
            raise ValueError("transition-model eviction counts are invalid")
        if self.evicted_slot_id is not None and self.evicted_slot_id < 0:
            raise ValueError("transition-model eviction slot ID is invalid")
        if self.accepted and self.destination_context_count != self.source_context_count - 1:
            raise ValueError("accepted transition-model eviction count is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("transition-model eviction receipt reason is missing")
        return self


@dataclass(frozen=True)
class ExternalTransitionModelCandidateReceipt:
    """Auditable result of promoting a staged external model candidate."""

    accepted: bool
    slot_index: int | None
    context_count: int
    heldout_error: float
    candidate_digest: str
    content_digest_before: str
    content_digest_after: str
    reason: str
    slot_id: int | None = None
    schema: str = EXTERNAL_TRANSITION_MODEL_CANDIDATE_SCHEMA

    def validate(self) -> ExternalTransitionModelCandidateReceipt:
        if self.schema != EXTERNAL_TRANSITION_MODEL_CANDIDATE_SCHEMA:
            raise ValueError("unsupported transition-model candidate schema")
        if self.context_count < 0:
            raise ValueError("transition-model candidate context count is invalid")
        if self.accepted and (self.slot_index is None or self.slot_index < 0):
            raise ValueError("accepted transition-model candidate has no slot")
        if self.slot_id is not None and self.slot_id < 0:
            raise ValueError("transition-model candidate slot ID is invalid")
        if not math.isfinite(self.heldout_error) and self.accepted:
            raise ValueError("accepted transition-model candidate error is invalid")
        if not isinstance(self.candidate_digest, str) or not self.candidate_digest:
            raise ValueError("transition-model candidate digest is missing")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("transition-model candidate reason is missing")
        return self


@dataclass(frozen=True)
class ExternalTransitionModelConsolidationReceipt:
    """Auditable result of safe parameter-sharing consolidation."""

    accepted: bool
    first: int
    second: int
    context_count: int
    physical_models_before: int
    physical_models_after: int
    max_heldout_difference: float
    content_digest_before: str
    content_digest_after: str
    reason: str
    schema: str = EXTERNAL_TRANSITION_MODEL_CONSOLIDATION_SCHEMA

    def validate(self) -> ExternalTransitionModelConsolidationReceipt:
        if self.schema != EXTERNAL_TRANSITION_MODEL_CONSOLIDATION_SCHEMA:
            raise ValueError("unsupported transition-model consolidation schema")
        if min(
            self.first,
            self.second,
            self.context_count,
            self.physical_models_before,
            self.physical_models_after,
        ) < 0:
            raise ValueError("transition-model consolidation counts are invalid")
        if self.first == self.second:
            raise ValueError("transition-model consolidation slots must differ")
        if not math.isfinite(self.max_heldout_difference) and self.accepted:
            raise ValueError("accepted transition-model consolidation difference is invalid")
        if self.accepted and self.physical_models_after >= self.physical_models_before:
            raise ValueError("accepted consolidation did not reduce physical models")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("transition-model consolidation reason is missing")
        return self


@dataclass(frozen=True)
class ExternalTransitionModelCompressionReceipt:
    """Auditable result of a held-out-verified storage codec candidate."""

    accepted: bool
    codec: str
    source_bytes: int
    compressed_bytes: int
    context_count: int
    physical_models: int
    candidate_digest: str
    reason: str
    schema: str = EXTERNAL_TRANSITION_MODEL_COMPRESSION_SCHEMA

    def validate(self) -> ExternalTransitionModelCompressionReceipt:
        if self.schema != EXTERNAL_TRANSITION_MODEL_COMPRESSION_SCHEMA:
            raise ValueError("unsupported transition-model compression schema")
        if min(self.source_bytes, self.compressed_bytes, self.context_count, self.physical_models) < 0:
            raise ValueError("transition-model compression counts are invalid")
        if self.accepted and self.compressed_bytes >= self.source_bytes:
            raise ValueError("accepted compression did not reduce storage bytes")
        if not isinstance(self.codec, str) or not self.codec:
            raise ValueError("transition-model compression codec is missing")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("transition-model compression reason is missing")
        return self


@dataclass(frozen=True)
class ExternalTransitionModelCompressionSelection:
    """Selection record for the smallest retained external codec."""

    accepted: bool
    selected_codec: str | None
    receipts: tuple[ExternalTransitionModelCompressionReceipt, ...]
    reason: str
    schema: str = EXTERNAL_TRANSITION_MODEL_COMPRESSION_SELECTION_SCHEMA

    def validate(self) -> ExternalTransitionModelCompressionSelection:
        if self.schema != EXTERNAL_TRANSITION_MODEL_COMPRESSION_SELECTION_SCHEMA:
            raise ValueError("unsupported transition-model compression selection schema")
        if not self.receipts:
            raise ValueError("compression selection has no candidate receipts")
        if self.accepted:
            if self.selected_codec is None:
                raise ValueError("accepted compression selection has no codec")
            if not any(
                receipt.accepted and receipt.codec == self.selected_codec
                for receipt in self.receipts
            ):
                raise ValueError("selected compression codec was not accepted")
        elif self.selected_codec is not None:
            raise ValueError("rejected compression selection has a codec")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("compression selection reason is missing")
        return self


class ExternalTransitionContextEncoder(nn.Module):
    """Encode opaque transition bundles into stable external context keys."""

    schema = EXTERNAL_TRANSITION_CONTEXT_ENCODER_SCHEMA

    def __init__(
        self,
        state_width: int,
        intention_width: int,
        *,
        hidden_width: int = 64,
        context_width: int = 32,
    ) -> None:
        super().__init__()
        if min(state_width, intention_width, hidden_width, context_width) < 1:
            raise ValueError("transition-context dimensions must be positive")
        self.state_width = int(state_width)
        self.intention_width = int(intention_width)
        self.hidden_width = int(hidden_width)
        self.context_width = int(context_width)
        token_width = self.state_width * 2 + self.intention_width + 1
        self.token_encoder = nn.Sequential(
            nn.Linear(token_width, self.hidden_width),
            nn.GELU(),
        )
        self.recurrent = nn.GRU(
            self.hidden_width,
            self.hidden_width,
            batch_first=True,
        )
        self.context_projection = nn.Linear(
            self.hidden_width,
            self.context_width,
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "hidden_width": self.hidden_width,
            "context_width": self.context_width,
            "input": "opaque_state_intention_next_state_confidence_v1",
            "training": "paired_noisy_transition_view_contrastive_v1",
            "inference": "read_only_context_key_v1",
        }

    def _validate_inputs(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        next_state: torch.Tensor,
        confidence: torch.Tensor | None,
    ) -> torch.Tensor:
        for name, value, width in (
            ("context state", state, self.state_width),
            ("context intention", intention, self.intention_width),
            ("context next state", next_state, self.state_width),
        ):
            _validate_tensor(value, name=name, ndim=3, width=width)
        if state.shape != next_state.shape:
            raise ValueError("context state and next-state shapes differ")
        if state.shape[:2] != intention.shape[:2]:
            raise ValueError("context state and intention batches differ")
        if confidence is None:
            return torch.ones(
                state.shape[:2],
                device=state.device,
                dtype=state.dtype,
            )
        if confidence.shape not in (state.shape[:2], (*state.shape[:2], 1)):
            raise ValueError("context confidence must match batch and time")
        values = confidence.reshape(state.shape[0], state.shape[1]).to(
            device=state.device,
            dtype=state.dtype,
        )
        if not bool(torch.isfinite(values).all()) or bool(torch.any(values < 0)):
            raise ValueError("context confidence must be finite and non-negative")
        return values

    def forward(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        next_state: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        confidence_values = self._validate_inputs(
            state,
            intention,
            next_state,
            confidence,
        )
        tokens = torch.cat(
            (
                state,
                intention,
                next_state,
                confidence_values.unsqueeze(-1),
            ),
            dim=-1,
        )
        sequence, _hidden = self.recurrent(self.token_encoder(tokens))
        return torch.nn.functional.normalize(
            self.context_projection(sequence[:, -1]),
            dim=-1,
        )

    def encode_observation(
        self,
        observation: ExternalTransitionObservation,
    ) -> torch.Tensor:
        observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )
        confidence = observation.confidence
        return self(
            observation.state.unsqueeze(0),
            observation.intention.unsqueeze(0),
            observation.next_state.unsqueeze(0),
            None if confidence is None else confidence.unsqueeze(0),
        )[0]

    @staticmethod
    def contrastive_loss(
        left_context: torch.Tensor,
        right_context: torch.Tensor,
        *,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        if left_context.ndim != 2 or right_context.shape != left_context.shape:
            raise ValueError("context views must have shape [batch, context_width]")
        if left_context.shape[0] < 2:
            raise ValueError("context contrastive loss needs at least two views")
        if temperature <= 0.0:
            raise ValueError("context contrastive temperature must be positive")
        if not bool(torch.isfinite(left_context).all()) or not bool(
            torch.isfinite(right_context).all()
        ):
            raise ValueError("context views must be finite")
        left = torch.nn.functional.normalize(left_context, dim=-1)
        right = torch.nn.functional.normalize(right_context, dim=-1)
        logits = left @ right.transpose(0, 1) / temperature
        labels = torch.arange(left.shape[0], device=left.device)
        return 0.5 * (
            nn.functional.cross_entropy(logits, labels)
            + nn.functional.cross_entropy(logits.transpose(0, 1), labels)
        )

    @staticmethod
    def prefix_alignment_loss(
        prefix_contexts: torch.Tensor,
        full_contexts: torch.Tensor,
        *,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Align variable-length evidence prefixes with full bundle keys.

        ``prefix_contexts`` has shape ``[regimes, prefixes, width]`` and
        ``full_contexts`` has shape ``[regimes, width]``. Each prefix is a
        positive view of its own full key and every other regime is a
        negative. The multi-positive full-key term keeps all prefixes for one
        regime together without requiring a particular prefix length at
        inference time.
        """

        if prefix_contexts.ndim != 3 or full_contexts.ndim != 2:
            raise ValueError("prefix and full contexts have invalid rank")
        if prefix_contexts.shape[0] != full_contexts.shape[0]:
            raise ValueError("prefix and full context regime counts differ")
        if prefix_contexts.shape[1] < 1 or prefix_contexts.shape[2] != full_contexts.shape[1]:
            raise ValueError("prefix and full context widths differ")
        if prefix_contexts.shape[0] < 2:
            raise ValueError("prefix alignment needs at least two regimes")
        if temperature <= 0.0:
            raise ValueError("prefix alignment temperature must be positive")
        if not bool(torch.isfinite(prefix_contexts).all()) or not bool(
            torch.isfinite(full_contexts).all()
        ):
            raise ValueError("prefix and full contexts must be finite")

        prefixes = nn.functional.normalize(prefix_contexts, dim=-1)
        full = nn.functional.normalize(full_contexts, dim=-1)
        regime_count, prefix_count, _width = prefixes.shape
        logits = prefixes.reshape(regime_count * prefix_count, -1) @ full.transpose(0, 1)
        logits = logits / temperature
        labels = torch.arange(regime_count, device=logits.device).repeat_interleave(
            prefix_count
        )
        prefix_to_full = nn.functional.cross_entropy(logits, labels)

        reverse_logits = full @ prefixes.reshape(regime_count * prefix_count, -1).transpose(0, 1)
        reverse_logits = reverse_logits / temperature
        positive_mask = torch.zeros_like(reverse_logits, dtype=torch.bool)
        for regime in range(regime_count):
            start = regime * prefix_count
            positive_mask[regime, start : start + prefix_count] = True
        positive = torch.logsumexp(
            reverse_logits.masked_fill(~positive_mask, float("-inf")), dim=-1
        )
        reverse = (-positive + torch.logsumexp(reverse_logits, dim=-1)).mean()
        return 0.5 * (prefix_to_full + reverse)

    def state_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "sha256": self.digest(),
        }

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        for name, value in sorted(self.state_dict().items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalTransitionContextEncoder:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported transition-context encoder payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("transition-context encoder payload is incomplete")
        encoder = cls(
            int(configuration["state_width"]),
            int(configuration["intention_width"]),
            hidden_width=int(configuration["hidden_width"]),
            context_width=int(configuration["context_width"]),
        )
        current = encoder.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("transition-context encoder state names differ")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("transition-context encoder state is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("transition-context encoder state is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("transition-context encoder state is not finite")
            normalized[name] = value.detach().clone()
        encoder.load_state_dict(normalized, strict=True)
        if payload.get("sha256") != encoder.digest():
            raise ValueError("transition-context encoder checksum mismatch")
        return encoder


@dataclass(frozen=True)
class ExternalOnlineTransitionContextResult:
    """One read/admit decision from the online transition router."""

    status: str
    slot_index: int | None
    context: torch.Tensor | None
    pending_observations: int
    prediction_error: float | None
    observation: ExternalTransitionObservation | None = None
    stable_slot_id: int | None = None
    schema: str = EXTERNAL_ONLINE_TRANSITION_CONTEXT_ROUTER_SCHEMA

    def validate(
        self,
        *,
        state_width: int,
        intention_width: int,
        context_width: int,
    ) -> ExternalOnlineTransitionContextResult:
        if self.schema != EXTERNAL_ONLINE_TRANSITION_CONTEXT_ROUTER_SCHEMA:
            raise ValueError("unsupported online transition-context result schema")
        if self.status not in {
            "matched",
            "continuation",
            "conflict",
            "staged",
            "pending",
            "admitted",
            "reused",
            "capacity",
        }:
            raise ValueError("unsupported online transition-context result status")
        if self.slot_index is not None and self.slot_index < 0:
            raise ValueError("online transition-context slot index is invalid")
        if self.stable_slot_id is not None and self.stable_slot_id < 0:
            raise ValueError("online transition-context stable slot ID is invalid")
        if self.context is not None:
            _validate_tensor(
                self.context,
                name="online transition-context result key",
                ndim=1,
                width=context_width,
            )
        if (
            not isinstance(self.pending_observations, int)
            or self.pending_observations < 0
        ):
            raise ValueError("online transition-context pending count is invalid")
        if self.prediction_error is not None and (
            not math.isfinite(self.prediction_error) or self.prediction_error < 0.0
        ):
            raise ValueError("online transition-context prediction error is invalid")
        if self.observation is not None:
            self.observation.validate(
                state_width=state_width,
                intention_width=intention_width,
            )
        return self


@dataclass
class _ProvisionalTransitionCandidate:
    """Mutable isolated candidate state held outside the committed bank."""

    context: torch.Tensor
    model: nn.Module
    model_family: str
    observations: list[ExternalTransitionObservation]
    alternatives: dict[str, nn.Module] = field(default_factory=dict)

    def models(self) -> dict[str, nn.Module]:
        return {self.model_family: self.model, **self.alternatives}


class ExternalOnlineTransitionContextRouter:
    """Route alternating opaque transitions and admit novel regimes online.

    Existing slots are selected by factual one-step prediction error. An
    unmatched row is provisional and is not written to a committed model.
    Once enough consecutive unmatched current-stream rows exist, the external
    context encoder forms an opaque key and stages an isolated candidate.
    Conflicting novel streams receive separate candidates while capacity
    permits. The controller is not involved; callers own optimizer updates
    through :meth:`adaptation_step` and promotion through
    :meth:`promote_staged_candidate`.
    """

    schema = EXTERNAL_ONLINE_TRANSITION_CONTEXT_ROUTER_SCHEMA

    def __init__(
        self,
        bank: ExternalTransitionModelBank,
        context_encoder: ExternalTransitionContextEncoder,
        *,
        match_tolerance: float = 0.05,
        match_margin: float = 0.01,
        admission_observations: int = 12,
        max_contexts: int | None = None,
        continuation_tolerance: float | None = None,
        conflict_patience: int = 1,
        defer_admission: bool = False,
        candidate_model_families: Sequence[str] | None = None,
        provisional_continuation_tolerance: float | None = None,
    ) -> None:
        if (
            bank.state_width != context_encoder.state_width
            or bank.intention_width != context_encoder.intention_width
        ):
            raise ValueError("router bank and context encoder widths differ")
        if bank.context_width != context_encoder.context_width:
            raise ValueError("router bank and context widths differ")
        if match_tolerance < 0.0:
            raise ValueError("online context match tolerance cannot be negative")
        if match_margin < 0.0:
            raise ValueError("online context match margin cannot be negative")
        if continuation_tolerance is not None and continuation_tolerance < 0.0:
            raise ValueError("online context continuation tolerance cannot be negative")
        if (
            provisional_continuation_tolerance is not None
            and provisional_continuation_tolerance < 0.0
        ):
            raise ValueError("provisional continuation tolerance cannot be negative")
        if conflict_patience < 1:
            raise ValueError("online context conflict patience must be positive")
        if admission_observations < 1:
            raise ValueError("online context admission count must be positive")
        if max_contexts is not None and max_contexts < 1:
            raise ValueError("online context maximum must be positive")
        allowed_families = {
            EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
            EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        }
        if candidate_model_families is None:
            families = (
                (
                    EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
                    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
                    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
                )
                if bank.model_family == EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY
                else (bank.model_family,)
            )
        else:
            families = tuple(str(family) for family in candidate_model_families)
        if not families or len(set(families)) != len(families):
            raise ValueError("candidate model families must be unique and nonempty")
        if any(family not in allowed_families for family in families):
            raise ValueError("candidate model family is unsupported")
        if len(families) > 1 and bank.model_family != EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY:
            raise ValueError("multiple candidate families require a mixed model bank")
        self.bank = bank
        self.context_encoder = context_encoder
        self.match_tolerance = float(match_tolerance)
        self.match_margin = float(match_margin)
        self.admission_observations = int(admission_observations)
        self.max_contexts = max_contexts
        self.continuation_tolerance = float(
            match_tolerance
            if continuation_tolerance is None
            else continuation_tolerance
        )
        self.provisional_continuation_tolerance = float(
            self.continuation_tolerance
            if provisional_continuation_tolerance is None
            else provisional_continuation_tolerance
        )
        self.conflict_patience = int(conflict_patience)
        self.defer_admission = bool(defer_admission)
        self.candidate_model_families = families
        self._pending: list[ExternalTransitionObservation] = []
        self._active_slot: int | None = None
        self._active_slot_id: int | None = None
        self._conflict_windows = 0
        self._provisional_candidates: list[_ProvisionalTransitionCandidate] = []

    @property
    def pending_observations(self) -> int:
        return len(self._pending)

    @property
    def provisional_model(self) -> nn.Module | None:
        """Return the first staged model for backward-compatible callers."""

        return (
            None
            if not self._provisional_candidates
            else self._provisional_candidates[0].model
        )

    @property
    def provisional_candidate_count(self) -> int:
        """Return the number of isolated candidates awaiting promotion."""

        return len(self._provisional_candidates)

    def provisional_model_at(self, candidate_index: int) -> nn.Module:
        """Return one isolated candidate for caller-owned optimization."""

        if not 0 <= candidate_index < len(self._provisional_candidates):
            raise IndexError("provisional candidate index is out of range")
        return self._provisional_candidates[candidate_index].model

    def provisional_context_at(self, candidate_index: int) -> torch.Tensor:
        """Return one detached opaque candidate key."""

        if not 0 <= candidate_index < len(self._provisional_candidates):
            raise IndexError("provisional candidate index is out of range")
        return self._provisional_candidates[candidate_index].context.detach().clone()

    def provisional_evidence_count(self, candidate_index: int) -> int:
        """Return the number of evidence bundles retained by one candidate."""

        if not 0 <= candidate_index < len(self._provisional_candidates):
            raise IndexError("provisional candidate index is out of range")
        return len(self._provisional_candidates[candidate_index].observations)

    @property
    def _provisional_context(self) -> torch.Tensor | None:
        return (
            None
            if not self._provisional_candidates
            else self._provisional_candidates[0].context
        )

    @property
    def _provisional_model(self) -> nn.Module | None:
        return self.provisional_model

    @property
    def _provisional_observations(self) -> list[ExternalTransitionObservation]:
        return (
            []
            if not self._provisional_candidates
            else self._provisional_candidates[0].observations
        )

    def configuration(self) -> dict[str, int | float | str | None]:
        return {
            "schema": self.schema,
            "state_width": self.bank.state_width,
            "intention_width": self.bank.intention_width,
            "context_width": self.bank.context_width,
            "match_tolerance": self.match_tolerance,
            "match_margin": self.match_margin,
            "admission_observations": self.admission_observations,
            "max_contexts": self.max_contexts,
            "continuation_tolerance": self.continuation_tolerance,
            "provisional_continuation_tolerance": self.provisional_continuation_tolerance,
            "conflict_patience": self.conflict_patience,
            "defer_admission": self.defer_admission,
            "candidate_model_families": list(self.candidate_model_families),
            "routing": "factual_prediction_error_then_bound_continuation_v2",
            "writes": "caller_owned_slot_only_v1",
            "provisional_evidence": "cumulative_verified_window_v1",
            "provisional_candidates": "isolated_indexed_copy_on_write_v1",
        }

    def grow_verified(
        self,
        destination_capacity: int,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
    ) -> ExternalTransitionModelGrowthReceipt:
        """Grow routing and bank capacity as one verified external transaction."""

        if self.max_contexts is None:
            raise ValueError("router requires an explicit maximum for verified growth")
        if self.bank.capacity != self.max_contexts:
            raise ValueError("router and bank capacities are out of sync")
        receipt = self.bank.grow_verified(destination_capacity, retention_probe)
        if receipt.accepted:
            self.max_contexts = destination_capacity
        return receipt

    def evict_verified(
        self,
        index: int,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
    ) -> ExternalTransitionModelEvictionReceipt:
        """Evict a bank tail and invalidate any stale active-slot reference."""

        receipt = self.bank.evict_verified(index, retention_probe)
        if receipt.accepted:
            self._refresh_active_slot()
        return receipt

    def evict_with_lifetime_policy_verified(
        self,
        policy: ExternalTransitionModelLifetimePolicy,
        usage: torch.Tensor,
        age: torch.Tensor,
        prediction_error: torch.Tensor,
        protected: torch.Tensor,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[
        ExternalTransitionModelLifetimeProposal,
        ExternalTransitionModelEvictionReceipt | None,
    ]:
        """Use a learned proposal while preserving router address repair."""

        proposal, receipt = policy.evict_verified(
            self.bank,
            usage,
            age,
            prediction_error,
            protected,
            retention_probe,
            optimizer,
        )
        if receipt is not None and receipt.accepted:
            self._refresh_active_slot()
        return proposal, receipt

    def evict_with_bank_lifetime_policy_verified(
        self,
        policy: ExternalTransitionModelLifetimePolicy,
        protected: torch.Tensor,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[
        ExternalTransitionModelLifetimeProposal,
        ExternalTransitionModelEvictionReceipt | None,
    ]:
        """Use bank-owned telemetry and repair the active logical reference."""

        proposal, receipt = policy.evict_from_bank_verified(
            self.bank,
            protected,
            retention_probe,
            optimizer,
        )
        if receipt is not None and receipt.accepted:
            self._refresh_active_slot()
        return proposal, receipt

    def evict_verified_id(
        self,
        slot_id: int,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
    ) -> ExternalTransitionModelEvictionReceipt:
        """Evict a logical slot and repair the router's physical cache."""

        receipt = self.bank.evict_verified_id(slot_id, retention_probe)
        if receipt.accepted:
            self._refresh_active_slot()
        return receipt

    def slot_id_at(self, index: int) -> int:
        """Expose a stable memory address for a physical slot."""

        return self.bank.slot_id_at(index)

    def physical_index_for_slot_id(self, slot_id: int) -> int:
        """Resolve a persisted logical address after memory reorganization."""

        return self.bank.physical_index_for_slot_id(slot_id)

    def _set_active_slot(self, index: int | None) -> None:
        self._active_slot = index
        self._active_slot_id = (
            None if index is None else self.bank.slot_id_at(index)
        )

    def _refresh_active_slot(self) -> None:
        if self._active_slot_id is None:
            self._set_active_slot(None)
            return
        try:
            self._active_slot = self.bank.physical_index_for_slot_id(
                self._active_slot_id
            )
        except KeyError:
            self._set_active_slot(None)

    @staticmethod
    def _clone_observation(
        observation: ExternalTransitionObservation,
    ) -> ExternalTransitionObservation:
        return ExternalTransitionObservation(
            state=observation.state.detach().clone(),
            intention=observation.intention.detach().clone(),
            next_state=observation.next_state.detach().clone(),
            confidence=(
                None
                if observation.confidence is None
                else observation.confidence.detach().clone()
            ),
        )

    @staticmethod
    def _merge_observations(
        observations: list[ExternalTransitionObservation],
    ) -> ExternalTransitionObservation:
        if not observations:
            raise ValueError("cannot merge an empty pending transition bundle")
        confidence = [
            torch.ones(item.state.shape[0], device=item.state.device)
            if item.confidence is None
            else item.confidence.reshape(-1)
            for item in observations
        ]
        return ExternalTransitionObservation(
            state=torch.cat([item.state for item in observations]),
            intention=torch.cat([item.intention for item in observations]),
            next_state=torch.cat([item.next_state for item in observations]),
            confidence=torch.cat(confidence),
        )

    def _best_slot(
        self,
        observation: ExternalTransitionObservation,
    ) -> tuple[int, float, float, torch.Tensor] | None:
        if self.bank.context_count == 0:
            return None
        candidates: list[tuple[float, int, torch.Tensor]] = []
        for index in range(self.bank.context_count):
            context = self.bank.context_at(index).to(observation.state)
            context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
            prediction = self.bank(
                observation.state,
                observation.intention,
                context_batch,
            )
            error = float(
                (prediction - observation.next_state).square().mean().detach()
            )
            candidates.append((error, index, context))
        error, index, context = min(candidates, key=lambda item: (item[0], item[1]))
        ordered_errors = sorted(item[0] for item in candidates)
        margin = (
            float("inf")
            if len(ordered_errors) == 1
            else ordered_errors[1] - ordered_errors[0]
        )
        if error > self.match_tolerance or margin < self.match_margin:
            return None
        return index, error, margin, context

    def _slot_error(
        self,
        index: int,
        observation: ExternalTransitionObservation,
    ) -> float:
        context = self.bank.context_at(index).to(observation.state)
        context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
        prediction = self.bank(
            observation.state,
            observation.intention,
            context_batch,
        )
        return float(
            (prediction - observation.next_state).square().mean().detach()
        )

    def _pending_result(
        self,
        *,
        status: str = "pending",
        prediction_error: float | None = None,
        slot_index: int | None = None,
        stable_slot_id: int | None = None,
        context: torch.Tensor | None = None,
        observation: ExternalTransitionObservation | None = None,
    ) -> ExternalOnlineTransitionContextResult:
        return ExternalOnlineTransitionContextResult(
            status=status,
            slot_index=slot_index,
            context=context,
            pending_observations=self.pending_observations,
            prediction_error=prediction_error,
            observation=observation,
            stable_slot_id=stable_slot_id,
        ).validate(
            state_width=self.bank.state_width,
            intention_width=self.bank.intention_width,
            context_width=self.bank.context_width,
        )

    def _stage_candidate(
        self,
        context: torch.Tensor,
        *,
        prior_index: int | None,
    ) -> int:
        models = {
            family: self.bank._new_model(family)
            for family in self.candidate_model_families
        }
        if prior_index is not None:
            prior_family = self.bank.model_family_at(prior_index)
            for family, model in models.items():
                if (
                    family == prior_family
                    and family == EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY
                ):
                    model.load_state_dict(self.bank.models[prior_index].state_dict())
        self._provisional_candidates.append(
            _ProvisionalTransitionCandidate(
                context=context.detach().clone(),
                model=models[self.candidate_model_families[0]],
                model_family=self.candidate_model_families[0],
                observations=[],
                alternatives={
                    family: model
                    for family, model in models.items()
                    if family != self.candidate_model_families[0]
                },
            )
        )
        return len(self._provisional_candidates) - 1

    def _staged_result(
        self,
        observation: ExternalTransitionObservation,
        *,
        candidate_index: int,
    ) -> ExternalOnlineTransitionContextResult:
        if not 0 <= candidate_index < len(self._provisional_candidates):
            raise RuntimeError("staged result requested without a candidate")
        candidate = self._provisional_candidates[candidate_index]
        candidate.observations.append(self._clone_observation(observation))
        return ExternalOnlineTransitionContextResult(
            status="staged",
            slot_index=candidate_index,
            context=candidate.context.detach().clone(),
            pending_observations=0,
            prediction_error=self._slot_error_from_model(
                candidate.model,
                observation,
            ),
            observation=observation,
        ).validate(
            state_width=self.bank.state_width,
            intention_width=self.bank.intention_width,
            context_width=self.bank.context_width,
        )

    @staticmethod
    def _slot_error_from_model(
        model: nn.Module,
        observation: ExternalTransitionObservation,
    ) -> float:
        prediction = model(observation.state, observation.intention)
        return float(
            (prediction - observation.next_state).square().mean().detach()
        )

    @classmethod
    def _candidate_error(
        cls,
        candidate: _ProvisionalTransitionCandidate,
        observation: ExternalTransitionObservation,
    ) -> float:
        return min(
            cls._slot_error_from_model(model, observation)
            for model in candidate.models().values()
        )

    def observe(
        self,
        observation: ExternalTransitionObservation,
    ) -> ExternalOnlineTransitionContextResult:
        """Route one current-stream row without accepting a regime label."""

        self._refresh_active_slot()
        observation.validate(
            state_width=self.bank.state_width,
            intention_width=self.bank.intention_width,
        )
        if observation.state.shape[0] != 1:
            raise ValueError("online context routing accepts one transition row")
        self._pending.append(self._clone_observation(observation))
        if self.pending_observations < self.admission_observations:
            return self._pending_result()

        bundle = self._merge_observations(self._pending)
        match = self._best_slot(bundle)
        if match is not None:
            index, error, _margin, context = match
            self._pending.clear()
            self._provisional_candidates = [
                candidate
                for candidate in self._provisional_candidates
                if self._candidate_error(candidate, bundle)
                > self.provisional_continuation_tolerance
            ]
            self._set_active_slot(index)
            self._conflict_windows = 0
            return ExternalOnlineTransitionContextResult(
                status="matched",
                slot_index=index,
                context=context,
                pending_observations=0,
                prediction_error=error,
                observation=bundle,
                stable_slot_id=self.bank.slot_id_at(index),
            ).validate(
                state_width=self.bank.state_width,
                intention_width=self.bank.intention_width,
                context_width=self.bank.context_width,
            )

        if self._active_slot is not None and self._active_slot < self.bank.context_count:
            active_error = self._slot_error(self._active_slot, bundle)
            active_context = self.bank.context_at(self._active_slot)
            if active_error <= self.continuation_tolerance:
                index = self._active_slot
                self._pending.clear()
                self._conflict_windows = 0
                return ExternalOnlineTransitionContextResult(
                    status="continuation",
                    slot_index=index,
                    context=active_context,
                    pending_observations=0,
                    prediction_error=active_error,
                    observation=bundle,
                    stable_slot_id=self.bank.slot_id_at(index),
                ).validate(
                    state_width=self.bank.state_width,
                    intention_width=self.bank.intention_width,
                    context_width=self.bank.context_width,
                )
            self._conflict_windows += 1
            self._pending.clear()
            if self._conflict_windows < self.conflict_patience:
                return self._pending_result(
                    status="conflict",
                    prediction_error=active_error,
                    observation=bundle,
                )
            self._active_slot = None
            self._conflict_windows = 0

        if self.defer_admission:
            if self._provisional_candidates:
                candidate_errors = [
                    (
                        self._candidate_error(candidate, bundle),
                        index,
                    )
                    for index, candidate in enumerate(self._provisional_candidates)
                ]
                candidate_error, candidate_index = min(
                    candidate_errors,
                    key=lambda item: (item[0], item[1]),
                )
                if candidate_error <= self.provisional_continuation_tolerance:
                    self._pending.clear()
                    return self._staged_result(
                        bundle,
                        candidate_index=candidate_index,
                    )
            if self.max_contexts is not None and (
                self.bank.context_count + len(self._provisional_candidates)
                >= self.max_contexts
            ):
                self._pending.clear()
                return self._pending_result(status="capacity")
            with torch.no_grad():
                candidate_context = self.context_encoder.encode_observation(
                    bundle
                ).detach()
            prior_index: int | None = None
            if self.bank.context_count:
                prior_index = int(
                    (
                        self.bank.contexts
                        @ candidate_context.to(self.bank.contexts)
                    ).argmax()
                )
            candidate_index = self._stage_candidate(
                candidate_context,
                prior_index=prior_index,
            )
            self._pending.clear()
            return self._staged_result(bundle, candidate_index=candidate_index)

        if self.max_contexts is not None and self.bank.context_count >= self.max_contexts:
            self._pending.clear()
            return self._pending_result(status="capacity")

        with torch.no_grad():
            context = self.context_encoder.encode_observation(bundle).detach()
        prior_index: int | None = None
        if self.bank.context_count:
            prior_index = int(
                (self.bank.contexts @ context.to(self.bank.contexts)).argmax()
            )
        before = self.bank.context_count
        index = self.bank.ensure_context(context, initialize_from=prior_index)
        status = "admitted" if index == before else "reused"
        self._pending.clear()
        self._set_active_slot(index)
        self._conflict_windows = 0
        return ExternalOnlineTransitionContextResult(
            status=status,
            slot_index=index,
            context=self.bank.context_at(index),
            pending_observations=0,
            prediction_error=None,
            observation=bundle,
            stable_slot_id=self.bank.slot_id_at(index),
        ).validate(
            state_width=self.bank.state_width,
            intention_width=self.bank.intention_width,
            context_width=self.bank.context_width,
        )

    def adaptation_step(
        self,
        result: ExternalOnlineTransitionContextResult,
        optimizer: (
            torch.optim.Optimizer
            | Mapping[str, torch.optim.Optimizer]
            | None
        ),
        *,
        replay_evidence: bool = True,
    ) -> float:
        """Update only the externally selected slot using current evidence.

        ``replay_evidence=True`` trains a provisional candidate against its
        retained evidence window. Setting it to ``False`` performs a strictly
        one-pass update on only the current staged observation; this is an
        explicit audit/control mode for replay-free learning claims.
        """

        result.validate(
            state_width=self.bank.state_width,
            intention_width=self.bank.intention_width,
            context_width=self.bank.context_width,
        )
        if result.status == "staged":
            candidate_index = 0 if result.slot_index is None else result.slot_index
            if (
                result.observation is None
                or not 0 <= candidate_index < len(self._provisional_candidates)
            ):
                return 0.0
            candidate = self._provisional_candidates[candidate_index]
            evidence = self._merge_observations(
                candidate.observations if replay_evidence else [result.observation]
            )
            primary_loss = candidate.model.loss(evidence)
            for family, model in candidate.models().items():
                if hasattr(model, "observe"):
                    with torch.no_grad():
                        model.observe(result.observation)
                    continue
                selected_optimizer = (
                    optimizer.get(family)
                    if isinstance(optimizer, Mapping)
                    else optimizer
                    if family == candidate.model_family
                    else None
                )
                if selected_optimizer is None:
                    selected_optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=self.bank.adaptation_learning_rate,
                    )
                selected_optimizer.zero_grad()
                model_loss = model.loss(evidence)
                model_loss.backward()
                selected_optimizer.step()
            return float(primary_loss.detach())
        if result.slot_index is None or result.context is None or result.observation is None:
            return 0.0
        context = result.context.to(result.observation.state)
        context_batch = context.unsqueeze(0).expand(
            result.observation.state.shape[0], -1
        )
        return self.bank.adaptation_step(
            result.observation,
            context_batch,
            optimizer,
        )

    def promote_staged_candidate(
        self,
        heldout_observation: ExternalTransitionObservation,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
        *,
        prediction_tolerance: float = 0.05,
        candidate_index: int = 0,
        destination_capacity: int | None = None,
    ) -> ExternalTransitionModelCandidateReceipt:
        """Commit a provisional model only after held-out and retention proof.

        When the committed bank is full, ``destination_capacity`` can request
        an atomic metadata-only capacity expansion as part of the same
        copy-on-write candidate transaction. Existing model content is copied
        before the candidate is tested; a failed candidate never changes the
        live capacity or slots.
        """

        if prediction_tolerance < 0.0:
            raise ValueError("candidate prediction tolerance cannot be negative")
        if destination_capacity is not None:
            if not isinstance(destination_capacity, int):
                raise TypeError("destination capacity must be an integer")
            if self.bank.capacity is None:
                raise ValueError(
                    "destination capacity requires a bounded model bank"
                )
            if destination_capacity <= self.bank.capacity:
                raise ValueError(
                    "destination capacity must exceed current model-bank capacity"
                )
        heldout_observation.validate(
            state_width=self.bank.state_width,
            intention_width=self.bank.intention_width,
        )
        if not callable(retention_probe):
            raise TypeError("candidate retention probe must be callable")
        before = self.bank.content_digest()
        if not 0 <= candidate_index < len(self._provisional_candidates):
            candidate = None
            context = None
            candidate_digest = "none"
        else:
            candidate = self._provisional_candidates[candidate_index]
            context = candidate.context
        if candidate is None or context is None:
            return ExternalTransitionModelCandidateReceipt(
                accepted=False,
                slot_index=None,
                context_count=self.bank.context_count,
                heldout_error=float("inf"),
                candidate_digest=candidate_digest,
                content_digest_before=before,
                content_digest_after=before,
                reason="no provisional candidate is staged",
            ).validate()
        candidate_models = candidate.models()
        selection = self.bank.select_model_family_verified(
            candidate_models,
            heldout_observation,
            prediction_tolerance=prediction_tolerance,
        )
        if not selection.accepted or selection.selected_family is None:
            best_error = min(
                receipt.heldout_error for receipt in selection.candidates
            )
            candidate_digest = candidate.model.digest()
            return ExternalTransitionModelCandidateReceipt(
                accepted=False,
                slot_index=None,
                context_count=self.bank.context_count,
                heldout_error=best_error,
                candidate_digest=candidate_digest,
                content_digest_before=before,
                content_digest_after=before,
                reason="no provisional model family passed held-out verification",
            ).validate()
        selected_family = selection.selected_family
        model = candidate_models[selected_family]
        candidate_digest = model.digest()
        if (
            destination_capacity is None
            and self.bank.capacity is not None
            and self.bank.context_count >= self.bank.capacity
        ):
            return ExternalTransitionModelCandidateReceipt(
                accepted=False,
                slot_index=None,
                context_count=self.bank.context_count,
                heldout_error=float("inf"),
                candidate_digest=candidate_digest,
                content_digest_before=before,
                content_digest_after=before,
                reason="committed model-bank capacity is full",
            ).validate()

        candidate_bank = ExternalTransitionModelBank.from_payload(self.bank.payload())
        target_capacity = (
            self.bank.capacity
            if destination_capacity is None
            else destination_capacity
        )
        if target_capacity is not None:
            candidate_content_before_growth = candidate_bank.content_digest()
            candidate_bank.capacity = target_capacity
            if candidate_bank.content_digest() != candidate_content_before_growth:
                raise RuntimeError("candidate capacity growth changed model content")
        candidate_count_before = candidate_bank.context_count
        bank_candidate_index = candidate_bank.ensure_context(
            context,
            model_family=selected_family,
        )
        if bank_candidate_index != candidate_count_before:
            return ExternalTransitionModelCandidateReceipt(
                accepted=False,
                slot_index=None,
                context_count=self.bank.context_count,
                heldout_error=float("inf"),
                candidate_digest=candidate_digest,
                content_digest_before=before,
                content_digest_after=before,
                reason="provisional context duplicates a committed context",
            ).validate()
        candidate_bank.models[bank_candidate_index].load_state_dict(model.state_dict())
        heldout_context = context.unsqueeze(0).expand(
            heldout_observation.state.shape[0], -1
        )
        heldout_error = float(
            candidate_bank.loss(heldout_observation, heldout_context).detach()
        )
        if heldout_error > prediction_tolerance:
            return ExternalTransitionModelCandidateReceipt(
                accepted=False,
                slot_index=None,
                context_count=self.bank.context_count,
                heldout_error=heldout_error,
                candidate_digest=candidate_digest,
                content_digest_before=before,
                content_digest_after=before,
                reason="held-out candidate prediction failed",
            ).validate()
        if not bool(retention_probe(candidate_bank)):
            return ExternalTransitionModelCandidateReceipt(
                accepted=False,
                slot_index=None,
                context_count=self.bank.context_count,
                heldout_error=heldout_error,
                candidate_digest=candidate_digest,
                content_digest_before=before,
                content_digest_after=before,
                reason="candidate retention probe failed",
            ).validate()

        promoted_model = self.bank._new_model(selected_family)
        promoted_model.load_state_dict(model.state_dict())
        self.bank._contexts.append(context.detach().cpu().clone())
        self.bank.models.append(promoted_model)
        self.bank._model_families.append(selected_family)
        self.bank._slot_ids.append(self.bank._next_slot_id)
        self.bank._initialize_lifetime_slot(self.bank._next_slot_id)
        self.bank._next_slot_id += 1
        if destination_capacity is not None:
            self.bank.capacity = destination_capacity
            if self.max_contexts is not None:
                self.max_contexts = destination_capacity
        after = self.bank.content_digest()
        slot_index = self.bank.context_count - 1
        self._set_active_slot(slot_index)
        self._conflict_windows = 0
        del self._provisional_candidates[candidate_index]
        return ExternalTransitionModelCandidateReceipt(
            accepted=True,
            slot_index=slot_index,
            context_count=self.bank.context_count,
            heldout_error=heldout_error,
            candidate_digest=candidate_digest,
            content_digest_before=before,
            content_digest_after=after,
            slot_id=self.bank.slot_id_at(slot_index),
            reason=(
                "held-out and retention-verified candidate promotion with "
                "capacity growth committed"
                if destination_capacity is not None
                else "held-out and retention-verified candidate promotion committed"
            ),
        ).validate()

    @staticmethod
    def _observation_payload(
        observation: ExternalTransitionObservation,
    ) -> dict[str, object]:
        return {
            "state": observation.state.detach().cpu().tolist(),
            "intention": observation.intention.detach().cpu().tolist(),
            "next_state": observation.next_state.detach().cpu().tolist(),
            "confidence": (
                None
                if observation.confidence is None
                else observation.confidence.detach().cpu().tolist()
            ),
        }

    @staticmethod
    def _observation_from_payload(
        payload: Mapping[str, Any],
    ) -> ExternalTransitionObservation:
        if not isinstance(payload, Mapping):
            raise TypeError("online transition pending row must be a mapping")
        confidence = payload.get("confidence")
        return ExternalTransitionObservation(
            state=torch.tensor(payload["state"], dtype=torch.float32),
            intention=torch.tensor(payload["intention"], dtype=torch.float32),
            next_state=torch.tensor(payload["next_state"], dtype=torch.float32),
            confidence=(
                None
                if confidence is None
                else torch.tensor(confidence, dtype=torch.float32)
            ),
        )

    @staticmethod
    def _model_from_payload(payload: Mapping[str, Any]) -> nn.Module:
        schema = payload.get("schema")
        if schema == EXTERNAL_TRANSITION_MODEL_SCHEMA:
            return ExternalTransitionModel.from_payload(payload)
        from .online_transition import (
            ExternalAffineTransitionStatistics,
            ExternalRandomFeatureTransitionStatistics,
        )

        if schema == ExternalAffineTransitionStatistics.schema:
            return ExternalAffineTransitionStatistics.from_payload(payload)
        if schema != ExternalRandomFeatureTransitionStatistics.schema:
            raise ValueError("unsupported online transition provisional model")
        return ExternalRandomFeatureTransitionStatistics.from_payload(payload)

    def state_payload(self) -> dict[str, object]:
        provisional_candidates = [
            {
                "context": candidate.context.detach().cpu().clone(),
                "model": candidate.model.state_payload(),
                "model_family": candidate.model_family,
                "models": {
                    family: model.state_payload()
                    for family, model in candidate.models().items()
                },
                "observations": [
                    self._observation_payload(row)
                    for row in candidate.observations
                ],
            }
            for candidate in self._provisional_candidates
        ]
        first_candidate = (
            None
            if not self._provisional_candidates
            else self._provisional_candidates[0]
        )
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "bank": self.bank.payload(),
            "context_encoder": self.context_encoder.state_payload(),
            "pending": [self._observation_payload(row) for row in self._pending],
            "active_slot": self._active_slot,
            "active_slot_id": self._active_slot_id,
            "conflict_windows": self._conflict_windows,
            "provisional_candidates": provisional_candidates,
            "provisional_context": (
                None
                if first_candidate is None
                else first_candidate.context.detach().cpu().clone()
            ),
            "provisional_model": (
                None
                if first_candidate is None
                else first_candidate.model.state_payload()
            ),
            "provisional_model_family": (
                None
                if first_candidate is None
                else first_candidate.model_family
            ),
            "provisional_observations": [
                self._observation_payload(row)
                for row in ([] if first_candidate is None else first_candidate.observations)
            ],
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalOnlineTransitionContextRouter:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported online transition-context router payload")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("online transition-context router configuration is missing")
        bank_payload = payload.get("bank")
        encoder_payload = payload.get("context_encoder")
        pending_payload = payload.get("pending")
        provisional_candidates_payload = payload.get("provisional_candidates")
        provisional_observations_payload = payload.get("provisional_observations", [])
        if not isinstance(bank_payload, Mapping) or not isinstance(
            encoder_payload, Mapping
        ):
            raise TypeError("online transition-context router components are missing")
        if not isinstance(pending_payload, list):
            raise TypeError("online transition-context router pending rows are invalid")
        if not isinstance(provisional_observations_payload, list):
            raise TypeError(
                "online transition-context provisional evidence is invalid"
            )
        if provisional_candidates_payload is not None and not isinstance(
            provisional_candidates_payload, list
        ):
            raise TypeError(
                "online transition-context provisional candidates are invalid"
            )
        bank = ExternalTransitionModelBank.from_payload(bank_payload)
        encoder = ExternalTransitionContextEncoder.from_payload(encoder_payload)
        router = cls(
            bank,
            encoder,
            match_tolerance=float(configuration["match_tolerance"]),
            match_margin=float(configuration["match_margin"]),
            admission_observations=int(configuration["admission_observations"]),
            max_contexts=(
                None
                if configuration.get("max_contexts") is None
                else int(configuration["max_contexts"])
            ),
            continuation_tolerance=float(
                configuration.get(
                    "continuation_tolerance", configuration["match_tolerance"]
                )
            ),
            provisional_continuation_tolerance=(
                None
                if configuration.get("provisional_continuation_tolerance") is None
                else float(configuration["provisional_continuation_tolerance"])
            ),
            conflict_patience=int(configuration.get("conflict_patience", 1)),
            defer_admission=bool(configuration.get("defer_admission", False)),
            candidate_model_families=(
                None
                if configuration.get("candidate_model_families") is None
                else tuple(str(family) for family in configuration["candidate_model_families"])
            ),
        )
        for row_payload in pending_payload:
            row = cls._observation_from_payload(row_payload)
            row.validate(
                state_width=bank.state_width,
                intention_width=bank.intention_width,
            )
            if row.state.shape[0] != 1:
                raise ValueError("online transition pending row is not scalar")
            router._pending.append(row)
        active_slot = payload.get("active_slot")
        if active_slot is not None and (
            not isinstance(active_slot, int) or not 0 <= active_slot < bank.context_count
        ):
            raise ValueError("online transition active slot is invalid")
        active_slot_id = payload.get("active_slot_id")
        if active_slot_id is not None and (
            not isinstance(active_slot_id, int)
            or isinstance(active_slot_id, bool)
            or active_slot_id < 0
        ):
            raise ValueError("online transition active stable slot ID is invalid")
        if active_slot_id is not None:
            try:
                resolved_active_slot = bank.physical_index_for_slot_id(active_slot_id)
            except KeyError as error:
                raise ValueError(
                    "online transition active stable slot ID is unknown"
                ) from error
            if active_slot is not None and active_slot != resolved_active_slot:
                raise ValueError("online transition active slot references disagree")
            active_slot = resolved_active_slot
        router._set_active_slot(active_slot)
        conflict_windows = payload.get("conflict_windows", 0)
        if not isinstance(conflict_windows, int) or conflict_windows < 0:
            raise ValueError("online transition conflict count is invalid")
        router._conflict_windows = conflict_windows
        if provisional_candidates_payload is None:
            provisional_context = payload.get("provisional_context")
            provisional_model = payload.get("provisional_model")
            if (provisional_context is None) != (provisional_model is None):
                raise ValueError("online transition provisional candidate is incomplete")
            if provisional_context is not None and not isinstance(
                provisional_context, torch.Tensor
            ):
                raise TypeError("online transition provisional context is not a tensor")
            provisional_candidates_payload = []
            if provisional_context is not None:
                provisional_candidates_payload.append(
                    {
                        "context": provisional_context,
                        "model": provisional_model,
                        "observations": provisional_observations_payload,
                    }
                )
        for candidate_payload in provisional_candidates_payload:
            if not isinstance(candidate_payload, Mapping):
                raise TypeError("online transition provisional candidate is invalid")
            provisional_context = candidate_payload.get("context")
            provisional_model = candidate_payload.get("model")
            candidate_models_payload = candidate_payload.get("models")
            candidate_model_family = candidate_payload.get("model_family")
            observations_payload = candidate_payload.get("observations", [])
            if not isinstance(provisional_context, torch.Tensor):
                raise TypeError("online transition provisional context is not a tensor")
            if not isinstance(provisional_model, Mapping):
                raise TypeError("online transition provisional model is invalid")
            if candidate_models_payload is not None and not isinstance(
                candidate_models_payload, Mapping
            ):
                raise TypeError("online transition provisional model candidates are invalid")
            if not isinstance(observations_payload, list):
                raise TypeError("online transition provisional evidence is invalid")
            _validate_tensor(
                provisional_context,
                name="online transition provisional context",
                ndim=1,
                width=bank.context_width,
            )
            if candidate_models_payload is None:
                family = str(
                    candidate_model_family
                    or (
                        EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY
                        if bank.model_family == EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY
                        else bank.model_family
                    )
                )
                candidate_models_payload = {family: provisional_model}
            restored_models: dict[str, nn.Module] = {}
            for family, model_payload in candidate_models_payload.items():
                if family not in {
                    EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
                    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
                    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
                }:
                    raise ValueError("online transition provisional family is invalid")
                if not isinstance(model_payload, Mapping):
                    raise TypeError("online transition provisional model candidate is invalid")
                restored_model = cls._model_from_payload(model_payload)
                if (
                    restored_model.state_width != bank.state_width
                    or restored_model.intention_width != bank.intention_width
                ):
                    raise ValueError("online transition provisional model is incompatible")
                if (
                    family == EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY
                    and not isinstance(restored_model, ExternalTransitionModel)
                ) or (
                    family == EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY
                    and isinstance(restored_model, ExternalTransitionModel)
                ):
                    raise ValueError("online transition provisional model family differs")
                restored_models[str(family)] = restored_model
            primary_family = str(candidate_model_family or next(iter(restored_models)))
            if primary_family not in restored_models:
                raise ValueError("online transition primary family is missing")
            observations: list[ExternalTransitionObservation] = []
            for observation_payload in observations_payload:
                observation = cls._observation_from_payload(observation_payload)
                observation.validate(
                    state_width=bank.state_width,
                    intention_width=bank.intention_width,
                )
                observations.append(observation)
            router._provisional_candidates.append(
                _ProvisionalTransitionCandidate(
                    context=provisional_context.detach().clone(),
                    model=restored_models[primary_family],
                    model_family=primary_family,
                    observations=observations,
                    alternatives={
                        family: model
                        for family, model in restored_models.items()
                        if family != primary_family
                    },
                )
            )
        return router


class ExternalGoalEvaluator(nn.Module):
    """A replaceable learned verifier for opaque terminal states."""

    schema = EXTERNAL_GOAL_EVALUATOR_SCHEMA

    def __init__(self, state_width: int, *, hidden_width: int = 64) -> None:
        super().__init__()
        if min(state_width, hidden_width) < 1:
            raise ValueError("goal-evaluator dimensions must be positive")
        self.state_width = int(state_width)
        self.hidden_width = int(hidden_width)
        self.network = nn.Sequential(
            nn.Linear(self.state_width * 3, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, 1),
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "hidden_width": self.hidden_width,
            "training": "scalar_verifier_outcomes_v1",
            "score": "sigmoid_success_probability_v1",
        }

    def _validate_inputs(self, state: torch.Tensor, goal_state: torch.Tensor) -> None:
        _validate_tensor(state, name="state", ndim=2, width=self.state_width)
        _validate_tensor(
            goal_state, name="goal_state", ndim=2, width=self.state_width
        )
        if state.shape[0] != goal_state.shape[0]:
            raise ValueError("state and goal-state batches differ")

    def forward(self, state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(state, goal_state)
        difference = (state - goal_state).square()
        return self.network(torch.cat((state, goal_state, difference), dim=-1)).squeeze(-1)

    def loss(
        self,
        state: torch.Tensor,
        goal_state: torch.Tensor,
        outcome: torch.Tensor,
    ) -> torch.Tensor:
        if outcome.shape not in ((state.shape[0],), (state.shape[0], 1)):
            raise ValueError("goal-evaluator outcomes must match the batch")
        if not bool(torch.isfinite(outcome).all()):
            raise ValueError("goal-evaluator outcomes must be finite")
        if bool(torch.any(outcome < 0) or torch.any(outcome > 1)):
            raise ValueError("goal-evaluator outcomes must lie in [0, 1]")
        targets = outcome.reshape(-1).to(
            device=state.device, dtype=state.dtype
        )
        return nn.functional.binary_cross_entropy_with_logits(
            self(state, goal_state), targets
        )

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        for name, value in sorted(self.state_dict().items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        return digest.hexdigest()


class ExternalTransitionEvidenceEvaluator(nn.Module):
    """Learned external verifier for noisy factual transition evidence."""

    schema = EXTERNAL_TRANSITION_EVIDENCE_EVALUATOR_SCHEMA

    def __init__(self, state_width: int, *, hidden_width: int = 64) -> None:
        super().__init__()
        if min(state_width, hidden_width) < 1:
            raise ValueError("transition-evidence dimensions must be positive")
        self.state_width = int(state_width)
        self.hidden_width = int(hidden_width)
        self.network = nn.Sequential(
            nn.Linear(self.state_width * 3 + 1, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, 1),
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "hidden_width": self.hidden_width,
            "training": "deterministic_transition_verifier_outcomes_v1",
            "behavior": "read_only_consistency_gate_v1",
        }

    def _validate_inputs(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        hit: torch.Tensor | None,
    ) -> torch.Tensor:
        _validate_tensor(
            prediction, name="transition prediction", ndim=2, width=self.state_width
        )
        _validate_tensor(
            observed, name="observed next state", ndim=2, width=self.state_width
        )
        if prediction.shape != observed.shape:
            raise ValueError("transition prediction and observation shapes differ")
        if hit is None:
            return torch.ones(
                prediction.shape[0], device=prediction.device, dtype=prediction.dtype
            )
        if hit.shape not in ((prediction.shape[0],), (prediction.shape[0], 1)):
            raise ValueError("transition hit flags must match the batch")
        hit_value = hit.reshape(-1).to(
            device=prediction.device, dtype=prediction.dtype
        )
        if not bool(torch.isfinite(hit_value).all()) or bool(
            torch.any(hit_value < 0) or torch.any(hit_value > 1)
        ):
            raise ValueError("transition hit flags must lie in [0, 1]")
        return hit_value

    def forward(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hit_value = self._validate_inputs(prediction, observed, hit)
        difference = (prediction - observed).square()
        features = torch.cat(
            (prediction, observed, difference, hit_value.unsqueeze(-1)), dim=-1
        )
        return self.network(features).squeeze(-1)

    def loss(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        outcome: torch.Tensor,
        hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if outcome.shape not in ((prediction.shape[0],), (prediction.shape[0], 1)):
            raise ValueError("transition-evidence outcomes must match the batch")
        if not bool(torch.isfinite(outcome).all()) or bool(
            torch.any(outcome < 0) or torch.any(outcome > 1)
        ):
            raise ValueError("transition-evidence outcomes must lie in [0, 1]")
        targets = outcome.reshape(-1).to(
            device=prediction.device, dtype=prediction.dtype
        )
        return nn.functional.binary_cross_entropy_with_logits(
            self(prediction, observed, hit), targets
        )

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        for name, value in sorted(self.state_dict().items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        return digest.hexdigest()


class ExternalTransitionEvidenceCalibrator(nn.Module):
    """Trainable scalar calibration state around a frozen evidence evaluator."""

    schema = EXTERNAL_TRANSITION_EVIDENCE_CALIBRATOR_SCHEMA

    def __init__(
        self,
        evaluator: ExternalTransitionEvidenceEvaluator,
        *,
        prior_strength: float = 0.01,
    ) -> None:
        super().__init__()
        if not isinstance(evaluator, ExternalTransitionEvidenceEvaluator):
            raise TypeError("calibrator requires ExternalTransitionEvidenceEvaluator")
        if prior_strength < 0.0:
            raise ValueError("calibration prior strength cannot be negative")
        self.evaluator = evaluator
        for parameter in self.evaluator.parameters():
            parameter.requires_grad_(False)
        self.state_width = evaluator.state_width
        self.prior_strength = float(prior_strength)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("reference_log_temperature", torch.zeros(()))
        self.register_buffer("reference_bias", torch.zeros(()))

    def configuration(self) -> dict[str, int | float | str | dict[str, object]]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "prior_strength": self.prior_strength,
            "trainable_state": "scalar_temperature_and_bias_v1",
            "frozen_base": self.evaluator.configuration(),
            "replay": "caller_owned_online_scalar_outcomes_v1",
        }

    def forward(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_logits = self.evaluator(prediction, observed, hit).detach()
        temperature = self.log_temperature.exp().clamp_min(1e-4)
        return base_logits / temperature + self.bias

    def loss(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        outcome: torch.Tensor,
        hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if outcome.shape not in ((prediction.shape[0],), (prediction.shape[0], 1)):
            raise ValueError("calibration outcomes must match the batch")
        if not bool(torch.isfinite(outcome).all()) or bool(
            torch.any(outcome < 0) or torch.any(outcome > 1)
        ):
            raise ValueError("calibration outcomes must lie in [0, 1]")
        targets = outcome.reshape(-1).to(
            device=prediction.device, dtype=prediction.dtype
        )
        prior = self.prior_strength * (
            (self.log_temperature - self.reference_log_temperature).square()
            + (self.bias - self.reference_bias).square()
        )
        return nn.functional.binary_cross_entropy_with_logits(
            self(prediction, observed, hit), targets
        ) + prior

    def calibration_step(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        outcome: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        hit: torch.Tensor | None = None,
    ) -> float:
        """Apply one caller-owned live scalar-outcome update."""

        optimizer.zero_grad()
        loss = self.loss(prediction, observed, outcome, hit)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        for name, value in sorted(self.state_dict().items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        return digest.hexdigest()


class ExternalContextualEvidenceCalibrator(nn.Module):
    """Append-only per-context calibration states around one frozen evaluator."""

    schema = EXTERNAL_CONTEXTUAL_EVIDENCE_CALIBRATOR_SCHEMA

    def __init__(
        self,
        evaluator: ExternalTransitionEvidenceEvaluator,
        context_width: int,
        *,
        matching_tolerance: float = 1e-4,
        prior_strength: float = 0.01,
    ) -> None:
        super().__init__()
        if context_width < 1:
            raise ValueError("context width must be positive")
        if matching_tolerance < 0.0:
            raise ValueError("context matching tolerance cannot be negative")
        if not isinstance(evaluator, ExternalTransitionEvidenceEvaluator):
            raise TypeError("contextual calibrator requires evidence evaluator")
        self.evaluator = evaluator
        for parameter in self.evaluator.parameters():
            parameter.requires_grad_(False)
        self.state_width = evaluator.state_width
        self.context_width = int(context_width)
        self.matching_tolerance = float(matching_tolerance)
        self.prior_strength = float(prior_strength)
        self.calibrators = nn.ModuleList()
        self._contexts: list[torch.Tensor] = []

    @property
    def context_count(self) -> int:
        return len(self._contexts)

    def _validate_context(self, context: torch.Tensor) -> torch.Tensor:
        _validate_tensor(
            context, name="calibration context", ndim=2, width=self.context_width
        )
        norms = torch.linalg.vector_norm(context, dim=-1)
        if bool(torch.any(norms <= 1e-12)):
            raise ValueError("calibration contexts must be non-zero")
        return torch.nn.functional.normalize(context.detach().to("cpu"), dim=-1)

    def ensure_context(self, context: torch.Tensor) -> int:
        """Return an existing calibration slot or append a new one."""

        normalized = self._validate_context(
            context if context.ndim == 2 else context.unsqueeze(0)
        )[0]
        if self._contexts:
            distances = torch.linalg.vector_norm(
                torch.stack(self._contexts) - normalized, dim=-1
            )
            nearest = int(distances.argmin())
            if float(distances[nearest]) <= self.matching_tolerance:
                return nearest
        self._contexts.append(normalized.clone())
        self.calibrators.append(
            ExternalTransitionEvidenceCalibrator(
                self.evaluator,
                prior_strength=self.prior_strength,
            )
        )
        return len(self._contexts) - 1

    def _context_indices(self, context: torch.Tensor) -> list[int]:
        normalized = self._validate_context(context)
        return [self.ensure_context(row) for row in normalized]

    def configuration(self) -> dict[str, int | float | str | dict[str, object]]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "context_width": self.context_width,
            "matching_tolerance": self.matching_tolerance,
            "prior_strength": self.prior_strength,
            "context_count": self.context_count,
            "growth": "append_only_context_calibration_v1",
            "frozen_base": self.evaluator.configuration(),
        }

    def score(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        hit: torch.Tensor | None,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape[0] != context.shape[0]:
            raise ValueError("calibration context batch differs")
        indices = self._context_indices(context)
        values = [
            self.calibrators[index](
                prediction[row : row + 1],
                observed[row : row + 1],
                None if hit is None else hit[row : row + 1],
            ).squeeze(0)
            for row, index in enumerate(indices)
        ]
        return torch.stack(values)

    def forward(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        hit: torch.Tensor | None,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.score(prediction, observed, hit, context)

    def loss(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        outcome: torch.Tensor,
        hit: torch.Tensor | None,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if outcome.shape not in ((prediction.shape[0],), (prediction.shape[0], 1)):
            raise ValueError("contextual calibration outcomes must match the batch")
        targets = outcome.reshape(-1).to(prediction)
        indices = self._context_indices(context)
        losses = [
            self.calibrators[index].loss(
                prediction[row : row + 1],
                observed[row : row + 1],
                targets[row : row + 1],
                None if hit is None else hit[row : row + 1],
            )
            for row, index in enumerate(indices)
        ]
        return torch.stack(losses).mean()

    def calibration_step(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        outcome: torch.Tensor,
        context: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        hit: torch.Tensor | None = None,
    ) -> float:
        """Apply one caller-owned live update to only the addressed slots."""

        optimizer.zero_grad()
        loss = self.loss(prediction, observed, outcome, hit, context)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        for name, value in sorted(self.state_dict().items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        for context in self._contexts:
            digest.update(context.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "matching_tolerance": self.matching_tolerance,
            "prior_strength": self.prior_strength,
            "contexts": [context.tolist() for context in self._contexts],
            "calibrators": [
                {
                    "log_temperature": float(calibrator.log_temperature.detach()),
                    "bias": float(calibrator.bias.detach()),
                }
                for calibrator in self.calibrators
            ],
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        evaluator: ExternalTransitionEvidenceEvaluator,
    ) -> ExternalContextualEvidenceCalibrator:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported contextual-calibrator payload")
        contexts = payload.get("contexts")
        calibrators = payload.get("calibrators")
        if not isinstance(contexts, list) or not isinstance(calibrators, list):
            raise TypeError("contextual-calibrator payload lists are invalid")
        if len(contexts) != len(calibrators):
            raise ValueError("contextual-calibrator payload lengths differ")
        restored = cls(
            evaluator,
            int(payload["context_width"]),
            matching_tolerance=float(payload["matching_tolerance"]),
            prior_strength=float(payload["prior_strength"]),
        )
        for values, state in zip(contexts, calibrators, strict=True):
            context = torch.tensor(values, dtype=torch.float32)
            index = restored.ensure_context(context)
            if not isinstance(state, Mapping):
                raise TypeError("contextual-calibrator scalar state is invalid")
            restored.calibrators[index].log_temperature.data.fill_(
                float(state["log_temperature"])
            )
            restored.calibrators[index].bias.data.fill_(float(state["bias"]))
        return restored


class ExternalTransitionMemory(nn.Module):
    """Append-only factual transition memory keyed by learned opaque context."""

    schema = EXTERNAL_TRANSITION_MEMORY_SCHEMA

    def __init__(
        self,
        state_width: int,
        intention_width: int,
        *,
        context_width: int = 0,
        read_match_threshold: float = 0.999,
        write_match_threshold: float = 0.999,
    ) -> None:
        super().__init__()
        if min(state_width, intention_width) < 1 or context_width < 0:
            raise ValueError("transition-memory dimensions are invalid")
        self.state_width = int(state_width)
        self.intention_width = int(intention_width)
        self.context_width = int(context_width)
        self.key_width = self.state_width + self.intention_width + self.context_width
        self.store = AppendOnlyContentAddressedMemory(
            self.key_width,
            read_match_threshold=read_match_threshold,
            write_match_threshold=write_match_threshold,
        )

    @property
    def record_count(self) -> int:
        return self.store.record_count

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "context_width": self.context_width,
            "storage": "append_only_content_addressed_transition_facts_v1",
            "behavior": "derived_by_external_search_not_stored_policy_v1",
            "store": self.store.configuration(),
        }

    def _validate_context(
        self, context: torch.Tensor | None, batch: int, device: torch.device
    ) -> torch.Tensor:
        if self.context_width == 0:
            if context is not None:
                raise ValueError("context is not configured for this transition memory")
            return torch.empty(batch, 0, device=device)
        if context is None:
            raise ValueError("context is required for this transition memory")
        _validate_tensor(context, name="context", ndim=2, width=self.context_width)
        if context.shape[0] != batch:
            raise ValueError("transition context batch differs")
        return context.to(device=device)

    def _key(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        _validate_tensor(state, name="state", ndim=2, width=self.state_width)
        _validate_tensor(
            intention, name="intention", ndim=2, width=self.intention_width
        )
        if state.shape[0] != intention.shape[0]:
            raise ValueError("state and intention batches differ")
        context_value = self._validate_context(context, state.shape[0], state.device)
        return torch.cat((context_value.to(state), state, intention), dim=-1)

    def _stored_value(self, next_state: torch.Tensor) -> torch.Tensor:
        padding = next_state.new_zeros(
            next_state.shape[0], self.key_width - self.state_width
        )
        return torch.cat((next_state, padding), dim=-1)

    @torch.no_grad()
    def write(
        self,
        observation: ExternalTransitionObservation,
        *,
        context: torch.Tensor | None = None,
    ) -> MemoryWriteReceipt:
        observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )
        key = self._key(observation.state, observation.intention, context)
        strength = (
            torch.ones(observation.state.shape[0], device=key.device)
            if observation.confidence is None
            else observation.confidence.reshape(-1).to(key)
        )
        return self.store.write(key, self._stored_value(observation.next_state), strength)

    @torch.no_grad()
    def predict_with_hit(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._key(state, intention, context)
        read = self.store.read(MemoryQuery(key=key))
        return read.value[:, : self.state_width], read.hit

    def forward(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_with_hit(state, intention, context=context)[0]

    def predict_with_context(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(state, intention, context)

    @torch.no_grad()
    def clear(self) -> None:
        self.store.clear()

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        for name, value in sorted(self.store.state_dict().items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class ExternalContextResolution:
    """Memory-side decision to reuse or allocate one opaque context address."""

    context: torch.Tensor
    reused: bool
    matched_observations: int
    mean_error: float
    schema: str = EXTERNAL_CONTEXT_ADDRESS_RESOLVER_SCHEMA

    def validate(self, *, context_width: int) -> ExternalContextResolution:
        if self.schema != EXTERNAL_CONTEXT_ADDRESS_RESOLVER_SCHEMA:
            raise ValueError("unsupported context-resolution schema")
        _validate_tensor(self.context, name="context", ndim=1, width=context_width)
        if not isinstance(self.reused, bool):
            raise TypeError("context-resolution reused flag must be boolean")
        if not isinstance(self.matched_observations, int) or self.matched_observations < 0:
            raise ValueError("context-resolution match count is invalid")
        if not isinstance(self.mean_error, (float, int)) or not math.isfinite(
            float(self.mean_error)
        ) or self.mean_error < 0.0:
            raise ValueError("context-resolution mean error is invalid")
        return self


class ExternalContextAddressResolver:
    """Infer opaque regime addresses from verified transition bundles.

    A candidate address is reusable only when its existing factual rows
    explain every observation in the bundle within ``match_tolerance``.  An
    unexplained bundle receives a fresh opaque handle.  The resolver has no
    task labels, modality fields, or controller path; it is a replaceable
    memory-side admission policy.
    """

    schema = EXTERNAL_CONTEXT_ADDRESS_RESOLVER_SCHEMA

    def __init__(
        self,
        context_width: int,
        *,
        match_tolerance: float = 1e-6,
        address_seed: int = 0,
    ) -> None:
        if context_width < 1:
            raise ValueError("context width must be positive")
        if match_tolerance < 0.0:
            raise ValueError("context match tolerance must be non-negative")
        if not isinstance(address_seed, int) or address_seed < 0:
            raise ValueError("context address seed must be a non-negative integer")
        self.context_width = int(context_width)
        self.match_tolerance = float(match_tolerance)
        self.address_seed = address_seed
        self._allocation_count = 0
        self._addresses: list[torch.Tensor] = []

    @property
    def context_count(self) -> int:
        return len(self._addresses)

    def addresses(self) -> torch.Tensor:
        if not self._addresses:
            return torch.empty(0, self.context_width)
        return torch.stack(self._addresses)

    def _new_address(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(
            self.address_seed + self._allocation_count
        )
        for _attempt in range(100):
            candidate = torch.nn.functional.normalize(
                torch.randn(self.context_width, generator=generator), dim=0
            )
            if not self._addresses:
                break
            distances = torch.linalg.vector_norm(
                torch.stack(self._addresses) - candidate, dim=-1
            )
            if bool(torch.all(distances > 1e-4)):
                break
        self._allocation_count += 1
        self._addresses.append(candidate.detach().clone())
        return candidate

    def resolve(
        self,
        observation: ExternalTransitionObservation,
        memory: ExternalTransitionMemory,
    ) -> ExternalContextResolution:
        """Reuse a factually consistent address or append a new one."""

        if not isinstance(memory, ExternalTransitionMemory):
            raise TypeError("context resolver requires ExternalTransitionMemory")
        observation.validate(
            state_width=memory.state_width,
            intention_width=memory.intention_width,
        )
        if memory.context_width != self.context_width:
            raise ValueError("resolver and transition memory context widths differ")
        if observation.state.shape[0] < 1:
            raise ValueError("context resolution requires at least one observation")

        best: tuple[float, int, torch.Tensor] | None = None
        for index, address in enumerate(self._addresses):
            context = address.to(observation.state)
            prediction, hits = memory.predict_with_hit(
                observation.state,
                observation.intention,
                context=context.unsqueeze(0).expand(observation.state.shape[0], -1),
            )
            matched = int(hits.sum())
            if matched != observation.state.shape[0]:
                continue
            error = float(
                (prediction - observation.next_state).square().mean().detach()
            )
            candidate = (error, index, address)
            if error <= self.match_tolerance and (
                best is None or candidate[:2] < best[:2]
            ):
                best = candidate
        if best is not None:
            resolution = ExternalContextResolution(
                context=best[2].clone(),
                reused=True,
                matched_observations=observation.state.shape[0],
                mean_error=best[0],
            )
            return resolution.validate(context_width=self.context_width)

        context = self._new_address()
        resolution = ExternalContextResolution(
            context=context,
            reused=False,
            matched_observations=0,
            mean_error=0.0,
        )
        return resolution.validate(context_width=self.context_width)

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "match_tolerance": self.match_tolerance,
            "address_seed": self.address_seed,
            "allocation_count": self._allocation_count,
            "addresses": [address.tolist() for address in self._addresses],
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> ExternalContextAddressResolver:
        if not isinstance(payload, Mapping):
            raise TypeError("context-resolver payload must be a mapping")
        if payload.get("schema") != cls.schema:
            raise ValueError("unsupported context-resolver schema")
        resolver = cls(
            int(payload["context_width"]),
            match_tolerance=float(payload["match_tolerance"]),
            address_seed=int(payload["address_seed"]),
        )
        addresses = payload.get("addresses")
        if not isinstance(addresses, list):
            raise TypeError("context-resolver addresses must be a list")
        for values in addresses:
            if not isinstance(values, list):
                raise TypeError("context-resolver address must be a list")
            address = torch.tensor(values, dtype=torch.float32)
            _validate_tensor(
                address, name="context address", ndim=1, width=resolver.context_width
            )
            address = torch.nn.functional.normalize(address, dim=0)
            resolver._addresses.append(address)
        allocation_count = payload.get("allocation_count", len(addresses))
        if not isinstance(allocation_count, int) or allocation_count < len(addresses):
            raise ValueError("context-resolver allocation count is invalid")
        resolver._allocation_count = allocation_count
        return resolver


@dataclass(frozen=True)
class ExternalOnlineContextResolution:
    """Result of one partial-evidence online address decision."""

    status: str
    context: torch.Tensor | None
    committed_observations: int
    pending_observations: int
    schema: str = EXTERNAL_ONLINE_CONTEXT_RESOLVER_SCHEMA

    def validate(
        self, *, context_width: int
    ) -> ExternalOnlineContextResolution:
        if self.schema != EXTERNAL_ONLINE_CONTEXT_RESOLVER_SCHEMA:
            raise ValueError("unsupported online context-resolution schema")
        if self.status not in {"uncertain", "conflict", "reused", "admitted"}:
            raise ValueError("unsupported online context-resolution status")
        if self.context is not None:
            _validate_tensor(
                self.context, name="online context", ndim=1, width=context_width
            )
        if (
            not isinstance(self.committed_observations, int)
            or self.committed_observations < 0
            or not isinstance(self.pending_observations, int)
            or self.pending_observations < 0
        ):
            raise ValueError("online context-resolution counts are invalid")
        return self


class ExternalOnlineContextAddressResolver(ExternalContextAddressResolver):
    """Accumulate partial verified evidence before changing memory addresses.

    New streams remain provisional until ``admission_observations`` consistent
    facts have accumulated.  An already-bound stream remains provisional on an
    unknown row and requires ``contradiction_observations`` contradictory hits
    before a new address is admitted.  Ambiguous rows are never written to the
    old address.  ``stream_key`` is an opaque transport binding, not a task or
    semantic label.
    """

    schema = EXTERNAL_ONLINE_CONTEXT_RESOLVER_SCHEMA

    def __init__(
        self,
        context_width: int,
        *,
        match_tolerance: float = 1e-6,
        address_seed: int = 0,
        admission_observations: int = 3,
        contradiction_observations: int = 2,
        evidence_evaluator: nn.Module | None = None,
        evidence_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            context_width,
            match_tolerance=match_tolerance,
            address_seed=address_seed,
        )
        if admission_observations < 1 or contradiction_observations < 1:
            raise ValueError("online admission thresholds must be positive")
        if not 0.0 < evidence_threshold < 1.0:
            raise ValueError("online evidence threshold must lie in (0, 1)")
        if evidence_evaluator is not None and evidence_evaluator.state_width < 1:
            raise ValueError("online evidence evaluator is invalid")
        self.admission_observations = int(admission_observations)
        self.contradiction_observations = int(contradiction_observations)
        self.evidence_evaluator = evidence_evaluator
        self.evidence_threshold = float(evidence_threshold)
        self._assigned: dict[tuple[float, ...], int] = {}
        self._pending: dict[
            tuple[float, ...], list[ExternalTransitionObservation]
        ] = {}
        self._contradiction_streak: dict[tuple[float, ...], int] = {}

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "match_tolerance": self.match_tolerance,
            "address_seed": self.address_seed,
            "admission_observations": self.admission_observations,
            "contradiction_observations": self.contradiction_observations,
            "evidence_threshold": self.evidence_threshold,
            "evidence_evaluator": (
                None
                if self.evidence_evaluator is None
                else self.evidence_evaluator.configuration()
            ),
            "behavior": "uncertain_rows_are_not_written_v1",
        }

    def _consistent(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        hits: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[bool, float]:
        error = float((prediction - observed).square().mean().detach())
        if self.evidence_evaluator is None:
            return error <= self.match_tolerance, error
        with torch.no_grad():
            scorer = getattr(self.evidence_evaluator, "score", None)
            if callable(scorer):
                logits = scorer(prediction, observed, hits, context)
            else:
                logits = self.evidence_evaluator(prediction, observed, hits)
            probability = float(
                torch.sigmoid(logits).mean()
            )
        return probability >= self.evidence_threshold, error

    def _stream_id(self, stream_key: torch.Tensor) -> tuple[float, ...]:
        _validate_tensor(
            stream_key, name="stream_key", ndim=1, width=self.context_width
        )
        if float(torch.linalg.vector_norm(stream_key)) <= 1e-12:
            raise ValueError("stream_key must be non-zero")
        normalized = torch.nn.functional.normalize(
            stream_key.detach().to(device="cpu", dtype=torch.float32), dim=0
        )
        return tuple(float(value) for value in normalized.tolist())

    @staticmethod
    def _clone_observation(
        observation: ExternalTransitionObservation,
    ) -> ExternalTransitionObservation:
        return ExternalTransitionObservation(
            state=observation.state.detach().clone(),
            intention=observation.intention.detach().clone(),
            next_state=observation.next_state.detach().clone(),
            confidence=(
                None
                if observation.confidence is None
                else observation.confidence.detach().clone()
            ),
        )

    @staticmethod
    def _merge(
        observations: list[ExternalTransitionObservation],
    ) -> ExternalTransitionObservation:
        if not observations:
            raise ValueError("cannot merge empty online observation list")
        confidence = [
            torch.ones(item.state.shape[0], device=item.state.device)
            if item.confidence is None
            else item.confidence.reshape(-1)
            for item in observations
        ]
        return ExternalTransitionObservation(
            state=torch.cat([item.state for item in observations]),
            intention=torch.cat([item.intention for item in observations]),
            next_state=torch.cat([item.next_state for item in observations]),
            confidence=torch.cat(confidence),
        )

    def _candidate(
        self,
        observation: ExternalTransitionObservation,
        memory: ExternalTransitionMemory,
    ) -> tuple[int, float] | None:
        best: tuple[int, float] | None = None
        for index, address in enumerate(self._addresses):
            prediction, hits = memory.predict_with_hit(
                observation.state,
                observation.intention,
                context=address.to(observation.state)
                .unsqueeze(0)
                .expand(observation.state.shape[0], -1),
            )
            if not bool(hits.all()):
                continue
            error = float(
                (prediction - observation.next_state).square().mean().detach()
            )
            consistent, _error = self._consistent(
                prediction,
                observation.next_state,
                hits,
                address.to(observation.state).unsqueeze(0).expand(
                    observation.state.shape[0], -1
                ),
            )
            if consistent and (
                best is None or error < best[1]
            ):
                best = (index, error)
        return best

    def _commit(
        self,
        stream_id: tuple[float, ...],
        memory: ExternalTransitionMemory,
        context_index: int,
        observations: list[ExternalTransitionObservation],
    ) -> int:
        if not observations:
            return 0
        memory.write(
            self._merge(observations),
            context=self._addresses[context_index]
            .to(observations[0].state)
            .unsqueeze(0)
            .expand(sum(item.state.shape[0] for item in observations), -1),
        )
        self._assigned[stream_id] = context_index
        return sum(item.state.shape[0] for item in observations)

    def observe(
        self,
        observation: ExternalTransitionObservation,
        stream_key: torch.Tensor,
        memory: ExternalTransitionMemory,
    ) -> ExternalOnlineContextResolution:
        """Consume one verified row without writing while its address is ambiguous."""

        if not isinstance(memory, ExternalTransitionMemory):
            raise TypeError("online resolver requires ExternalTransitionMemory")
        observation.validate(
            state_width=memory.state_width,
            intention_width=memory.intention_width,
        )
        if observation.state.shape[0] != 1:
            raise ValueError("online resolver expects one observation per call")
        if memory.context_width != self.context_width:
            raise ValueError("resolver and transition memory context widths differ")
        if (
            self.evidence_evaluator is not None
            and self.evidence_evaluator.state_width != memory.state_width
        ):
            raise ValueError("evidence evaluator and transition memory widths differ")
        stream_id = self._stream_id(stream_key)
        current = self._clone_observation(observation)
        pending = self._pending.setdefault(stream_id, [])

        if stream_id not in self._assigned:
            candidate = self._candidate(current, memory)
            if candidate is not None and not pending:
                self._assigned[stream_id] = candidate[0]
                return ExternalOnlineContextResolution(
                    status="reused",
                    context=self._addresses[candidate[0]].clone(),
                    committed_observations=0,
                    pending_observations=0,
                ).validate(context_width=self.context_width)
            pending.append(current)
            if len(pending) < self.admission_observations:
                return ExternalOnlineContextResolution(
                    status="uncertain",
                    context=None,
                    committed_observations=0,
                    pending_observations=len(pending),
                ).validate(context_width=self.context_width)
            context = self._new_address()
            context_index = len(self._addresses) - 1
            committed = self._commit(stream_id, memory, context_index, pending)
            pending.clear()
            return ExternalOnlineContextResolution(
                status="admitted",
                context=context.clone(),
                committed_observations=committed,
                pending_observations=0,
            ).validate(context_width=self.context_width)

        context_index = self._assigned[stream_id]
        context = self._addresses[context_index]
        prediction, hits = memory.predict_with_hit(
            current.state,
            current.intention,
            context=context.to(current.state).unsqueeze(0),
        )
        if bool(hits.all()):
            consistent, _error = self._consistent(
                prediction,
                current.next_state,
                hits,
                context.to(current.state).unsqueeze(0),
            )
            if consistent:
                pending.clear()
                self._contradiction_streak.pop(stream_id, None)
                return ExternalOnlineContextResolution(
                    status="reused",
                    context=context.clone(),
                    committed_observations=0,
                    pending_observations=0,
                ).validate(context_width=self.context_width)
            self._contradiction_streak[stream_id] = (
                self._contradiction_streak.get(stream_id, 0) + 1
            )
            pending.append(current)
            if (
                self._contradiction_streak[stream_id]
                < self.contradiction_observations
            ):
                return ExternalOnlineContextResolution(
                    status="conflict",
                    context=None,
                    committed_observations=0,
                    pending_observations=len(pending),
                ).validate(context_width=self.context_width)
            new_context = self._new_address()
            new_index = len(self._addresses) - 1
            committed = self._commit(stream_id, memory, new_index, pending)
            pending.clear()
            self._contradiction_streak[stream_id] = 0
            return ExternalOnlineContextResolution(
                status="admitted",
                context=new_context.clone(),
                committed_observations=committed,
                pending_observations=0,
            ).validate(context_width=self.context_width)

        pending.append(current)
        return ExternalOnlineContextResolution(
            status="uncertain",
            context=None,
            committed_observations=0,
            pending_observations=len(pending),
        ).validate(context_width=self.context_width)

    def pending_observations(self, stream_key: torch.Tensor) -> int:
        return len(self._pending.get(self._stream_id(stream_key), []))

    def payload(self) -> dict[str, object]:
        base = super().payload()
        base["schema"] = self.schema
        base["admission_observations"] = self.admission_observations
        base["contradiction_observations"] = self.contradiction_observations
        base["assigned"] = [
            {"stream_key": list(stream), "address_index": index}
            for stream, index in self._assigned.items()
        ]
        base["contradiction_streak"] = [
            {"stream_key": list(stream), "count": count}
            for stream, count in self._contradiction_streak.items()
        ]
        base["pending"] = [
            {
                "stream_key": list(stream),
                "observations": [
                    {
                        "state": item.state.tolist(),
                        "intention": item.intention.tolist(),
                        "next_state": item.next_state.tolist(),
                        "confidence": (
                            None
                            if item.confidence is None
                            else item.confidence.tolist()
                        ),
                    }
                    for item in observations
                ],
            }
            for stream, observations in self._pending.items()
            if observations
        ]
        return base

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        evidence_evaluator: nn.Module | None = None,
    ) -> ExternalOnlineContextAddressResolver:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported online context-resolver payload")
        resolver = cls(
            int(payload["context_width"]),
            match_tolerance=float(payload["match_tolerance"]),
            address_seed=int(payload["address_seed"]),
            admission_observations=int(payload["admission_observations"]),
            contradiction_observations=int(payload["contradiction_observations"]),
            evidence_threshold=float(payload.get("evidence_threshold", 0.5)),
            evidence_evaluator=evidence_evaluator,
        )
        addresses = payload.get("addresses")
        if not isinstance(addresses, list):
            raise TypeError("online context-resolver addresses must be a list")
        for values in addresses:
            address = torch.tensor(values, dtype=torch.float32)
            _validate_tensor(
                address, name="online context address", ndim=1, width=resolver.context_width
            )
            resolver._addresses.append(torch.nn.functional.normalize(address, dim=0))
        allocation_count = payload.get("allocation_count", len(addresses))
        if not isinstance(allocation_count, int) or allocation_count < len(addresses):
            raise ValueError("online context-resolver allocation count is invalid")
        resolver._allocation_count = allocation_count
        assigned = payload.get("assigned", [])
        if not isinstance(assigned, list):
            raise TypeError("online context-resolver assignments must be a list")
        for item in assigned:
            stream = resolver._stream_id(torch.tensor(item["stream_key"]))
            index = int(item["address_index"])
            if not 0 <= index < resolver.context_count:
                raise ValueError("online context-resolver address index is invalid")
            resolver._assigned[stream] = index
        streaks = payload.get("contradiction_streak", [])
        if not isinstance(streaks, list):
            raise TypeError("online context-resolver streaks must be a list")
        for item in streaks:
            resolver._contradiction_streak[resolver._stream_id(torch.tensor(item["stream_key"]))] = int(item["count"])
        pending = payload.get("pending", [])
        if not isinstance(pending, list):
            raise TypeError("online context-resolver pending state must be a list")
        for item in pending:
            stream = resolver._stream_id(torch.tensor(item["stream_key"]))
            rows: list[ExternalTransitionObservation] = []
            for row in item["observations"]:
                confidence = row.get("confidence")
                rows.append(
                    ExternalTransitionObservation(
                        state=torch.tensor(row["state"], dtype=torch.float32),
                        intention=torch.tensor(row["intention"], dtype=torch.float32),
                        next_state=torch.tensor(row["next_state"], dtype=torch.float32),
                        confidence=(
                            None
                            if confidence is None
                            else torch.tensor(confidence, dtype=torch.float32)
                        ),
                    )
                )
            resolver._pending[stream] = rows
        return resolver


@dataclass(frozen=True)
class ModelBasedPlanningResult:
    """Opaque search output returned to an intention decoder."""

    intentions: torch.Tensor
    predicted_states: torch.Tensor
    scores: torch.Tensor
    expanded_nodes: int
    schema: str = EXTERNAL_MODEL_PLANNER_SCHEMA

    def validate(
        self,
        *,
        batch: int,
        horizon: int,
        state_width: int,
        intention_width: int,
    ) -> ModelBasedPlanningResult:
        if self.schema != EXTERNAL_MODEL_PLANNER_SCHEMA:
            raise ValueError("unsupported planner-result schema")
        if self.intentions.shape != (batch, horizon, intention_width):
            raise ValueError("planner intentions have the wrong shape")
        if self.predicted_states.shape != (batch, horizon, state_width):
            raise ValueError("planner states have the wrong shape")
        if self.scores.shape != (batch,):
            raise ValueError("planner scores have the wrong shape")
        for name, value in (
            ("intentions", self.intentions),
            ("predicted_states", self.predicted_states),
            ("scores", self.scores),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"planner {name} must be finite")
        if self.expanded_nodes < 1:
            raise ValueError("planner must expand at least one node")
        return self


@dataclass(frozen=True)
class GoalConditionedModelSelection:
    """Goal-reachability ranking over stable external model addresses."""

    selected_slot_id: int
    candidate_slot_ids: tuple[int, ...]
    scores: torch.Tensor
    planning: ModelBasedPlanningResult
    schema: str = EXTERNAL_GOAL_CONDITIONED_MODEL_SELECTION_SCHEMA

    def validate(
        self,
        *,
        state_width: int,
        intention_width: int,
    ) -> GoalConditionedModelSelection:
        if self.schema != EXTERNAL_GOAL_CONDITIONED_MODEL_SELECTION_SCHEMA:
            raise ValueError("unsupported goal-conditioned model selection schema")
        if self.scores.ndim != 1 or self.scores.shape[0] != len(
            self.candidate_slot_ids
        ):
            raise ValueError("goal-conditioned model scores are misaligned")
        if not bool(torch.isfinite(self.scores).all()):
            raise ValueError("goal-conditioned model scores are not finite")
        if len(set(self.candidate_slot_ids)) != len(self.candidate_slot_ids):
            raise ValueError("goal-conditioned model slot IDs are duplicated")
        if any(slot_id < 0 for slot_id in self.candidate_slot_ids):
            raise ValueError("goal-conditioned model slot ID is invalid")
        if self.selected_slot_id not in self.candidate_slot_ids:
            raise ValueError("goal-conditioned selected model is not a candidate")
        self.planning.validate(
            batch=1,
            horizon=self.planning.intentions.shape[1],
            state_width=state_width,
            intention_width=intention_width,
        )
        return self


class ExternalModelBasedPlanner:
    """Beam search over opaque candidate intentions and a frozen model."""

    schema = EXTERNAL_MODEL_PLANNER_SCHEMA

    def __init__(
        self,
        model: nn.Module,
        *,
        beam_width: int = 4,
        goal_evaluator: ExternalGoalEvaluator | None = None,
    ) -> None:
        if not isinstance(model, nn.Module) or not all(
            isinstance(getattr(model, name, None), int)
            for name in ("state_width", "intention_width")
        ):
            raise TypeError("planner requires a compatible external transition model")
        if beam_width < 1:
            raise ValueError("planner beam_width must be positive")
        if goal_evaluator is not None and goal_evaluator.state_width != model.state_width:
            raise ValueError("goal evaluator width must match transition model state width")
        self.model = model
        self.beam_width = int(beam_width)
        self.goal_evaluator = goal_evaluator

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "beam_width": self.beam_width,
            "search": "opaque_candidate_beam_rollout_v1",
            "objective": (
                "learned_verifier_terminal_match_v1"
                if self.goal_evaluator is not None
                else "terminal_opaque_goal_state_match_v1"
            ),
            "policy": "none_behavior_derived_at_inference_v1",
        }

    def _predict(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        if context is None:
            return self.model(state, intention)
        predictor = getattr(self.model, "predict_with_context", None)
        if not callable(predictor):
            raise TypeError("transition model does not expose contextual prediction")
        return predictor(state, intention, context)

    def plan(
        self,
        state: torch.Tensor,
        goal_state: torch.Tensor,
        candidate_intentions: torch.Tensor,
        *,
        horizon: int,
        beam_width: int | None = None,
        transition_context: torch.Tensor | None = None,
    ) -> ModelBasedPlanningResult:
        """Return the lowest predicted-distance candidate sequence.

        Candidate intentions may be shared as ``[candidates, width]`` or
        supplied per batch as ``[batch, candidates, width]``.  Candidate
        count is therefore runtime-variable and never changes model shapes.
        The planner is inference-only and does not mutate the model.
        """

        _validate_tensor(
            state,
            name="state",
            ndim=2,
            width=self.model.state_width,
        )
        _validate_tensor(
            goal_state,
            name="goal_state",
            ndim=2,
            width=self.model.state_width,
        )
        if goal_state.shape[0] != state.shape[0]:
            raise ValueError("state and goal_state batches differ")
        if transition_context is not None:
            context_width = getattr(self.model, "context_width", None)
            if not isinstance(context_width, int) or context_width < 1:
                raise ValueError("transition model does not accept a context")
            _validate_tensor(
                transition_context,
                name="transition_context",
                ndim=2,
                width=context_width,
            )
            if transition_context.shape[0] != state.shape[0]:
                raise ValueError("transition context batch differs")
        _validate_tensor(
            candidate_intentions,
            name="candidate_intentions",
            ndim=2 if candidate_intentions.ndim == 2 else 3,
            width=self.model.intention_width,
        )
        if candidate_intentions.ndim == 2:
            candidates = candidate_intentions.unsqueeze(0).expand(
                state.shape[0], -1, -1
            )
        elif candidate_intentions.shape[0] == state.shape[0]:
            candidates = candidate_intentions
        else:
            raise ValueError("per-batch candidate intentions have the wrong batch")
        if candidates.shape[1] < 1:
            raise ValueError("planner requires at least one candidate intention")
        if horizon < 1:
            raise ValueError("planner horizon must be positive")
        width = self.beam_width if beam_width is None else int(beam_width)
        if width < 1:
            raise ValueError("planner beam_width must be positive")

        batch = state.shape[0]
        chosen_intentions: list[torch.Tensor] = []
        chosen_states: list[torch.Tensor] = []
        chosen_scores: list[torch.Tensor] = []
        expanded_nodes = 0
        with torch.no_grad():
            for row in range(batch):
                # Each item is (cumulative score, state, intention sequence,
                # predicted-state sequence).  Lower is better.
                beams: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]] = [
                    (torch.zeros((), device=state.device, dtype=state.dtype), state[row], [], [])
                ]
                row_candidates = candidates[row]
                for _step in range(horizon):
                    parent_states = torch.stack([item[1] for item in beams])
                    parent_count = parent_states.shape[0]
                    candidate_count = row_candidates.shape[0]
                    expanded_nodes += parent_count * candidate_count
                    next_states = self._predict(
                        parent_states.repeat_interleave(candidate_count, dim=0),
                        row_candidates.repeat(parent_count, 1),
                        None
                        if transition_context is None
                        else transition_context[row]
                        .unsqueeze(0)
                        .expand(parent_count * candidate_count, -1),
                    ).reshape(parent_count, candidate_count, -1)
                    if self.goal_evaluator is None:
                        terminal_scores = (
                            next_states - goal_state[row].unsqueeze(0).unsqueeze(0)
                        ).square().mean(dim=-1)
                    else:
                        terminal_scores = -torch.sigmoid(
                            self.goal_evaluator(
                                next_states.reshape(-1, self.model.state_width),
                                goal_state[row]
                                .unsqueeze(0)
                                .expand(parent_count * candidate_count, -1),
                            )
                        ).reshape(parent_count, candidate_count)
                    expanded: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]] = []
                    for parent_index, parent in enumerate(beams):
                        for candidate_index in range(candidate_count):
                            score = (
                                terminal_scores[parent_index, candidate_index]
                                if _step == horizon - 1
                                else torch.zeros_like(
                                    terminal_scores[parent_index, candidate_index]
                                )
                            )
                            expanded.append(
                                (
                                    score,
                                    next_states[parent_index, candidate_index],
                                    [*parent[2], row_candidates[candidate_index]],
                                    [*parent[3], next_states[parent_index, candidate_index]],
                                )
                            )
                    expanded.sort(key=lambda item: float(item[0]))
                    beams = expanded[:width]
                best = beams[0]
                chosen_scores.append(best[0])
                chosen_intentions.append(torch.stack(best[2]))
                chosen_states.append(torch.stack(best[3]))

        result = ModelBasedPlanningResult(
            intentions=torch.stack(chosen_intentions),
            predicted_states=torch.stack(chosen_states),
            scores=torch.stack(chosen_scores),
            expanded_nodes=expanded_nodes,
        )
        return result.validate(
            batch=batch,
            horizon=horizon,
            state_width=self.model.state_width,
            intention_width=self.model.intention_width,
        )

    @torch.no_grad()
    def select_bank_model(
        self,
        bank: ExternalTransitionModelBank,
        state: torch.Tensor,
        goal_state: torch.Tensor,
        candidate_intentions: torch.Tensor,
        *,
        horizon: int,
        beam_width: int | None = None,
    ) -> GoalConditionedModelSelection:
        """Select the model whose factual rollout best reaches the goal.

        This is inference-time search over facts, not a stored task policy.
        Candidate model count remains runtime-variable; the controller and
        each model interface remain unchanged. Stable logical addresses are
        returned so physical bank reorganization cannot stale the selection.
        """

        if not isinstance(bank, ExternalTransitionModelBank):
            raise TypeError("goal-conditioned selection requires an external model bank")
        if bank.state_width != self.model.state_width or bank.intention_width != self.model.intention_width:
            raise ValueError("model bank dimensions do not match planner")
        if state.shape[0] != 1 or goal_state.shape[0] != 1:
            raise ValueError("goal-conditioned bank selection accepts one query row")
        if bank.context_count < 1:
            raise ValueError("goal-conditioned bank selection needs one model")
        results: list[ModelBasedPlanningResult] = []
        for index in range(bank.context_count):
            context = bank.context_at(index).to(state)
            results.append(
                self.plan(
                    state,
                    goal_state,
                    candidate_intentions,
                    horizon=horizon,
                    beam_width=beam_width,
                    transition_context=context.unsqueeze(0),
                )
            )
        scores = torch.stack([result.scores[0] for result in results])
        selected_index = int(scores.argmin())
        return GoalConditionedModelSelection(
            selected_slot_id=bank.slot_id_at(selected_index),
            candidate_slot_ids=bank.slot_ids,
            scores=scores.detach().clone(),
            planning=results[selected_index],
        ).validate(
            state_width=bank.state_width,
            intention_width=bank.intention_width,
        )


__all__ = [
    "EXTERNAL_CONTEXTUAL_EVIDENCE_CALIBRATOR_SCHEMA",
    "EXTERNAL_CONTEXT_ADDRESS_RESOLVER_SCHEMA",
    "EXTERNAL_GOAL_CONDITIONED_MODEL_SELECTION_SCHEMA",
    "EXTERNAL_GOAL_EVALUATOR_SCHEMA",
    "EXTERNAL_MODEL_PLANNER_SCHEMA",
    "EXTERNAL_ONLINE_CONTEXT_RESOLVER_SCHEMA",
    "EXTERNAL_ONLINE_TRANSITION_CONTEXT_ROUTER_SCHEMA",
    "EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY",
    "EXTERNAL_TRANSITION_CONTEXT_ENCODER_SCHEMA",
    "EXTERNAL_TRANSITION_EVIDENCE_CALIBRATOR_SCHEMA",
    "EXTERNAL_TRANSITION_MEMORY_SCHEMA",
    "EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY",
    "EXTERNAL_TRANSITION_MODEL_BANK_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_CANDIDATE_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_COMPRESSION_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_COMPRESSION_SELECTION_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_CONSOLIDATION_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_FAMILY_SELECTION_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_GROWTH_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_SCHEMA",
    "EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY",
    "EXTERNAL_TRANSITION_OBSERVATION_SCHEMA",
    "EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY",
    "ExternalContextAddressResolver",
    "ExternalContextResolution",
    "ExternalContextualEvidenceCalibrator",
    "ExternalGoalEvaluator",
    "ExternalModelBasedPlanner",
    "ExternalOnlineContextAddressResolver",
    "ExternalOnlineContextResolution",
    "ExternalOnlineTransitionContextResult",
    "ExternalOnlineTransitionContextRouter",
    "ExternalTransitionContextEncoder",
    "ExternalTransitionEvidenceCalibrator",
    "ExternalTransitionEvidenceEvaluator",
    "ExternalTransitionMemory",
    "ExternalTransitionModel",
    "ExternalTransitionModelBank",
    "ExternalTransitionModelCandidateReceipt",
    "ExternalTransitionModelCompressionReceipt",
    "ExternalTransitionModelCompressionSelection",
    "ExternalTransitionModelConsolidationReceipt",
    "ExternalTransitionModelGrowthReceipt",
    "ExternalTransitionObservation",
    "GoalConditionedModelSelection",
    "ModelBasedPlanningResult",
]
