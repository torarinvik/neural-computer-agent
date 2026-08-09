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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .memory import (
    AppendOnlyContentAddressedMemory,
    MemoryQuery,
    MemoryWriteReceipt,
)

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
EXTERNAL_TRANSITION_MODEL_BANK_SCHEMA = (
    "neural-computer.external-transition-model-bank.v1"
)
EXTERNAL_MODEL_PLANNER_SCHEMA = "neural-computer.external-model-planner.v1"


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
    ) -> None:
        super().__init__()
        if min(state_width, intention_width, context_width, hidden_width) < 1:
            raise ValueError("transition-model bank dimensions must be positive")
        if matching_tolerance < 0.0:
            raise ValueError("transition-model context tolerance cannot be negative")
        self.state_width = int(state_width)
        self.intention_width = int(intention_width)
        self.context_width = int(context_width)
        self.hidden_width = int(hidden_width)
        self.matching_tolerance = float(matching_tolerance)
        self.models = nn.ModuleList()
        self._contexts: list[torch.Tensor] = []

    @property
    def context_count(self) -> int:
        return len(self._contexts)

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
        model = ExternalTransitionModel(
            self.state_width,
            self.intention_width,
            hidden_width=self.hidden_width,
        )
        if initialize_from is not None:
            model.load_state_dict(self.models[initialize_from].state_dict())
        self._contexts.append(normalized.clone())
        self.models.append(model)
        return self.context_count - 1

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

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "context_width": self.context_width,
            "hidden_width": self.hidden_width,
            "matching_tolerance": self.matching_tolerance,
            "growth": "append_only_isolated_model_slots_v1",
            "behavior": "derived_by_external_model_search_v1",
            "updates": "caller_selected_slot_only_v1",
        }

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

    def adaptation_step(
        self,
        observation: ExternalTransitionObservation,
        context: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Update only parameters selected by the supplied context batch."""

        optimizer.zero_grad()
        loss = self.loss(observation, context)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        digest.update(repr(self.configuration()).encode("utf-8"))
        for context in self._contexts:
            detached = context.detach().cpu().contiguous()
            digest.update(detached.numpy().tobytes())
        for index, model in enumerate(self.models):
            digest.update(str(index).encode("utf-8"))
            digest.update(model.digest().encode("utf-8"))
        return digest.hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "contexts": [context.tolist() for context in self._contexts],
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
        if not isinstance(configuration, Mapping):
            raise TypeError("transition-model bank configuration is missing")
        if not isinstance(contexts, list) or not isinstance(models, list):
            raise TypeError("transition-model bank payload lists are invalid")
        if len(contexts) != len(models):
            raise ValueError("transition-model bank payload lengths differ")
        bank = cls(
            int(configuration["state_width"]),
            int(configuration["intention_width"]),
            int(configuration["context_width"]),
            hidden_width=int(configuration["hidden_width"]),
            matching_tolerance=float(configuration["matching_tolerance"]),
        )
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
            bank.models.append(
                ExternalTransitionModel(
                    bank.state_width,
                    bank.intention_width,
                    hidden_width=bank.hidden_width,
                )
            )
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
        if payload.get("sha256") != bank.digest():
            raise ValueError("transition-model bank checksum mismatch")
        return bank


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


__all__ = [
    "EXTERNAL_CONTEXTUAL_EVIDENCE_CALIBRATOR_SCHEMA",
    "EXTERNAL_CONTEXT_ADDRESS_RESOLVER_SCHEMA",
    "EXTERNAL_GOAL_EVALUATOR_SCHEMA",
    "EXTERNAL_MODEL_PLANNER_SCHEMA",
    "EXTERNAL_ONLINE_CONTEXT_RESOLVER_SCHEMA",
    "EXTERNAL_TRANSITION_CONTEXT_ENCODER_SCHEMA",
    "EXTERNAL_TRANSITION_EVIDENCE_CALIBRATOR_SCHEMA",
    "EXTERNAL_TRANSITION_MEMORY_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_BANK_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_SCHEMA",
    "EXTERNAL_TRANSITION_OBSERVATION_SCHEMA",
    "ExternalContextAddressResolver",
    "ExternalContextResolution",
    "ExternalContextualEvidenceCalibrator",
    "ExternalGoalEvaluator",
    "ExternalModelBasedPlanner",
    "ExternalOnlineContextAddressResolver",
    "ExternalOnlineContextResolution",
    "ExternalTransitionContextEncoder",
    "ExternalTransitionEvidenceCalibrator",
    "ExternalTransitionEvidenceEvaluator",
    "ExternalTransitionMemory",
    "ExternalTransitionModel",
    "ExternalTransitionModelBank",
    "ExternalTransitionObservation",
    "ModelBasedPlanningResult",
]
