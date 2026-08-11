"""Factored factual transition models with external residual memory."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .world_model import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    ExternalModelBasedPlanner,
    ExternalSparseTransitionEvidenceIndex,
    ExternalTransitionContextEncoder,
    ExternalTransitionMemory,
    ExternalTransitionModel,
    ExternalTransitionModelBank,
    ExternalTransitionModelCompressionSelection,
    ExternalTransitionModelEvictionReceipt,
    ExternalTransitionModelGrowthReceipt,
    ExternalTransitionObservation,
    ExternalTransitionRollout,
)

EXTERNAL_FACTORED_TRANSITION_MODEL_SCHEMA = (
    "neural-computer.external-factored-transition-model.v1"
)
EXTERNAL_FACTORED_TRANSITION_EXACT_RESIDUAL_MODE = "exact_residual_memory_v1"
EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE = "learned_residual_function_v1"
EXTERNAL_FACTORED_TRANSITION_ROUTER_SCHEMA = (
    "neural-computer.external-factored-transition-router.v1"
)
EXTERNAL_FACTORED_TRANSITION_PROMOTION_SCHEMA = (
    "neural-computer.external-factored-transition-promotion.v1"
)
EXTERNAL_FACTORED_TRANSITION_QUARANTINE_SCHEMA = (
    "neural-computer.external-factored-transition-quarantine.v1"
)


class ExternalFactoredTransitionModel(nn.Module):
    """A frozen factual base plus external context-local residual state.

    The base learns reusable transition structure.  Once frozen, new regime
    evidence is written only to external residual state addressed by an
    opaque context.  The compatibility mode stores exact residual facts;
    learned mode stores a trainable residual function. Planning sees the sum
    of the base prediction and the context-local residual; it never sees a
    task policy or protocol-specific action meaning.
    """

    schema = EXTERNAL_FACTORED_TRANSITION_MODEL_SCHEMA

    def __init__(
        self,
        state_width: int,
        intention_width: int,
        context_width: int,
        *,
        hidden_width: int = 64,
        residual_read_match_threshold: float = 0.999,
        residual_write_match_threshold: float = 0.999,
        residual_mode: str = EXTERNAL_FACTORED_TRANSITION_EXACT_RESIDUAL_MODE,
        residual_model_family: str = EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
        residual_hidden_width: int | None = None,
        residual_learning_rate: float = 0.01,
        residual_ridge: float = 1e-5,
        residual_random_feature_width: int = 128,
        residual_random_feature_seed: int = 0,
        residual_capacity: int | None = None,
        base_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if min(state_width, intention_width, context_width, hidden_width) < 1:
            raise ValueError("factored transition dimensions must be positive")
        self.state_width = int(state_width)
        self.intention_width = int(intention_width)
        self.context_width = int(context_width)
        self.hidden_width = int(hidden_width)
        if residual_mode not in {
            EXTERNAL_FACTORED_TRANSITION_EXACT_RESIDUAL_MODE,
            EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        }:
            raise ValueError("unsupported factored transition residual mode")
        if residual_hidden_width is None:
            residual_hidden_width = hidden_width
        if residual_hidden_width < 1:
            raise ValueError("factored residual hidden width must be positive")
        if residual_learning_rate <= 0.0:
            raise ValueError("factored residual learning rate must be positive")
        if residual_ridge <= 0.0:
            raise ValueError("factored residual ridge must be positive")
        if residual_random_feature_width < 1:
            raise ValueError("factored residual random-feature width must be positive")
        if residual_capacity is not None and residual_capacity < 1:
            raise ValueError("factored residual capacity must be positive")
        if residual_model_family not in {
            EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
            EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        }:
            raise ValueError("unsupported factored residual model family")
        self.residual_mode = str(residual_mode)
        self.residual_model_family = str(residual_model_family)
        self.residual_hidden_width = int(residual_hidden_width)
        self.residual_learning_rate = float(residual_learning_rate)
        self.residual_ridge = float(residual_ridge)
        self.residual_random_feature_width = int(residual_random_feature_width)
        self.residual_random_feature_seed = int(residual_random_feature_seed)
        self.residual_capacity = (
            None if residual_capacity is None else int(residual_capacity)
        )
        self.base = (
            ExternalTransitionModel(
                self.state_width,
                self.intention_width,
                hidden_width=self.hidden_width,
            )
            if base_model is None
            else base_model
        )
        self._validate_base_model(self.base)
        self.residual_memory = ExternalTransitionMemory(
            self.state_width,
            self.intention_width,
            context_width=self.context_width,
            read_match_threshold=residual_read_match_threshold,
            write_match_threshold=residual_write_match_threshold,
        )
        self.residual_bank = (
            None
            if self.residual_mode == EXTERNAL_FACTORED_TRANSITION_EXACT_RESIDUAL_MODE
            else ExternalTransitionModelBank(
                self.state_width,
                self.intention_width,
                self.context_width,
                hidden_width=self.residual_hidden_width,
                model_family=self.residual_model_family,
                matching_tolerance=1e-6,
                affine_ridge=self.residual_ridge,
                adaptation_learning_rate=self.residual_learning_rate,
                random_feature_width=self.residual_random_feature_width,
                random_feature_seed=self.residual_random_feature_seed,
                capacity=self.residual_capacity,
            )
        )

    def configuration(self) -> dict[str, int | float | str | dict[str, object]]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "context_width": self.context_width,
            "hidden_width": self.hidden_width,
            "residual_mode": self.residual_mode,
            "residual_model_family": self.residual_model_family,
            "residual_hidden_width": self.residual_hidden_width,
            "residual_learning_rate": self.residual_learning_rate,
            "residual_ridge": self.residual_ridge,
            "residual_random_feature_width": self.residual_random_feature_width,
            "residual_random_feature_seed": self.residual_random_feature_seed,
            "residual_capacity": self.residual_capacity,
            "representation": "frozen_shared_base_plus_opaque_context_residual_v2",
            "behavior": "derived_by_external_search_not_stored_policy_v1",
            "base": self.base.configuration(),
            "base_model_schema": str(
                getattr(self.base, "schema", type(self.base).__name__)
            ),
            "base_persistence": "versioned_external_base_payload_v1",
            "residual_memory": self.residual_memory.configuration(),
            "residual_bank": (
                None
                if self.residual_bank is None
                else self.residual_bank.configuration()
            ),
        }

    @property
    def base_frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.base.parameters())

    @property
    def residual_record_count(self) -> int:
        if self.residual_bank is not None:
            return self.residual_bank.context_count
        return self.residual_memory.record_count

    @property
    def residual_context_count(self) -> int:
        return 0 if self.residual_bank is None else self.residual_bank.context_count

    def freeze_base(self) -> None:
        """Freeze reusable computation before online residual adaptation."""

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def _residual_observation(
        self,
        observation: ExternalTransitionObservation,
    ) -> ExternalTransitionObservation:
        observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )
        with torch.no_grad():
            residual = observation.next_state - self.base(
                observation.state,
                observation.intention,
            )
        return ExternalTransitionObservation(
            state=observation.state.detach(),
            intention=observation.intention.detach(),
            next_state=residual.detach(),
            confidence=(
                None
                if observation.confidence is None
                else observation.confidence.detach()
            ),
        )

    @torch.no_grad()
    def write_residual(
        self,
        observation: ExternalTransitionObservation,
        *,
        context: torch.Tensor,
    ) -> object:
        """Store or address one current-regime correction without base updates."""

        residual = self._residual_observation(observation)
        context_batch = context
        if context_batch.ndim == 1:
            context_batch = context_batch.unsqueeze(0).expand(
                observation.state.shape[0], -1
            )
        if self.residual_bank is not None:
            return self.residual_bank.ensure_context(context)
        return self.residual_memory.write(residual, context=context_batch)

    def fit_residual(
        self,
        observation: ExternalTransitionObservation,
        *,
        context: torch.Tensor,
        updates: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[float, int]:
        """Learn a context-local residual function outside the frozen base."""

        if self.residual_bank is None:
            raise RuntimeError("exact residual memory has no trainable residual function")
        if not self.base_frozen:
            raise RuntimeError("factored residual learning requires a frozen base")
        if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
            raise ValueError("factored residual updates must be a positive integer")
        observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )
        slot_index = self.residual_bank.ensure_context(context)
        normalized_context = self.residual_bank.context_at(slot_index)
        context_batch = normalized_context.to(observation.state).unsqueeze(0).expand(
            observation.state.shape[0], -1
        )
        residual = self._residual_observation(observation)
        final_loss = float("inf")
        if hasattr(self.residual_bank.models[slot_index], "observe"):
            final_loss = self.residual_bank.adaptation_step(
                residual,
                context_batch,
                None,
            )
            return final_loss, 1
        selected_optimizer = optimizer or torch.optim.Adam(
            self.residual_bank.models[slot_index].parameters(),
            lr=self.residual_learning_rate,
        )
        for _update in range(updates):
            final_loss = self.residual_bank.adaptation_step(
                residual,
                context_batch,
                selected_optimizer,
            )
        return final_loss, updates

    def predict_with_context(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        base_prediction = self.base(state, intention)
        if self.residual_bank is not None:
            residual_prediction = self.residual_bank(
                state,
                intention,
                context,
            )
            return base_prediction + residual_prediction.to(base_prediction)
        residual, hit = self.residual_memory.predict_with_hit(
            state,
            intention,
            context=context,
        )
        return base_prediction + torch.where(
            hit.unsqueeze(-1),
            residual.to(base_prediction),
            torch.zeros_like(base_prediction),
        )

    def forward(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None:
            return self.base(state, intention)
        return self.predict_with_context(state, intention, context)

    def loss(
        self,
        observation: ExternalTransitionObservation,
        *,
        context: torch.Tensor,
    ) -> torch.Tensor:
        observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )
        prediction = self.predict_with_context(
            observation.state,
            observation.intention,
            context,
        )
        errors = (prediction - observation.next_state).square().mean(dim=-1)
        if observation.confidence is None:
            return errors.mean()
        confidence = observation.confidence.reshape(-1).to(errors)
        return (errors * confidence).sum() / confidence.sum().clamp_min(1e-12)

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        digest.update(self.base.digest().encode("utf-8"))
        digest.update(self.residual_memory.digest().encode("utf-8"))
        if self.residual_bank is not None:
            digest.update(self.residual_bank.digest().encode("utf-8"))
        return digest.hexdigest()

    def _validate_base_model(self, base: nn.Module) -> None:
        """Validate the narrow, versioned interface of a replaceable base."""

        if not isinstance(base, nn.Module):
            raise TypeError("factored transition base must be a torch module")
        for attribute in ("configuration", "state_payload", "digest", "loss"):
            if not callable(getattr(base, attribute, None)):
                raise TypeError(
                    f"factored transition base must provide {attribute}()"
                )
        schema = getattr(base, "schema", None)
        if not isinstance(schema, str) or not schema:
            raise ValueError("factored transition base must expose a schema")
        state_width = getattr(base, "state_width", None)
        intention_width = getattr(base, "intention_width", None)
        if not isinstance(state_width, int) or isinstance(state_width, bool):
            raise TypeError("factored transition base state width is invalid")
        if not isinstance(intention_width, int) or isinstance(intention_width, bool):
            raise TypeError("factored transition base intention width is invalid")
        if state_width != self.state_width:
            raise ValueError("factored transition base state width differs")
        if intention_width != self.intention_width:
            raise ValueError("factored transition base intention width differs")

    @staticmethod
    def _base_from_payload(payload: Mapping[str, Any]) -> nn.Module:
        """Restore only explicitly supported, versioned external base models."""

        if not isinstance(payload, Mapping):
            raise TypeError("factored transition base payload must be a mapping")
        schema = payload.get("schema")
        if schema == ExternalTransitionModel.schema:
            return ExternalTransitionModel.from_payload(payload)
        from .online_transition import (
            ExternalAffineTransitionStatistics,
            ExternalRandomFeatureTransitionStatistics,
        )

        if schema == ExternalAffineTransitionStatistics.schema:
            return ExternalAffineTransitionStatistics.from_payload(payload)
        if schema == ExternalRandomFeatureTransitionStatistics.schema:
            return ExternalRandomFeatureTransitionStatistics.from_payload(payload)
        raise ValueError("unsupported factored transition base schema")

    @staticmethod
    def _state_payload(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()
        }

    @staticmethod
    def _load_state(module: nn.Module, state: Mapping[str, Any]) -> None:
        current = module.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("factored transition state names do not match")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("factored transition state values must be tensors")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"factored transition state {name!r} is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"factored transition state {name!r} is non-finite")
            normalized[name] = value.detach().clone()
        module.load_state_dict(normalized, strict=True)

    def state_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "configuration": self.configuration(),
            "base_payload": self.base.state_payload(),
            "base_state": self._state_payload(self.base),
            "residual_state": self._state_payload(self.residual_memory),
            "residual_bank": (
                None if self.residual_bank is None else self.residual_bank.payload()
            ),
            "base_frozen": self.base_frozen,
            "sha256": self.digest(),
        }
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalFactoredTransitionModel:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported factored transition payload")
        configuration = payload.get("configuration")
        base_payload = payload.get("base_payload")
        base_state = payload.get("base_state")
        residual_state = payload.get("residual_state")
        if not isinstance(configuration, Mapping):
            raise TypeError("factored transition configuration is missing")
        if not isinstance(base_state, Mapping) or not isinstance(
            residual_state, Mapping
        ):
            raise TypeError("factored transition states are missing")
        residual_configuration = configuration.get("residual_memory")
        if not isinstance(residual_configuration, Mapping):
            raise TypeError("factored residual configuration is missing")
        residual_mode = str(
            configuration.get(
                "residual_mode",
                EXTERNAL_FACTORED_TRANSITION_EXACT_RESIDUAL_MODE,
            )
        )
        restored_base = (
            None
            if base_payload is None
            else cls._base_from_payload(base_payload)
        )
        model = cls(
            int(configuration["state_width"]),
            int(configuration["intention_width"]),
            int(configuration["context_width"]),
            hidden_width=int(configuration["hidden_width"]),
            residual_read_match_threshold=float(
                residual_configuration["store"]["read_match_threshold"]
            ),
            residual_write_match_threshold=float(
                residual_configuration["store"]["write_match_threshold"]
            ),
            residual_mode=residual_mode,
            residual_model_family=str(
                configuration.get(
                    "residual_model_family",
                    EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
                )
            ),
            residual_hidden_width=int(
                configuration.get(
                    "residual_hidden_width",
                    configuration["hidden_width"],
                )
            ),
            residual_learning_rate=float(
                configuration.get("residual_learning_rate", 0.01)
            ),
            residual_ridge=float(configuration.get("residual_ridge", 1e-5)),
            residual_random_feature_width=int(
                configuration.get("residual_random_feature_width", 128)
            ),
            residual_random_feature_seed=int(
                configuration.get("residual_random_feature_seed", 0)
            ),
            residual_capacity=(
                None
                if configuration.get("residual_capacity") is None
                else int(configuration["residual_capacity"])
            ),
            base_model=restored_base,
        )
        cls._load_state(model.base, base_state)
        residual_store_state = {
            name.removeprefix("store."): value
            for name, value in residual_state.items()
        }
        model.residual_memory.store.load_state_dict(
            residual_store_state,
            strict=True,
        )
        residual_bank_payload = payload.get("residual_bank")
        if model.residual_bank is None:
            if residual_bank_payload is not None:
                raise ValueError("exact residual mode cannot contain a residual bank")
        else:
            if not isinstance(residual_bank_payload, Mapping):
                raise TypeError("learned residual mode requires a residual bank")
            model.residual_bank = ExternalTransitionModelBank.from_payload(
                residual_bank_payload
            )
        if bool(payload.get("base_frozen", False)):
            model.freeze_base()
        if payload.get("sha256") != model.digest():
            raise ValueError("factored transition checksum mismatch")
        return model


@dataclass(frozen=True)
class FactoredTransitionRouteResult:
    """Opaque read/admission result from the factored-memory router."""

    status: str
    slot_id: int | None
    context: torch.Tensor | None
    pending_observations: int
    schema: str = EXTERNAL_FACTORED_TRANSITION_ROUTER_SCHEMA
    quarantine_accepted: bool | None = None

    def validate(self, *, context_width: int) -> FactoredTransitionRouteResult:
        if self.schema != EXTERNAL_FACTORED_TRANSITION_ROUTER_SCHEMA:
            raise ValueError("unsupported factored transition route schema")
        if self.status not in {
            "pending",
            "matched",
            "staged",
            "ambiguous",
            "reliability_veto",
        }:
            raise ValueError("unsupported factored transition route status")
        if self.slot_id is not None and self.slot_id < 0:
            raise ValueError("factored transition route slot ID is invalid")
        if self.context is not None and (
            self.context.ndim != 1 or self.context.shape[0] != context_width
        ):
            raise ValueError("factored transition route context has wrong shape")
        if self.pending_observations < 0:
            raise ValueError("factored transition pending count is invalid")
        if self.quarantine_accepted is not None and not isinstance(
            self.quarantine_accepted, bool
        ):
            raise ValueError("factored transition quarantine status is invalid")
        return self


@dataclass(frozen=True)
class FactoredTransitionPromotionReceipt:
    """Verifier receipt for committing an isolated residual candidate."""

    accepted: bool
    slot_id: int | None
    heldout_error: float
    reason: str
    schema: str = EXTERNAL_FACTORED_TRANSITION_PROMOTION_SCHEMA
    heldout_rollout_error: float | None = None

    def validate(self) -> FactoredTransitionPromotionReceipt:
        if self.schema != EXTERNAL_FACTORED_TRANSITION_PROMOTION_SCHEMA:
            raise ValueError("unsupported factored transition promotion schema")
        if self.slot_id is not None and self.slot_id < 0:
            raise ValueError("factored promotion slot ID is invalid")
        if not torch.isfinite(torch.tensor(self.heldout_error)) or self.heldout_error < 0.0:
            raise ValueError("factored promotion heldout error is invalid")
        if self.heldout_rollout_error is not None and (
            not torch.isfinite(torch.tensor(self.heldout_rollout_error))
            or self.heldout_rollout_error < 0.0
        ):
            raise ValueError("factored promotion rollout error is invalid")
        if not self.reason:
            raise ValueError("factored promotion reason is missing")
        return self


@dataclass(frozen=True)
class FactoredTransitionQuarantineReceipt:
    """Receipt for explicitly storing unresolved external evidence."""

    accepted: bool
    rows_added: int
    total_rows: int
    capacity: int
    digest_before: str
    digest_after: str
    reason: str
    schema: str = EXTERNAL_FACTORED_TRANSITION_QUARANTINE_SCHEMA

    def validate(self) -> FactoredTransitionQuarantineReceipt:
        if self.schema != EXTERNAL_FACTORED_TRANSITION_QUARANTINE_SCHEMA:
            raise ValueError("unsupported factored transition quarantine schema")
        if min(self.rows_added, self.total_rows) < 0:
            raise ValueError("factored quarantine row counts cannot be negative")
        if self.capacity < 0 or self.total_rows > self.capacity:
            raise ValueError("factored quarantine capacity is invalid")
        if not self.digest_before or not self.digest_after:
            raise ValueError("factored quarantine digests are missing")
        if not self.reason:
            raise ValueError("factored quarantine reason is missing")
        return self


class ExternalFactoredTransitionRouter:
    """Infer opaque residual addresses and promote candidates copy-on-write.

    Existing contexts are read-only during routing.  A novel stream is held
    outside the committed model until enough evidence is accumulated; its
    residual writes occur on an isolated model copy.  Only a caller-owned
    held-out retention/behavior probe can commit that copy.
    """

    schema = EXTERNAL_FACTORED_TRANSITION_ROUTER_SCHEMA

    def __init__(
        self,
        model: ExternalFactoredTransitionModel,
        context_encoder: ExternalTransitionContextEncoder,
        *,
        match_tolerance: float = 0.05,
        match_margin: float = 0.01,
        admission_observations: int = 8,
        max_contexts: int | None = None,
        auto_grow: bool = False,
        residual_adaptation_updates: int = 16,
        quarantine_capacity: int = 0,
        evidence_evaluator: nn.Module | None = None,
        evidence_threshold: float = 0.5,
        evidence_gate_min_evidence: int = 0,
        committed_evidence_gate: bool = False,
        sparse_evidence: ExternalSparseTransitionEvidenceIndex | None = None,
    ) -> None:
        if not isinstance(model, ExternalFactoredTransitionModel):
            raise TypeError("factored router requires a factored transition model")
        if (
            model.state_width != context_encoder.state_width
            or model.intention_width != context_encoder.intention_width
            or model.context_width != context_encoder.context_width
        ):
            raise ValueError("factored router model and encoder widths differ")
        if match_tolerance < 0.0 or match_margin < 0.0:
            raise ValueError("factored router tolerances cannot be negative")
        if admission_observations < 1:
            raise ValueError("factored router admission observations must be positive")
        if max_contexts is not None and max_contexts < 1:
            raise ValueError("factored router capacity must be positive")
        if not isinstance(auto_grow, bool):
            raise TypeError("factored router auto-growth flag must be boolean")
        bank_capacity = (
            None if model.residual_bank is None else model.residual_bank.capacity
        )
        if max_contexts is None and bank_capacity is not None:
            max_contexts = bank_capacity
        elif (
            max_contexts is not None
            and bank_capacity is not None
            and max_contexts != bank_capacity
        ):
            raise ValueError("factored router and residual-bank capacities differ")
        if (
            not isinstance(residual_adaptation_updates, int)
            or isinstance(residual_adaptation_updates, bool)
            or residual_adaptation_updates < 1
        ):
            raise ValueError("factored residual adaptation updates must be positive")
        if (
            not isinstance(quarantine_capacity, int)
            or isinstance(quarantine_capacity, bool)
            or quarantine_capacity < 0
        ):
            raise ValueError("factored quarantine capacity must be non-negative")
        if not 0.0 < evidence_threshold < 1.0:
            raise ValueError("factored evidence threshold must lie in (0, 1)")
        if evidence_gate_min_evidence < 0:
            raise ValueError("factored evidence gate warm-up cannot be negative")
        if not isinstance(committed_evidence_gate, bool):
            raise TypeError("factored committed evidence gate must be boolean")
        if evidence_evaluator is not None and (
            not hasattr(evidence_evaluator, "state_width")
            or int(evidence_evaluator.state_width) != model.state_width
        ):
            raise ValueError("factored evidence evaluator and model widths differ")
        if sparse_evidence is not None and (
            sparse_evidence.state_width != model.state_width
            or sparse_evidence.intention_width != model.intention_width
        ):
            raise ValueError("factored sparse evidence and model widths differ")
        self.model = model
        self.context_encoder = context_encoder
        self.match_tolerance = float(match_tolerance)
        self.match_margin = float(match_margin)
        self.admission_observations = int(admission_observations)
        self.max_contexts = max_contexts
        self.auto_grow = auto_grow
        self.residual_adaptation_updates = int(residual_adaptation_updates)
        self.quarantine_capacity = int(quarantine_capacity)
        self.evidence_evaluator = evidence_evaluator
        self.evidence_threshold = float(evidence_threshold)
        self.evidence_gate_min_evidence = int(evidence_gate_min_evidence)
        self.committed_evidence_gate = committed_evidence_gate
        self.sparse_evidence = sparse_evidence
        self._contexts: list[torch.Tensor] = []
        self._slot_ids: list[int] = []
        self._next_slot_id = 0
        self._pending: list[ExternalTransitionObservation] = []
        self._quarantine: list[ExternalTransitionObservation] = []
        self._candidate_model: ExternalFactoredTransitionModel | None = None
        self._candidate_context: torch.Tensor | None = None
        self._candidate_evidence: list[ExternalTransitionObservation] = []
        self._last_reliability_veto = False

    @property
    def contexts(self) -> torch.Tensor:
        if not self._contexts:
            return torch.empty((0, self.model.context_width))
        return torch.stack(self._contexts).detach().clone()

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(self._slot_ids)

    @property
    def candidate_active(self) -> bool:
        return self._candidate_model is not None

    @property
    def pending_observations(self) -> int:
        return sum(item.state.shape[0] for item in self._pending)

    @property
    def quarantined_observations(self) -> int:
        """Return the number of unresolved rows retained outside candidates."""

        return sum(item.state.shape[0] for item in self._quarantine)

    @property
    def quarantined_bundles(self) -> int:
        """Return the number of unresolved evidence bundles retained."""

        return len(self._quarantine)

    def grow_verified(
        self,
        destination_capacity: int,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
    ) -> ExternalTransitionModelGrowthReceipt:
        """Grow the factored router and external bank as one verified transaction."""

        if self.max_contexts is None:
            raise ValueError("factored router requires an explicit maximum for growth")
        if self.model.residual_bank is None:
            raise ValueError("factored growth requires a learned residual bank")
        if self.model.residual_bank.capacity != self.max_contexts:
            raise ValueError("factored router and residual-bank capacities are out of sync")
        receipt = self.model.residual_bank.grow_verified(
            destination_capacity,
            retention_probe,
        )
        if receipt.accepted:
            self.max_contexts = destination_capacity
            self.model.residual_capacity = destination_capacity
        return receipt

    def evict_verified_id(
        self,
        slot_id: int,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
    ) -> ExternalTransitionModelEvictionReceipt:
        """Evict one logical residual slot and repair the factored route cache."""

        if self.model.residual_bank is None:
            raise ValueError("factored eviction requires a learned residual bank")
        if tuple(self._slot_ids) != self.model.residual_bank.slot_ids:
            raise RuntimeError("factored router and residual-bank slot addresses differ")
        context = self._contexts[self._slot_ids.index(slot_id)].clone()
        receipt = self.model.residual_bank.evict_verified_id(
            slot_id,
            retention_probe,
        )
        if receipt.accepted:
            try:
                index = self._slot_ids.index(slot_id)
            except ValueError as error:
                raise RuntimeError(
                    "factored eviction removed an unknown router slot"
                ) from error
            del self._slot_ids[index]
            del self._contexts[index]
            if self.sparse_evidence is not None:
                self.sparse_evidence.unregister_slot(slot_id)
            evict_context = getattr(self.evidence_evaluator, "evict_context", None)
            if callable(evict_context):
                evict_context(context)
        return receipt

    def select_compression_verified(
        self,
        codecs: Sequence[torch.dtype | str],
        *,
        retention_probe: Callable[[ExternalTransitionModelBank], bool] | None = None,
    ) -> ExternalTransitionModelCompressionSelection:
        """Select the smallest retained storage codec for external residuals."""

        if self.model.residual_bank is None:
            raise ValueError("factored compression requires a learned residual bank")
        return self.model.residual_bank.select_compression_verified(
            codecs,
            retention_probe=retention_probe,
        )

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
            raise ValueError("cannot merge empty factored router evidence")
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

    def _route_existing(
        self,
        observation: ExternalTransitionObservation,
        *,
        match_tolerance: float | None = None,
        match_margin: float | None = None,
    ) -> tuple[int, float, float] | None:
        self._last_reliability_veto = False
        if not self._contexts:
            return None
        errors: list[float] = []
        for context in self._contexts:
            context_batch = context.to(observation.state).unsqueeze(0)
            prediction = self.model.predict_with_context(
                observation.state,
                observation.intention,
                context_batch.expand(observation.state.shape[0], -1),
            )
            error = float((prediction - observation.next_state).square().mean().detach())
            if self.committed_evidence_gate and not self._evidence_allows(
                prediction,
                observation.next_state,
                context=context,
                candidate_evidence_count=None,
            ):
                if error <= (
                    self.match_tolerance
                    if match_tolerance is None
                    else match_tolerance
                ):
                    self._last_reliability_veto = True
                errors.append(float("inf"))
            else:
                errors.append(error)
        ordering = sorted(range(len(errors)), key=lambda index: (errors[index], index))
        best = ordering[0]
        margin = (
            float("inf") if len(ordering) == 1 else errors[ordering[1]] - errors[best]
        )
        tolerance = self.match_tolerance if match_tolerance is None else match_tolerance
        margin_floor = self.match_margin if match_margin is None else match_margin
        if tolerance < 0.0 or margin_floor < 0.0:
            raise ValueError("factored route tolerances cannot be negative")
        if errors[best] > tolerance or margin < margin_floor:
            return None
        return self._slot_ids[best], errors[best], margin

    def _evidence_probability(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
    ) -> float:
        """Read caller-owned reliability state without changing factual memory."""

        if self.evidence_evaluator is None:
            return 1.0
        hits = torch.ones(
            prediction.shape[0],
            device=prediction.device,
            dtype=prediction.dtype,
        )
        scorer = getattr(self.evidence_evaluator, "score", None)
        with torch.no_grad():
            if callable(scorer):
                if context is not None and hasattr(
                    self.evidence_evaluator, "context_width"
                ):
                    context_batch = (
                        context.unsqueeze(0).expand(prediction.shape[0], -1)
                        if context.ndim == 1
                        else context
                    )
                    logits = scorer(prediction, observed, hits, context_batch)
                else:
                    logits = scorer(prediction, observed, hits)
            else:
                logits = self.evidence_evaluator(prediction, observed, hits)
            return float(torch.sigmoid(logits).mean())

    def _evidence_allows(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        candidate_evidence_count: int | None,
    ) -> bool:
        """Conservatively veto unreliable evidence after explicit warm-up."""

        if self.evidence_evaluator is None:
            return True
        if candidate_evidence_count is None:
            observed_count = getattr(
                self.evidence_evaluator, "observation_count", None
            )
            ready = (
                True
                if observed_count is None
                else int(observed_count) >= self.evidence_gate_min_evidence
            )
        else:
            ready = candidate_evidence_count >= self.evidence_gate_min_evidence
        return (
            not ready
            or self._evidence_probability(prediction, observed, context=context)
            >= self.evidence_threshold
        )

    def _quarantine_reliability_veto(
        self,
        observation: ExternalTransitionObservation,
    ) -> FactoredTransitionRouteResult:
        receipt = self.quarantine_partial_bundle((observation,))
        return FactoredTransitionRouteResult(
            status="reliability_veto",
            slot_id=None,
            context=None,
            pending_observations=0,
            quarantine_accepted=receipt.accepted,
        ).validate(context_width=self.model.context_width)

    def _sparse_proposal(
        self,
        observation: ExternalTransitionObservation,
    ) -> int | None:
        """Return a contradiction-free sparse factual identity proposal."""

        if self.sparse_evidence is None or not self._slot_ids:
            return None
        proposal = self.sparse_evidence.propose(observation, self._slot_ids)
        slot_id = proposal.selected_slot_id
        if slot_id is None or not self.committed_evidence_gate:
            return slot_id
        context = self._contexts[self._slot_ids.index(slot_id)]
        prediction = self.model.predict_with_context(
            observation.state,
            observation.intention,
            context.to(observation.state).unsqueeze(0).expand(
                observation.state.shape[0], -1
            ),
        )
        error = float((prediction - observation.next_state).square().mean().detach())
        if not self._evidence_allows(
            prediction,
            observation.next_state,
            context=context,
            candidate_evidence_count=None,
        ):
            if error <= self.match_tolerance:
                self._last_reliability_veto = True
            return None
        return slot_id

    def _stage_candidate(self) -> FactoredTransitionRouteResult:
        evidence = self._merge(self._pending)
        context = self.context_encoder.encode_observation(evidence).detach()
        candidate = ExternalFactoredTransitionModel.from_payload(
            self.model.state_payload()
        )
        if (
            self.auto_grow
            and self.max_contexts is not None
            and len(self._contexts) >= self.max_contexts
            and candidate.residual_bank is not None
        ):
            candidate.residual_bank.capacity = self.max_contexts + 1
            candidate.residual_capacity = self.max_contexts + 1
        candidate.write_residual(evidence, context=context)
        if candidate.residual_bank is not None:
            candidate.fit_residual(
                evidence,
                context=context,
                updates=self.residual_adaptation_updates,
            )
        self._candidate_model = candidate
        self._candidate_context = context
        self._candidate_evidence = [self._clone_observation(evidence)]
        self._pending.clear()
        return FactoredTransitionRouteResult(
            status="staged",
            slot_id=None,
            context=context.clone(),
            pending_observations=0,
        ).validate(context_width=self.model.context_width)

    def observe(
        self,
        observation: ExternalTransitionObservation,
    ) -> FactoredTransitionRouteResult:
        """Route one opaque row without mutating committed memory on novelty."""

        observation.validate(
            state_width=self.model.state_width,
            intention_width=self.model.intention_width,
        )
        if observation.state.shape[0] != 1:
            raise ValueError("factored router expects one observation per call")
        if self._candidate_model is not None:
            if self._candidate_context is None:
                raise RuntimeError("factored candidate context is missing")
            candidate_prediction = self._candidate_model.predict_with_context(
                observation.state,
                observation.intention,
                self._candidate_context.to(observation.state)
                .unsqueeze(0)
                .expand(observation.state.shape[0], -1),
            )
            if self.committed_evidence_gate and not self._evidence_allows(
                candidate_prediction,
                observation.next_state,
                context=self._candidate_context,
                candidate_evidence_count=sum(
                    item.state.shape[0] for item in self._candidate_evidence
                ),
            ):
                return self._quarantine_reliability_veto(observation)
            self._candidate_model.write_residual(
                observation,
                context=self._candidate_context,
            )
            if self._candidate_model.residual_bank is not None:
                self._candidate_model.fit_residual(
                    observation,
                    context=self._candidate_context,
                    updates=self.residual_adaptation_updates,
                )
            self._candidate_evidence.append(self._clone_observation(observation))
            return FactoredTransitionRouteResult(
                status="staged",
                slot_id=None,
                context=self._candidate_context.clone(),
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        matched = self._route_existing(observation)
        if matched is None:
            sparse_slot = self._sparse_proposal(observation)
            if sparse_slot is not None:
                matched = (sparse_slot, 0.0, float("inf"))
        if matched is not None:
            slot_id, _error, _margin = matched
            if self.sparse_evidence is not None:
                self.sparse_evidence.record(slot_id, observation)
            return FactoredTransitionRouteResult(
                status="matched",
                slot_id=slot_id,
                context=self._contexts[self._slot_ids.index(slot_id)].clone(),
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        if self._last_reliability_veto:
            self._pending.clear()
            return self._quarantine_reliability_veto(observation)
        self._pending.append(self._clone_observation(observation))
        if self.pending_observations < self.admission_observations:
            return FactoredTransitionRouteResult(
                status="pending",
                slot_id=None,
                context=None,
                pending_observations=self.pending_observations,
            ).validate(context_width=self.model.context_width)
        if (
            self.max_contexts is not None
            and not self.auto_grow
            and len(self._contexts) >= self.max_contexts
        ):
            return FactoredTransitionRouteResult(
                status="ambiguous",
                slot_id=None,
                context=None,
                pending_observations=self.pending_observations,
            ).validate(context_width=self.model.context_width)
        return self._stage_candidate()

    def route_bundle(
        self,
        observations: Sequence[ExternalTransitionObservation],
        *,
        match_tolerance: float | None = None,
        match_margin: float | None = None,
    ) -> FactoredTransitionRouteResult:
        """Route an opaque evidence bundle as one atomic stream transaction.

        A bundle is scored against each retained factual slot before any
        pending state is created.  This prevents a shared single transition
        from deciding identity and prevents interleaved novel streams from
        being merged into the router's row-wise pending buffer.
        """

        if self._candidate_model is not None:
            raise RuntimeError("cannot route a bundle while a candidate is staged")
        if self._pending:
            raise RuntimeError("cannot route a bundle with pending row evidence")
        if not observations:
            raise ValueError("factored route bundle cannot be empty")
        cloned = [self._clone_observation(item) for item in observations]
        for item in cloned:
            item.validate(
                state_width=self.model.state_width,
                intention_width=self.model.intention_width,
            )
        merged = self._merge(cloned)
        matched = self._route_existing(
            merged,
            match_tolerance=match_tolerance,
            match_margin=match_margin,
        )
        if matched is not None:
            slot_id, _error, _margin = matched
            if self.sparse_evidence is not None:
                self.sparse_evidence.record(slot_id, merged)
            return FactoredTransitionRouteResult(
                status="matched",
                slot_id=slot_id,
                context=self._contexts[self._slot_ids.index(slot_id)].clone(),
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        if self._last_reliability_veto:
            return self._quarantine_reliability_veto(merged)
        sparse_slot = self._sparse_proposal(merged)
        if self._last_reliability_veto:
            return self._quarantine_reliability_veto(merged)
        if sparse_slot is not None:
            self.sparse_evidence.record(sparse_slot, merged)
            return FactoredTransitionRouteResult(
                status="matched",
                slot_id=sparse_slot,
                context=self._contexts[self._slot_ids.index(sparse_slot)].clone(),
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        if (
            self.max_contexts is not None
            and not self.auto_grow
            and len(self._contexts) >= self.max_contexts
        ):
            return FactoredTransitionRouteResult(
                status="ambiguous",
                slot_id=None,
                context=None,
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        self._pending = cloned
        return self._stage_candidate()

    def route_partial_bundle(
        self,
        observations: Sequence[ExternalTransitionObservation],
        *,
        min_match_fraction: float = 1.0,
        match_tolerance: float = 0.05,
        contradiction_tolerance: float | None = None,
        match_margin: float | None = None,
    ) -> FactoredTransitionRouteResult:
        """Read-route partial evidence without admitting a new memory slot.

        This path is deliberately read-only.  It can match a known slot from
        a subset of its expected evidence, but it never stages a candidate from
        incomplete evidence.  A row that is far outside every candidate's
        factual tolerance is treated as contradictory rather than being
        averaged away by a majority of agreeing rows.  Empty evidence returns
        ``ambiguous`` as an explicit no-op result.
        """

        if not 0.0 < min_match_fraction <= 1.0:
            raise ValueError("partial route match fraction must lie in (0, 1]")
        if match_tolerance < 0.0:
            raise ValueError("partial route match tolerance cannot be negative")
        contradiction_floor = (
            match_tolerance
            if contradiction_tolerance is None
            else contradiction_tolerance
        )
        if contradiction_floor < match_tolerance:
            raise ValueError(
                "partial route contradiction tolerance cannot be below match tolerance"
            )
        if match_margin is not None and match_margin < 0.0:
            raise ValueError("partial route match margin cannot be negative")
        if self._candidate_model is not None or self._pending:
            raise RuntimeError("cannot partial-route while factored evidence is staged")
        if not observations or not self._contexts:
            return FactoredTransitionRouteResult(
                status="ambiguous",
                slot_id=None,
                context=None,
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        cloned = [self._clone_observation(item) for item in observations]
        for item in cloned:
            item.validate(
                state_width=self.model.state_width,
                intention_width=self.model.intention_width,
            )
        merged = self._merge(cloned)
        self._last_reliability_veto = False
        sparse_slot = self._sparse_proposal(merged)
        if self._last_reliability_veto:
            return FactoredTransitionRouteResult(
                status="reliability_veto",
                slot_id=None,
                context=None,
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        if sparse_slot is not None:
            context_index = self._slot_ids.index(sparse_slot)
            return FactoredTransitionRouteResult(
                status="matched",
                slot_id=sparse_slot,
                context=self._contexts[context_index].clone(),
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        required_matches = max(1, math.ceil(merged.state.shape[0] * min_match_fraction))
        eligible: list[tuple[int, float, float]] = []
        for index, context in enumerate(self._contexts):
            context_batch = context.to(merged.state).unsqueeze(0).expand(
                merged.state.shape[0], -1
            )
            prediction = self.model.predict_with_context(
                merged.state,
                merged.intention,
                context_batch,
            )
            errors = (prediction - merged.next_state).square().mean(dim=-1)
            if self.committed_evidence_gate and not self._evidence_allows(
                prediction,
                merged.next_state,
                context=context,
                candidate_evidence_count=None,
            ):
                continue
            matched = int((errors <= match_tolerance).sum())
            contradictory = bool(torch.any(errors > contradiction_floor))
            if matched < required_matches or contradictory:
                continue
            eligible.append(
                (
                    self._slot_ids[index],
                    float(errors.mean().detach()),
                    float(errors.max().detach()),
                )
            )
        if not eligible:
            return FactoredTransitionRouteResult(
                status="ambiguous",
                slot_id=None,
                context=None,
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        eligible.sort(key=lambda value: (value[1], value[0]))
        best_slot, best_error, _best_max = eligible[0]
        margin = (
            float("inf")
            if len(eligible) == 1
            else eligible[1][1] - best_error
        )
        margin_floor = self.match_margin if match_margin is None else match_margin
        if margin < margin_floor:
            return FactoredTransitionRouteResult(
                status="ambiguous",
                slot_id=None,
                context=None,
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        context_index = self._slot_ids.index(best_slot)
        return FactoredTransitionRouteResult(
            status="matched",
            slot_id=best_slot,
            context=self._contexts[context_index].clone(),
            pending_observations=0,
        ).validate(context_width=self.model.context_width)

    def quarantine_partial_bundle(
        self,
        observations: Sequence[ExternalTransitionObservation],
    ) -> FactoredTransitionQuarantineReceipt:
        """Persist unresolved evidence without routing, training, or promotion."""

        if self._candidate_model is not None or self._pending:
            raise RuntimeError("cannot quarantine while factored evidence is staged")
        if not observations:
            raise ValueError("factored quarantine bundle cannot be empty")
        cloned = [self._clone_observation(item) for item in observations]
        for item in cloned:
            item.validate(
                state_width=self.model.state_width,
                intention_width=self.model.intention_width,
            )
        rows_added = sum(item.state.shape[0] for item in cloned)
        before = self.digest()
        total = self.quarantined_observations + rows_added
        if self.quarantine_capacity == 0 or total > self.quarantine_capacity:
            return FactoredTransitionQuarantineReceipt(
                accepted=False,
                rows_added=0,
                total_rows=self.quarantined_observations,
                capacity=self.quarantine_capacity,
                digest_before=before,
                digest_after=before,
                reason="quarantine capacity rejected unresolved evidence",
            ).validate()
        self._quarantine.extend(cloned)
        after = self.digest()
        return FactoredTransitionQuarantineReceipt(
            accepted=True,
            rows_added=rows_added,
            total_rows=total,
            capacity=self.quarantine_capacity,
            digest_before=before,
            digest_after=after,
            reason="unresolved evidence retained outside committed memory",
        ).validate()

    def peek_quarantine(self) -> tuple[ExternalTransitionObservation, ...]:
        """Read retained bundles without combining or mutating them."""

        return tuple(self._clone_observation(item) for item in self._quarantine)

    def drain_quarantine(self) -> tuple[ExternalTransitionObservation, ...]:
        """Release retained bundles for an explicit caller-owned decision."""

        if self._candidate_model is not None or self._pending:
            raise RuntimeError("cannot drain quarantine while factored evidence is staged")
        if not self._quarantine:
            return ()
        released = self.peek_quarantine()
        self._quarantine.clear()
        return released

    def resolve_quarantine(
        self,
        *,
        min_match_fraction: float = 1.0,
        match_tolerance: float = 0.05,
        contradiction_tolerance: float | None = None,
        match_margin: float | None = None,
    ) -> tuple[int, ...]:
        """Re-test quarantined bundles against committed factual versions.

        Only bundles that independently match one existing slot are removed.
        Unresolved and contradictory bundles remain quarantined, preserving
        their boundaries for a later caller-owned admission decision.
        """

        if self._candidate_model is not None or self._pending:
            raise RuntimeError("cannot resolve quarantine while factored evidence is staged")
        resolved: list[int] = []
        unresolved: list[ExternalTransitionObservation] = []
        for item in self._quarantine:
            result = self.route_partial_bundle(
                (item,),
                min_match_fraction=min_match_fraction,
                match_tolerance=match_tolerance,
                contradiction_tolerance=contradiction_tolerance,
                match_margin=match_margin,
            )
            if result.status == "matched" and result.slot_id is not None:
                resolved.append(result.slot_id)
            else:
                unresolved.append(item)
        self._quarantine = unresolved
        return tuple(resolved)

    def resolve_quarantine_to_candidate(
        self,
        *,
        match_tolerance: float = 0.05,
    ) -> int:
        """Consume quarantined bundles that now fit the isolated candidate.

        A later evidence bundle may make a novel stream identifiable and
        create ``_candidate_model`` while an earlier bundle remains in
        quarantine.  This method tests each retained bundle against that
        candidate, then writes only independently agreeing bundles to the
        copy-on-write model.  The live model, committed contexts, and route
        addresses are unchanged until ``promote_staged_candidate`` passes its
        independent held-out and retention gates.

        The quarantined bundle itself is never used as its own promotion gate;
        callers must still provide separate held-out evidence when promoting.
        """

        if self._candidate_model is None or self._candidate_context is None:
            raise RuntimeError("factored router has no staged candidate")
        if self._pending:
            raise RuntimeError(
                "cannot resolve quarantine with pending factored evidence"
            )
        if match_tolerance < 0.0:
            raise ValueError("candidate quarantine match tolerance cannot be negative")
        resolved_rows = 0
        unresolved: list[ExternalTransitionObservation] = []
        for item in self._quarantine:
            context_batch = self._candidate_context.to(item.state).unsqueeze(0)
            prediction = self._candidate_model.predict_with_context(
                item.state,
                item.intention,
                context_batch.expand(item.state.shape[0], -1),
            )
            errors = (prediction - item.next_state).square().mean(dim=-1)
            if bool(torch.any(errors > match_tolerance)):
                unresolved.append(item)
                continue
            self._candidate_model.write_residual(
                item,
                context=self._candidate_context,
            )
            if self._candidate_model.residual_bank is not None:
                self._candidate_model.fit_residual(
                    item,
                    context=self._candidate_context,
                    updates=self.residual_adaptation_updates,
                )
            self._candidate_evidence.append(self._clone_observation(item))
            resolved_rows += int(item.state.shape[0])
        self._quarantine = unresolved
        return resolved_rows

    def promote_staged_candidate(
        self,
        heldout: ExternalTransitionObservation,
        retention_probe: Any,
        *,
        prediction_tolerance: float = 0.05,
        heldout_rollout: ExternalTransitionRollout | None = None,
        rollout_error_tolerance: float | None = None,
    ) -> FactoredTransitionPromotionReceipt:
        """Commit only a candidate that passes current and retention probes."""

        if self._candidate_model is None or self._candidate_context is None:
            raise RuntimeError("factored router has no staged candidate")
        if rollout_error_tolerance is not None and rollout_error_tolerance < 0.0:
            raise ValueError("factored candidate rollout tolerance cannot be negative")
        if heldout_rollout is None and rollout_error_tolerance is not None:
            raise ValueError(
                "rollout error tolerance requires a held-out transition rollout"
            )
        heldout.validate(
            state_width=self.model.state_width,
            intention_width=self.model.intention_width,
        )
        context_batch = self._candidate_context.to(heldout.state).unsqueeze(0)
        prediction = self._candidate_model.predict_with_context(
            heldout.state,
            heldout.intention,
            context_batch.expand(heldout.state.shape[0], -1),
        )
        error = float((prediction - heldout.next_state).square().mean().detach())
        heldout_rollout_error: float | None = None
        if heldout_rollout is not None:
            heldout_rollout.validate(
                state_width=self.model.state_width,
                intention_width=self.model.intention_width,
            )
            rollout_error_tolerance = (
                prediction_tolerance
                if rollout_error_tolerance is None
                else rollout_error_tolerance
            )
            heldout_rollout_error = ExternalModelBasedPlanner(
                self._candidate_model,
                beam_width=1,
            ).rollout_error(
                heldout_rollout,
                transition_context=self._candidate_context.unsqueeze(0),
            )
        grows_capacity = (
            self.auto_grow
            and self.max_contexts is not None
            and len(self._contexts) >= self.max_contexts
        )
        destination_capacity = (
            None if not grows_capacity else self.max_contexts + 1
        )
        if (
            destination_capacity is not None
            and self._candidate_model.residual_bank is not None
        ):
            self._candidate_model.residual_bank.capacity = destination_capacity
            self._candidate_model.residual_capacity = destination_capacity
        accepted = (
            error <= prediction_tolerance
            and (
                heldout_rollout_error is None
                or heldout_rollout_error <= rollout_error_tolerance
            )
            and bool(retention_probe(self._candidate_model))
        )
        rollout_failed = (
            heldout_rollout_error is not None
            and rollout_error_tolerance is not None
            and heldout_rollout_error > rollout_error_tolerance
        )
        if accepted:
            slot_id = self._next_slot_id
            self._next_slot_id += 1
            self._contexts.append(self._candidate_context.detach().clone())
            self._slot_ids.append(slot_id)
            if self.sparse_evidence is not None:
                self.sparse_evidence.register_slot(slot_id)
                for evidence in self._candidate_evidence:
                    self.sparse_evidence.record(slot_id, evidence)
            self.model = self._candidate_model
            if destination_capacity is not None:
                self.max_contexts = destination_capacity
            self._candidate_model = None
            self._candidate_context = None
            self._candidate_evidence = []
            return FactoredTransitionPromotionReceipt(
                accepted=True,
                slot_id=slot_id,
                heldout_error=error,
                reason=(
                    "candidate passed factual, recursive rollout, and retention probes with capacity growth"
                    if destination_capacity is not None and heldout_rollout_error is not None
                    else "candidate passed factual, recursive rollout, and retention probes"
                    if heldout_rollout_error is not None
                    else "candidate passed factual and retention probes with capacity growth"
                    if destination_capacity is not None
                    else "candidate passed factual and retention probes"
                ),
                heldout_rollout_error=heldout_rollout_error,
            ).validate()
        self._candidate_model = None
        self._candidate_context = None
        self._candidate_evidence = []
        return FactoredTransitionPromotionReceipt(
            accepted=False,
            slot_id=None,
            heldout_error=error,
            reason=(
                "candidate failed recursive rollout probe"
                if rollout_failed
                else "candidate failed factual or retention probe"
            ),
            heldout_rollout_error=heldout_rollout_error,
        ).validate()

    def update_bound_slot(
        self,
        slot_id: int,
        observation: ExternalTransitionObservation,
        retention_probe: Callable[[ExternalFactoredTransitionModel], bool],
        *,
        heldout: ExternalTransitionObservation,
        prediction_tolerance: float = 0.05,
    ) -> FactoredTransitionPromotionReceipt:
        """Add a verified unseen fact to an already-bound logical slot.

        The update is copy-on-write.  A caller that owns an opaque stream
        binding can extend its slot without asking the identity router to
        rediscover the context from one incomplete row.  Existing behavior is
        protected by ``retention_probe`` before the new residual is committed.
        """

        if not isinstance(slot_id, int) or isinstance(slot_id, bool) or slot_id < 0:
            raise ValueError("bound factored slot ID must be a non-negative integer")
        if not callable(retention_probe):
            raise TypeError("bound factored retention probe must be callable")
        if prediction_tolerance < 0.0:
            raise ValueError("bound factored prediction tolerance cannot be negative")
        observation.validate(
            state_width=self.model.state_width,
            intention_width=self.model.intention_width,
        )
        try:
            context_index = self._slot_ids.index(slot_id)
        except ValueError as error:
            raise KeyError(f"unknown factored slot ID: {slot_id}") from error
        context = self._contexts[context_index]
        candidate = ExternalFactoredTransitionModel.from_payload(
            self.model.state_payload()
        )
        candidate.write_residual(observation, context=context)
        if candidate.residual_bank is not None:
            candidate.fit_residual(
                observation,
                context=context,
                updates=self.residual_adaptation_updates,
            )
        heldout.validate(
            state_width=self.model.state_width,
            intention_width=self.model.intention_width,
        )
        context_batch = context.to(heldout.state).unsqueeze(0)
        prediction = candidate.predict_with_context(
            heldout.state,
            heldout.intention,
            context_batch.expand(heldout.state.shape[0], -1),
        )
        error = float(
            (prediction - heldout.next_state).square().mean().detach()
        )
        accepted = error <= prediction_tolerance and bool(retention_probe(candidate))
        if accepted:
            self.model = candidate
            if self.sparse_evidence is not None:
                self.sparse_evidence.record(slot_id, observation)
            return FactoredTransitionPromotionReceipt(
                accepted=True,
                slot_id=slot_id,
                heldout_error=error,
                reason="bound residual passed factual and retention probes",
            ).validate()
        return FactoredTransitionPromotionReceipt(
            accepted=False,
            slot_id=None,
            heldout_error=error,
            reason="bound residual failed factual or retention probe",
        ).validate()

    def configuration(self) -> dict[str, int | float | str | None]:
        return {
            "schema": self.schema,
            "match_tolerance": self.match_tolerance,
            "match_margin": self.match_margin,
            "admission_observations": self.admission_observations,
            "max_contexts": self.max_contexts,
            "auto_grow": self.auto_grow,
            "residual_adaptation_updates": self.residual_adaptation_updates,
            "quarantine_capacity": self.quarantine_capacity,
            "evidence_threshold": self.evidence_threshold,
            "evidence_gate_min_evidence": self.evidence_gate_min_evidence,
            "committed_evidence_gate": self.committed_evidence_gate,
            "evidence_evaluator": (
                None
                if self.evidence_evaluator is None
                else self.evidence_evaluator.configuration()
            ),
            "sparse_evidence": (
                None
                if self.sparse_evidence is None
                else self.sparse_evidence.configuration()
            ),
            "quarantined_observations": self.quarantined_observations,
            "slot_ids": list(self._slot_ids),
            "behavior": "copy_on_write_residual_admission_v1",
        }

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        digest.update(self.model.digest().encode("utf-8"))
        digest.update(self.context_encoder.digest().encode("utf-8"))
        for context, slot_id in zip(self._contexts, self._slot_ids, strict=True):
            digest.update(context.detach().cpu().contiguous().numpy().tobytes())
            digest.update(str(slot_id).encode("utf-8"))
        if self._candidate_context is not None:
            digest.update(self._candidate_context.detach().cpu().contiguous().numpy().tobytes())
        if self._candidate_model is not None:
            digest.update(self._candidate_model.digest().encode("utf-8"))
        digest.update(str(self.pending_observations).encode("utf-8"))
        for item in self._quarantine:
            for value in (item.state, item.intention, item.next_state):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            if item.confidence is not None:
                digest.update(
                    item.confidence.detach().cpu().contiguous().numpy().tobytes()
                )
        if self.sparse_evidence is not None:
            digest.update(self.sparse_evidence.digest().encode("utf-8"))
        for item in self._candidate_evidence:
            for value in (item.state, item.intention, item.next_state):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def state_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "model": self.model.state_payload(),
            "context_encoder": self.context_encoder.state_payload(),
            "contexts": [context.tolist() for context in self._contexts],
            "slot_ids": list(self._slot_ids),
            "next_slot_id": self._next_slot_id,
            "candidate_model": (
                None
                if self._candidate_model is None
                else self._candidate_model.state_payload()
            ),
            "candidate_context": (
                None
                if self._candidate_context is None
                else self._candidate_context.tolist()
            ),
            "pending": [
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
                for item in self._pending
            ],
            "quarantine": [
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
                for item in self._quarantine
            ],
            "candidate_evidence": [
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
                for item in self._candidate_evidence
            ],
            "sparse_evidence": (
                None
                if self.sparse_evidence is None
                else self.sparse_evidence.state_payload()
            ),
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        evidence_evaluator: nn.Module | None = None,
    ) -> ExternalFactoredTransitionRouter:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported factored router payload")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("factored router configuration is missing")
        serialized_evaluator = configuration.get("evidence_evaluator")
        if serialized_evaluator is not None and evidence_evaluator is None:
            raise ValueError("factored router payload requires its external evidence evaluator")
        model = ExternalFactoredTransitionModel.from_payload(payload["model"])
        encoder = ExternalTransitionContextEncoder.from_payload(payload["context_encoder"])
        router = cls(
            model,
            encoder,
            match_tolerance=float(configuration["match_tolerance"]),
            match_margin=float(configuration["match_margin"]),
            admission_observations=int(configuration["admission_observations"]),
            max_contexts=(
                None
                if configuration.get("max_contexts") is None
                else int(configuration["max_contexts"])
            ),
            auto_grow=bool(configuration.get("auto_grow", False)),
            residual_adaptation_updates=int(
                configuration.get("residual_adaptation_updates", 16)
            ),
            quarantine_capacity=int(configuration.get("quarantine_capacity", 0)),
            evidence_evaluator=evidence_evaluator,
            evidence_threshold=float(configuration.get("evidence_threshold", 0.5)),
            evidence_gate_min_evidence=int(
                configuration.get("evidence_gate_min_evidence", 0)
            ),
            committed_evidence_gate=bool(
                configuration.get("committed_evidence_gate", False)
            ),
            sparse_evidence=(
                None
                if payload.get("sparse_evidence") is None
                else ExternalSparseTransitionEvidenceIndex.from_payload(
                    payload["sparse_evidence"]
                )
            ),
        )
        router._contexts = [
            torch.tensor(values, dtype=torch.float32)
            for values in payload.get("contexts", [])
        ]
        router._slot_ids = [int(value) for value in payload.get("slot_ids", [])]
        router._next_slot_id = int(payload.get("next_slot_id", len(router._slot_ids)))
        candidate_model = payload.get("candidate_model")
        router._candidate_model = (
            None
            if candidate_model is None
            else ExternalFactoredTransitionModel.from_payload(candidate_model)
        )
        candidate_context = payload.get("candidate_context")
        router._candidate_context = (
            None
            if candidate_context is None
            else torch.tensor(candidate_context, dtype=torch.float32)
        )
        router._candidate_evidence = []
        for item in payload.get("candidate_evidence", []):
            router._candidate_evidence.append(
                ExternalTransitionObservation(
                    state=torch.tensor(item["state"], dtype=torch.float32),
                    intention=torch.tensor(item["intention"], dtype=torch.float32),
                    next_state=torch.tensor(item["next_state"], dtype=torch.float32),
                    confidence=(
                        None
                        if item.get("confidence") is None
                        else torch.tensor(item["confidence"], dtype=torch.float32)
                    ),
                )
            )
        router._pending = []
        for item in payload.get("pending", []):
            router._pending.append(
                ExternalTransitionObservation(
                    state=torch.tensor(item["state"], dtype=torch.float32),
                    intention=torch.tensor(item["intention"], dtype=torch.float32),
                    next_state=torch.tensor(item["next_state"], dtype=torch.float32),
                    confidence=(
                        None
                        if item.get("confidence") is None
                        else torch.tensor(item["confidence"], dtype=torch.float32)
                    ),
                )
            )
        router._quarantine = []
        for item in payload.get("quarantine", []):
            router._quarantine.append(
                ExternalTransitionObservation(
                    state=torch.tensor(item["state"], dtype=torch.float32),
                    intention=torch.tensor(item["intention"], dtype=torch.float32),
                    next_state=torch.tensor(item["next_state"], dtype=torch.float32),
                    confidence=(
                        None
                        if item.get("confidence") is None
                        else torch.tensor(item["confidence"], dtype=torch.float32)
                    ),
                )
            )
        if router.quarantined_observations > router.quarantine_capacity:
            raise ValueError("factored quarantine payload exceeds capacity")
        if payload.get("sha256") != router.digest():
            raise ValueError("factored router checksum mismatch")
        return router
