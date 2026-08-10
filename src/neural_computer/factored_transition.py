"""Factored factual transition models with external residual memory."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .world_model import (
    ExternalTransitionContextEncoder,
    ExternalTransitionMemory,
    ExternalTransitionModel,
    ExternalTransitionObservation,
)

EXTERNAL_FACTORED_TRANSITION_MODEL_SCHEMA = (
    "neural-computer.external-factored-transition-model.v1"
)
EXTERNAL_FACTORED_TRANSITION_ROUTER_SCHEMA = (
    "neural-computer.external-factored-transition-router.v1"
)
EXTERNAL_FACTORED_TRANSITION_PROMOTION_SCHEMA = (
    "neural-computer.external-factored-transition-promotion.v1"
)


class ExternalFactoredTransitionModel(nn.Module):
    """A frozen factual base plus append-only context-local residual facts.

    The base learns reusable transition structure.  Once frozen, new regime
    evidence is written only to the external residual memory addressed by an
    opaque context.  Planning sees the sum of the base prediction and a
    residual correction when an exact residual fact is available; it never
    sees a task policy or protocol-specific action meaning.
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
    ) -> None:
        super().__init__()
        if min(state_width, intention_width, context_width, hidden_width) < 1:
            raise ValueError("factored transition dimensions must be positive")
        self.state_width = int(state_width)
        self.intention_width = int(intention_width)
        self.context_width = int(context_width)
        self.hidden_width = int(hidden_width)
        self.base = ExternalTransitionModel(
            self.state_width,
            self.intention_width,
            hidden_width=self.hidden_width,
        )
        self.residual_memory = ExternalTransitionMemory(
            self.state_width,
            self.intention_width,
            context_width=self.context_width,
            read_match_threshold=residual_read_match_threshold,
            write_match_threshold=residual_write_match_threshold,
        )

    def configuration(self) -> dict[str, int | float | str | dict[str, object]]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "context_width": self.context_width,
            "hidden_width": self.hidden_width,
            "representation": "frozen_shared_base_plus_opaque_context_residual_v1",
            "behavior": "derived_by_external_search_not_stored_policy_v1",
            "base": self.base.configuration(),
            "residual_memory": self.residual_memory.configuration(),
        }

    @property
    def base_frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.base.parameters())

    @property
    def residual_record_count(self) -> int:
        return self.residual_memory.record_count

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
        """Store one verified current-regime correction without base updates."""

        residual = self._residual_observation(observation)
        context_batch = context
        if context_batch.ndim == 1:
            context_batch = context_batch.unsqueeze(0).expand(
                observation.state.shape[0], -1
            )
        return self.residual_memory.write(residual, context=context_batch)

    def predict_with_context(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        base_prediction = self.base(state, intention)
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
        return digest.hexdigest()

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
            "base_state": self._state_payload(self.base),
            "residual_state": self._state_payload(self.residual_memory),
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

    def validate(self, *, context_width: int) -> FactoredTransitionRouteResult:
        if self.schema != EXTERNAL_FACTORED_TRANSITION_ROUTER_SCHEMA:
            raise ValueError("unsupported factored transition route schema")
        if self.status not in {"pending", "matched", "staged", "ambiguous"}:
            raise ValueError("unsupported factored transition route status")
        if self.slot_id is not None and self.slot_id < 0:
            raise ValueError("factored transition route slot ID is invalid")
        if self.context is not None and (
            self.context.ndim != 1 or self.context.shape[0] != context_width
        ):
            raise ValueError("factored transition route context has wrong shape")
        if self.pending_observations < 0:
            raise ValueError("factored transition pending count is invalid")
        return self


@dataclass(frozen=True)
class FactoredTransitionPromotionReceipt:
    """Verifier receipt for committing an isolated residual candidate."""

    accepted: bool
    slot_id: int | None
    heldout_error: float
    reason: str
    schema: str = EXTERNAL_FACTORED_TRANSITION_PROMOTION_SCHEMA

    def validate(self) -> FactoredTransitionPromotionReceipt:
        if self.schema != EXTERNAL_FACTORED_TRANSITION_PROMOTION_SCHEMA:
            raise ValueError("unsupported factored transition promotion schema")
        if self.slot_id is not None and self.slot_id < 0:
            raise ValueError("factored promotion slot ID is invalid")
        if not torch.isfinite(torch.tensor(self.heldout_error)) or self.heldout_error < 0.0:
            raise ValueError("factored promotion heldout error is invalid")
        if not self.reason:
            raise ValueError("factored promotion reason is missing")
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
        self.model = model
        self.context_encoder = context_encoder
        self.match_tolerance = float(match_tolerance)
        self.match_margin = float(match_margin)
        self.admission_observations = int(admission_observations)
        self.max_contexts = max_contexts
        self._contexts: list[torch.Tensor] = []
        self._slot_ids: list[int] = []
        self._next_slot_id = 0
        self._pending: list[ExternalTransitionObservation] = []
        self._candidate_model: ExternalFactoredTransitionModel | None = None
        self._candidate_context: torch.Tensor | None = None

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
    ) -> tuple[int, float, float] | None:
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
            errors.append(
                float((prediction - observation.next_state).square().mean().detach())
            )
        ordering = sorted(range(len(errors)), key=lambda index: (errors[index], index))
        best = ordering[0]
        margin = (
            float("inf") if len(ordering) == 1 else errors[ordering[1]] - errors[best]
        )
        if errors[best] > self.match_tolerance or margin < self.match_margin:
            return None
        return self._slot_ids[best], errors[best], margin

    def _stage_candidate(self) -> FactoredTransitionRouteResult:
        evidence = self._merge(self._pending)
        context = self.context_encoder.encode_observation(evidence).detach()
        candidate = ExternalFactoredTransitionModel.from_payload(
            self.model.state_payload()
        )
        candidate.write_residual(evidence, context=context)
        self._candidate_model = candidate
        self._candidate_context = context
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
            self._candidate_model.write_residual(
                observation,
                context=self._candidate_context,
            )
            return FactoredTransitionRouteResult(
                status="staged",
                slot_id=None,
                context=self._candidate_context.clone(),
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        matched = self._route_existing(observation)
        if matched is not None:
            slot_id, _error, _margin = matched
            return FactoredTransitionRouteResult(
                status="matched",
                slot_id=slot_id,
                context=self._contexts[self._slot_ids.index(slot_id)].clone(),
                pending_observations=0,
            ).validate(context_width=self.model.context_width)
        self._pending.append(self._clone_observation(observation))
        if self.pending_observations < self.admission_observations:
            return FactoredTransitionRouteResult(
                status="pending",
                slot_id=None,
                context=None,
                pending_observations=self.pending_observations,
            ).validate(context_width=self.model.context_width)
        if self.max_contexts is not None and len(self._contexts) >= self.max_contexts:
            return FactoredTransitionRouteResult(
                status="ambiguous",
                slot_id=None,
                context=None,
                pending_observations=self.pending_observations,
            ).validate(context_width=self.model.context_width)
        return self._stage_candidate()

    def promote_staged_candidate(
        self,
        heldout: ExternalTransitionObservation,
        retention_probe: Any,
        *,
        prediction_tolerance: float = 0.05,
    ) -> FactoredTransitionPromotionReceipt:
        """Commit only a candidate that passes current and retention probes."""

        if self._candidate_model is None or self._candidate_context is None:
            raise RuntimeError("factored router has no staged candidate")
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
        accepted = error <= prediction_tolerance and bool(retention_probe(self._candidate_model))
        if accepted:
            slot_id = self._next_slot_id
            self._next_slot_id += 1
            self._contexts.append(self._candidate_context.detach().clone())
            self._slot_ids.append(slot_id)
            self.model = self._candidate_model
            self._candidate_model = None
            self._candidate_context = None
            return FactoredTransitionPromotionReceipt(
                accepted=True,
                slot_id=slot_id,
                heldout_error=error,
                reason="candidate passed factual and retention probes",
            ).validate()
        self._candidate_model = None
        self._candidate_context = None
        return FactoredTransitionPromotionReceipt(
            accepted=False,
            slot_id=None,
            heldout_error=error,
            reason="candidate failed factual or retention probe",
        ).validate()

    def configuration(self) -> dict[str, int | float | str | None]:
        return {
            "schema": self.schema,
            "match_tolerance": self.match_tolerance,
            "match_margin": self.match_margin,
            "admission_observations": self.admission_observations,
            "max_contexts": self.max_contexts,
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
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalFactoredTransitionRouter:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported factored router payload")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("factored router configuration is missing")
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
        if payload.get("sha256") != router.digest():
            raise ValueError("factored router checksum mismatch")
        return router
