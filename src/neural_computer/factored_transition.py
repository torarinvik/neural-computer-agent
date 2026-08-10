"""Factored factual transition models with external residual memory."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .world_model import (
    ExternalTransitionMemory,
    ExternalTransitionModel,
    ExternalTransitionObservation,
)

EXTERNAL_FACTORED_TRANSITION_MODEL_SCHEMA = (
    "neural-computer.external-factored-transition-model.v1"
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
