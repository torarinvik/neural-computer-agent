"""Replay-free sufficient-statistics transition memory.

This module is an intentionally narrow external-memory primitive. It learns an
affine mapping from opaque state/intention tensors to opaque next-state tensors
by accumulating weighted normal-equation statistics. It never stores or
replays individual observations, so it is useful as a pressure test for the
one-pass continual-learning boundary. It is not a replacement for the
general nonlinear transition model.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .world_model import ExternalTransitionObservation

EXTERNAL_TRANSITION_AFFINE_STATISTICS_SCHEMA = (
    "neural-computer.external-transition-affine-statistics.v1"
)


class ExternalAffineTransitionStatistics(nn.Module):
    """Compact online memory for an opaque affine transition function."""

    schema = EXTERNAL_TRANSITION_AFFINE_STATISTICS_SCHEMA

    def __init__(
        self,
        state_width: int,
        intention_width: int,
        *,
        ridge: float = 1e-5,
    ) -> None:
        super().__init__()
        if min(state_width, intention_width) < 1:
            raise ValueError("affine transition dimensions must be positive")
        if ridge <= 0.0 or not math.isfinite(ridge):
            raise ValueError("affine transition ridge must be finite and positive")
        self.state_width = int(state_width)
        self.intention_width = int(intention_width)
        self.ridge = float(ridge)
        feature_width = self.state_width + self.intention_width + 1
        self.register_buffer(
            "normal_matrix",
            torch.eye(feature_width, dtype=torch.float32) * self.ridge,
        )
        self.register_buffer(
            "target_matrix",
            torch.zeros(feature_width, self.state_width, dtype=torch.float32),
        )
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.long))

    @property
    def feature_width(self) -> int:
        return self.state_width + self.intention_width + 1

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "ridge": self.ridge,
            "representation": "opaque_affine_sufficient_statistics_v1",
            "updates": "single_pass_weighted_normal_equations_v1",
            "storage": "normal_and_target_matrices_only_v1",
        }

    def _features(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.state_width:
            raise ValueError("affine transition state has the wrong shape")
        if intention.ndim != 2 or intention.shape[-1] != self.intention_width:
            raise ValueError("affine transition intention has the wrong shape")
        if state.shape[0] != intention.shape[0]:
            raise ValueError("affine transition state and intention batches differ")
        if not bool(torch.isfinite(state).all()) or not bool(
            torch.isfinite(intention).all()
        ):
            raise ValueError("affine transition inputs must be finite")
        ones = torch.ones(state.shape[0], 1, device=state.device, dtype=state.dtype)
        return torch.cat((state, intention, ones), dim=-1)

    def _validate_observation(self, observation: ExternalTransitionObservation) -> None:
        observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )

    def observe(self, observation: ExternalTransitionObservation) -> None:
        """Consume verified evidence once without retaining the rows."""

        self._validate_observation(observation)
        features = self._features(observation.state, observation.intention).to(
            self.normal_matrix
        )
        targets = observation.next_state.to(self.normal_matrix)
        if observation.confidence is None:
            weights = torch.ones(features.shape[0], device=features.device)
        else:
            weights = observation.confidence.reshape(-1).to(features)
        if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0)):
            raise ValueError("affine transition confidence must be finite and non-negative")
        self.normal_matrix.add_(features.transpose(0, 1) @ (features * weights[:, None]))
        self.target_matrix.add_(features.transpose(0, 1) @ (targets * weights[:, None]))
        self.sample_count.add_(observation.state.shape[0])

    def _weights(self) -> torch.Tensor:
        return torch.linalg.solve(self.normal_matrix, self.target_matrix)

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        features = self._features(state, intention).to(self.normal_matrix)
        return features @ self._weights()

    def loss(self, observation: ExternalTransitionObservation) -> torch.Tensor:
        self._validate_observation(observation)
        prediction = self(observation.state, observation.intention)
        errors = (prediction - observation.next_state.to(prediction)).square().mean(dim=-1)
        if observation.confidence is None:
            return errors.mean()
        weights = observation.confidence.reshape(-1).to(errors)
        return (errors * weights).sum() / weights.sum().clamp_min(1e-12)

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        digest.update(str(self.state_width).encode("utf-8"))
        digest.update(str(self.intention_width).encode("utf-8"))
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

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

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalAffineTransitionStatistics:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported affine transition statistics payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("affine transition statistics payload is incomplete")
        model = cls(
            int(configuration["state_width"]),
            int(configuration["intention_width"]),
            ridge=float(configuration["ridge"]),
        )
        current = model.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("affine transition statistics state names differ")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("affine transition statistics state is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("affine transition statistics state is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("affine transition statistics state is not finite")
            normalized[name] = value.detach().clone()
        model.load_state_dict(normalized, strict=True)
        if payload.get("sha256") != model.digest():
            raise ValueError("affine transition statistics checksum mismatch")
        return model


__all__ = [
    "EXTERNAL_TRANSITION_AFFINE_STATISTICS_SCHEMA",
    "ExternalAffineTransitionStatistics",
]
