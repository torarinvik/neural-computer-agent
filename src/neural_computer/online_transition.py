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
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .world_model import (
    EXTERNAL_TRANSITION_MODEL_FAMILY_SELECTION_SCHEMA,
    ExternalTransitionObservation,
)

EXTERNAL_TRANSITION_AFFINE_STATISTICS_SCHEMA = (
    "neural-computer.external-transition-affine-statistics.v1"
)
EXTERNAL_TRANSITION_RANDOM_FEATURE_STATISTICS_SCHEMA = (
    "neural-computer.external-transition-random-feature-statistics.v1"
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


class ExternalRandomFeatureTransitionStatistics(nn.Module):
    """Replay-free nonlinear transition memory with a frozen feature map.

    The random feature projection is fixed at construction and persisted as
    part of the external artifact. Only normal-equation sufficient statistics
    change during adaptation, so individual observations are not retained or
    replayed. This is a bounded nonlinear basis, not unrestricted computation.
    """

    schema = EXTERNAL_TRANSITION_RANDOM_FEATURE_STATISTICS_SCHEMA

    def __init__(
        self,
        state_width: int,
        intention_width: int,
        *,
        feature_width: int = 128,
        ridge: float = 1e-5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if min(state_width, intention_width, feature_width) < 1:
            raise ValueError("random-feature transition dimensions must be positive")
        if ridge <= 0.0 or not math.isfinite(ridge):
            raise ValueError("random-feature transition ridge must be finite and positive")
        self.state_width = int(state_width)
        self.intention_width = int(intention_width)
        self.feature_width = int(feature_width)
        self.ridge = float(ridge)
        self.seed = int(seed)
        input_width = self.state_width + self.intention_width
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        self.register_buffer(
            "projection",
            torch.randn(
                input_width,
                self.feature_width,
                generator=generator,
                dtype=torch.float32,
            )
            / math.sqrt(input_width),
        )
        self.register_buffer(
            "bias",
            torch.rand(
                self.feature_width,
                generator=generator,
                dtype=torch.float32,
            )
            * (2.0 * math.pi),
        )
        statistics_width = self.feature_width + 1
        self.register_buffer(
            "normal_matrix",
            torch.eye(statistics_width, dtype=torch.float32) * self.ridge,
        )
        self.register_buffer(
            "target_matrix",
            torch.zeros(statistics_width, self.state_width, dtype=torch.float32),
        )
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.long))

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "feature_width": self.feature_width,
            "ridge": self.ridge,
            "seed": self.seed,
            "representation": "opaque_frozen_random_features_v1",
            "updates": "single_pass_weighted_normal_equations_v1",
            "storage": "frozen_features_and_sufficient_statistics_v1",
        }

    def _features(
        self,
        state: torch.Tensor,
        intention: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.state_width:
            raise ValueError("random-feature transition state has the wrong shape")
        if intention.ndim != 2 or intention.shape[-1] != self.intention_width:
            raise ValueError("random-feature transition intention has the wrong shape")
        if state.shape[0] != intention.shape[0]:
            raise ValueError("random-feature transition batches differ")
        if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(intention).all()):
            raise ValueError("random-feature transition inputs must be finite")
        inputs = torch.cat((state, intention), dim=-1).to(self.projection)
        features = torch.cos(inputs @ self.projection + self.bias)
        return torch.cat((features, torch.ones(features.shape[0], 1)), dim=-1)

    def _validate_observation(self, observation: ExternalTransitionObservation) -> None:
        observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )

    def observe(self, observation: ExternalTransitionObservation) -> None:
        """Consume verified evidence once without retaining the rows."""

        self._validate_observation(observation)
        features = self._features(observation.state, observation.intention)
        targets = observation.next_state.to(self.normal_matrix)
        if observation.confidence is None:
            weights = torch.ones(features.shape[0], device=features.device)
        else:
            weights = observation.confidence.reshape(-1).to(features)
        if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0)):
            raise ValueError("random-feature transition confidence is invalid")
        self.normal_matrix.add_(features.transpose(0, 1) @ (features * weights[:, None]))
        self.target_matrix.add_(features.transpose(0, 1) @ (targets * weights[:, None]))
        self.sample_count.add_(observation.state.shape[0])

    def _weights(self) -> torch.Tensor:
        return torch.linalg.solve(self.normal_matrix, self.target_matrix)

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        return self._features(state, intention) @ self._weights()

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
        for name, value in sorted(self.state_dict().items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
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
    ) -> ExternalRandomFeatureTransitionStatistics:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported random-feature transition payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("random-feature transition payload is incomplete")
        model = cls(
            int(configuration["state_width"]),
            int(configuration["intention_width"]),
            feature_width=int(configuration["feature_width"]),
            ridge=float(configuration["ridge"]),
            seed=int(configuration["seed"]),
        )
        current = model.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("random-feature transition state names differ")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("random-feature transition state is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("random-feature transition state is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("random-feature transition state is not finite")
            normalized[name] = value.detach().clone()
        model.load_state_dict(normalized, strict=True)
        if payload.get("sha256") != model.digest():
            raise ValueError("random-feature transition checksum mismatch")
        return model


@dataclass(frozen=True)
class ExternalTransitionModelFamilyCandidateReceipt:
    """Verifier result for one independently trained opaque model candidate."""

    model_family: str
    accepted: bool
    heldout_error: float
    storage_bytes: int
    candidate_digest: str
    reason: str

    def validate(self) -> ExternalTransitionModelFamilyCandidateReceipt:
        if not self.model_family:
            raise ValueError("model-family candidate name must be nonempty")
        if not math.isfinite(self.heldout_error) or self.heldout_error < 0.0:
            raise ValueError("model-family held-out error is invalid")
        if self.storage_bytes < 1:
            raise ValueError("model-family candidate storage must be positive")
        if not self.candidate_digest or not self.reason:
            raise ValueError("model-family candidate receipt is incomplete")
        return self


@dataclass(frozen=True)
class ExternalTransitionModelFamilySelection:
    """Auditable smallest-accepted candidate selection."""

    accepted: bool
    selected_family: str | None
    candidates: tuple[ExternalTransitionModelFamilyCandidateReceipt, ...]
    reason: str
    schema: str = EXTERNAL_TRANSITION_MODEL_FAMILY_SELECTION_SCHEMA

    def validate(self) -> ExternalTransitionModelFamilySelection:
        if self.schema != EXTERNAL_TRANSITION_MODEL_FAMILY_SELECTION_SCHEMA:
            raise ValueError("unsupported model-family selection schema")
        if not self.candidates:
            raise ValueError("model-family selection must include candidates")
        for candidate in self.candidates:
            candidate.validate()
        accepted = [candidate.model_family for candidate in self.candidates if candidate.accepted]
        if self.accepted != bool(accepted):
            raise ValueError("model-family selection acceptance is inconsistent")
        if self.selected_family not in accepted and self.selected_family is not None:
            raise ValueError("selected model family was not accepted")
        if self.accepted and self.selected_family is None:
            raise ValueError("accepted model-family selection lacks a winner")
        if not self.reason:
            raise ValueError("model-family selection reason is missing")
        return self


def _candidate_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    schema = str(getattr(model, "schema", type(model).__name__))
    digest.update(schema.encode("utf-8"))
    for name, value in sorted(model.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def select_verified_transition_model_family(
    candidates: Mapping[str, nn.Module],
    heldout_observation: ExternalTransitionObservation,
    *,
    prediction_tolerance: float = 0.05,
    retention_probe: Any = None,
) -> ExternalTransitionModelFamilySelection:
    """Choose the smallest opaque candidate that passes held-out verification.

    Candidate training is deliberately outside this function. Each candidate
    must already have been adapted from the same verified stream. Selection
    only measures held-out factual prediction and an optional retention probe;
    it does not infer a semantic model family or mutate any candidate.
    """

    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("model-family candidates must be a nonempty mapping")
    if any(not isinstance(name, str) or not name for name in candidates):
        raise ValueError("model-family candidate names must be nonempty strings")
    if prediction_tolerance < 0.0 or not math.isfinite(prediction_tolerance):
        raise ValueError("model-family prediction tolerance is invalid")
    if retention_probe is not None and not callable(retention_probe):
        raise TypeError("model-family retention probe must be callable")
    first_model = next(iter(candidates.values()))
    if not isinstance(first_model, nn.Module) or not hasattr(first_model, "loss"):
        raise TypeError("model-family candidates must be learned modules with loss()")
    heldout_observation.validate(
        state_width=int(first_model.state_width),
        intention_width=int(first_model.intention_width),
    )
    receipts: list[ExternalTransitionModelFamilyCandidateReceipt] = []
    for name, model in candidates.items():
        if not isinstance(model, nn.Module) or not hasattr(model, "loss"):
            raise TypeError("model-family candidates must be learned modules with loss()")
        heldout_error = float(model.loss(heldout_observation).detach())
        storage_bytes = sum(
            value.numel() * value.element_size() for value in model.state_dict().values()
        )
        retained = retention_probe is None or bool(retention_probe(model))
        accepted = heldout_error <= prediction_tolerance and retained
        receipts.append(
            ExternalTransitionModelFamilyCandidateReceipt(
                model_family=name,
                accepted=accepted,
                heldout_error=heldout_error,
                storage_bytes=storage_bytes,
                candidate_digest=_candidate_digest(model),
                reason=(
                    "held-out and retention-verified candidate accepted"
                    if accepted
                    else (
                        "held-out candidate prediction failed"
                        if heldout_error > prediction_tolerance
                        else "candidate retention probe failed"
                    )
                ),
            ).validate()
        )
    accepted = [receipt for receipt in receipts if receipt.accepted]
    selected = (
        min(accepted, key=lambda receipt: (receipt.storage_bytes, receipt.heldout_error, receipt.model_family))
        if accepted
        else None
    )
    return ExternalTransitionModelFamilySelection(
        accepted=selected is not None,
        selected_family=None if selected is None else selected.model_family,
        candidates=tuple(receipts),
        reason=(
            "smallest held-out and retention-verified model family selected"
            if selected is not None
            else "no model-family candidate passed verification"
        ),
    ).validate()


__all__ = [
    "EXTERNAL_TRANSITION_AFFINE_STATISTICS_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_FAMILY_SELECTION_SCHEMA",
    "EXTERNAL_TRANSITION_RANDOM_FEATURE_STATISTICS_SCHEMA",
    "ExternalAffineTransitionStatistics",
    "ExternalRandomFeatureTransitionStatistics",
    "ExternalTransitionModelFamilyCandidateReceipt",
    "ExternalTransitionModelFamilySelection",
    "select_verified_transition_model_family",
]
