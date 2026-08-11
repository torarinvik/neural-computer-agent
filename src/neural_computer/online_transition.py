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
from collections.abc import Callable, Mapping, Sequence
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
EXTERNAL_TRANSITION_RANDOM_FEATURE_GROWTH_SCHEMA = (
    "neural-computer.external-transition-random-feature-growth.v1"
)

EXTERNAL_GOAL_EVALUATOR_STATISTICS_SCHEMA = (
    "neural-computer.external-goal-evaluator-statistics.v2"
)


class ExternalGoalEvaluatorStatistics(nn.Module):
    """Replay-free sufficient statistics for a graded opaque goal score.

    The component stores only normal-equation state. Its features are generic
    pairwise relations between standardized state and goal tensors; no task
    label or protocol field is introduced. Bounded distance features make the
    one-pass fit robust to small representation noise and prevent harmful
    extrapolation outside the observed goal range. The output is a logit so
    the planner can use it anywhere it accepts an ``ExternalGoalEvaluator``.
    """

    schema = EXTERNAL_GOAL_EVALUATOR_STATISTICS_SCHEMA

    def __init__(
        self,
        state_width: int,
        *,
        ridge: float = 1e-5,
        distance_clip: float = 0.1,
    ) -> None:
        super().__init__()
        if state_width < 1:
            raise ValueError("goal-evaluator statistics width must be positive")
        if ridge <= 0.0 or not math.isfinite(ridge):
            raise ValueError("goal-evaluator statistics ridge must be finite and positive")
        if distance_clip <= 0.0 or not math.isfinite(distance_clip):
            raise ValueError(
                "goal-evaluator statistics distance clip must be finite and positive"
            )
        self.state_width = int(state_width)
        self.ridge = float(ridge)
        self.distance_clip = float(distance_clip)
        self.feature_width = self.state_width * 2 + 1
        self.register_buffer(
            "normal_matrix",
            torch.eye(self.feature_width, dtype=torch.float32) * self.ridge,
        )
        self.register_buffer(
            "target_vector",
            torch.zeros(self.feature_width, 1, dtype=torch.float32),
        )
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.long))

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "ridge": self.ridge,
            "distance_clip": self.distance_clip,
            "representation": "opaque_goal_distance_features_v2",
            "score": "graded_verifier_logit_v1",
            "updates": "single_pass_weighted_normal_equations_v1",
            "storage": "normal_and_target_statistics_only_v1",
        }

    def _features(
        self,
        state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.state_width:
            raise ValueError("goal-evaluator state has the wrong shape")
        if goal_state.ndim != 2 or goal_state.shape != state.shape:
            raise ValueError("goal-evaluator goal has the wrong shape")
        if not bool(torch.isfinite(state).all()) or not bool(
            torch.isfinite(goal_state).all()
        ):
            raise ValueError("goal-evaluator inputs must be finite")
        difference = (state - goal_state).abs()
        bounded_difference = difference.clamp(max=self.distance_clip)
        ones = torch.ones(state.shape[0], 1, device=state.device, dtype=state.dtype)
        return torch.cat((difference, bounded_difference, ones), dim=-1)

    def observe(
        self,
        state: torch.Tensor,
        goal_state: torch.Tensor,
        outcome: torch.Tensor,
    ) -> None:
        """Consume graded verifier outcomes once without retaining rows."""

        features = self._features(state, goal_state).to(self.normal_matrix)
        if outcome.shape not in ((state.shape[0],), (state.shape[0], 1)):
            raise ValueError("goal-evaluator outcomes must match the batch")
        values = outcome.reshape(-1).to(features)
        if not bool(torch.isfinite(values).all()) or bool(
            torch.any(values < 0.0) or torch.any(values > 1.0)
        ):
            raise ValueError("goal-evaluator outcomes must lie in [0, 1]")
        logits = torch.logit(values.clamp(1e-4, 1.0 - 1e-4)).unsqueeze(-1)
        self.normal_matrix.add_(features.transpose(0, 1) @ features)
        self.target_vector.add_(features.transpose(0, 1) @ logits)
        self.sample_count.add_(state.shape[0])

    def _weights(self) -> torch.Tensor:
        return torch.linalg.solve(self.normal_matrix, self.target_vector)

    def forward(self, state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        return (self._features(state, goal_state).to(self.normal_matrix) @ self._weights()).squeeze(-1)

    def loss(
        self,
        state: torch.Tensor,
        goal_state: torch.Tensor,
        outcome: torch.Tensor,
    ) -> torch.Tensor:
        values = outcome.reshape(-1).to(self.normal_matrix)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            self(state, goal_state).to(values), values
        )

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
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
    ) -> ExternalGoalEvaluatorStatistics:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported goal-evaluator statistics payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("goal-evaluator statistics payload is incomplete")
        model = cls(
            int(configuration["state_width"]),
            ridge=float(configuration["ridge"]),
            distance_clip=float(configuration["distance_clip"]),
        )
        current = model.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("goal-evaluator statistics state names differ")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("goal-evaluator statistics state is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("goal-evaluator statistics state is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("goal-evaluator statistics state is not finite")
            normalized[name] = value.detach().clone()
        model.load_state_dict(normalized, strict=True)
        if payload.get("sha256") != model.digest():
            raise ValueError("goal-evaluator statistics checksum mismatch")
        return model


EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_SCHEMA = (
    "neural-computer.external-goal-representation-alignment.v1"
)
EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_VERIFICATION_SCHEMA = (
    "neural-computer.external-goal-representation-alignment-verification.v1"
)


@dataclass(frozen=True)
class ExternalGoalRepresentationAlignmentReceipt:
    """Auditable held-out decision for an alignment candidate."""

    accepted: bool
    source_width: int
    target_width: int
    query_count: int
    max_heldout_mse: float
    alignment_digest: str
    heldout_digest: str
    reason: str
    schema: str = EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_VERIFICATION_SCHEMA

    def validate(self) -> ExternalGoalRepresentationAlignmentReceipt:
        if self.schema != EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_VERIFICATION_SCHEMA:
            raise ValueError("unsupported goal alignment verification schema")
        if min(self.source_width, self.target_width, self.query_count) < 1:
            raise ValueError("goal alignment verification dimensions are invalid")
        if self.max_heldout_mse < 0.0 or not math.isfinite(self.max_heldout_mse):
            raise ValueError("goal alignment held-out error is invalid")
        for name, value in (
            ("alignment_digest", self.alignment_digest),
            ("heldout_digest", self.heldout_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"goal alignment verification {name} is missing")
        return self


class ExternalGoalRepresentationAlignmentStatistics(nn.Module):
    """Replay-free linear alignment from a replacement goal basis.

    The adapter consumes paired new/old representation tensors once and keeps
    only normal-equation state. It is deliberately separate from the goal
    evaluator: a frontend can be replaced while the old verifier memory stays
    frozen and retains its original representation contract.
    """

    schema = EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_SCHEMA

    def __init__(
        self,
        source_width: int,
        target_width: int,
        *,
        ridge: float = 1e-5,
    ) -> None:
        super().__init__()
        if min(source_width, target_width) < 1:
            raise ValueError("goal alignment widths must be positive")
        if ridge <= 0.0 or not math.isfinite(ridge):
            raise ValueError("goal alignment ridge must be finite and positive")
        self.source_width = int(source_width)
        self.target_width = int(target_width)
        self.ridge = float(ridge)
        self.feature_width = self.source_width + 1
        self.register_buffer(
            "normal_matrix",
            torch.eye(self.feature_width, dtype=torch.float32) * self.ridge,
        )
        self.register_buffer(
            "target_matrix",
            torch.zeros(self.feature_width, self.target_width, dtype=torch.float32),
        )
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.long))

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "ridge": self.ridge,
            "representation": "opaque_linear_alignment_with_bias_v1",
            "updates": "single_pass_weighted_normal_equations_v1",
            "storage": "normal_and_target_statistics_only_v1",
        }

    def _features(self, source: torch.Tensor) -> torch.Tensor:
        if source.ndim != 2 or source.shape[-1] != self.source_width:
            raise ValueError("goal alignment source has the wrong shape")
        if not bool(torch.isfinite(source).all()):
            raise ValueError("goal alignment source must be finite")
        ones = torch.ones(source.shape[0], 1, device=source.device, dtype=source.dtype)
        return torch.cat((source, ones), dim=-1)

    def observe(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Consume paired replacement/source representations once."""

        features = self._features(source).to(self.normal_matrix)
        if target.ndim != 2 or target.shape != (source.shape[0], self.target_width):
            raise ValueError("goal alignment target has the wrong shape")
        values = target.to(features)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("goal alignment target must be finite")
        self.normal_matrix.add_(features.transpose(0, 1) @ features)
        self.target_matrix.add_(features.transpose(0, 1) @ values)
        self.sample_count.add_(source.shape[0])

    def _weights(self) -> torch.Tensor:
        return torch.linalg.solve(self.normal_matrix, self.target_matrix)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self._features(source).to(self.normal_matrix) @ self._weights()

    def verify_heldout(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        prediction_tolerance: float,
    ) -> ExternalGoalRepresentationAlignmentReceipt:
        """Check a held-out paired set without changing adapter state."""

        if prediction_tolerance < 0.0 or not math.isfinite(prediction_tolerance):
            raise ValueError("goal alignment prediction tolerance is invalid")
        self._features(source)
        if target.ndim != 2 or target.shape != (source.shape[0], self.target_width):
            raise ValueError("goal alignment held-out target has the wrong shape")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("goal alignment held-out target must be finite")
        with torch.no_grad():
            prediction = self(source)
            errors = (prediction - target.to(prediction)).square().mean(dim=-1)
        max_error = float(errors.max().detach())
        heldout_digest = hashlib.sha256(
            target.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        accepted = max_error <= prediction_tolerance
        return ExternalGoalRepresentationAlignmentReceipt(
            accepted=accepted,
            source_width=self.source_width,
            target_width=self.target_width,
            query_count=int(source.shape[0]),
            max_heldout_mse=max_error,
            alignment_digest=self.digest(),
            heldout_digest=heldout_digest,
            reason=(
                "held-out alignment behavior remained within tolerance"
                if accepted
                else "held-out alignment behavior exceeded tolerance"
            ),
        ).validate()

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
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
    ) -> ExternalGoalRepresentationAlignmentStatistics:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported goal alignment payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("goal alignment payload is incomplete")
        model = cls(
            int(configuration["source_width"]),
            int(configuration["target_width"]),
            ridge=float(configuration["ridge"]),
        )
        current = model.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("goal alignment state names differ")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("goal alignment state is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("goal alignment state is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("goal alignment state is not finite")
            normalized[name] = value.detach().clone()
        model.load_state_dict(normalized, strict=True)
        if payload.get("sha256") != model.digest():
            raise ValueError("goal alignment checksum mismatch")
        return model


EXTERNAL_GOAL_REPRESENTATION_RANDOM_FEATURE_ALIGNMENT_SCHEMA = (
    "neural-computer.external-goal-representation-random-feature-alignment.v1"
)
EXTERNAL_GOAL_REPRESENTATION_RANDOM_FEATURE_GROWTH_SCHEMA = (
    "neural-computer.external-goal-representation-random-feature-growth.v1"
)


@dataclass(frozen=True)
class ExternalGoalRepresentationRandomFeatureGrowthReceipt:
    """Auditable copy-on-write nonlinear alignment basis growth."""

    accepted: bool
    source_feature_width: int
    destination_feature_width: int
    max_retention_mse: float
    source_digest: str
    target_digest: str
    reason: str
    schema: str = EXTERNAL_GOAL_REPRESENTATION_RANDOM_FEATURE_GROWTH_SCHEMA

    def validate(self) -> ExternalGoalRepresentationRandomFeatureGrowthReceipt:
        if self.schema != EXTERNAL_GOAL_REPRESENTATION_RANDOM_FEATURE_GROWTH_SCHEMA:
            raise ValueError("unsupported random-feature alignment growth schema")
        if self.source_feature_width < 1 or self.destination_feature_width < 1:
            raise ValueError("random-feature alignment growth widths are invalid")
        if self.accepted and self.destination_feature_width <= self.source_feature_width:
            raise ValueError("accepted random-feature alignment did not grow")
        if self.max_retention_mse < 0.0 or not math.isfinite(self.max_retention_mse):
            raise ValueError("random-feature alignment retention error is invalid")
        for name, value in (
            ("source_digest", self.source_digest),
            ("target_digest", self.target_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"random-feature alignment growth {name} is missing")
        return self


class ExternalGoalRepresentationRandomFeatureAlignmentStatistics(nn.Module):
    """Replay-free nonlinear alignment with a frozen random feature basis."""

    schema = EXTERNAL_GOAL_REPRESENTATION_RANDOM_FEATURE_ALIGNMENT_SCHEMA

    def __init__(
        self,
        source_width: int,
        target_width: int,
        *,
        feature_width: int = 64,
        ridge: float = 1e-4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if min(source_width, target_width, feature_width) < 1:
            raise ValueError("random-feature alignment dimensions must be positive")
        if ridge <= 0.0 or not math.isfinite(ridge):
            raise ValueError("random-feature alignment ridge must be finite and positive")
        self.source_width = int(source_width)
        self.target_width = int(target_width)
        self.feature_width = int(feature_width)
        self.ridge = float(ridge)
        self.seed = int(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        self.register_buffer(
            "projection",
            torch.randn(
                self.source_width,
                self.feature_width,
                generator=generator,
                dtype=torch.float32,
            )
            / math.sqrt(self.source_width),
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
            torch.zeros(statistics_width, self.target_width, dtype=torch.float32),
        )
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.long))

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "feature_width": self.feature_width,
            "ridge": self.ridge,
            "seed": self.seed,
            "representation": "opaque_frozen_random_features_v1",
            "updates": "single_pass_weighted_normal_equations_v1",
            "storage": "frozen_features_and_sufficient_statistics_v1",
        }

    def _features(self, source: torch.Tensor) -> torch.Tensor:
        if source.ndim != 2 or source.shape[-1] != self.source_width:
            raise ValueError("random-feature alignment source has the wrong shape")
        if not bool(torch.isfinite(source).all()):
            raise ValueError("random-feature alignment source must be finite")
        values = source.to(self.projection)
        features = torch.cos(values @ self.projection + self.bias)
        return torch.cat((features, torch.ones(features.shape[0], 1)), dim=-1)

    def observe(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Consume paired nonlinear alignment evidence once."""

        features = self._features(source)
        if target.ndim != 2 or target.shape != (source.shape[0], self.target_width):
            raise ValueError("random-feature alignment target has the wrong shape")
        values = target.to(features)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("random-feature alignment target must be finite")
        self.normal_matrix.add_(features.transpose(0, 1) @ features)
        self.target_matrix.add_(features.transpose(0, 1) @ values)
        self.sample_count.add_(source.shape[0])

    def _weights(self) -> torch.Tensor:
        return torch.linalg.solve(self.normal_matrix, self.target_matrix)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self._features(source) @ self._weights()

    def _grow_features(self, destination_width: int) -> None:
        if not isinstance(destination_width, int):
            raise TypeError("random-feature alignment destination width must be an integer")
        if destination_width <= self.feature_width:
            raise ValueError("random-feature alignment destination must grow")
        source_width = self.feature_width
        added_width = destination_width - source_width
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + source_width * 100_003)
        projection = torch.randn(
            self.source_width,
            added_width,
            generator=generator,
            dtype=self.projection.dtype,
        ) / math.sqrt(self.source_width)
        bias = torch.rand(
            added_width,
            generator=generator,
            dtype=self.bias.dtype,
        ) * (2.0 * math.pi)
        self.projection = torch.cat((self.projection, projection), dim=-1)
        self.bias = torch.cat((self.bias, bias), dim=-1)

        old_normal = self.normal_matrix
        old_target = self.target_matrix
        new_statistics_width = destination_width + 1
        new_normal = torch.eye(
            new_statistics_width,
            dtype=old_normal.dtype,
            device=old_normal.device,
        ) * self.ridge
        new_normal[:source_width, :source_width] = old_normal[:source_width, :source_width]
        new_normal[:source_width, destination_width] = old_normal[
            :source_width, source_width
        ]
        new_normal[destination_width, :source_width] = old_normal[
            source_width, :source_width
        ]
        new_normal[destination_width, destination_width] = old_normal[
            source_width, source_width
        ]
        new_target = torch.zeros(
            new_statistics_width,
            self.target_width,
            dtype=old_target.dtype,
            device=old_target.device,
        )
        new_target[:source_width] = old_target[:source_width]
        new_target[destination_width] = old_target[source_width]
        self.normal_matrix = new_normal
        self.target_matrix = new_target
        self.feature_width = destination_width

    def grow_features_verified(
        self,
        destination_width: int,
        retention_source: torch.Tensor,
        *,
        retention_tolerance: float,
    ) -> ExternalGoalRepresentationRandomFeatureGrowthReceipt:
        """Grow copy-on-write only when old alignment behavior is retained."""

        if retention_tolerance < 0.0 or not math.isfinite(retention_tolerance):
            raise ValueError("random-feature alignment retention tolerance is invalid")
        if retention_source.ndim != 2 or retention_source.shape[-1] != self.source_width:
            raise ValueError("random-feature alignment retention source has the wrong shape")
        if not bool(torch.isfinite(retention_source).all()):
            raise ValueError("random-feature alignment retention source must be finite")
        source_feature_width = self.feature_width
        source_digest = self.digest()
        with torch.no_grad():
            source_prediction = self(retention_source)
        candidate = self.from_payload(self.state_payload())
        candidate._grow_features(destination_width)
        with torch.no_grad():
            candidate_prediction = candidate(retention_source)
            max_error = float(
                (candidate_prediction - source_prediction).square().mean(dim=-1).max()
            )
        if max_error > retention_tolerance:
            return ExternalGoalRepresentationRandomFeatureGrowthReceipt(
                accepted=False,
                source_feature_width=source_feature_width,
                destination_feature_width=source_feature_width,
                max_retention_mse=max_error,
                source_digest=source_digest,
                target_digest=source_digest,
                reason="candidate nonlinear basis failed retention verification",
            ).validate()
        target_digest = candidate.digest()
        self.feature_width = candidate.feature_width
        self.projection = candidate.projection
        self.bias = candidate.bias
        self.normal_matrix = candidate.normal_matrix
        self.target_matrix = candidate.target_matrix
        return ExternalGoalRepresentationRandomFeatureGrowthReceipt(
            accepted=True,
            source_feature_width=source_feature_width,
            destination_feature_width=destination_width,
            max_retention_mse=max_error,
            source_digest=source_digest,
            target_digest=target_digest,
            reason="retention-verified nonlinear alignment basis growth committed",
        ).validate()

    def verify_heldout(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        prediction_tolerance: float,
    ) -> ExternalGoalRepresentationAlignmentReceipt:
        if prediction_tolerance < 0.0 or not math.isfinite(prediction_tolerance):
            raise ValueError("random-feature alignment prediction tolerance is invalid")
        self._features(source)
        if target.ndim != 2 or target.shape != (source.shape[0], self.target_width):
            raise ValueError("random-feature alignment held-out target has the wrong shape")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("random-feature alignment held-out target must be finite")
        with torch.no_grad():
            errors = (self(source) - target.to(self.normal_matrix)).square().mean(dim=-1)
        max_error = float(errors.max().detach())
        heldout_digest = hashlib.sha256(
            target.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        accepted = max_error <= prediction_tolerance
        return ExternalGoalRepresentationAlignmentReceipt(
            accepted=accepted,
            source_width=self.source_width,
            target_width=self.target_width,
            query_count=int(source.shape[0]),
            max_heldout_mse=max_error,
            alignment_digest=self.digest(),
            heldout_digest=heldout_digest,
            reason=(
                "held-out nonlinear alignment remained within tolerance"
                if accepted
                else "held-out nonlinear alignment exceeded tolerance"
            ),
        ).validate()

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
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
    ) -> ExternalGoalRepresentationRandomFeatureAlignmentStatistics:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported random-feature alignment payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("random-feature alignment payload is incomplete")
        model = cls(
            int(configuration["source_width"]),
            int(configuration["target_width"]),
            feature_width=int(configuration["feature_width"]),
            ridge=float(configuration["ridge"]),
            seed=int(configuration["seed"]),
        )
        current = model.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("random-feature alignment state names differ")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("random-feature alignment state is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("random-feature alignment state is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("random-feature alignment state is not finite")
            normalized[name] = value.detach().clone()
        model.load_state_dict(normalized, strict=True)
        if payload.get("sha256") != model.digest():
            raise ValueError("random-feature alignment checksum mismatch")
        return model


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
            "regularization": "analytic_copy_on_write_ridge_reparameterization_v1",
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

    def predictor_weights(self) -> torch.Tensor:
        """Return the solved factual predictor for representation-aware storage."""

        return self._weights().detach().clone()

    def reparameterized_ridge(
        self,
        ridge: float,
    ) -> ExternalAffineTransitionStatistics:
        """Return a copy with a new ridge without replaying observations.

        The sufficient statistics contain the accumulated unregularized
        normal matrix plus the construction ridge.  Adjusting its diagonal is
        therefore an analytic regularization change, not another pass over
        stored examples.
        """

        if ridge <= 0.0 or not math.isfinite(ridge):
            raise ValueError("affine transition ridge must be finite and positive")
        candidate = self.from_payload(self.state_payload())
        delta = float(ridge) - candidate.ridge
        identity = torch.eye(
            candidate.normal_matrix.shape[0],
            device=candidate.normal_matrix.device,
            dtype=candidate.normal_matrix.dtype,
        )
        candidate.normal_matrix = candidate.normal_matrix + delta * identity
        candidate.ridge = float(ridge)
        return candidate

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


@dataclass(frozen=True)
class ExternalRandomFeatureGrowthReceipt:
    """Auditable result of verifier-gated nonlinear basis growth."""

    accepted: bool
    source_width: int
    destination_width: int
    content_digest_before: str
    content_digest_after: str
    reason: str
    schema: str = EXTERNAL_TRANSITION_RANDOM_FEATURE_GROWTH_SCHEMA

    def validate(self) -> ExternalRandomFeatureGrowthReceipt:
        if self.schema != EXTERNAL_TRANSITION_RANDOM_FEATURE_GROWTH_SCHEMA:
            raise ValueError("unsupported random-feature growth schema")
        if self.source_width < 1 or self.destination_width < self.source_width:
            raise ValueError("random-feature growth widths are invalid")
        if not self.content_digest_before or not self.content_digest_after:
            raise ValueError("random-feature growth digests are missing")
        if not self.reason:
            raise ValueError("random-feature growth reason is missing")
        if self.accepted and self.destination_width == self.source_width:
            raise ValueError("accepted random-feature growth did not grow")
        return self


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
            "regularization": "analytic_copy_on_write_ridge_reparameterization_v1",
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

    def predictor_weights(self) -> torch.Tensor:
        """Return the solved factual predictor for representation-aware storage."""

        return self._weights().detach().clone()

    def reparameterized_ridge(
        self,
        ridge: float,
    ) -> ExternalRandomFeatureTransitionStatistics:
        """Return a copy with a new ridge without replaying observations."""

        if ridge <= 0.0 or not math.isfinite(ridge):
            raise ValueError(
                "random-feature transition ridge must be finite and positive"
            )
        candidate = self.from_payload(self.state_payload())
        delta = float(ridge) - candidate.ridge
        identity = torch.eye(
            candidate.normal_matrix.shape[0],
            device=candidate.normal_matrix.device,
            dtype=candidate.normal_matrix.dtype,
        )
        candidate.normal_matrix = candidate.normal_matrix + delta * identity
        candidate.ridge = float(ridge)
        return candidate

    def _grow_features(self, destination_width: int) -> None:
        if not isinstance(destination_width, int):
            raise TypeError("random-feature destination width must be an integer")
        if destination_width <= self.feature_width:
            raise ValueError("random-feature destination width must grow")
        source_width = self.feature_width
        added_width = destination_width - source_width
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + source_width * 100003)
        input_width = self.state_width + self.intention_width
        projection = torch.randn(
            input_width,
            added_width,
            generator=generator,
            dtype=self.projection.dtype,
        ) / math.sqrt(input_width)
        bias = torch.rand(
            added_width,
            generator=generator,
            dtype=self.bias.dtype,
        ) * (2.0 * math.pi)
        self.projection = torch.cat((self.projection, projection), dim=-1)
        self.bias = torch.cat((self.bias, bias), dim=-1)

        old_normal = self.normal_matrix
        old_target = self.target_matrix
        new_statistics_width = destination_width + 1
        new_normal = torch.eye(
            new_statistics_width,
            dtype=old_normal.dtype,
            device=old_normal.device,
        ) * self.ridge
        new_normal[:source_width, :source_width] = old_normal[:source_width, :source_width]
        new_normal[:source_width, destination_width] = old_normal[:source_width, source_width]
        new_normal[destination_width, :source_width] = old_normal[source_width, :source_width]
        new_normal[destination_width, destination_width] = old_normal[source_width, source_width]
        new_target = torch.zeros(
            new_statistics_width,
            self.state_width,
            dtype=old_target.dtype,
            device=old_target.device,
        )
        new_target[:source_width] = old_target[:source_width]
        new_target[destination_width] = old_target[source_width]
        self.normal_matrix = new_normal
        self.target_matrix = new_target
        self.feature_width = destination_width

    def grow_features_verified(
        self,
        destination_width: int,
        retention_probe: Callable[[ExternalRandomFeatureTransitionStatistics], bool],
    ) -> ExternalRandomFeatureGrowthReceipt:
        """Expand the fixed basis only after proving old behavior is retained."""

        if not callable(retention_probe):
            raise TypeError("random-feature growth retention probe must be callable")
        source_width = self.feature_width
        before = self.digest()
        if not bool(retention_probe(self)):
            return ExternalRandomFeatureGrowthReceipt(
                accepted=False,
                source_width=source_width,
                destination_width=source_width,
                content_digest_before=before,
                content_digest_after=before,
                reason="pre-growth retention probe failed",
            ).validate()
        candidate = self.from_payload(self.state_payload())
        candidate._grow_features(destination_width)
        after = candidate.digest()
        if not bool(retention_probe(candidate)):
            return ExternalRandomFeatureGrowthReceipt(
                accepted=False,
                source_width=source_width,
                destination_width=source_width,
                content_digest_before=before,
                content_digest_after=before,
                reason="post-growth retention probe failed",
            ).validate()
        self.feature_width = candidate.feature_width
        self.projection = candidate.projection
        self.bias = candidate.bias
        self.normal_matrix = candidate.normal_matrix
        self.target_matrix = candidate.target_matrix
        return ExternalRandomFeatureGrowthReceipt(
            accepted=True,
            source_width=source_width,
            destination_width=destination_width,
            content_digest_before=before,
            content_digest_after=after,
            reason="retention-verified nonlinear basis growth committed",
        ).validate()

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
    heldout_observations: Sequence[ExternalTransitionObservation] | None = None,
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
    additional_observations = (
        () if heldout_observations is None else tuple(heldout_observations)
    )
    observations = (heldout_observation, *additional_observations)
    for observation in observations:
        observation.validate(
            state_width=int(first_model.state_width),
            intention_width=int(first_model.intention_width),
        )
    receipts: list[ExternalTransitionModelFamilyCandidateReceipt] = []
    for name, model in candidates.items():
        if not isinstance(model, nn.Module) or not hasattr(model, "loss"):
            raise TypeError("model-family candidates must be learned modules with loss()")
        heldout_error = max(
            float(model.loss(observation).detach()) for observation in observations
        )
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
                    "all held-out and retention-verified candidates accepted"
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
            "smallest all-held-out and retention-verified model family selected"
            if selected is not None
            else "no model-family candidate passed verification"
        ),
    ).validate()


__all__ = [
    "EXTERNAL_TRANSITION_AFFINE_STATISTICS_SCHEMA",
    "EXTERNAL_TRANSITION_MODEL_FAMILY_SELECTION_SCHEMA",
    "EXTERNAL_TRANSITION_RANDOM_FEATURE_GROWTH_SCHEMA",
    "EXTERNAL_TRANSITION_RANDOM_FEATURE_STATISTICS_SCHEMA",
    "ExternalAffineTransitionStatistics",
    "ExternalRandomFeatureGrowthReceipt",
    "ExternalRandomFeatureTransitionStatistics",
    "ExternalTransitionModelFamilyCandidateReceipt",
    "ExternalTransitionModelFamilySelection",
    "select_verified_transition_model_family",
]
