"""Generic learned transport policies.

These policies contain no modality names or task labels.  They can be trained
from scalar utility/outcome signals and are intentionally separate from the
controller's recurrent state so their causal contribution can be audited.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .interface import EventTokenWindow


class EventReliabilityPolicy(nn.Module):
    """Estimate per-event reliability from latent content and transport facts."""

    def __init__(
        self,
        event_width: int,
        *,
        source_key_width: int = 0,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        if min(event_width, hidden) < 1 or source_key_width < 0:
            raise ValueError("reliability policy dimensions are invalid")
        self.event_width = event_width
        self.source_key_width = source_key_width
        input_width = event_width + 5 + source_key_width
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def features(self, window: EventTokenWindow) -> torch.Tensor:
        window.validate(width=self.event_width, source_key_width=self.source_key_width)
        present = window.present.to(window.payload.dtype)
        count = present.sum(dim=1, keepdim=True).clamp_min(1.0)
        reference = (window.payload * present.unsqueeze(-1)).sum(dim=1) / count
        agreement = torch.nn.functional.cosine_similarity(
            window.payload, reference.unsqueeze(1), dim=-1
        ) * present
        transport = torch.stack(
            [
                window.confidence,
                window.age,
                window.duration,
                window.timestamp_present.to(window.payload.dtype),
                agreement,
            ],
            dim=-1,
        )
        parts = [window.payload, transport]
        if self.source_key_width:
            assert window.source_key is not None
            parts.append(window.source_key)
        return torch.cat(parts, dim=-1)

    def forward(self, window: EventTokenWindow) -> torch.Tensor:
        trust = torch.sigmoid(self.network(self.features(window)).squeeze(-1))
        return torch.where(window.present, trust, torch.zeros_like(trust))


class EventWaitPolicy(nn.Module):
    """Predict whether an incomplete timestamp window should keep waiting."""

    feature_width = 5

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError("wait-policy hidden width must be positive")
        self.network = nn.Sequential(
            nn.Linear(self.feature_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": "neural-computer.wait-policy.v1",
            "hidden": self.network[0].out_features,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_width:
            raise ValueError(f"wait features must have shape [batch, {self.feature_width}]")
        return torch.sigmoid(self.network(features)).squeeze(-1)

    @classmethod
    def features(
        cls,
        *,
        age: torch.Tensor,
        present_fraction: torch.Tensor,
        complete: torch.Tensor,
        arrival_count: torch.Tensor,
        arrival_delta: torch.Tensor,
    ) -> torch.Tensor:
        values = [age, present_fraction, complete, arrival_count, arrival_delta]
        shaped = [value.reshape(-1, 1).to(dtype=torch.float32) for value in values]
        return torch.cat(shaped, dim=-1)


EVENT_WAIT_STATISTICS_SCHEMA = "neural-computer.event-wait-statistics.v1"


class EventWaitStatistics(nn.Module):
    """Replay-free external state for learned delay/absence decisions.

    The feature map is fixed and generic: quantized transport values are
    represented by one-hot main effects and pairwise interactions.  Learning
    updates only ridge-regression normal equations, so scalar outcomes can be
    consumed once without retaining examples or optimizer state.  Output is
    the estimated probability that waiting is useful; the window buffer keeps
    its existing ``< 0.5`` release convention.
    """

    schema = EVENT_WAIT_STATISTICS_SCHEMA
    feature_width = EventWaitPolicy.feature_width

    def __init__(
        self,
        *,
        bin_count: int = 4,
        ridge: float = 1e-2,
        age_scale: float = 4.0,
        arrival_count_scale: float = 16.0,
        outcome_scale: float = 4.0,
        minimum_context_observations: int = 1,
    ) -> None:
        super().__init__()
        if bin_count < 2:
            raise ValueError("wait-statistics bin count must be at least two")
        if ridge <= 0.0 or not math.isfinite(ridge):
            raise ValueError("wait-statistics ridge must be finite and positive")
        if age_scale <= 0.0 or not math.isfinite(age_scale):
            raise ValueError("wait-statistics age scale must be finite and positive")
        if arrival_count_scale <= 0.0 or not math.isfinite(arrival_count_scale):
            raise ValueError(
                "wait-statistics arrival-count scale must be finite and positive"
            )
        if outcome_scale <= 0.0 or not math.isfinite(outcome_scale):
            raise ValueError(
                "wait-statistics outcome scale must be finite and positive"
            )
        if minimum_context_observations < 1:
            raise ValueError(
                "wait-statistics minimum context observations must be positive"
            )
        self.bin_count = int(bin_count)
        self.ridge = float(ridge)
        self.age_scale = float(age_scale)
        self.arrival_count_scale = float(arrival_count_scale)
        self.outcome_scale = float(outcome_scale)
        self.minimum_context_observations = int(minimum_context_observations)
        self.register_buffer(
            "normal_matrix",
            torch.eye(self.basis_width, dtype=torch.float32) * self.ridge,
        )
        self.register_buffer(
            "target_vector",
            torch.zeros(self.basis_width, dtype=torch.float32),
        )
        self.register_buffer(
            "context_counts",
            torch.zeros(self.bin_count**self.feature_width, dtype=torch.long),
        )
        self.register_buffer("sample_count", torch.zeros((), dtype=torch.long))

    @property
    def basis_width(self) -> int:
        main_effects = self.feature_width * self.bin_count
        pairwise_effects = (
            self.feature_width
            * (self.feature_width - 1)
            // 2
            * self.bin_count
            * self.bin_count
        )
        return 1 + main_effects + pairwise_effects

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "bin_count": self.bin_count,
            "ridge": self.ridge,
            "age_scale": self.age_scale,
            "arrival_count_scale": self.arrival_count_scale,
            "outcome_scale": self.outcome_scale,
            "minimum_context_observations": self.minimum_context_observations,
            "feature_width": self.feature_width,
            "basis_width": self.basis_width,
            "representation": "quantized_transport_pairwise_basis_v1",
            "updates": "single_pass_ridge_sufficient_statistics_v1",
            "storage": "normal_and_target_statistics_only_v1",
            "target": "scalar_wait_utility_v1",
        }

    def _validate_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_width:
            raise ValueError(
                f"wait-statistics features must have shape [batch, {self.feature_width}]"
            )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("wait-statistics features must be finite")
        if bool(torch.any(features[:, 0] < 0)) or bool(
            torch.any(features[:, 1] < 0) or torch.any(features[:, 1] > 1)
        ):
            raise ValueError("wait-statistics age and presence features are invalid")
        if bool(torch.any(features[:, 2] < 0)) or bool(
            torch.any(features[:, 2] > 1)
        ):
            raise ValueError("wait-statistics completeness features are invalid")
        if bool(torch.any(features[:, 3] < 0)) or bool(
            torch.any(features[:, 4] < 0)
        ):
            raise ValueError("wait-statistics transport counts are invalid")
        return features

    def _indices(self, features: torch.Tensor) -> torch.Tensor:
        values = self._validate_features(features).to(self.normal_matrix)
        normalized = torch.stack(
            (
                values[:, 0].clamp_max(self.age_scale) / self.age_scale,
                values[:, 1].clamp(0.0, 1.0),
                values[:, 2].clamp(0.0, 1.0),
                (
                    torch.log1p(values[:, 3].clamp_min(0.0))
                    / math.log1p(self.arrival_count_scale)
                ).clamp(0.0, 1.0),
                values[:, 4].clamp_max(self.age_scale) / self.age_scale,
            ),
            dim=-1,
        )
        return torch.floor(normalized * self.bin_count).long().clamp_max(
            self.bin_count - 1
        )

    def _basis(self, features: torch.Tensor) -> torch.Tensor:
        indices = self._indices(features)
        one_hot = torch.nn.functional.one_hot(
            indices,
            num_classes=self.bin_count,
        ).to(self.normal_matrix)
        parts = [
            torch.ones(
                features.shape[0],
                1,
                device=self.normal_matrix.device,
                dtype=self.normal_matrix.dtype,
            ),
            one_hot.reshape(features.shape[0], -1),
        ]
        for first in range(self.feature_width):
            for second in range(first + 1, self.feature_width):
                parts.append(
                    (
                        one_hot[:, first, :, None]
                        * one_hot[:, second, None, :]
                    ).reshape(features.shape[0], -1)
                )
        basis = torch.cat(parts, dim=-1)
        if basis.shape[1] != self.basis_width:
            raise RuntimeError("wait-statistics basis width is inconsistent")
        return basis

    def _context_ids(self, features: torch.Tensor) -> torch.Tensor:
        indices = self._indices(features)
        context_ids = torch.zeros(
            indices.shape[0],
            dtype=torch.long,
            device=indices.device,
        )
        for column in range(self.feature_width):
            context_ids = context_ids * self.bin_count + indices[:, column]
        return context_ids

    def _weights(self) -> torch.Tensor:
        return torch.linalg.solve(self.normal_matrix, self.target_vector)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        basis = self._basis(features)
        probability = torch.sigmoid(basis @ self._weights())
        context_ids = self._context_ids(features)
        counts = self.context_counts.to(context_ids.device)[context_ids]
        known = counts >= self.minimum_context_observations
        return torch.where(known, probability, torch.full_like(probability, 0.5))

    def loss(self, features: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
        if outcome.shape not in ((features.shape[0],), (features.shape[0], 1)):
            raise ValueError("wait-statistics outcomes must match the batch")
        targets = outcome.reshape(-1).to(self.normal_matrix)
        if not bool(torch.isfinite(targets).all()) or bool(
            torch.any(targets < 0) or torch.any(targets > 1)
        ):
            raise ValueError("wait-statistics outcomes must lie in [0, 1]")
        return torch.nn.functional.binary_cross_entropy(
            self(features).clamp(1e-6, 1.0 - 1e-6),
            targets,
        )

    def observe(
        self,
        features: torch.Tensor,
        outcome: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> None:
        """Consume scalar wait utility once without retaining the feature row."""

        basis = self._basis(features)
        if outcome.shape not in ((features.shape[0],), (features.shape[0], 1)):
            raise ValueError("wait-statistics outcomes must match the batch")
        targets = outcome.reshape(-1).to(self.normal_matrix)
        if not bool(torch.isfinite(targets).all()) or bool(
            torch.any(targets < 0) or torch.any(targets > 1)
        ):
            raise ValueError("wait-statistics outcomes must lie in [0, 1]")
        if weight is None:
            weights = torch.ones(features.shape[0], device=basis.device)
        else:
            if weight.shape not in ((features.shape[0],), (features.shape[0], 1)):
                raise ValueError("wait-statistics weights must match the batch")
            weights = weight.reshape(-1).to(basis)
        if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0)):
            raise ValueError("wait-statistics weights must be finite and non-negative")
        self.normal_matrix.add_(basis.transpose(0, 1) @ (basis * weights[:, None]))
        # Fit signed utility so a verified "do not wait" outcome supplies
        # negative evidence instead of merely leaving an unseen probability at
        # the neutral prior.
        signed_targets = targets.mul(2.0).sub(1.0).mul(self.outcome_scale)
        self.target_vector.add_(basis.transpose(0, 1) @ (signed_targets * weights))
        context_ids = self._context_ids(features).to(self.context_counts.device)
        self.context_counts.index_add_(
            0,
            context_ids,
            torch.ones(
                features.shape[0],
                device=self.context_counts.device,
                dtype=torch.long,
            ),
        )
        self.sample_count.add_(features.shape[0])

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr(self.configuration()).encode("utf-8"))
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

    payload = state_payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EventWaitStatistics:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported wait-statistics payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("wait-statistics payload is incomplete")
        model = cls(
            bin_count=int(configuration["bin_count"]),
            ridge=float(configuration["ridge"]),
            age_scale=float(configuration["age_scale"]),
            arrival_count_scale=float(configuration["arrival_count_scale"]),
            outcome_scale=float(configuration["outcome_scale"]),
            minimum_context_observations=int(
                configuration["minimum_context_observations"]
            ),
        )
        current = model.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("wait-statistics state names differ")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("wait-statistics state is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("wait-statistics state is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("wait-statistics state is not finite")
            normalized[name] = value.detach().clone()
        model.load_state_dict(normalized, strict=True)
        if payload.get("sha256") != model.digest():
            raise ValueError("wait-statistics checksum mismatch")
        return model
