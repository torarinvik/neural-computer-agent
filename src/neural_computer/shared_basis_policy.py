"""Outcome-trained external policy for shared-basis compression candidates.

The policy ranks candidate representations from generic storage statistics.  It
never receives raw modality data, task labels, verifier targets, or memory
contents; the caller supplies candidate features derived from the opaque
learned value boundary.  Its proposal is advisory and must still pass the
memory backend's independent retention verifier before commit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

SHARED_BASIS_COMPRESSION_POLICY_SCHEMA = (
    "neural-computer.opaque-shared-basis-compression-policy.v1"
)
SHARED_BASIS_STRUCTURE_POLICY_SCHEMA = (
    "neural-computer.opaque-shared-basis-structure-policy.v1"
)


@dataclass(frozen=True)
class SharedBasisCompressionPolicyOutput:
    """Candidate logits emitted by the external compression policy."""

    logits: torch.Tensor


@dataclass(frozen=True)
class SharedBasisCompressionPlan:
    """One advisory candidate selection from opaque storage features."""

    candidate_index: int
    score: torch.Tensor


@dataclass(frozen=True)
class SharedBasisStructurePlan:
    """One advisory candidate selection from opaque value rows."""

    candidate_index: int
    score: torch.Tensor


class OpaqueSharedBasisCompressionPolicy(nn.Module):
    """Learn a generic rank/representation preference from scalar utility.

    Candidate rows are scored independently, so the policy accepts any runtime
    candidate count without resizing the controller.  The feature contract is
    deliberately storage-generic; a caller may include rank, reconstruction
    error, physical-size ratio, and other non-semantic memory statistics.
    """

    schema = SHARED_BASIS_COMPRESSION_POLICY_SCHEMA

    def __init__(
        self,
        *,
        feature_width: int,
        hidden: int = 64,
        learning_rate: float = 1e-2,
    ) -> None:
        super().__init__()
        if min(feature_width, hidden) < 1:
            raise ValueError("shared-basis policy dimensions must be positive")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("shared-basis policy learning rate must be positive")
        self.feature_width = int(feature_width)
        self.hidden = int(hidden)
        self.learning_rate = float(learning_rate)
        self.scorer = nn.Sequential(
            nn.Linear(self.feature_width, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "feature_width": self.feature_width,
            "hidden": self.hidden,
            "learning_rate": self.learning_rate,
            "features": "opaque_rank_error_storage_occupancy_statistics_v1",
            "updates": "single_scalar_verifier_utility_without_replay_v1",
            "proposal": "candidate_index_only_v1",
        }

    def _validate_features(self, features: torch.Tensor) -> None:
        if features.ndim != 3 or features.shape[-1] != self.feature_width:
            raise ValueError(
                "shared-basis policy features must have shape "
                f"[batch, candidates, {self.feature_width}]"
            )
        if features.shape[0] < 1 or features.shape[1] < 1:
            raise ValueError("shared-basis policy needs non-empty feature batches")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("shared-basis policy features must be finite")

    def forward(self, features: torch.Tensor) -> SharedBasisCompressionPolicyOutput:
        self._validate_features(features)
        logits = self.scorer(features).squeeze(-1)
        return SharedBasisCompressionPolicyOutput(logits=logits)

    @torch.no_grad()
    def propose(
        self,
        features: torch.Tensor,
        *,
        explore: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> SharedBasisCompressionPlan | tuple[SharedBasisCompressionPlan, ...]:
        """Return one candidate proposal per batch row."""

        if not isinstance(explore, bool):
            raise TypeError("shared-basis policy explore flag must be boolean")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("shared-basis policy temperature must be positive")
        logits = self(features).logits
        plans: list[SharedBasisCompressionPlan] = []
        for row in logits:
            if explore:
                probabilities = torch.softmax(row / temperature, dim=-1)
                index = int(
                    torch.multinomial(probabilities, 1, generator=generator)
                )
            else:
                index = int(row.argmax())
            plans.append(
                SharedBasisCompressionPlan(
                    candidate_index=index,
                    score=row[index].detach().clone(),
                )
            )
        return plans[0] if len(plans) == 1 else tuple(plans)

    def adaptation_step(
        self,
        features: torch.Tensor,
        plan: SharedBasisCompressionPlan,
        verifier_utility: float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        """Update from one external scalar utility without retaining replay."""

        self._validate_features(features)
        if features.shape[0] != 1:
            raise ValueError("shared-basis policy adaptation needs one feature bank")
        if not isinstance(plan, SharedBasisCompressionPlan):
            raise TypeError("shared-basis policy plan is invalid")
        candidate_count = features.shape[1]
        if not 0 <= plan.candidate_index < candidate_count:
            raise ValueError("shared-basis policy candidate index is invalid")
        if not math.isfinite(verifier_utility) or not 0.0 <= verifier_utility <= 1.0:
            raise ValueError("shared-basis policy utility must lie in [0, 1]")
        logits = self(features).logits[0]
        log_probability = torch.log_softmax(logits, dim=-1)[plan.candidate_index]
        loss = -(verifier_utility - 0.5) * log_probability
        selected_optimizer = optimizer
        if selected_optimizer is None:
            selected_optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate
            )
        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        return float(loss.detach())


class OpaqueSharedBasisStructurePolicy(nn.Module):
    """Learn compression preference from opaque values, not candidate errors.

    The policy computes a fixed-width singular-spectrum summary internally from
    the external value rows.  Candidate reconstruction errors are never passed
    through the ABI.  The summary is structural and permutation-invariant over
    rows; the candidate count and ranks remain runtime inputs.
    """

    schema = SHARED_BASIS_STRUCTURE_POLICY_SCHEMA

    def __init__(
        self,
        *,
        value_width: int,
        hidden: int = 64,
        max_spectral_bins: int = 8,
        learning_rate: float = 1e-2,
    ) -> None:
        super().__init__()
        if min(value_width, hidden, max_spectral_bins) < 1:
            raise ValueError("shared-basis structure policy dimensions are invalid")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("shared-basis structure policy learning rate is invalid")
        self.value_width = int(value_width)
        self.hidden = int(hidden)
        self.max_spectral_bins = int(max_spectral_bins)
        self.learning_rate = float(learning_rate)
        self.global_feature_width = 2 * self.max_spectral_bins + 3
        self.candidate_feature_width = self.global_feature_width + 5
        self.scorer = nn.Sequential(
            nn.Linear(self.candidate_feature_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "value_width": self.value_width,
            "hidden": self.hidden,
            "max_spectral_bins": self.max_spectral_bins,
            "learning_rate": self.learning_rate,
            "features": "opaque_singular_spectrum_row_permutation_invariant_v1",
            "forbidden_features": "precomputed_candidate_reconstruction_error_v1",
            "updates": "single_scalar_verifier_utility_without_replay_v1",
            "proposal": "candidate_index_only_v1",
        }

    def _validate_inputs(
        self,
        values: torch.Tensor,
        occupied: torch.Tensor,
        candidate_ranks: torch.Tensor,
    ) -> None:
        if values.ndim != 3 or values.shape[-1] != self.value_width:
            raise ValueError(
                "shared-basis structure values must have shape "
                f"[batch, rows, {self.value_width}]"
            )
        if occupied.shape != values.shape[:2] or occupied.dtype != torch.bool:
            raise ValueError("shared-basis structure occupancy has the wrong shape")
        if candidate_ranks.ndim != 1 or candidate_ranks.numel() < 1:
            raise ValueError("shared-basis structure candidate ranks are invalid")
        if candidate_ranks.dtype != torch.long:
            raise ValueError("shared-basis structure candidate ranks must be int64")
        if bool(torch.any(candidate_ranks < 1)) or bool(
            torch.any(candidate_ranks > self.value_width)
        ):
            raise ValueError("shared-basis structure candidate ranks are out of range")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("shared-basis structure values must be finite")
        if not bool(occupied.any(dim=-1).all()):
            raise ValueError("shared-basis structure needs an occupied row")

    def _candidate_features(
        self,
        values: torch.Tensor,
        occupied: torch.Tensor,
        candidate_ranks: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(values, occupied, candidate_ranks)
        masked_values = values * occupied.unsqueeze(-1).to(values.dtype)
        singular = torch.linalg.svdvals(masked_values)
        bins = min(self.max_spectral_bins, singular.shape[-1])
        singular = singular[..., :bins]
        total_energy = singular.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        normalized = singular / singular[..., :1].clamp_min(1e-6)
        energy = singular.square() / total_energy
        cumulative = energy.cumsum(dim=-1)
        normalized_padded = torch.zeros(
            values.shape[0], self.max_spectral_bins,
            device=values.device,
            dtype=values.dtype,
        )
        cumulative_padded = torch.ones_like(normalized_padded)
        normalized_padded[:, :bins] = normalized
        cumulative_padded[:, :bins] = cumulative
        row_count = occupied.sum(dim=-1, keepdim=True).to(values.dtype)
        occupied_fraction = row_count / values.shape[1]
        mean_norm = singular.square().sum(dim=-1, keepdim=True).sqrt() / row_count.sqrt().clamp_min(1.0)
        global_features = torch.cat(
            (
                normalized_padded,
                cumulative_padded,
                row_count / 16.0,
                occupied_fraction,
                mean_norm,
            ),
            dim=-1,
        )
        max_rank = candidate_ranks.max().to(values.dtype).clamp_min(1.0)
        rank_values = candidate_ranks.to(values.dtype)
        rank_indices = (candidate_ranks - 1).clamp_max(singular.shape[-1] - 1)
        rank_energy = cumulative[:, rank_indices]
        rank_energy = torch.where(
            candidate_ranks[None, :] > singular.shape[-1],
            torch.ones_like(rank_energy),
            rank_energy,
        )
        storage_fraction = (
            rank_values[None, :] * (self.value_width + row_count)
            / (row_count * self.value_width).clamp_min(1.0)
        )
        candidate_features = torch.stack(
            (
                rank_values[None, :].expand(values.shape[0], -1) / max_rank,
                rank_energy.expand(values.shape[0], -1),
                storage_fraction.expand(values.shape[0], -1),
                1.0 - storage_fraction.expand(values.shape[0], -1),
                rank_values[None, :].expand(values.shape[0], -1)
                / self.value_width,
            ),
            dim=-1,
        )
        return torch.cat(
            (
                global_features[:, None, :].expand(
                    -1, candidate_ranks.numel(), -1
                ),
                candidate_features,
            ),
            dim=-1,
        )

    def forward(
        self,
        values: torch.Tensor,
        occupied: torch.Tensor,
        candidate_ranks: torch.Tensor,
    ) -> SharedBasisCompressionPolicyOutput:
        features = self._candidate_features(values, occupied, candidate_ranks)
        return SharedBasisCompressionPolicyOutput(
            logits=self.scorer(features).squeeze(-1)
        )

    @torch.no_grad()
    def propose(
        self,
        values: torch.Tensor,
        occupied: torch.Tensor,
        candidate_ranks: torch.Tensor,
        *,
        explore: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> SharedBasisStructurePlan | tuple[SharedBasisStructurePlan, ...]:
        if not isinstance(explore, bool):
            raise TypeError("shared-basis structure explore flag must be boolean")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("shared-basis structure temperature is invalid")
        logits = self(values, occupied, candidate_ranks).logits
        plans: list[SharedBasisStructurePlan] = []
        for row in logits:
            if explore:
                probabilities = torch.softmax(row / temperature, dim=-1)
                index = int(
                    torch.multinomial(probabilities, 1, generator=generator)
                )
            else:
                index = int(row.argmax())
            plans.append(
                SharedBasisStructurePlan(
                    candidate_index=index,
                    score=row[index].detach().clone(),
                )
            )
        return plans[0] if len(plans) == 1 else tuple(plans)

    def adaptation_step(
        self,
        values: torch.Tensor,
        occupied: torch.Tensor,
        candidate_ranks: torch.Tensor,
        plan: SharedBasisStructurePlan,
        verifier_utility: float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        self._validate_inputs(values, occupied, candidate_ranks)
        if values.shape[0] != 1:
            raise ValueError("shared-basis structure adaptation needs one bank")
        if not isinstance(plan, SharedBasisStructurePlan):
            raise TypeError("shared-basis structure plan is invalid")
        if not 0 <= plan.candidate_index < candidate_ranks.numel():
            raise ValueError("shared-basis structure candidate index is invalid")
        if not math.isfinite(verifier_utility) or not 0.0 <= verifier_utility <= 1.0:
            raise ValueError("shared-basis structure utility must lie in [0, 1]")
        logits = self(values, occupied, candidate_ranks).logits[0]
        log_probability = torch.log_softmax(logits, dim=-1)[plan.candidate_index]
        loss = -(verifier_utility - 0.5) * log_probability
        selected_optimizer = optimizer
        if selected_optimizer is None:
            selected_optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate
            )
        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        return float(loss.detach())


__all__ = [
    "SHARED_BASIS_COMPRESSION_POLICY_SCHEMA",
    "SHARED_BASIS_STRUCTURE_POLICY_SCHEMA",
    "OpaqueSharedBasisCompressionPolicy",
    "OpaqueSharedBasisStructurePolicy",
    "SharedBasisCompressionPlan",
    "SharedBasisCompressionPolicyOutput",
    "SharedBasisStructurePlan",
]
