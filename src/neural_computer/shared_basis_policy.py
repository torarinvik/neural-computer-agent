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


@dataclass(frozen=True)
class SharedBasisCompressionPolicyOutput:
    """Candidate logits emitted by the external compression policy."""

    logits: torch.Tensor


@dataclass(frozen=True)
class SharedBasisCompressionPlan:
    """One advisory candidate selection from opaque storage features."""

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


__all__ = [
    "SHARED_BASIS_COMPRESSION_POLICY_SCHEMA",
    "OpaqueSharedBasisCompressionPolicy",
    "SharedBasisCompressionPlan",
    "SharedBasisCompressionPolicyOutput",
]
