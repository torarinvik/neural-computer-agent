"""External raw-value regime-boundary policy.

This policy is deliberately outside the controller.  It compares opaque
current and incoming value banks through permutation-invariant structural
summaries and emits only a keep/replace decision.  A scalar verifier utility
is the sole learning signal; the policy never receives task labels,
reconstruction errors, or a hand-written regime identifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

REGIME_CHANGE_POLICY_SCHEMA = "neural-computer.opaque-regime-change-policy.v1"


@dataclass(frozen=True)
class RegimeChangePolicyOutput:
    """Keep/replace logits emitted by the external detector."""

    logits: torch.Tensor


@dataclass(frozen=True)
class RegimeChangePlan:
    """One advisory keep/replace proposal."""

    replace: bool
    score: torch.Tensor


class OpaqueRegimeChangePolicy(nn.Module):
    """Learn whether an incoming opaque bank warrants a replacement."""

    schema = REGIME_CHANGE_POLICY_SCHEMA

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
            raise ValueError("regime policy dimensions must be positive")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("regime policy learning rate must be positive")
        self.value_width = int(value_width)
        self.hidden = int(hidden)
        self.max_spectral_bins = int(max_spectral_bins)
        self.learning_rate = float(learning_rate)
        self.summary_width = 2 * self.max_spectral_bins + 2
        self.feature_width = 2 * self.summary_width + 7
        self.scorer = nn.Sequential(
            nn.Linear(self.feature_width, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 2),
        )

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "value_width": self.value_width,
            "hidden": self.hidden,
            "max_spectral_bins": self.max_spectral_bins,
            "learning_rate": self.learning_rate,
            "features": "opaque_spectral_cross_bank_structure_v1",
            "forbidden_features": (
                "task_labels_regime_ids_candidate_reconstruction_error_v1"
            ),
            "updates": "single_scalar_verifier_utility_without_replay_v1",
            "proposal": "keep_or_replace_v1",
        }

    def _validate_bank(
        self,
        values: torch.Tensor,
        occupied: torch.Tensor,
    ) -> None:
        if values.ndim != 3 or values.shape[-1] != self.value_width:
            raise ValueError(
                "regime policy values must have shape "
                f"[batch, rows, {self.value_width}]"
            )
        if occupied.shape != values.shape[:2] or occupied.dtype != torch.bool:
            raise ValueError("regime policy occupancy has the wrong shape")
        if values.shape[0] < 1 or not bool(occupied.any(dim=-1).all()):
            raise ValueError("regime policy needs non-empty banks")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("regime policy values must be finite")

    def _summary(
        self,
        values: torch.Tensor,
        occupied: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_bank(values, occupied)
        masked = values * occupied.unsqueeze(-1).to(values.dtype)
        singular = torch.linalg.svdvals(masked)
        bins = min(self.max_spectral_bins, singular.shape[-1])
        singular = singular[..., :bins]
        energy = singular.square() / singular.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        cumulative = energy.cumsum(dim=-1)
        normalized = singular / singular[..., :1].clamp_min(1e-6)
        normalized_padded = torch.zeros(
            values.shape[0], self.max_spectral_bins,
            dtype=values.dtype,
            device=values.device,
        )
        cumulative_padded = torch.ones_like(normalized_padded)
        normalized_padded[:, :bins] = normalized
        cumulative_padded[:, :bins] = cumulative
        row_count = occupied.sum(dim=-1, keepdim=True).to(values.dtype)
        mean_norm = singular.square().sum(dim=-1, keepdim=True).sqrt() / row_count.sqrt().clamp_min(1.0)
        return torch.cat(
            (
                normalized_padded,
                cumulative_padded,
                row_count / float(self.value_width),
                mean_norm,
            ),
            dim=-1,
        )

    def _cross_summary(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
    ) -> torch.Tensor:
        current_norm = current_values / current_values.square().sum(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        incoming_norm = incoming_values / incoming_values.square().sum(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        similarity = torch.abs(current_norm @ incoming_norm.transpose(-1, -2))
        mask = current_occupied.unsqueeze(-1) & incoming_occupied.unsqueeze(-2)
        count = mask.sum(dim=(-1, -2)).to(current_values.dtype)
        safe_count = count.to(torch.long).clamp_min(1)
        selected = similarity.masked_fill(~mask, float("inf")).reshape(
            current_values.shape[0], -1
        ).sort(dim=-1).values
        indices = torch.stack(
            (
                (safe_count - 1) // 4,
                (safe_count - 1) // 2,
                (safe_count - 1) * 3 // 4,
            ),
            dim=-1,
        )
        quantiles = selected.gather(1, indices)
        quantiles = torch.where(count.unsqueeze(-1) > 0, quantiles, torch.zeros_like(quantiles))
        weighted = similarity * mask.to(similarity.dtype)
        mean = weighted.sum(dim=(-1, -2)) / count.clamp_min(1.0)
        second = weighted.square().sum(dim=(-1, -2)) / count.clamp_min(1.0)
        std = (second - mean.square()).clamp_min(0.0).sqrt()
        maximum = weighted.masked_fill(~mask, 0.0).amax(dim=(-1, -2))
        current_count = current_occupied.sum(dim=-1).to(current_values.dtype)
        incoming_count = incoming_occupied.sum(dim=-1).to(current_values.dtype)
        return torch.cat(
            (
                mean.unsqueeze(-1),
                std.unsqueeze(-1),
                maximum.unsqueeze(-1),
                quantiles,
                (incoming_count / current_count.clamp_min(1.0)).unsqueeze(-1),
            ),
            dim=-1,
        )

    def _features(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_bank(current_values, current_occupied)
        self._validate_bank(incoming_values, incoming_occupied)
        if current_values.shape[0] != incoming_values.shape[0]:
            raise ValueError("regime policy banks must have matching batches")
        return torch.cat(
            (
                self._summary(current_values, current_occupied),
                self._summary(incoming_values, incoming_occupied),
                self._cross_summary(
                    current_values,
                    current_occupied,
                    incoming_values,
                    incoming_occupied,
                ),
            ),
            dim=-1,
        )

    def forward(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
    ) -> RegimeChangePolicyOutput:
        logits = self.scorer(
            self._features(
                current_values,
                current_occupied,
                incoming_values,
                incoming_occupied,
            )
        )
        return RegimeChangePolicyOutput(logits=logits)

    @torch.no_grad()
    def propose(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
        *,
        explore: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> RegimeChangePlan | tuple[RegimeChangePlan, ...]:
        if not isinstance(explore, bool):
            raise TypeError("regime policy explore flag must be boolean")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("regime policy temperature must be positive")
        logits = self(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
        ).logits
        plans: list[RegimeChangePlan] = []
        for row in logits:
            if explore:
                probabilities = torch.softmax(row / temperature, dim=-1)
                index = int(torch.multinomial(probabilities, 1, generator=generator))
            else:
                index = int(row.argmax())
            plans.append(
                RegimeChangePlan(
                    replace=index == 1,
                    score=row[index].detach().clone(),
                )
            )
        return plans[0] if len(plans) == 1 else tuple(plans)

    def adaptation_step(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
        plan: RegimeChangePlan,
        verifier_utility: float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        self._validate_bank(current_values, current_occupied)
        self._validate_bank(incoming_values, incoming_occupied)
        if current_values.shape[0] != 1 or incoming_values.shape[0] != 1:
            raise ValueError("regime policy adaptation needs one bank pair")
        if not isinstance(plan, RegimeChangePlan):
            raise TypeError("regime policy plan is invalid")
        if not math.isfinite(verifier_utility) or not 0.0 <= verifier_utility <= 1.0:
            raise ValueError("regime policy utility must lie in [0, 1]")
        logits = self(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
        ).logits[0]
        index = int(plan.replace)
        log_probability = torch.log_softmax(logits, dim=-1)[index]
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
    "REGIME_CHANGE_POLICY_SCHEMA",
    "OpaqueRegimeChangePolicy",
    "RegimeChangePlan",
    "RegimeChangePolicyOutput",
]
