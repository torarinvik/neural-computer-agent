"""Generic learned transport policies.

These policies contain no modality names or task labels.  They can be trained
from scalar utility/outcome signals and are intentionally separate from the
controller's recurrent state so their causal contribution can be audited.
"""

from __future__ import annotations

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
