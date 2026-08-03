"""Generic, modality-independent arrival prediction for adaptive waiting."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .amodal_runtime import AmodalEventWindowStatus

AMODAL_WAIT_FEATURES = 5


def arrival_features(
    status: AmodalEventWindowStatus,
    history: Sequence[bool],
    *,
    deadline: float,
) -> torch.Tensor:
    """Encode only transport metadata, never event payload or stream names."""
    if deadline <= 0:
        raise ValueError("deadline must be positive")
    if not status.present:
        raise ValueError("a wait-policy status needs at least one stream")
    history_size = max(1, len(history))
    arrivals = sum(bool(value) for value in history)
    gaps = []
    for index, value in enumerate(reversed(history), start=1):
        if value:
            gaps.append(index)
            break
    last_gap = gaps[0] if gaps else history_size + 1
    return torch.tensor(
        [
            sum(status.present) / len(status.present),
            min(status.age / deadline, 1.0),
            arrivals / history_size,
            min(last_gap / (history_size + 1), 1.0),
            float(status.complete),
        ],
        dtype=torch.float32,
    )


class AmodalArrivalPredictor(nn.Module):
    """Predict whether an absent event will arrive before the deadline."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.hidden = hidden
        self.network = nn.Sequential(
            nn.Linear(AMODAL_WAIT_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != AMODAL_WAIT_FEATURES:
            raise ValueError(
                f"features must have shape [batch, {AMODAL_WAIT_FEATURES}]"
            )
        return torch.sigmoid(self.network(features)).squeeze(-1)


class AmodalWaitDecisionPolicy(nn.Module):
    """Metadata-only policy whose probability means ``wait``."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.hidden = hidden
        self.network = nn.Sequential(
            nn.Linear(AMODAL_WAIT_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != AMODAL_WAIT_FEATURES:
            raise ValueError(
                f"features must have shape [batch, {AMODAL_WAIT_FEATURES}]"
            )
        return torch.sigmoid(self.network(features)).squeeze(-1)
