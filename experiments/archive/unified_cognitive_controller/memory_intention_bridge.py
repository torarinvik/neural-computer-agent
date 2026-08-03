"""Small reusable bridge from generic memory rows to amodal intentions."""
from __future__ import annotations

import torch
from torch import nn


class MemoryCodeBridge(nn.Module):
    """Compress a controller-created memory row into a two-state code."""

    def __init__(self, memory_width: int, hidden_width: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(memory_width),
            nn.Linear(memory_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 2),
        )

    def forward(self, memory_value: torch.Tensor) -> torch.Tensor:
        return self.network(memory_value)


class MemoryActionComposer(nn.Module):
    """Combine a query intention and a memory code into protocol logits."""

    def __init__(
            self, intention_width: int, code_width: int = 2,
            hidden_width: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(intention_width + code_width),
            nn.Linear(intention_width + code_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 2),
        )

    def forward(
            self, intention: torch.Tensor, code: torch.Tensor
            ) -> torch.Tensor:
        return self.network(torch.cat((intention, code), dim=-1))


class MemoryIntentionReader(nn.Module):
    """Translate a generic memory row into an amodal intention residual."""

    def __init__(self, memory_width: int, intention_width: int,
                 hidden_width: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(memory_width + intention_width),
            nn.Linear(memory_width + intention_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, intention_width),
        )
        # Before training, the optional reader is an exact no-op.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
            self, intention: torch.Tensor, memory_value: torch.Tensor
            ) -> torch.Tensor:
        return self.network(torch.cat((intention, memory_value), dim=-1))
