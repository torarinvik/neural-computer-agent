"""Task-agnostic routing over opaque external-memory addresses.

The router is a memory-side component. It receives learned controller query
vectors, opaque candidate address rows, an attempted row, and a scalar
outcome during training. It does not receive task identifiers, semantic
labels, or correct unattempted actions, and it does not add a reasoning path
to the frozen controller.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class OpaqueAddressRouter(nn.Module):
    """Permutation-equivariant scorer for a variable set of memory rows."""

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError("width and hidden must be positive")
        self.width = int(width)
        self.net = nn.Sequential(
            nn.Linear(width * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        """Return one route score per candidate row."""
        if query.ndim != 2 or query.shape[1] != self.width:
            raise ValueError("query must have shape [batch, width]")
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        if (
            keys.ndim != 3
            or keys.shape[0] != query.shape[0]
            or keys.shape[2] != self.width
            or keys.shape[1] < 1
        ):
            raise ValueError(
                "keys must have shape [batch, rows, width] or [rows, width]"
            )
        query_rows = query.unsqueeze(1).expand(-1, keys.shape[1], -1)
        pair = torch.cat(
            (query_rows, keys, (query_rows - keys).abs(), query_rows * keys),
            dim=-1,
        )
        return self.net(pair).squeeze(-1)


class FactorizedOpaqueAddressRouter(nn.Module):
    """Learned query/key addressing with a permutation-equivariant score.

    The query and each opaque memory key are encoded independently into a
    shared latent space, then matched by a scaled dot product.  This gives a
    memory-side learner a direct way to discover a reusable address relation
    from scalar attempted-row outcomes without assigning meaning to key
    coordinates or adding a candidate-specific reasoning branch.
    """

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError("width and hidden must be positive")
        self.width = int(width)
        self.hidden = int(hidden)
        self.query_encoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        """Return one learned compatibility score per candidate row."""
        if query.ndim != 2 or query.shape[1] != self.width:
            raise ValueError("query must have shape [batch, width]")
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        if (
            keys.ndim != 3
            or keys.shape[0] != query.shape[0]
            or keys.shape[2] != self.width
            or keys.shape[1] < 1
        ):
            raise ValueError(
                "keys must have shape [batch, rows, width] or [rows, width]"
            )
        query_latent = self.query_encoder(query)
        key_latent = self.key_encoder(keys)
        return torch.einsum("bh,brh->br", query_latent, key_latent) / self.hidden**0.5


def attempted_outcome_loss(
    logits: torch.Tensor,
    attempted: torch.Tensor,
    outcomes: torch.Tensor,
) -> torch.Tensor:
    """Train from only the attempted row and its scalar binary outcome."""
    if logits.ndim != 2 or attempted.ndim != 1 or outcomes.ndim != 1:
        raise ValueError("invalid router transition shapes")
    if logits.shape[0] != attempted.shape[0] or outcomes.shape != attempted.shape:
        raise ValueError("router transition batch lengths differ")
    if not bool(((attempted >= 0) & (attempted < logits.shape[1])).all()):
        raise ValueError("attempted row is out of range")
    if not bool(((outcomes == 0) | (outcomes == 1)).all()):
        raise ValueError("outcomes must be binary scalar rewards")
    selected = logits.gather(1, attempted[:, None]).squeeze(1)
    return F.binary_cross_entropy_with_logits(selected, outcomes)


def selector_distillation_loss(
    current_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Preserve old opaque route behavior during memory-side updates."""
    if current_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and current router shapes must match")
    return F.mse_loss(current_logits, teacher_logits.detach())
