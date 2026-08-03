"""Task-agnostic learned routing over opaque skill-bank addresses.

The router sees only a controller-produced query and the bank's controller-
produced row keys.  Training can use an attempted row and its scalar verifier
outcome, but never a task name, span, or correct-row label.  It is deliberately
separate from :class:`SkillArtifactBank` so a diagnostic router cannot silently
change the promoted nearest-key behavior.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SkillAddressSelector(nn.Module):
    """Permutation-equivariant pairwise scorer for candidate skill rows."""

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
        # The diagnostic starts neutral; verifier outcomes must earn a route.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
            self, query: torch.Tensor, keys: torch.Tensor,
            ) -> torch.Tensor:
        """Return one score per candidate row.

        ``query`` has shape ``[batch, width]`` and ``keys`` has shape
        ``[batch, rows, width]``.  A single key matrix ``[rows, width]`` is
        accepted and broadcast across the batch for convenient bank reads.
        """
        if query.ndim != 2 or query.shape[1] != self.width:
            raise ValueError("query must have shape [batch, width]")
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        if (
                keys.ndim != 3 or keys.shape[0] != query.shape[0]
                or keys.shape[2] != self.width or keys.shape[1] < 1):
            raise ValueError(
                "keys must have shape [batch, rows, width] or [rows, width]")
        query_rows = query.unsqueeze(1).expand(-1, keys.shape[1], -1)
        pair = torch.cat((
            query_rows,
            keys,
            (query_rows - keys).abs(),
            query_rows * keys,
        ), dim=-1)
        return self.net(pair).squeeze(-1)


def attempted_outcome_loss(
        logits: torch.Tensor, attempted: torch.Tensor,
        outcomes: torch.Tensor,
        ) -> torch.Tensor:
    """Use only the attempted candidate and its scalar binary outcome."""
    if logits.ndim != 2 or attempted.ndim != 1 or outcomes.ndim != 1:
        raise ValueError("invalid router transition shapes")
    if logits.shape[0] != attempted.shape[0] or outcomes.shape != (
            attempted.shape[0],):
        raise ValueError("router transition batch lengths differ")
    if not bool(((attempted >= 0) & (attempted < logits.shape[1])).all()):
        raise ValueError("attempted candidate is out of range")
    if not bool(((outcomes == 0) | (outcomes == 1)).all()):
        raise ValueError("outcomes must be binary scalar rewards")
    selected = logits.gather(1, attempted[:, None]).squeeze(1)
    return F.binary_cross_entropy_with_logits(selected, outcomes)


def selector_distillation_loss(
        current_logits: torch.Tensor, teacher_logits: torch.Tensor,
        ) -> torch.Tensor:
    """Preserve an earlier selector's opaque decisions during new learning."""
    if current_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and current selector shapes must match")
    return F.mse_loss(current_logits, teacher_logits.detach())
