from __future__ import annotations

import torch
from torch import nn

from .memory import PersistentMemory


class ActiveContextSelector(nn.Module):
    """Select zero or one latent long-term row for the fast active context."""

    def __init__(self, width: int, hidden: int = 128):
        super().__init__()
        self.width = width
        row_width = width * 6 + 2
        self.row_score = nn.Sequential(nn.Linear(row_width, hidden), nn.GELU(),
                                       nn.Linear(hidden, 1))
        self.null_score = nn.Sequential(nn.Linear(width * 3 + 1, hidden), nn.GELU(),
                                        nn.Linear(hidden, 1))

    def _row_features(self, memory: PersistentMemory,
                      indices: torch.Tensor) -> torch.Tensor:
        usage = memory.usage[indices, None]
        age = (memory.clock - memory.age[indices]).to(memory.keys.dtype)[:, None]
        age = age / age.max().clamp_min(1.0)
        return torch.cat((memory.keys[indices], memory.values[indices], usage, age), -1)

    def forward(self, sensory: torch.Tensor,
                memory: PersistentMemory) -> tuple[torch.Tensor, torch.Tensor]:
        if sensory.shape != (self.width,):
            raise ValueError("sensory query must have shape [memory width]")
        indices = memory.valid.nonzero(as_tuple=False).squeeze(1)
        if not indices.numel():
            return sensory.new_zeros(1), indices
        rows = self._row_features(memory, indices)
        queries = sensory.unsqueeze(0).expand(indices.numel(), -1)
        keys = memory.keys[indices]
        values = memory.values[indices]
        interactions = torch.cat((queries, rows, queries * keys,
                                  (queries - keys).abs(), queries * values), -1)
        row_logits = self.row_score(interactions).squeeze(-1)
        keys = keys.mean(0)
        values = values.mean(0)
        count = sensory.new_tensor([float(indices.numel())]).log1p()
        null_logit = self.null_score(torch.cat((sensory, keys, values, count))).reshape(1)
        return torch.cat((null_logit, row_logits)), indices

    @torch.no_grad()
    def select(self, sensory: torch.Tensor,
               memory: PersistentMemory) -> tuple[PersistentMemory, int | None]:
        logits, indices = self(sensory, memory)
        choice = int(logits.argmax())
        if choice == 0:
            return memory.select([]), None
        index = int(indices[choice - 1])
        return memory.select([index]), index
