from __future__ import annotations

import torch
from torch import nn

from experiments.syllogimous_neural_computer.training_memory import DifferentiableBatchMemory


class LatentConsolidator(nn.Module):
    """Compress a short sequence of learned latent rows into one latent row.

    The module sees no task identifiers, labels, text, or game state. Its only
    input is the neural computer's own key/value/strength memory stream.
    """

    def __init__(self, width: int, *, heads: int = 5, layers: int = 2,
                 max_rows: int = 8):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.max_rows = max_rows
        self.row_projection = nn.Sequential(
            nn.Linear(width * 2 + 2, width), nn.LayerNorm(width), nn.GELU())
        self.positions = nn.Parameter(torch.randn(1, max_rows, width) * 0.02)
        self.summary = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width))
        self.key_head = nn.Linear(width, width)
        self.value_head = nn.Linear(width, width)
        self.strength_head = nn.Linear(width, 1)

    def forward(self, memory: DifferentiableBatchMemory) -> DifferentiableBatchMemory:
        if memory.count < 1 or memory.count > self.max_rows:
            raise ValueError("memory row count outside consolidator capacity")
        scalar = torch.stack((memory.strengths, memory.admissions), dim=-1)
        rows = self.row_projection(torch.cat((memory.keys, memory.values, scalar), dim=-1))
        rows = rows + self.positions[:, :memory.count]
        summary = self.summary.expand(memory.batch, -1, -1)
        encoded = self.encoder(torch.cat((summary, rows), dim=1))[:, 0]
        key = self.key_head(encoded)
        value = self.value_head(encoded)
        strength = torch.sigmoid(self.strength_head(encoded)).squeeze(-1).clamp_min(1e-4)
        return DifferentiableBatchMemory(
            memory.batch, memory.width, device=memory.device,
            keys=key[:, None], values=value[:, None],
            strengths=strength[:, None], admissions=torch.ones_like(strength[:, None]))

