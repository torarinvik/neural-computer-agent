"""Disk-backed long-term memory interface for the unified controller.

The first few-shot rung intentionally uses only differentiable RAM/VRAM state.
This wrapper establishes the long-term boundary without giving the controller
semantic fields or task-specific storage.
"""
from __future__ import annotations

from pathlib import Path

import torch

from experiments.syllogimous_neural_computer.memory import PersistentMemory


class DiskLatentMemory:
    """Serializable controller-created key/value rows stored outside weights."""

    def __init__(
            self, width: int, capacity: int = 1024, *,
            device: torch.device | str = "cpu") -> None:
        self.store = PersistentMemory.empty(
            capacity, width, device=device, growth_chunk=capacity)

    @property
    def count(self) -> int:
        return self.store.count

    def retrieve(
            self, queries: torch.Tensor, top_k: int = 4,
            confidence_mode: str = "ranked",
            record_access: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.store.read(
            queries, top_k=top_k,
            confidence_mode=confidence_mode,
            record_access=record_access)

    def retrieve_with_features(
            self, queries: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard top-1 values and task-agnostic match statistics."""
        if queries.ndim != 2 or queries.shape[1] != self.store.width:
            raise ValueError("queries must have shape [batch, memory width]")
        if self.count == 0:
            return (
                torch.zeros_like(queries),
                queries.new_zeros(queries.shape[0], 4))
        indices = self.store.valid.nonzero(
            as_tuple=False).squeeze(1)
        keys = torch.nn.functional.normalize(
            self.store.keys[indices], dim=-1)
        normalized_queries = torch.nn.functional.normalize(
            queries, dim=-1)
        cosine = normalized_queries @ keys.T
        usage = self.store.usage[indices].clamp_min(1e-6)
        ranked = cosine + usage.log().unsqueeze(0)
        selected_count = min(2, indices.numel())
        scores, local = ranked.topk(selected_count, dim=-1)
        selected = local[:, 0]
        read = self.store.values[indices[selected]]
        confidence = torch.gather(
            cosine, 1, selected.unsqueeze(1)).squeeze(1)
        if selected_count == 1:
            margin = torch.ones_like(confidence)
        else:
            margin = scores[:, 0] - scores[:, 1]
        selected_usage = usage[selected]
        occupancy = torch.full_like(
            confidence, self.count / self.store.capacity)
        features = torch.stack((
            confidence, margin, selected_usage, occupancy), dim=-1)
        return read, features

    def commit(
            self, keys: torch.Tensor, values: torch.Tensor,
            strengths: torch.Tensor, threshold: float = 0.5) -> int:
        return self.store.write(
            keys, values, strengths, threshold=threshold)

    @torch.no_grad()
    def replace(
            self, index: int, key: torch.Tensor, value: torch.Tensor,
            strength: torch.Tensor | float) -> None:
        """Replace one valid row without growing the bounded physical store."""
        if not 0 <= index < self.store.capacity:
            raise IndexError("replacement index is outside memory capacity")
        if not bool(self.store.valid[index]):
            raise ValueError("replacement requires a valid occupied row")
        if key.shape != (self.store.width,) or value.shape != (self.store.width,):
            raise ValueError("replacement key and value must match memory width")
        self.store.clock += 1
        self.store.keys[index].copy_(key.detach())
        self.store.values[index].copy_(value.detach())
        self.store.usage[index] = torch.as_tensor(
            strength, device=self.store.usage.device,
            dtype=self.store.usage.dtype)
        self.store.age[index] = self.store.clock
        self.store.access_count[index] = 0
        self.store.success_count[index] = 0
        self.store.failure_count[index] = 0
        self.store.valid[index] = True

    def save(self, path: Path) -> None:
        self.store.save(path)

    @classmethod
    def load(
            cls, path: Path, *, device: torch.device | str = "cpu"
            ) -> "DiskLatentMemory":
        instance = cls.__new__(cls)
        instance.store = PersistentMemory.load(path, device=device)
        return instance
