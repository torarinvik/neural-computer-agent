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
            self, queries: torch.Tensor, top_k: int = 4
            ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.store.read(queries, top_k=top_k)

    def commit(
            self, keys: torch.Tensor, values: torch.Tensor,
            strengths: torch.Tensor, threshold: float = 0.5) -> int:
        return self.store.write(
            keys, values, strengths, threshold=threshold)

    def save(self, path: Path) -> None:
        self.store.save(path)

    @classmethod
    def load(
            cls, path: Path, *, device: torch.device | str = "cpu"
            ) -> "DiskLatentMemory":
        instance = cls.__new__(cls)
        instance.store = PersistentMemory.load(path, device=device)
        return instance
