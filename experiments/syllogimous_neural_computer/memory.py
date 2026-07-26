from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class PersistentMemory:
    """A growing, serializable key-value memory outside the network weights.

    This object contains no task labels or game state. The controller supplies
    every key, value, and write strength. Rows are append-only at this layer:
    consolidation may later write a better abstraction, but raw experience is
    not silently destroyed. Tensor storage grows in chunks and can be saved to
    disk independently of model weights.
    """

    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    age: torch.Tensor
    valid: torch.Tensor
    clock: int = 0
    growth_chunk: int = 1024

    @classmethod
    def empty(cls, capacity: int, width: int, *, device: torch.device | str = "cpu",
              dtype: torch.dtype = torch.float32,
              growth_chunk: int = 1024) -> "PersistentMemory":
        if capacity < 1 or width < 1:
            raise ValueError("capacity and width must be positive")
        return cls(
            torch.zeros(capacity, width, device=device, dtype=dtype),
            torch.zeros(capacity, width, device=device, dtype=dtype),
            torch.zeros(capacity, device=device, dtype=dtype),
            torch.zeros(capacity, device=device, dtype=torch.long),
            torch.zeros(capacity, device=device, dtype=torch.bool), 0, growth_chunk,
        )

    @property
    def capacity(self) -> int:
        return self.keys.shape[0]

    @property
    def width(self) -> int:
        return self.keys.shape[1]

    @property
    def count(self) -> int:
        return int(self.valid.sum())

    def to(self, device: torch.device | str) -> "PersistentMemory":
        return PersistentMemory(self.keys.to(device), self.values.to(device),
                                self.usage.to(device), self.age.to(device),
                                self.valid.to(device), self.clock, self.growth_chunk)

    def clone(self) -> "PersistentMemory":
        return PersistentMemory(self.keys.clone(), self.values.clone(), self.usage.clone(),
                                self.age.clone(), self.valid.clone(), self.clock,
                                self.growth_chunk)

    def select(self, indices: torch.Tensor | list[int]) -> "PersistentMemory":
        """Copy selected valid rows into a compact active-memory store."""
        selected = torch.as_tensor(indices, device=self.keys.device, dtype=torch.long)
        if selected.ndim != 1:
            raise ValueError("memory indices must be one-dimensional")
        if selected.numel() and (not bool(self.valid[selected].all())):
            raise ValueError("active memory can contain only valid long-term rows")
        active = PersistentMemory.empty(
            max(1, int(selected.numel())), self.width, device=self.keys.device,
            dtype=self.keys.dtype, growth_chunk=self.growth_chunk)
        if selected.numel():
            count = int(selected.numel())
            active.keys[:count].copy_(self.keys[selected])
            active.values[:count].copy_(self.values[selected])
            active.usage[:count].copy_(self.usage[selected])
            active.age[:count].copy_(self.age[selected])
            active.valid[:count] = True
        active.clock = self.clock
        return active

    def _grow(self) -> None:
        amount = max(1, self.growth_chunk)
        self.keys = torch.cat((self.keys, self.keys.new_zeros(amount, self.width)))
        self.values = torch.cat((self.values, self.values.new_zeros(amount, self.width)))
        self.usage = torch.cat((self.usage, self.usage.new_zeros(amount)))
        self.age = torch.cat((self.age, self.age.new_zeros(amount)))
        self.valid = torch.cat((self.valid, self.valid.new_zeros(amount)))

    def read(self, queries: torch.Tensor, top_k: int = 4,
             temperature: torch.Tensor | float = 1.0,
             confidence_mode: str = "ranked"
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Content-addressed sparse reads; returns values and confidence."""
        if queries.ndim != 2 or queries.shape[1] != self.width:
            raise ValueError("queries must have shape [batch, memory width]")
        if self.count == 0:
            return torch.zeros_like(queries), queries.new_zeros(queries.shape[0])
        indices = self.valid.nonzero(as_tuple=False).squeeze(1)
        keys = torch.nn.functional.normalize(self.keys[indices], dim=-1)
        queries = torch.nn.functional.normalize(queries, dim=-1)
        cosine_similarity = queries @ keys.T
        similarity = cosine_similarity * temperature
        # Learned write strength is both an admission decision and a soft
        # retrieval prior. This lets delayed reward train the write gate in the
        # differentiable lifetime implementation used by the benchmark.
        similarity = similarity + self.usage[indices].clamp_min(1e-6).log().unsqueeze(0)
        selected = min(top_k, indices.numel())
        scores, local_indices = similarity.topk(selected, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        values = self.values[indices[local_indices]]
        read = (weights.unsqueeze(-1) * values).sum(dim=1)
        if confidence_mode == "ranked":
            confidence = scores[:, 0]
        elif confidence_mode == "cosine":
            confidence = torch.gather(
                cosine_similarity, 1, local_indices[:, :1]).squeeze(1)
        else:
            raise ValueError("unsupported confidence mode")
        return read, confidence

    @torch.no_grad()
    def write(self, keys: torch.Tensor, values: torch.Tensor, strengths: torch.Tensor,
              *, threshold: float = 0.5) -> int:
        """Commit sampled/thresholded controller writes between episodes."""
        if keys.shape != values.shape or keys.ndim != 2 or keys.shape[1] != self.width:
            raise ValueError("keys and values must share shape [writes, memory width]")
        if strengths.shape != (keys.shape[0],):
            raise ValueError("strengths must have one value per write")
        committed = 0
        for key, value, strength in zip(keys, values, strengths):
            if float(strength) < threshold:
                continue
            empty = (~self.valid).nonzero(as_tuple=False)
            if not empty.numel():
                self._grow()
                empty = (~self.valid).nonzero(as_tuple=False)
            slot = int(empty[0, 0])
            self.clock += 1
            self.keys[slot].copy_(key.detach())
            self.values[slot].copy_(value.detach())
            self.usage[slot] = strength.detach()
            self.age[slot] = self.clock
            self.valid[slot] = True
            committed += 1
        return committed

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "schema": "syllogimous-neural-computer-memory-v1",
            "keys": self.keys.detach().cpu(), "values": self.values.detach().cpu(),
            "usage": self.usage.detach().cpu(), "age": self.age.detach().cpu(),
            "valid": self.valid.detach().cpu(), "clock": self.clock,
            "growth_chunk": self.growth_chunk,
        }, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, *, device: torch.device | str = "cpu") -> "PersistentMemory":
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("schema") != "syllogimous-neural-computer-memory-v1":
            raise ValueError("unsupported persistent-memory schema")
        return cls(payload["keys"], payload["values"], payload["usage"],
                   payload["age"], payload["valid"], int(payload["clock"]),
                   int(payload.get("growth_chunk", 1024)))
