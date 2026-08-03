"""Versioned content-addressed long-term memory for the production runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .interface import MEMORY_SCHEMA


@dataclass(frozen=True)
class MemoryQuery:
    key: torch.Tensor
    top_k: int = 1

    def validate(self, *, width: int, batch: int | None = None) -> MemoryQuery:
        if self.key.ndim != 2 or self.key.shape[1] != width:
            raise ValueError(f"memory query key must have shape [batch, {width}]")
        if batch is not None and self.key.shape[0] != batch:
            raise ValueError("memory query batch does not match controller batch")
        if self.top_k < 1:
            raise ValueError("memory query top_k must be positive")
        return self


@dataclass(frozen=True)
class MemoryRead:
    value: torch.Tensor
    scores: torch.Tensor
    indices: torch.Tensor
    hit: torch.Tensor


@dataclass(frozen=True)
class MemoryWriteReceipt:
    committed: torch.Tensor
    indices: torch.Tensor
    version: int


class ContentAddressedMemory(nn.Module):
    """Bounded learned-latent store with exact save/load round trips.

    The store has no modality or task fields.  Keys and values use the same
    learned width as the controller, and writes are committed only through
    this contract so disk-backed and in-process memory have identical behavior.
    """

    schema = MEMORY_SCHEMA

    def __init__(
        self,
        width: int,
        capacity: int = 256,
        *,
        write_threshold: float = 0.5,
        query_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if min(width, capacity) < 1:
            raise ValueError("memory width and capacity must be positive")
        if not 0.0 <= write_threshold <= 1.0 or query_temperature <= 0.0:
            raise ValueError("memory thresholds are invalid")
        self.width = width
        self.capacity = capacity
        self.write_threshold = float(write_threshold)
        self.query_temperature = float(query_temperature)
        self.register_buffer("keys", torch.zeros(capacity, width))
        self.register_buffer("values", torch.zeros(capacity, width))
        self.register_buffer("strengths", torch.zeros(capacity))
        self.register_buffer("timestamps", torch.zeros(capacity))
        self.register_buffer("occupied", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("store_version", torch.zeros((), dtype=torch.long))

    def _validate_batch(self, value: torch.Tensor, batch: int, name: str) -> None:
        if value.ndim != 2 or value.shape != (batch, self.width):
            raise ValueError(f"{name} must have shape [{batch}, {self.width}]")

    @torch.no_grad()
    def read(self, query: MemoryQuery) -> MemoryRead:
        query.validate(width=self.width)
        keys = torch.nn.functional.normalize(self.keys, dim=-1)
        query_keys = torch.nn.functional.normalize(query.key.detach(), dim=-1)
        scores = query_keys @ keys.T
        scores = scores.masked_fill(~self.occupied.unsqueeze(0), -torch.inf)
        top_k = min(query.top_k, self.capacity)
        top_scores, indices = torch.topk(scores, k=top_k, dim=-1)
        finite = torch.isfinite(top_scores)
        hit = finite.any(dim=-1)
        weights = torch.softmax(top_scores / self.query_temperature, dim=-1)
        weights = torch.where(finite, weights, torch.zeros_like(weights))
        value = torch.einsum("bk,bkw->bw", weights, self.values[indices])
        return MemoryRead(value=value, scores=top_scores, indices=indices, hit=hit)

    @torch.no_grad()
    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: torch.Tensor,
        *,
        timestamp: torch.Tensor | None = None,
    ) -> MemoryWriteReceipt:
        if key.ndim != 2 or value.ndim != 2 or key.shape != value.shape:
            raise ValueError("memory key and value must have equal [batch, width] shape")
        batch = key.shape[0]
        self._validate_batch(key, batch, "key")
        self._validate_batch(value, batch, "value")
        strength = strength.reshape(batch).detach().to(self.strengths)
        if timestamp is None:
            timestamp = torch.zeros(batch, device=key.device, dtype=key.dtype)
        timestamp = timestamp.reshape(batch).detach().to(self.timestamps)
        committed = strength >= self.write_threshold
        indices = torch.full((batch,), -1, device=key.device, dtype=torch.long)
        for row in range(batch):
            if not bool(committed[row]):
                continue
            free = torch.nonzero(~self.occupied, as_tuple=False).reshape(-1)
            index = int(free[0]) if free.numel() else int(torch.argmin(self.strengths))
            self.keys[index].copy_(key[row].detach().to(self.keys))
            self.values[index].copy_(value[row].detach().to(self.values))
            self.strengths[index].copy_(strength[row].to(self.strengths))
            self.timestamps[index].copy_(timestamp[row].to(self.timestamps))
            self.occupied[index] = True
            indices[row] = index
        if bool(committed.any()):
            self.store_version.add_(1)
        return MemoryWriteReceipt(
            committed=committed,
            indices=indices,
            version=int(self.store_version.item()),
        )

    @torch.no_grad()
    def clear(self) -> None:
        """Reset persistent rows while preserving the component schema."""
        self.keys.zero_()
        self.values.zero_()
        self.strengths.zero_()
        self.timestamps.zero_()
        self.occupied.zero_()
        self.store_version.add_(1)

    def snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": self.schema,
                "configuration": self.configuration(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    def load_snapshot(self, path: Path, *, map_location: torch.device | str = "cpu") -> None:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("schema") != self.schema:
            raise ValueError("unsupported memory snapshot schema")
        if payload.get("configuration") != self.configuration():
            raise ValueError("memory snapshot configuration does not match store")
        self.load_state_dict(payload["state_dict"])

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "width": self.width,
            "capacity": self.capacity,
            "write_threshold": self.write_threshold,
            "query_temperature": self.query_temperature,
        }
