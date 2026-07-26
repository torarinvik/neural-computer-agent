"""Task-agnostic dynamic working memory and capability accounting.

This module deliberately contains no task identities or semantic labels.  The
pool exposes generic memory operations; a future learned controller may choose
their scores.  Until that controller is enabled, callers can keep the active
mask fixed and reproduce the existing four-slot control exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch


@dataclass
class MemoryOperationStats:
    reads: int = 0
    writes: int = 0
    activations: int = 0
    deactivations: int = 0
    evictions: int = 0
    active_slot_steps: int = 0
    capacity_steps: int = 0

    @property
    def mean_occupancy(self) -> float:
        if self.capacity_steps == 0:
            return 0.0
        return self.active_slot_steps / self.capacity_steps


class DynamicWorkingMemory:
    """A bounded physical RAM pool with a learned-ready active mask."""

    schema = "unified-controller-dynamic-working-memory-v1"

    def __init__(
            self, *, capacity: int = 8, width: int,
            fixed_active_slots: int = 0,
            device: torch.device | str = "cpu",
            dtype: torch.dtype = torch.float32) -> None:
        if capacity < 1 or width < 1:
            raise ValueError("working-memory dimensions must be positive")
        if not 0 <= fixed_active_slots <= capacity:
            raise ValueError("fixed active slots must fit within capacity")
        self.capacity = capacity
        self.width = width
        self.values = torch.zeros(
            capacity, width, device=device, dtype=dtype)
        self.active = torch.zeros(capacity, device=device, dtype=torch.bool)
        self.active[:fixed_active_slots] = True
        self.occupied = torch.zeros(
            capacity, device=device, dtype=torch.bool)
        self.usage = torch.zeros(capacity, device=device, dtype=torch.long)
        self.age = torch.zeros(capacity, device=device, dtype=torch.long)
        self.clock = 0
        self.stats = MemoryOperationStats()

    @property
    def active_count(self) -> int:
        return int(self.active.sum())

    def record_step(self) -> None:
        self.stats.active_slot_steps += self.active_count
        self.stats.capacity_steps += self.capacity
        self.clock += 1
        self.age[self.active] += 1

    def set_active_from_scores(
            self, scores: torch.Tensor, *, threshold: float = 0.0,
            minimum: int = 0) -> torch.Tensor:
        """Apply a generic learned mask; scores contain no task semantics."""
        if scores.shape != (self.capacity,):
            raise ValueError("activation scores have wrong width")
        if not 0 <= minimum <= self.capacity:
            raise ValueError("minimum must fit within capacity")
        selected = scores >= threshold
        if int(selected.sum()) < minimum:
            selected = torch.zeros_like(selected)
            selected[scores.topk(minimum).indices] = True
        self._set_active(selected)
        return self.active.clone()

    def _set_active(self, selected: torch.Tensor) -> None:
        activated = selected & ~self.active
        deactivated = self.active & ~selected
        self.stats.activations += int(activated.sum())
        self.stats.deactivations += int(deactivated.sum())
        self.active.copy_(selected)

    def write(self, slot: int, value: torch.Tensor) -> None:
        if not 0 <= slot < self.capacity:
            raise IndexError("working-memory slot out of range")
        if value.shape != (self.width,):
            raise ValueError("working-memory value has wrong width")
        if not self.active[slot]:
            self.active[slot] = True
            self.stats.activations += 1
        if self.occupied[slot]:
            self.stats.evictions += 1
        self.values[slot].copy_(value)
        self.occupied[slot] = True
        self.usage[slot] += 1
        self.age[slot] = 0
        self.stats.writes += 1

    def read(self, slot: int) -> torch.Tensor:
        if not 0 <= slot < self.capacity:
            raise IndexError("working-memory slot out of range")
        if not self.active[slot]:
            raise ValueError("cannot read an inactive working-memory slot")
        self.usage[slot] += 1
        self.stats.reads += 1
        return self.values[slot].clone()

    def active_values(self) -> torch.Tensor:
        self.stats.reads += self.active_count
        self.usage[self.active] += 1
        return self.values[self.active].clone()

    def save(self, path: Path) -> None:
        torch.save({
            "schema": self.schema,
            "capacity": self.capacity,
            "width": self.width,
            "values": self.values,
            "active": self.active,
            "occupied": self.occupied,
            "usage": self.usage,
            "age": self.age,
            "clock": self.clock,
            "stats": asdict(self.stats),
        }, path)

    @classmethod
    def load(
            cls, path: Path, *,
            device: torch.device | str = "cpu") -> "DynamicWorkingMemory":
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload["schema"] != cls.schema:
            raise ValueError("unsupported dynamic-memory schema")
        memory = cls(
            capacity=payload["capacity"], width=payload["width"],
            device=device, dtype=payload["values"].dtype)
        for field in ("values", "active", "occupied", "usage", "age"):
            getattr(memory, field).copy_(payload[field].to(device))
        memory.clock = int(payload["clock"])
        memory.stats = MemoryOperationStats(**payload["stats"])
        return memory


@dataclass
class CapabilityLedger:
    """Auditable resource and capability totals for one experimental arm."""

    unique_verifier_bits: int = 0
    unique_logical_lifetimes: int = 0
    optimizer_updates: int = 0
    replayed_examples: int = 0
    candidate_verifier_bits: int = 0
    memory_reads: int = 0
    memory_writes: int = 0
    memory_evictions: int = 0
    active_slot_steps: int = 0
    capacity_steps: int = 0
    thought_steps: int = 0
    disk_bytes_read: int = 0
    disk_bytes_written: int = 0
    latency_seconds: float = 0.0
    gpu_seconds: float = 0.0

    @property
    def mean_active_fraction(self) -> float:
        if self.capacity_steps == 0:
            return 0.0
        return self.active_slot_steps / self.capacity_steps

    def absorb_memory(self, stats: MemoryOperationStats) -> None:
        self.memory_reads += stats.reads
        self.memory_writes += stats.writes
        self.memory_evictions += stats.evictions
        self.active_slot_steps += stats.active_slot_steps
        self.capacity_steps += stats.capacity_steps

    def as_report(self) -> dict[str, int | float]:
        report = asdict(self)
        report["mean_active_fraction"] = self.mean_active_fraction
        return report


class LatencyTimer:
    """Small context manager for ledger-compatible wall latency."""

    def __init__(self, ledger: CapabilityLedger) -> None:
        self.ledger = ledger
        self.started = 0.0

    def __enter__(self) -> "LatencyTimer":
        self.started = perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.ledger.latency_seconds += perf_counter() - self.started
