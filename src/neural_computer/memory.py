"""Versioned content-addressed long-term memory for the production runtime."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .interface import MEMORY_SCHEMA
from .representation import (
    DEFAULT_MEMORY_KEY_SPACE_ID,
    DEFAULT_MEMORY_VALUE_SPACE_ID,
    validate_representation_space_id,
)
from .retention import CapabilityRetentionLedger

MEMORY_BACKEND_FORMAT = "neural-computer.memory-backend.v1"
APPEND_ONLY_MEMORY_BACKEND_FORMAT = "neural-computer.append-only-memory-backend.v1"
LEGACY_MEMORY_SNAPSHOT_FORMAT = "neural-computer.memory-snapshot.v1"
MEMORY_SNAPSHOT_FORMAT = "neural-computer.memory-snapshot.v2"
APPEND_ONLY_MEMORY_SNAPSHOT_FORMAT = "neural-computer.append-only-memory-snapshot.v1"
MEMORY_MIGRATION_SCHEMA = "neural-computer.memory-representation-migration.v1"
# Writes reject collisions more strictly than reads reject near-misses. Keeping
# these contracts separate prevents a noisy query from hallucinating the
# nearest occupied row while preserving exact content-addressed writes.
MEMORY_READ_MATCH_THRESHOLD = 0.75


@dataclass(frozen=True)
class MemoryQuery:
    key: torch.Tensor
    top_k: int = 1
    schema: str = MEMORY_SCHEMA
    scope: torch.Tensor | None = None

    def validate(self, *, width: int, batch: int | None = None) -> MemoryQuery:
        if self.schema != MEMORY_SCHEMA:
            raise ValueError(f"unsupported memory query schema: {self.schema}")
        if not isinstance(self.top_k, int):
            raise TypeError("memory query top_k must be an integer")
        if self.key.ndim != 2 or self.key.shape[1] != width:
            raise ValueError(f"memory query key must have shape [batch, {width}]")
        if not bool(torch.isfinite(self.key).all()):
            raise ValueError("memory query key must contain only finite values")
        if batch is not None and self.key.shape[0] != batch:
            raise ValueError("memory query batch does not match controller batch")
        if self.top_k < 1:
            raise ValueError("memory query top_k must be positive")
        if self.scope is not None:
            if self.scope.ndim == 0 or self.scope.numel() != self.key.shape[0]:
                raise ValueError(
                    "memory query scope must have one int64 value per batch row"
                )
            if self.scope.dtype != torch.long:
                raise ValueError("memory query scope must be int64")
        return self


@dataclass(frozen=True)
class MemoryRead:
    value: torch.Tensor
    scores: torch.Tensor
    indices: torch.Tensor
    hit: torch.Tensor
    schema: str = MEMORY_SCHEMA

    def validate(self, *, width: int, batch: int | None = None) -> MemoryRead:
        if self.schema != MEMORY_SCHEMA:
            raise ValueError(f"unsupported memory read schema: {self.schema}")
        if self.value.ndim != 2 or self.value.shape[1] != width:
            raise ValueError(f"memory value must have shape [batch, {width}]")
        if self.scores.ndim != 2 or self.scores.shape[0] != self.value.shape[0]:
            raise ValueError("memory scores must have shape [batch, top_k]")
        if self.indices.shape != self.scores.shape or self.indices.dtype != torch.long:
            raise ValueError("memory indices must match scores and be int64")
        if self.hit.shape != (self.value.shape[0],) or self.hit.dtype != torch.bool:
            raise ValueError("memory hit must have shape [batch] and be boolean")
        if batch is not None and self.value.shape[0] != batch:
            raise ValueError("memory read batch does not match controller batch")
        valid_scores = torch.isfinite(self.scores) | torch.isneginf(self.scores)
        if not bool(torch.isfinite(self.value).all()) or not bool(valid_scores.all()):
            raise ValueError("memory read contains non-finite values")
        return self


@dataclass(frozen=True)
class MemoryWriteReceipt:
    committed: torch.Tensor
    indices: torch.Tensor
    version: int
    schema: str = MEMORY_SCHEMA

    def validate(self, *, batch: int | None = None) -> MemoryWriteReceipt:
        if self.schema != MEMORY_SCHEMA:
            raise ValueError(f"unsupported memory receipt schema: {self.schema}")
        if self.committed.ndim != 1 or self.committed.dtype != torch.bool:
            raise ValueError("memory committed must be boolean [batch]")
        if self.indices.shape != self.committed.shape or self.indices.dtype != torch.long:
            raise ValueError("memory indices must be int64 [batch]")
        if batch is not None and self.committed.shape[0] != batch:
            raise ValueError("memory receipt batch does not match controller batch")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("memory version cannot be negative")
        return self


@dataclass(frozen=True)
class MemoryCandidates:
    """Opaque physical-row candidates exposed to replaceable memory policies."""

    keys: torch.Tensor
    values: torch.Tensor
    strengths: torch.Tensor
    timestamps: torch.Tensor
    occupied: torch.Tensor
    schema: str = MEMORY_SCHEMA

    def validate(
        self, *, width: int, capacity: int, batch: int | None = None
    ) -> MemoryCandidates:
        expected = (self.keys.shape[0], capacity, width)
        if self.schema != MEMORY_SCHEMA:
            raise ValueError(f"unsupported memory candidates schema: {self.schema}")
        if self.keys.shape != expected or self.values.shape != expected:
            raise ValueError("memory candidate keys and values have the wrong shape")
        scalar_shape = (self.keys.shape[0], capacity)
        if self.strengths.shape != scalar_shape or self.timestamps.shape != scalar_shape:
            raise ValueError("memory candidate scalars have the wrong shape")
        if self.occupied.shape != scalar_shape or self.occupied.dtype != torch.bool:
            raise ValueError("memory candidate occupancy has the wrong shape and dtype")
        if batch is not None and self.keys.shape[0] != batch:
            raise ValueError("memory candidate batch does not match request")
        for tensor in (self.keys, self.values, self.strengths, self.timestamps):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("memory candidates must contain finite values")
        if bool(torch.any(self.strengths < 0) or torch.any(self.strengths > 1)):
            raise ValueError("memory candidate strengths must lie in [0, 1]")
        return self


@dataclass(frozen=True)
class MemoryMigrationExample:
    """One paired query across source and replacement memory spaces."""

    source_query: MemoryQuery
    target_query: MemoryQuery


@dataclass(frozen=True)
class MemoryMigrationReceipt:
    """Verifier-gated copy-on-write memory representation migration."""

    accepted: bool
    source_key_space_id: str
    target_key_space_id: str
    source_value_space_id: str
    target_value_space_id: str
    address_count: int
    protected_count: int
    query_count: int
    max_value_difference: float
    source_digest: str
    target_digest: str
    reason: str
    schema: str = MEMORY_MIGRATION_SCHEMA

    def validate(self) -> MemoryMigrationReceipt:
        if self.schema != MEMORY_MIGRATION_SCHEMA:
            raise ValueError("unsupported memory migration schema")
        if min(self.address_count, self.protected_count, self.query_count) < 1:
            raise ValueError("memory migration evidence counts are invalid")
        if self.max_value_difference < 0.0 or (
            self.accepted
            and not torch.isfinite(torch.tensor(self.max_value_difference))
        ):
            raise ValueError("memory migration value difference is invalid")
        for name, value in (
            ("source_key_space_id", self.source_key_space_id),
            ("target_key_space_id", self.target_key_space_id),
            ("source_value_space_id", self.source_value_space_id),
            ("target_value_space_id", self.target_value_space_id),
            ("source_digest", self.source_digest),
            ("target_digest", self.target_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"memory migration {name} is missing")
        return self


class MemoryBackend(nn.Module):
    """Replaceable memory component contract owned by the runtime.

    Implementations remain neural modules so runtime checkpoints can load them
    independently, but the controller depends only on query/read/write
    semantics and configuration—not on a concrete storage medium.
    """

    format = MEMORY_BACKEND_FORMAT
    schema = MEMORY_SCHEMA

    def __init__(self, width: int) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("memory width must be positive")
        self.width = width

    def configuration(self) -> dict[str, Any]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def validate_state(self) -> None:
        """Validate the currently loaded backend state before it is served."""
        return

    @contextmanager
    def differentiable_transaction(self) -> Iterator[MemoryBackend]:
        """Scope optional gradients through writes and later reads.

        The base contract is a no-op so backends that already provide a fully
        differentiable implementation need no special handling. Stateful
        backends may retain only transaction-local tensors and must release
        them when the context exits.
        """
        yield self

    def read(self, query: MemoryQuery) -> MemoryRead:
        raise NotImplementedError

    def candidates(self, scope: torch.Tensor | None = None) -> MemoryCandidates:
        """Return opaque physical rows for an independent memory-side policy."""
        raise NotImplementedError

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: torch.Tensor,
        *,
        timestamp: torch.Tensor | None = None,
        scope: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
    ) -> MemoryWriteReceipt:
        raise NotImplementedError


class ContentAddressedMemory(MemoryBackend):
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
        write_match_threshold: float = 0.95,
        scope_capacity: int = 1,
        retention_ledger: CapabilityRetentionLedger | None = None,
        key_space_id: str = DEFAULT_MEMORY_KEY_SPACE_ID,
        value_space_id: str = DEFAULT_MEMORY_VALUE_SPACE_ID,
    ) -> None:
        super().__init__(width)
        if not isinstance(scope_capacity, int) or min(width, capacity, scope_capacity) < 1:
            raise ValueError("memory width and capacity must be positive")
        if (
            not 0.0 <= write_threshold <= 1.0
            or query_temperature <= 0.0
            or not 0.0 <= write_match_threshold <= 1.0
        ):
            raise ValueError("memory thresholds are invalid")
        self.capacity = capacity
        self.write_threshold = float(write_threshold)
        self.query_temperature = float(query_temperature)
        self.write_match_threshold = float(write_match_threshold)
        self.scope_capacity = scope_capacity
        if retention_ledger is not None and retention_ledger.width != width:
            raise ValueError("retention ledger width must match memory")
        key_space_id = validate_representation_space_id(
            key_space_id, name="memory key_space_id"
        )
        value_space_id = validate_representation_space_id(
            value_space_id, name="memory value_space_id"
        )
        self.retention = retention_ledger or CapabilityRetentionLedger(width)
        self.key_space_id = key_space_id
        self.value_space_id = value_space_id
        self._transaction_depth = 0
        self._pending_writes: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        key_shape = (
            (capacity, width)
            if scope_capacity == 1
            else (scope_capacity, capacity, width)
        )
        scalar_shape = (
            (capacity,) if scope_capacity == 1 else (scope_capacity, capacity)
        )
        self.register_buffer("keys", torch.zeros(key_shape))
        self.register_buffer("values", torch.zeros(key_shape))
        self.register_buffer("strengths", torch.zeros(scalar_shape))
        self.register_buffer("timestamps", torch.zeros(scalar_shape))
        self.register_buffer("occupied", torch.zeros(scalar_shape, dtype=torch.bool))
        self.register_buffer("store_version", torch.zeros((), dtype=torch.long))

    def _bank(self, tensor: torch.Tensor) -> torch.Tensor:
        """Expose the legacy single-scope layout as a banked view."""
        return tensor.unsqueeze(0) if self.scope_capacity == 1 else tensor

    def _scope_ids(
        self,
        scope: torch.Tensor | None,
        batch: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if scope is None:
            return torch.zeros(batch, device=device, dtype=torch.long)
        if scope.ndim == 0 or scope.numel() != batch:
            raise ValueError("memory scope must have one int64 value per batch row")
        if scope.dtype != torch.long:
            raise ValueError("memory scope must be int64")
        scope = scope.reshape(batch).to(device=device)
        if torch.any(scope < 0) or torch.any(scope >= self.scope_capacity):
            raise ValueError("memory scope is outside the configured scope capacity")
        return scope

    def _validate_batch(self, value: torch.Tensor, batch: int, name: str) -> None:
        if value.ndim != 2 or value.shape != (batch, self.width):
            raise ValueError(f"{name} must have shape [{batch}, {self.width}]")

    def _validate_state(self, state: dict[str, torch.Tensor] | None = None) -> None:
        values = self.state_dict() if state is None else state
        expected = {"keys", "values", "strengths", "timestamps", "occupied", "store_version"}
        if set(values) != expected:
            raise ValueError("memory state has an incompatible field set")
        expected_key_shape = (
            (self.capacity, self.width)
            if self.scope_capacity == 1
            else (self.scope_capacity, self.capacity, self.width)
        )
        expected_scalar_shape = (
            (self.capacity,)
            if self.scope_capacity == 1
            else (self.scope_capacity, self.capacity)
        )
        if values["keys"].shape != expected_key_shape:
            raise ValueError("memory key state has the wrong shape")
        if values["values"].shape != expected_key_shape:
            raise ValueError("memory value state has the wrong shape")
        for name in ("keys", "values", "strengths", "timestamps"):
            if not bool(torch.isfinite(values[name]).all()):
                raise ValueError(f"memory state {name} must be finite")
        if values["strengths"].shape != expected_scalar_shape:
            raise ValueError("memory strengths state has the wrong shape")
        if torch.any(values["strengths"] < 0) or torch.any(values["strengths"] > 1):
            raise ValueError("memory strengths must lie in [0, 1]")
        if values["timestamps"].shape != expected_scalar_shape:
            raise ValueError("memory timestamps state has the wrong shape")
        if (
            values["occupied"].shape != expected_scalar_shape
            or values["occupied"].dtype != torch.bool
        ):
            raise ValueError("memory occupied state has the wrong shape and dtype")
        if values["store_version"].shape != torch.Size([]) or values["store_version"].dtype != torch.long:
            raise ValueError("memory store_version must be a scalar int64")
        if int(values["store_version"].item()) < 0:
            raise ValueError("memory store_version cannot be negative")

    def validate_state(self) -> None:
        self._validate_state()

    @contextmanager
    def differentiable_transaction(self) -> Iterator[MemoryBackend]:
        if self._transaction_depth:
            raise RuntimeError("nested differentiable memory transactions are unsupported")
        self._transaction_depth = 1
        self._pending_writes = {}
        try:
            yield self
        finally:
            self._pending_writes.clear()
            self._transaction_depth = 0

    @staticmethod
    def _state_checksum(state: dict[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def read(self, query: MemoryQuery) -> MemoryRead:
        query.validate(width=self.width)
        self.validate_state()
        batch = query.key.shape[0]
        scope_ids = self._scope_ids(query.scope, batch, device=self.keys.device)
        bank_keys = self._bank(self.keys)
        bank_values = self._bank(self.values)
        bank_occupied = self._bank(self.occupied)
        key_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        occupied_rows: list[torch.Tensor] = []
        for scope_id in scope_ids.tolist():
            durable_keys = bank_keys[scope_id]
            durable_values = bank_values[scope_id]
            durable_occupied = bank_occupied[scope_id]
            row_keys = [durable_keys[index] for index in range(self.capacity)]
            row_values = [durable_values[index] for index in range(self.capacity)]
            row_occupied = durable_occupied.clone()
            for index in range(self.capacity):
                pending = self._pending_writes.get((scope_id, index))
                if pending is None:
                    continue
                pending_key, pending_value, _pending_strength = pending
                pending_key = pending_key.to(
                    device=self.keys.device, dtype=self.keys.dtype
                )
                pending_value = pending_value.to(
                    device=self.values.device, dtype=self.values.dtype
                )
                row_keys[index] = pending_key
                # The pending row is already the differentiable soft state.
                # Mixing it with the durable row again would erase the write
                # gate's gradient as soon as a committed write copied the
                # same value into durable storage.
                row_values[index] = pending_value
                row_occupied[index] = True
            key_rows.append(torch.stack(row_keys, dim=0))
            value_rows.append(torch.stack(row_values, dim=0))
            occupied_rows.append(row_occupied)
        keys = torch.stack(key_rows, dim=0)
        values = torch.stack(value_rows, dim=0)
        occupied = torch.stack(occupied_rows, dim=0)
        keys = torch.nn.functional.normalize(keys, dim=-1)
        query_keys = torch.nn.functional.normalize(
            query.key.to(device=self.keys.device, dtype=self.keys.dtype), dim=-1
        )
        scores = torch.einsum("bw,bkw->bk", query_keys, keys)
        scores = scores.masked_fill(~occupied, -torch.inf)
        top_k = min(query.top_k, self.capacity)
        top_scores, indices = torch.topk(scores, k=top_k, dim=-1)
        finite = torch.isfinite(top_scores)
        matches = finite & (top_scores >= MEMORY_READ_MATCH_THRESHOLD)
        hit = matches.any(dim=-1)
        weights = torch.softmax(top_scores / self.query_temperature, dim=-1)
        weights = torch.where(matches, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        selected_values = torch.gather(
            values,
            1,
            indices.unsqueeze(-1).expand(-1, -1, self.width),
        )
        value = torch.einsum("bk,bkw->bw", weights, selected_values)
        return MemoryRead(
            value=value,
            scores=top_scores,
            indices=indices,
            hit=hit,
        ).validate(width=self.width, batch=query.key.shape[0])

    @torch.no_grad()
    def candidates(self, scope: torch.Tensor | None = None) -> MemoryCandidates:
        batch = 1 if scope is None else int(scope.numel())
        scope_ids = self._scope_ids(scope, batch, device=self.keys.device)
        bank_keys = self._bank(self.keys)
        bank_values = self._bank(self.values)
        bank_strengths = self._bank(self.strengths)
        bank_timestamps = self._bank(self.timestamps)
        bank_occupied = self._bank(self.occupied)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        strengths: list[torch.Tensor] = []
        timestamps: list[torch.Tensor] = []
        occupied: list[torch.Tensor] = []
        for scope_id in scope_ids.tolist():
            row_keys = [bank_keys[scope_id, index] for index in range(self.capacity)]
            row_values = [
                bank_values[scope_id, index] for index in range(self.capacity)
            ]
            row_strengths = bank_strengths[scope_id].clone()
            row_timestamps = bank_timestamps[scope_id].clone()
            row_occupied = bank_occupied[scope_id].clone()
            for index in range(self.capacity):
                pending = self._pending_writes.get((scope_id, index))
                if pending is None:
                    continue
                pending_key, pending_value, pending_strength = pending
                row_keys[index] = pending_key
                row_values[index] = pending_value
                row_strengths[index] = pending_strength
                row_occupied[index] = True
            keys.append(torch.stack(row_keys).detach())
            values.append(torch.stack(row_values).detach())
            strengths.append(row_strengths.detach())
            timestamps.append(row_timestamps.detach())
            occupied.append(row_occupied.detach())
        return MemoryCandidates(
            keys=torch.stack(keys),
            values=torch.stack(values),
            strengths=torch.stack(strengths),
            timestamps=torch.stack(timestamps),
            occupied=torch.stack(occupied),
        ).validate(width=self.width, capacity=self.capacity, batch=batch)

    def migrate_representation_verified(
        self,
        candidate: ContentAddressedMemory,
        address_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
        query_pairs: Sequence[MemoryMigrationExample],
        *,
        prediction_tolerance: float = 1e-6,
        value_alignment: Callable[[torch.Tensor], torch.Tensor] | None = None,
        retention_probe: Callable[[ContentAddressedMemory], bool] | None = None,
    ) -> MemoryMigrationReceipt:
        """Approve a copy-on-write memory replacement across key/value spaces.

        ``address_pairs`` are opaque source/target key correspondences, not
        semantic labels. Every occupied source row must be paired exactly once
        and every protected source row must already be protected at its target
        key. ``query_pairs`` verify retrieval behavior on held-out queries. A
        caller may supply a learned value-space alignment, but it is evaluated
        as an external candidate and never changes either memory during this
        probe.
        """

        if not isinstance(candidate, ContentAddressedMemory):
            raise TypeError("memory migration candidate is invalid")
        if prediction_tolerance < 0.0 or not torch.isfinite(
            torch.tensor(prediction_tolerance)
        ):
            raise ValueError("memory migration tolerance is invalid")
        if not address_pairs or not query_pairs:
            raise ValueError("memory migration needs address and query evidence")
        if (
            self.width != candidate.width
            or self.capacity != candidate.capacity
            or self.scope_capacity != candidate.scope_capacity
            or self.write_threshold != candidate.write_threshold
            or self.query_temperature != candidate.query_temperature
            or self.write_match_threshold != candidate.write_match_threshold
        ):
            raise ValueError("memory migration structural configuration differs")
        if (
            self.key_space_id == candidate.key_space_id
            and self.value_space_id == candidate.value_space_id
        ):
            raise ValueError("memory migration does not replace a representation space")
        source_candidates = self.candidates()
        target_candidates = candidate.candidates()
        source_occupied = source_candidates.occupied[0]
        target_occupied = target_candidates.occupied[0]
        if int(source_occupied.sum()) != len(address_pairs):
            raise ValueError("memory migration does not map every source address")
        if int(target_occupied.sum()) != len(address_pairs):
            raise ValueError("memory migration does not map every target address")

        def normalized_digest(key: torch.Tensor) -> str:
            normalized = torch.nn.functional.normalize(
                key.reshape(1, -1).to(dtype=torch.float32), dim=-1
            )[0]
            return hashlib.sha256(normalized.cpu().contiguous().numpy().tobytes()).hexdigest()

        source_keys: set[str] = set()
        target_keys: set[str] = set()
        protected_count = 0
        for source_key, target_key in address_pairs:
            if source_key.ndim != 1 or target_key.ndim != 1:
                raise ValueError("memory migration address keys must be one-dimensional")
            if source_key.shape[0] != self.width or target_key.shape[0] != self.width:
                raise ValueError("memory migration address keys have the wrong width")
            source_digest = normalized_digest(source_key)
            target_digest = normalized_digest(target_key)
            if source_digest in source_keys or target_digest in target_keys:
                raise ValueError("memory migration address mapping is not one-to-one")
            source_keys.add(source_digest)
            target_keys.add(target_digest)

            source_scores = torch.nn.functional.normalize(
                source_candidates.keys[0], dim=-1
            ) @ torch.nn.functional.normalize(source_key.reshape(1, -1), dim=-1)[0]
            target_scores = torch.nn.functional.normalize(
                target_candidates.keys[0], dim=-1
            ) @ torch.nn.functional.normalize(target_key.reshape(1, -1), dim=-1)[0]
            source_index = int(source_scores.argmax())
            target_index = int(target_scores.argmax())
            if (
                not bool(source_occupied[source_index])
                or float(source_scores[source_index]) < self.write_match_threshold
                or not bool(target_occupied[target_index])
                or float(target_scores[target_index]) < candidate.write_match_threshold
            ):
                raise ValueError("memory migration address pair is not stored")
            if self.retention.is_protected(source_key):
                protected_count += 1
                if not candidate.retention.is_protected(target_key):
                    raise ValueError("protected memory evidence was not transferred")

        if protected_count < 1:
            raise ValueError("memory migration requires protected retention evidence")
        max_difference = 0.0
        for pair in query_pairs:
            source_query = pair.source_query.validate(width=self.width)
            target_query = pair.target_query.validate(width=self.width)
            with torch.no_grad():
                source_read = self.read(source_query)
                target_read = candidate.read(target_query)
                aligned_target = (
                    target_read.value
                    if value_alignment is None
                    else value_alignment(target_read.value)
                )
            if aligned_target.shape != source_read.value.shape:
                raise ValueError("memory value alignment returned the wrong shape")
            if not bool(torch.equal(source_read.hit, target_read.hit)):
                return MemoryMigrationReceipt(
                    accepted=False,
                    source_key_space_id=self.key_space_id,
                    target_key_space_id=candidate.key_space_id,
                    source_value_space_id=self.value_space_id,
                    target_value_space_id=candidate.value_space_id,
                    address_count=len(address_pairs),
                    protected_count=protected_count,
                    query_count=len(query_pairs),
                    max_value_difference=float("inf"),
                    source_digest=self._migration_digest(),
                    target_digest=candidate._migration_digest(),
                    reason="held-out memory hit behavior changed",
                ).validate()
            max_difference = max(
                max_difference,
                float((source_read.value - aligned_target).square().mean()),
            )
        if max_difference > prediction_tolerance:
            accepted = False
            reason = "held-out memory values changed"
        else:
            if retention_probe is not None and not callable(retention_probe):
                raise TypeError("memory migration retention probe is invalid")
            accepted = retention_probe is None or bool(retention_probe(candidate))
            reason = (
                "candidate passed address, retention, and held-out query checks"
                if accepted
                else "candidate retention probe failed"
            )
        return MemoryMigrationReceipt(
            accepted=accepted,
            source_key_space_id=self.key_space_id,
            target_key_space_id=candidate.key_space_id,
            source_value_space_id=self.value_space_id,
            target_value_space_id=candidate.value_space_id,
            address_count=len(address_pairs),
            protected_count=protected_count,
            query_count=len(query_pairs),
            max_value_difference=max_difference,
            source_digest=self._migration_digest(),
            target_digest=candidate._migration_digest(),
            reason=reason,
        ).validate()

    def _migration_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr(self.configuration()).encode("utf-8"))
        digest.update(self._state_checksum(self.state_dict()).encode("utf-8"))
        digest.update(repr(self.retention.payload()).encode("utf-8"))
        return digest.hexdigest()

    def observe_retention(
        self, key: torch.Tensor, outcome: float | torch.Tensor
    ) -> None:
        """Update persistent mastery state from one scalar verifier outcome."""

        if key.ndim != 1:
            raise ValueError("retention key must be one-dimensional")
        self.retention.observe(key, outcome)
        self._save_retention()

    def _save_retention(self) -> None:
        """Hook for persistent backends; in-memory state needs no sidecar."""

        return

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: torch.Tensor,
        *,
        timestamp: torch.Tensor | None = None,
        scope: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
    ) -> MemoryWriteReceipt:
        # Create the view before entering the no-grad durable-state section so
        # the pending transaction path can still differentiate to the caller's
        # write-strength tensor.
        with torch.enable_grad():
            strength_input = strength.reshape(-1)
        with torch.no_grad():
            if key.ndim != 2 or value.ndim != 2 or key.shape != value.shape:
                raise ValueError("memory key and value must have equal [batch, width] shape")
            if not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all()):
                raise ValueError("memory key and value must contain only finite values")
            batch = key.shape[0]
            scope_ids = self._scope_ids(scope, batch, device=self.keys.device)
            self._validate_batch(key, batch, "key")
            self._validate_batch(value, batch, "value")
            if strength.numel() != batch:
                raise ValueError("memory write strength must have one value per batch row")
            if target_index is not None:
                if target_index.numel() != batch or target_index.dtype != torch.long:
                    raise ValueError(
                        "memory target index must be int64 with one value per batch row"
                    )
                target_index = target_index.reshape(batch).to(device=key.device)
                if bool(torch.any(target_index < -1)) or bool(
                    torch.any(target_index >= self.capacity)
                ):
                    raise ValueError("memory target index is outside the capacity")
            strength = strength_input.detach().to(self.strengths)
            if not bool(torch.isfinite(strength).all()):
                raise ValueError("memory write strength must contain only finite values")
            if torch.any(strength < 0) or torch.any(strength > 1):
                raise ValueError("memory write strength must lie in [0, 1]")
            if timestamp is None:
                timestamp = torch.zeros(batch, device=key.device, dtype=key.dtype)
            if timestamp.numel() != batch:
                raise ValueError("memory timestamp must have one value per batch row")
            timestamp = timestamp.reshape(batch).detach().to(self.timestamps)
            if not bool(torch.isfinite(timestamp).all()):
                raise ValueError("memory timestamp must contain only finite values")
            # Equality stays on the differentiable pending side of the
            # boundary. This prevents a zero-initialized write policy at the
            # threshold from committing immediately and losing its gradient
            # path through the transaction.
            committed = strength > self.write_threshold
            indices = torch.full((batch,), -1, device=key.device, dtype=torch.long)
            bank_keys = self._bank(self.keys)
            bank_values = self._bank(self.values)
            bank_strengths = self._bank(self.strengths)
            bank_timestamps = self._bank(self.timestamps)
            bank_occupied = self._bank(self.occupied)
            for row in range(batch):
                scope_id = int(scope_ids[row])
                normalized_keys = torch.nn.functional.normalize(
                    bank_keys[scope_id], dim=-1
                )
                normalized_key = torch.nn.functional.normalize(
                    key[row].detach().to(self.keys), dim=-1
                )
                match_scores = normalized_keys @ normalized_key
                match_scores = match_scores.masked_fill(
                    ~bank_occupied[scope_id], -torch.inf
                )
                best_score, best_index = torch.max(match_scores, dim=0)
                if (
                    bool(torch.isfinite(best_score))
                    and float(best_score) >= self.write_match_threshold
                ):
                    index = int(best_index)
                elif target_index is not None and int(target_index[row]) >= 0:
                    index = int(target_index[row])
                    if bool(bank_occupied[scope_id, index]) and self.retention.is_protected(
                        bank_keys[scope_id, index]
                    ):
                        raise MemoryError(
                            "explicit memory write would evict a protected "
                            "capability; grow or consolidate the memory bank"
                        )
                else:
                    free = torch.nonzero(
                        ~bank_occupied[scope_id], as_tuple=False
                    ).reshape(-1)
                    if free.numel():
                        index = int(free[0])
                    else:
                        candidate_indices = torch.arange(
                            self.capacity, device=self.keys.device
                        )
                        candidate_keys = bank_keys[scope_id]
                        # In the absence of a learned eviction scorer, weak
                        # rows are the disposable baseline. The retention
                        # ledger masks mastered rows before this fallback.
                        candidate_scores = 1.0 - bank_strengths[scope_id]
                        candidate_position = self.retention.choose_eviction_index(
                            candidate_keys, candidate_scores
                        )
                        if candidate_position is None:
                            raise MemoryError(
                                "all occupied memory capabilities are protected; "
                                "grow or consolidate the memory bank"
                            )
                        index = int(candidate_indices[candidate_position])
                if self._transaction_depth:
                    # Keep a continuous effective row for the duration of
                    # the transaction. Durable storage remains discrete, but
                    # reads used by the loss must retain a gradient path even
                    # after this write crosses the commit threshold. Chaining
                    # against an earlier pending row also handles repeated
                    # writes to the same address without dropping the path.
                    with torch.enable_grad():
                        previous = self._pending_writes.get((scope_id, index))
                        if previous is None:
                            previous_key = bank_keys[scope_id, index].detach().clone()
                            previous_value = bank_values[scope_id, index].detach().clone()
                        else:
                            previous_key, previous_value, _ = previous
                        gate = strength_input[row].to(
                            device=self.keys.device, dtype=self.keys.dtype
                        )
                        effective_key = gate * key[row].to(
                            device=self.keys.device, dtype=self.keys.dtype
                        ) + (1.0 - gate) * previous_key.to(
                            device=self.keys.device, dtype=self.keys.dtype
                        )
                        effective_value = gate * value[row].to(
                            device=self.values.device, dtype=self.values.dtype
                        ) + (1.0 - gate) * previous_value.to(
                            device=self.values.device, dtype=self.values.dtype
                        )
                    self._pending_writes[(scope_id, index)] = (
                        effective_key,
                        effective_value,
                        gate,
                    )
                if not bool(committed[row]):
                    continue
                bank_keys[scope_id, index].copy_(key[row].detach().to(self.keys))
                bank_values[scope_id, index].copy_(value[row].detach().to(self.values))
                bank_strengths[scope_id, index].copy_(strength[row].to(self.strengths))
                bank_timestamps[scope_id, index].copy_(timestamp[row].to(self.timestamps))
                bank_occupied[scope_id, index] = True
                indices[row] = index
            if bool(committed.any()):
                self.store_version.add_(1)
            self.validate_state()
            return MemoryWriteReceipt(
                committed=committed,
                indices=indices,
                version=int(self.store_version.item()),
            ).validate(batch=batch)

    @torch.no_grad()
    def clear(self) -> None:
        """Reset persistent rows while preserving the component schema."""
        self._pending_writes.clear()
        self.keys.zero_()
        self.values.zero_()
        self.strengths.zero_()
        self.timestamps.zero_()
        self.occupied.zero_()
        self.store_version.add_(1)
        self.validate_state()

    def snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(
                {
                    "format": MEMORY_SNAPSHOT_FORMAT,
                    "schema": self.schema,
                    "configuration": self.configuration(),
                    "state_dict": self.state_dict(),
                    "state_checksum": self._state_checksum(self.state_dict()),
                },
                temporary_path,
            )
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary_path.replace(path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load_snapshot(self, path: Path, *, map_location: torch.device | str = "cpu") -> None:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        snapshot_format = payload.get("format")
        if snapshot_format not in {
            LEGACY_MEMORY_SNAPSHOT_FORMAT,
            MEMORY_SNAPSHOT_FORMAT,
        }:
            raise ValueError("unsupported memory snapshot format")
        if payload.get("schema") != self.schema:
            raise ValueError("unsupported memory snapshot schema")
        snapshot_configuration = payload.get("configuration")
        expected_configuration = self.configuration()
        if not isinstance(snapshot_configuration, dict):
            raise TypeError("memory snapshot configuration is invalid")
        snapshot_configuration = dict(snapshot_configuration)
        snapshot_configuration.pop("persistence", None)
        expected_configuration.pop("persistence", None)
        legacy_configuration = dict(expected_configuration)
        legacy_configuration.pop("write_match_threshold", None)
        if self.scope_capacity == 1:
            legacy_configuration.pop("scope_capacity", None)
        scope_legacy_configuration = dict(expected_configuration)
        if self.scope_capacity == 1:
            scope_legacy_configuration.pop("scope_capacity", None)
        threshold_legacy_configuration = dict(expected_configuration)
        threshold_legacy_configuration.pop("write_match_threshold", None)
        read_legacy_configuration = dict(expected_configuration)
        read_legacy_configuration.pop("read_match_threshold", None)
        read_legacy_scope_configuration = dict(read_legacy_configuration)
        if self.scope_capacity == 1:
            read_legacy_scope_configuration.pop("scope_capacity", None)
        read_legacy_write_configuration = dict(read_legacy_configuration)
        read_legacy_write_configuration.pop("write_match_threshold", None)
        read_legacy_threshold_configuration = dict(read_legacy_write_configuration)
        if self.scope_capacity == 1:
            read_legacy_threshold_configuration.pop("scope_capacity", None)
        accepted_configurations = (
            (
                expected_configuration,
                legacy_configuration,
                scope_legacy_configuration,
                threshold_legacy_configuration,
                read_legacy_configuration,
                read_legacy_scope_configuration,
                read_legacy_write_configuration,
                read_legacy_threshold_configuration,
            )
            if snapshot_format == LEGACY_MEMORY_SNAPSHOT_FORMAT
            else (
                expected_configuration,
                scope_legacy_configuration,
                read_legacy_configuration,
                read_legacy_scope_configuration,
            )
        )
        # v1/v2 snapshots predate explicit key/value representation spaces.
        # Accept their historical configuration while assigning the runtime
        # defaults supplied by this constructor.
        accepted_configurations = accepted_configurations + tuple(
            {
                key: value
                for key, value in configuration.items()
                if key not in {"key_space_id", "value_space_id"}
            }
            for configuration in accepted_configurations
        )
        if snapshot_configuration not in accepted_configurations:
            raise ValueError("memory snapshot configuration does not match store")
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise TypeError("memory snapshot state is invalid")
        self._validate_state(state)
        if snapshot_format == MEMORY_SNAPSHOT_FORMAT and payload.get(
            "state_checksum"
        ) != self._state_checksum(state):
            raise ValueError("memory snapshot checksum mismatch")
        self.load_state_dict(state)
        self._validate_state()

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "format": self.format,
            "schema": self.schema,
            "width": self.width,
            "key_space_id": self.key_space_id,
            "value_space_id": self.value_space_id,
            "capacity": self.capacity,
            "write_threshold": self.write_threshold,
            "query_temperature": self.query_temperature,
            "write_match_threshold": self.write_match_threshold,
            "read_match_threshold": MEMORY_READ_MATCH_THRESHOLD,
            "scope_capacity": self.scope_capacity,
        }


class PersistentContentAddressedMemory(ContentAddressedMemory):
    """Disk-backed implementation of the same memory contract.

    The in-memory index remains the hot read path; every committed write and
    clear is atomically snapshotted. A second runtime can open the same path
    and retrieve the exact learned keys/values without knowing the producer's
    adapter or task.
    """

    def __init__(self, width: int, capacity: int, path: Path, **kwargs: Any) -> None:
        self.path = Path(path)
        retention_ledger = kwargs.get("retention_ledger")
        super().__init__(width, capacity, **kwargs)
        if self.path.exists():
            self.load_snapshot(self.path)
        retention_path = self._retention_path()
        if retention_ledger is None and retention_path.exists():
            self.retention = CapabilityRetentionLedger.load(retention_path)

    def _retention_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".retention.json")

    def _save_retention(self) -> None:
        self.retention.save(self._retention_path())

    def configuration(self) -> dict[str, int | float | str]:
        configuration = super().configuration()
        configuration["persistence"] = "atomic_snapshot"
        return configuration

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: torch.Tensor,
        *,
        timestamp: torch.Tensor | None = None,
        scope: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
    ) -> MemoryWriteReceipt:
        previous = {name: value.detach().clone() for name, value in self.state_dict().items()}
        previous_pending = dict(self._pending_writes)
        receipt = super().write(
            key,
            value,
            strength,
            timestamp=timestamp,
            scope=scope,
            target_index=target_index,
        )
        if bool(receipt.committed.any()):
            try:
                self.snapshot(self.path)
                self._save_retention()
            except Exception:
                self.load_state_dict(previous)
                self._pending_writes = previous_pending
                raise
        return receipt

    @torch.no_grad()
    def clear(self) -> None:
        previous = {name: value.detach().clone() for name, value in self.state_dict().items()}
        previous_pending = dict(self._pending_writes)
        super().clear()
        try:
            self.snapshot(self.path)
            self._save_retention()
        except Exception:
            self.load_state_dict(previous)
            self._pending_writes = previous_pending
            raise


class AppendOnlyContentAddressedMemory(MemoryBackend):
    """Variable-capacity append-only learned-latent memory.

    Unlike :class:`ContentAddressedMemory`, this backend never chooses a row
    to evict. A new unmatched committed key appends a record, while a matching
    key is updated in place. The logical record count therefore grows without
    changing the controller or backend interface shapes. Record indices are
    opaque receipts and are never part of the controller's learned input.
    """

    format = APPEND_ONLY_MEMORY_BACKEND_FORMAT

    def __init__(
        self,
        width: int,
        *,
        write_threshold: float = 0.5,
        query_temperature: float = 0.1,
        write_match_threshold: float = 0.95,
        read_match_threshold: float = MEMORY_READ_MATCH_THRESHOLD,
        scope_capacity: int = 1,
    ) -> None:
        super().__init__(width)
        if not isinstance(scope_capacity, int) or scope_capacity < 1:
            raise ValueError("memory scope capacity must be positive")
        if (
            not 0.0 <= write_threshold <= 1.0
            or query_temperature <= 0.0
            or not 0.0 <= write_match_threshold <= 1.0
            or not 0.0 <= read_match_threshold <= 1.0
        ):
            raise ValueError("memory thresholds are invalid")
        self.write_threshold = float(write_threshold)
        self.query_temperature = float(query_temperature)
        self.write_match_threshold = float(write_match_threshold)
        self.read_match_threshold = float(read_match_threshold)
        self.scope_capacity = scope_capacity
        self._transaction_depth = 0
        self._pending_writes: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self.register_buffer("keys", torch.empty((0, width)))
        self.register_buffer("values", torch.empty((0, width)))
        self.register_buffer("strengths", torch.empty((0,)))
        self.register_buffer("timestamps", torch.empty((0,)))
        self.register_buffer("scopes", torch.empty((0,), dtype=torch.long))
        self.register_buffer("occupied", torch.empty((0,), dtype=torch.bool))
        self.register_buffer("store_version", torch.zeros((), dtype=torch.long))

    @property
    def record_count(self) -> int:
        return int(self.keys.shape[0])

    def _validate_state(self, state: dict[str, torch.Tensor] | None = None) -> None:
        values = self.state_dict() if state is None else state
        expected = {
            "keys",
            "values",
            "strengths",
            "timestamps",
            "scopes",
            "occupied",
            "store_version",
        }
        if set(values) != expected:
            raise ValueError("append-only memory state has an incompatible field set")
        record_count = values["keys"].shape[0]
        if values["keys"].ndim != 2 or values["keys"].shape[1] != self.width:
            raise ValueError("append-only memory keys have the wrong shape")
        if values["values"].shape != (record_count, self.width):
            raise ValueError("append-only memory values have the wrong shape")
        for name in ("keys", "values", "strengths", "timestamps"):
            if not bool(torch.isfinite(values[name]).all()):
                raise ValueError(f"append-only memory state {name} must be finite")
        for name in ("strengths", "timestamps", "scopes", "occupied"):
            if values[name].shape != (record_count,):
                raise ValueError(f"append-only memory state {name} has the wrong shape")
        if torch.any(values["strengths"] < 0) or torch.any(values["strengths"] > 1):
            raise ValueError("append-only memory strengths must lie in [0, 1]")
        if values["scopes"].dtype != torch.long:
            raise ValueError("append-only memory scopes must be int64")
        if bool(torch.any(values["scopes"] < 0)) or bool(
            torch.any(values["scopes"] >= self.scope_capacity)
        ):
            raise ValueError("append-only memory scopes are outside the configured range")
        if values["occupied"].dtype != torch.bool:
            raise ValueError("append-only memory occupancy must be boolean")
        if (
            values["store_version"].shape != torch.Size([])
            or values["store_version"].dtype != torch.long
            or int(values["store_version"].item()) < 0
        ):
            raise ValueError("append-only memory version must be a non-negative int64 scalar")

    def validate_state(self) -> None:
        self._validate_state()

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        expected = {
            "keys",
            "values",
            "strengths",
            "timestamps",
            "scopes",
            "occupied",
            "store_version",
        }
        if expected.issubset(state_dict):
            self._validate_state({name: state_dict[name] for name in expected})
            for name in expected:
                current = self._buffers[name]
                self._buffers[name] = torch.empty_like(
                    state_dict[name], device=current.device
                )
        self._pending_writes.clear()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _scope_ids(
        self,
        scope: torch.Tensor | None,
        batch: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if scope is None:
            return torch.zeros(batch, device=device, dtype=torch.long)
        if scope.ndim == 0 or scope.numel() != batch:
            raise ValueError("memory scope must have one int64 value per batch row")
        if scope.dtype != torch.long:
            raise ValueError("memory scope must be int64")
        scope = scope.reshape(batch).to(device=device)
        if bool(torch.any(scope < 0)) or bool(torch.any(scope >= self.scope_capacity)):
            raise ValueError("memory scope is outside the configured scope capacity")
        return scope

    def _record_indices(self, scope_id: int) -> torch.Tensor:
        return torch.nonzero(
            self.occupied & (self.scopes == scope_id), as_tuple=False
        ).reshape(-1)

    def _row_tensors(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = [
            self._pending_writes[int(index)][0]
            if int(index) in self._pending_writes
            else self.keys[index]
            for index in indices.tolist()
        ]
        values = [
            self._pending_writes[int(index)][1]
            if int(index) in self._pending_writes
            else self.values[index]
            for index in indices.tolist()
        ]
        if not keys:
            empty = self.keys.new_empty((0, self.width))
            return (
                empty,
                empty.clone(),
                self.strengths.new_empty((0,)),
                self.timestamps.new_empty((0,)),
            )
        strengths = self.strengths[indices].clone()
        timestamps = self.timestamps[indices].clone()
        for position, index in enumerate(indices.tolist()):
            pending = self._pending_writes.get(int(index))
            if pending is not None:
                strengths[position] = pending[2]
        return (
            torch.stack(keys, dim=0),
            torch.stack(values, dim=0),
            strengths,
            timestamps,
        )

    def read(self, query: MemoryQuery) -> MemoryRead:
        query.validate(width=self.width)
        self.validate_state()
        batch = query.key.shape[0]
        scope_ids = self._scope_ids(query.scope, batch, device=self.keys.device)
        query_keys = torch.nn.functional.normalize(
            query.key.to(device=self.keys.device, dtype=self.keys.dtype), dim=-1
        )
        counts = [int(self._record_indices(int(scope_id)).numel()) for scope_id in scope_ids]
        top_k = min(query.top_k, max(counts, default=0))
        if top_k == 0:
            return MemoryRead(
                value=torch.zeros(batch, self.width, device=self.keys.device),
                scores=torch.empty(batch, 0, device=self.keys.device),
                indices=torch.empty(batch, 0, device=self.keys.device, dtype=torch.long),
                hit=torch.zeros(batch, device=self.keys.device, dtype=torch.bool),
            ).validate(width=self.width, batch=batch)

        score_rows: list[torch.Tensor] = []
        index_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        hit_rows: list[torch.Tensor] = []
        for row, scope_id in enumerate(scope_ids.tolist()):
            record_indices = self._record_indices(scope_id)
            row_keys, row_values, _strengths, _timestamps = self._row_tensors(
                record_indices
            )
            row_keys = torch.nn.functional.normalize(row_keys, dim=-1)
            row_scores = row_keys @ query_keys[row]
            selected_scores, selected_positions = torch.topk(
                row_scores, k=min(top_k, row_scores.shape[0])
            )
            selected_indices = record_indices[selected_positions]
            selected_values = row_values[selected_positions]
            if selected_scores.shape[0] < top_k:
                pad = top_k - selected_scores.shape[0]
                selected_scores = torch.cat(
                    [
                        selected_scores,
                        torch.full(
                            (pad,),
                            -torch.inf,
                            dtype=selected_scores.dtype,
                            device=selected_scores.device,
                        ),
                    ]
                )
                selected_indices = torch.cat(
                    [selected_indices, torch.full((pad,), -1, device=selected_indices.device)]
                )
                selected_values = torch.cat(
                    [selected_values, torch.zeros(pad, self.width, device=row_values.device)]
                )
            finite = torch.isfinite(selected_scores)
            matches = finite & (selected_scores >= self.read_match_threshold)
            safe_scores = torch.where(
                finite, selected_scores, torch.zeros_like(selected_scores)
            )
            weights = torch.softmax(safe_scores / self.query_temperature, dim=0)
            weights = torch.where(matches, weights, torch.zeros_like(weights))
            weights = weights / weights.sum().clamp_min(1e-12)
            score_rows.append(selected_scores)
            index_rows.append(selected_indices)
            value_rows.append(torch.einsum("k,kw->w", weights, selected_values))
            hit_rows.append(matches.any())
        return MemoryRead(
            value=torch.stack(value_rows),
            scores=torch.stack(score_rows),
            indices=torch.stack(index_rows),
            hit=torch.stack(hit_rows),
        ).validate(width=self.width, batch=batch)

    @torch.no_grad()
    def candidates(self, scope: torch.Tensor | None = None) -> MemoryCandidates:
        batch = 1 if scope is None else int(scope.numel())
        scope_ids = self._scope_ids(scope, batch, device=self.keys.device)
        record_indices = [self._record_indices(int(scope_id)) for scope_id in scope_ids]
        capacity = max((int(indices.numel()) for indices in record_indices), default=0)
        key_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        strength_rows: list[torch.Tensor] = []
        timestamp_rows: list[torch.Tensor] = []
        occupied_rows: list[torch.Tensor] = []
        for indices in record_indices:
            keys, values, strengths, timestamps = self._row_tensors(indices)
            pad = capacity - keys.shape[0]
            if pad:
                keys = torch.cat([keys, self.keys.new_zeros((pad, self.width))])
                values = torch.cat([values, self.values.new_zeros((pad, self.width))])
                strengths = torch.cat([strengths, self.strengths.new_zeros((pad,))])
                timestamps = torch.cat(
                    [timestamps, self.timestamps.new_zeros((pad,))]
                )
            key_rows.append(keys.detach())
            value_rows.append(values.detach())
            strength_rows.append(strengths.detach())
            timestamp_rows.append(timestamps.detach())
            occupied_rows.append(
                torch.cat(
                    [
                        torch.ones(indices.shape[0], dtype=torch.bool, device=self.keys.device),
                        torch.zeros(pad, dtype=torch.bool, device=self.keys.device),
                    ]
                )
            )
        return MemoryCandidates(
            keys=torch.stack(key_rows),
            values=torch.stack(value_rows),
            strengths=torch.stack(strength_rows),
            timestamps=torch.stack(timestamp_rows),
            occupied=torch.stack(occupied_rows),
        ).validate(width=self.width, capacity=capacity, batch=batch)

    def _append_record(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: torch.Tensor,
        timestamp: torch.Tensor,
        scope_id: torch.Tensor,
    ) -> int:
        index = self.record_count
        self._buffers["keys"] = torch.cat(
            [self.keys, key.reshape(1, self.width).detach().to(self.keys)]
        )
        self._buffers["values"] = torch.cat(
            [self.values, value.reshape(1, self.width).detach().to(self.values)]
        )
        self._buffers["strengths"] = torch.cat(
            [self.strengths, strength.reshape(1).detach().to(self.strengths)]
        )
        self._buffers["timestamps"] = torch.cat(
            [self.timestamps, timestamp.reshape(1).detach().to(self.timestamps)]
        )
        self._buffers["scopes"] = torch.cat(
            [self.scopes, scope_id.reshape(1).detach().to(self.scopes)]
        )
        self._buffers["occupied"] = torch.cat(
            [self.occupied, torch.ones(1, dtype=torch.bool, device=self.keys.device)]
        )
        return index

    @torch.no_grad()
    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: torch.Tensor,
        *,
        timestamp: torch.Tensor | None = None,
        scope: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
    ) -> MemoryWriteReceipt:
        if key.ndim != 2 or value.ndim != 2 or key.shape != value.shape:
            raise ValueError("memory key and value must have equal [batch, width] shape")
        if key.shape[1] != self.width:
            raise ValueError(f"memory key and value must have width {self.width}")
        if not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all()):
            raise ValueError("memory key and value must contain only finite values")
        batch = key.shape[0]
        scope_ids = self._scope_ids(scope, batch, device=self.keys.device)
        if strength.numel() != batch:
            raise ValueError("memory write strength must have one value per batch row")
        with torch.enable_grad():
            strength_input = strength.reshape(-1)
        strength_value = strength_input.detach().to(self.strengths)
        if not bool(torch.isfinite(strength_value).all()) or bool(
            torch.any(strength_value < 0) or torch.any(strength_value > 1)
        ):
            raise ValueError("memory write strength must lie in [0, 1]")
        if target_index is not None and bool(torch.any(target_index != -1)):
            raise ValueError("append-only memory does not support target_index replacement")
        if timestamp is None:
            timestamp = torch.zeros(batch, device=key.device, dtype=key.dtype)
        if timestamp.numel() != batch:
            raise ValueError("memory timestamp must have one value per batch row")
        timestamp = timestamp.reshape(batch).detach().to(self.timestamps)
        if not bool(torch.isfinite(timestamp).all()):
            raise ValueError("memory timestamp must contain only finite values")
        committed = strength_value > self.write_threshold
        indices = torch.full((batch,), -1, device=key.device, dtype=torch.long)
        for row in range(batch):
            if not bool(committed[row]):
                continue
            scope_id = int(scope_ids[row])
            normalized_key = torch.nn.functional.normalize(
                key[row].detach().to(self.keys), dim=0
            )
            record_indices = self._record_indices(scope_id)
            if record_indices.numel():
                row_keys, _row_values, _row_strengths, _row_timestamps = self._row_tensors(
                    record_indices
                )
                scores = torch.nn.functional.normalize(row_keys, dim=-1) @ normalized_key
                best_score, best_position = torch.max(scores, dim=0)
            else:
                best_score = torch.tensor(-torch.inf, device=self.keys.device)
                best_position = torch.tensor(0, device=self.keys.device, dtype=torch.long)
            if bool(torch.isfinite(best_score)) and float(best_score) >= self.write_match_threshold:
                index = int(record_indices[best_position])
                pending_previous = self._pending_writes.get(index)
                previous_key = (
                    pending_previous[0].detach().clone()
                    if pending_previous is not None
                    else self.keys[index].detach().clone()
                )
                previous_value = (
                    pending_previous[1].detach().clone()
                    if pending_previous is not None
                    else self.values[index].detach().clone()
                )
                self._buffers["keys"][index].copy_(key[row].detach().to(self.keys))
                self._buffers["values"][index].copy_(value[row].detach().to(self.values))
                self._buffers["strengths"][index].copy_(strength_value[row])
                self._buffers["timestamps"][index].copy_(timestamp[row])
            else:
                previous_key = self.keys.new_zeros((self.width,))
                previous_value = self.values.new_zeros((self.width,))
                index = self._append_record(
                    key[row], value[row], strength_value[row], timestamp[row], scope_ids[row]
                )
            if self._transaction_depth:
                with torch.enable_grad():
                    gate = strength_input[row].to(
                        device=self.values.device, dtype=self.values.dtype
                    )
                    effective_key = gate * key[row].to(self.keys) + (
                        1.0 - gate
                    ) * previous_key
                    effective_value = gate * value[row].to(self.values) + (
                        1.0 - gate
                    ) * previous_value
                self._pending_writes[index] = (effective_key, effective_value, gate)
            indices[row] = index
        if bool(committed.any()):
            self.store_version.add_(1)
        self.validate_state()
        return MemoryWriteReceipt(
            committed=committed.to(device=key.device),
            indices=indices,
            version=int(self.store_version.item()),
        ).validate(batch=batch)

    @contextmanager
    def differentiable_transaction(self) -> Iterator[MemoryBackend]:
        if self._transaction_depth:
            raise RuntimeError("nested differentiable memory transactions are unsupported")
        self._transaction_depth = 1
        self._pending_writes = {}
        try:
            yield self
        finally:
            self._pending_writes.clear()
            self._transaction_depth = 0

    @torch.no_grad()
    def clear(self) -> None:
        self._pending_writes.clear()
        device = self.keys.device
        self._buffers["keys"] = torch.empty((0, self.width), device=device)
        self._buffers["values"] = torch.empty((0, self.width), device=device)
        self._buffers["strengths"] = torch.empty((0,), device=device)
        self._buffers["timestamps"] = torch.empty((0,), device=device)
        self._buffers["scopes"] = torch.empty((0,), dtype=torch.long, device=device)
        self._buffers["occupied"] = torch.empty((0,), dtype=torch.bool, device=device)
        self.store_version.add_(1)
        self.validate_state()

    @staticmethod
    def _state_checksum(state: dict[str, torch.Tensor]) -> str:
        return ContentAddressedMemory._state_checksum(state)

    def snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            state = self.state_dict()
            torch.save(
                {
                    "format": APPEND_ONLY_MEMORY_SNAPSHOT_FORMAT,
                    "schema": self.schema,
                    "configuration": self.configuration(),
                    "state_dict": state,
                    "state_checksum": self._state_checksum(state),
                },
                temporary_path,
            )
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary_path.replace(path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load_snapshot(
        self, path: Path, *, map_location: torch.device | str = "cpu"
    ) -> None:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("format") != APPEND_ONLY_MEMORY_SNAPSHOT_FORMAT:
            raise ValueError("unsupported append-only memory snapshot format")
        if payload.get("schema") != self.schema:
            raise ValueError("unsupported memory snapshot schema")
        if payload.get("configuration") != self.configuration():
            raise ValueError("append-only memory snapshot configuration does not match store")
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise TypeError("append-only memory snapshot state is invalid")
        self._validate_state(state)
        if payload.get("state_checksum") != self._state_checksum(state):
            raise ValueError("append-only memory snapshot checksum mismatch")
        self.load_state_dict(state)
        self.validate_state()

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "format": self.format,
            "schema": self.schema,
            "width": self.width,
            "storage": "append_only",
            "write_threshold": self.write_threshold,
            "query_temperature": self.query_temperature,
            "write_match_threshold": self.write_match_threshold,
            "read_match_threshold": self.read_match_threshold,
            "scope_capacity": self.scope_capacity,
        }


class PersistentAppendOnlyContentAddressedMemory(AppendOnlyContentAddressedMemory):
    """Atomically persisted variable-capacity append-only memory."""

    def __init__(self, width: int, path: Path, **kwargs: Any) -> None:
        self.path = Path(path)
        super().__init__(width, **kwargs)
        if self.path.exists():
            self.load_snapshot(self.path)

    def configuration(self) -> dict[str, int | float | str]:
        configuration = super().configuration()
        configuration["persistence"] = "atomic_snapshot"
        return configuration

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: torch.Tensor,
        *,
        timestamp: torch.Tensor | None = None,
        scope: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
    ) -> MemoryWriteReceipt:
        previous = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }
        previous_pending = dict(self._pending_writes)
        receipt = super().write(
            key,
            value,
            strength,
            timestamp=timestamp,
            scope=scope,
            target_index=target_index,
        )
        if bool(receipt.committed.any()):
            try:
                self.snapshot(self.path)
            except Exception:
                self.load_state_dict(previous)
                self._pending_writes = previous_pending
                raise
        return receipt

    @torch.no_grad()
    def clear(self) -> None:
        previous = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }
        previous_pending = dict(self._pending_writes)
        super().clear()
        try:
            self.snapshot(self.path)
        except Exception:
            self.load_state_dict(previous)
            self._pending_writes = previous_pending
            raise
