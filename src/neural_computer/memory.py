"""Versioned content-addressed long-term memory for the production runtime."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .interface import MEMORY_SCHEMA

MEMORY_BACKEND_FORMAT = "neural-computer.memory-backend.v1"
LEGACY_MEMORY_SNAPSHOT_FORMAT = "neural-computer.memory-snapshot.v1"
MEMORY_SNAPSHOT_FORMAT = "neural-computer.memory-snapshot.v2"
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

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: torch.Tensor,
        *,
        timestamp: torch.Tensor | None = None,
        scope: torch.Tensor | None = None,
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

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: torch.Tensor,
        *,
        timestamp: torch.Tensor | None = None,
        scope: torch.Tensor | None = None,
    ) -> MemoryWriteReceipt:
        # Create the view before entering the no-grad durable-state section so
        # the pending transaction path can still differentiate to the caller's
        # write-strength tensor.
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
                else:
                    free = torch.nonzero(
                        ~bank_occupied[scope_id], as_tuple=False
                    ).reshape(-1)
                    index = (
                        int(free[0])
                        if free.numel()
                        else int(torch.argmin(bank_strengths[scope_id]))
                    )
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
        super().__init__(width, capacity, **kwargs)
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
    ) -> MemoryWriteReceipt:
        previous = {name: value.detach().clone() for name, value in self.state_dict().items()}
        previous_pending = dict(self._pending_writes)
        receipt = super().write(
            key, value, strength, timestamp=timestamp, scope=scope
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
        previous = {name: value.detach().clone() for name, value in self.state_dict().items()}
        previous_pending = dict(self._pending_writes)
        super().clear()
        try:
            self.snapshot(self.path)
        except Exception:
            self.load_state_dict(previous)
            self._pending_writes = previous_pending
            raise
