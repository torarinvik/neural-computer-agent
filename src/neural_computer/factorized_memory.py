"""Shared-basis content memory for verifier-gated external compression.

This backend keeps content-addressed keys and logical records independent while
factorizing only the stored value payloads.  A shared orthonormal basis is
external, appendable state; each record stores coefficients over that basis.
Reads materialize ordinary learned values before crossing the memory ABI, so
the controller does not know whether storage is dense or factorized.

Basis reduction is never implicit.  A caller builds a copy-on-write candidate,
checks held-out behavior, and commits it through ``replace_from_candidate``
with an optional expected store version.  This makes representation
compression independently replaceable and prevents a memory-side optimizer
from overwriting newer evidence.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .memory import (
    ContentAddressedMemory,
    MemoryBackend,
    MemoryCandidates,
    MemoryQuery,
    MemoryRead,
    MemoryWriteReceipt,
)

SHARED_BASIS_MEMORY_BACKEND_FORMAT = (
    "neural-computer.shared-basis-memory-backend.v1"
)
SHARED_BASIS_MEMORY_SNAPSHOT_FORMAT = (
    "neural-computer.shared-basis-memory-snapshot.v1"
)
SHARED_BASIS_MEMORY_SCHEMA = "neural-computer.shared-basis-memory.v1"
SHARED_BASIS_COMPRESSION_SCHEMA = "neural-computer.shared-basis-compression.v1"
SHARED_BASIS_REWRITE_SCHEMA = "neural-computer.shared-basis-rewrite.v1"


@dataclass(frozen=True)
class SharedBasisCompressionReceipt:
    """Auditable replacement of factorized value storage."""

    accepted: bool
    rows_before: int
    rows_after: int
    basis_rows_before: int
    basis_rows_after: int
    max_value_error: float
    version: int
    reason: str
    schema: str = SHARED_BASIS_COMPRESSION_SCHEMA

    def validate(self) -> SharedBasisCompressionReceipt:
        if self.schema != SHARED_BASIS_COMPRESSION_SCHEMA:
            raise ValueError("unsupported shared-basis compression schema")
        if min(
            self.rows_before,
            self.rows_after,
            self.basis_rows_before,
            self.basis_rows_after,
            self.version,
        ) < 0:
            raise ValueError("shared-basis compression counts are invalid")
        if self.max_value_error < 0.0 or not bool(
            torch.isfinite(torch.tensor(self.max_value_error))
        ):
            raise ValueError("shared-basis compression error is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("shared-basis compression reason is missing")
        return self


@dataclass(frozen=True)
class SharedBasisRewriteReceipt:
    """Auditable verifier-gated logical record replacement."""

    accepted: bool
    rows_before: int
    rows_after: int
    basis_rows_before: int
    basis_rows_after: int
    version: int
    reason: str
    schema: str = SHARED_BASIS_REWRITE_SCHEMA

    def validate(self) -> SharedBasisRewriteReceipt:
        if self.schema != SHARED_BASIS_REWRITE_SCHEMA:
            raise ValueError("unsupported shared-basis rewrite schema")
        if min(
            self.rows_before,
            self.rows_after,
            self.basis_rows_before,
            self.basis_rows_after,
            self.version,
        ) < 0:
            raise ValueError("shared-basis rewrite counts are invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("shared-basis rewrite reason is missing")
        return self


class SharedBasisContentAddressedMemory(MemoryBackend):
    """Variable-capacity content memory with shared factorized values.

    Keys remain one independent learned address per logical record.  Values
    are represented as ``coefficients @ basis`` and are materialized into the
    ordinary :class:`MemoryRead` and :class:`MemoryCandidates` contracts.
    New writes append an orthogonal basis direction only when the current
    basis cannot represent the value within ``basis_tolerance``.  Existing
    coefficient rows receive zero padding and therefore retain their values
    during basis growth.
    """

    format = SHARED_BASIS_MEMORY_BACKEND_FORMAT
    schema = SHARED_BASIS_MEMORY_SCHEMA

    def __init__(
        self,
        width: int,
        *,
        write_threshold: float = 0.5,
        write_match_threshold: float = 0.95,
        read_match_threshold: float = 0.75,
        basis_tolerance: float = 1e-6,
        scope_capacity: int = 1,
    ) -> None:
        super().__init__(width)
        if width < 1 or not isinstance(scope_capacity, int) or scope_capacity < 1:
            raise ValueError("shared-basis memory dimensions are invalid")
        if not (
            0.0 <= write_threshold <= 1.0
            and 0.0 <= write_match_threshold <= 1.0
            and 0.0 <= read_match_threshold <= 1.0
            and basis_tolerance > 0.0
        ):
            raise ValueError("shared-basis memory thresholds are invalid")
        self.write_threshold = float(write_threshold)
        self.write_match_threshold = float(write_match_threshold)
        self.read_match_threshold = float(read_match_threshold)
        self.basis_tolerance = float(basis_tolerance)
        self.scope_capacity = int(scope_capacity)
        self.register_buffer("basis", torch.empty((0, width)))
        self.register_buffer("keys", torch.empty((0, width)))
        self.register_buffer("coefficients", torch.empty((0, 0)))
        self.register_buffer("strengths", torch.empty((0,)))
        self.register_buffer("timestamps", torch.empty((0,)))
        self.register_buffer("scopes", torch.empty((0,), dtype=torch.long))
        self.register_buffer("occupied", torch.empty((0,), dtype=torch.bool))
        self.register_buffer("store_version", torch.zeros((), dtype=torch.long))

    @property
    def record_count(self) -> int:
        return int(self.keys.shape[0])

    @property
    def basis_count(self) -> int:
        return int(self.basis.shape[0])

    @property
    def physical_value_scalar_count(self) -> int:
        """Number of stored scalars for basis plus coefficient payloads."""

        return int(self.basis.numel() + self.coefficients.numel())

    @property
    def dense_value_scalar_count(self) -> int:
        return self.record_count * self.width

    def _validate_state(
        self, state: dict[str, torch.Tensor] | None = None
    ) -> None:
        values = self.state_dict() if state is None else state
        expected = {
            "basis",
            "keys",
            "coefficients",
            "strengths",
            "timestamps",
            "scopes",
            "occupied",
            "store_version",
        }
        if set(values) != expected:
            raise ValueError("shared-basis memory state has an incompatible field set")
        basis = values["basis"]
        keys = values["keys"]
        coefficients = values["coefficients"]
        record_count = keys.shape[0]
        basis_count = basis.shape[0]
        if basis.ndim != 2 or basis.shape[1] != self.width:
            raise ValueError("shared-basis memory basis has the wrong shape")
        if basis_count > self.width:
            raise ValueError("shared-basis memory basis exceeds value width")
        if keys.ndim != 2 or keys.shape != (record_count, self.width):
            raise ValueError("shared-basis memory keys have the wrong shape")
        if coefficients.shape != (record_count, basis_count):
            raise ValueError("shared-basis memory coefficients have the wrong shape")
        for name in (
            "basis",
            "keys",
            "coefficients",
            "strengths",
            "timestamps",
        ):
            if not bool(torch.isfinite(values[name]).all()):
                raise ValueError(f"shared-basis memory {name} must be finite")
        if basis_count:
            gram = basis @ basis.transpose(0, 1)
            identity = torch.eye(basis_count, device=basis.device, dtype=basis.dtype)
            if not bool(torch.allclose(gram, identity, atol=2e-4, rtol=2e-4)):
                raise ValueError("shared-basis memory basis must be orthonormal")
        for name in ("strengths", "timestamps", "scopes", "occupied"):
            if values[name].shape != (record_count,):
                raise ValueError(f"shared-basis memory {name} has the wrong shape")
        if bool(torch.any(values["strengths"] < 0)) or bool(
            torch.any(values["strengths"] > 1)
        ):
            raise ValueError("shared-basis memory strengths must lie in [0, 1]")
        if values["scopes"].dtype != torch.long:
            raise ValueError("shared-basis memory scopes must be int64")
        if bool(torch.any(values["scopes"] < 0)) or bool(
            torch.any(values["scopes"] >= self.scope_capacity)
        ):
            raise ValueError("shared-basis memory scopes are outside the range")
        if values["occupied"].dtype != torch.bool:
            raise ValueError("shared-basis memory occupancy must be boolean")
        if (
            values["store_version"].shape != torch.Size([])
            or values["store_version"].dtype != torch.long
            or int(values["store_version"].item()) < 0
        ):
            raise ValueError("shared-basis memory version is invalid")

    def validate_state(self) -> None:
        self._validate_state()

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        expected = {
            "basis",
            "keys",
            "coefficients",
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
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _scope_ids(
        self,
        scope: torch.Tensor | None,
        batch: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if scope is None:
            return torch.zeros(batch, dtype=torch.long, device=device)
        if scope.ndim == 0 or scope.numel() != batch or scope.dtype != torch.long:
            raise ValueError("shared-basis memory scope must be int64 per batch row")
        scope = scope.reshape(batch).to(device=device)
        if bool(torch.any(scope < 0)) or bool(
            torch.any(scope >= self.scope_capacity)
        ):
            raise ValueError("shared-basis memory scope is outside the range")
        return scope

    def _record_indices(self, scope_id: int) -> torch.Tensor:
        return torch.nonzero(
            self.occupied & (self.scopes == scope_id), as_tuple=False
        ).reshape(-1)

    def _materialized_values(self) -> torch.Tensor:
        if not self.basis_count:
            return self.keys.new_zeros((self.record_count, self.width))
        return self.coefficients @ self.basis

    def _values_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if not indices.numel():
            return self.keys.new_empty((0, self.width))
        if not self.basis_count:
            return self.keys.new_zeros((indices.numel(), self.width))
        return self.coefficients[indices] @ self.basis

    def read(self, query: MemoryQuery) -> MemoryRead:
        query.validate(width=self.width)
        self.validate_state()
        batch = query.key.shape[0]
        scope_ids = self._scope_ids(query.scope, batch, device=self.keys.device)
        query_keys = torch.nn.functional.normalize(
            query.key.to(device=self.keys.device, dtype=self.keys.dtype), dim=-1
        )
        counts = [
            int(self._record_indices(int(scope_id)).numel()) for scope_id in scope_ids
        ]
        top_k = min(query.top_k, max(counts, default=0))
        if top_k == 0:
            return MemoryRead(
                value=torch.zeros(batch, self.width, device=self.keys.device),
                scores=torch.empty(batch, 0, device=self.keys.device),
                indices=torch.empty(
                    batch, 0, device=self.keys.device, dtype=torch.long
                ),
                hit=torch.zeros(batch, device=self.keys.device, dtype=torch.bool),
            ).validate(width=self.width, batch=batch)
        score_rows: list[torch.Tensor] = []
        index_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        hit_rows: list[torch.Tensor] = []
        for row, scope_id in enumerate(scope_ids.tolist()):
            record_indices = self._record_indices(scope_id)
            row_keys = torch.nn.functional.normalize(
                self.keys[record_indices], dim=-1
            )
            row_values = self._values_for_indices(record_indices)
            row_scores = row_keys @ query_keys[row]
            selected_scores, selected_positions = torch.topk(
                row_scores, k=min(top_k, row_scores.shape[0])
            )
            selected_indices = record_indices[selected_positions]
            selected_values = row_values[selected_positions]
            if selected_scores.shape[0] < top_k:
                padding = top_k - selected_scores.shape[0]
                selected_scores = torch.cat(
                    [
                        selected_scores,
                        torch.full(
                            (padding,),
                            -torch.inf,
                            device=selected_scores.device,
                        ),
                    ]
                )
                selected_indices = torch.cat(
                    [
                        selected_indices,
                        torch.full(
                            (padding,),
                            -1,
                            dtype=torch.long,
                            device=selected_indices.device,
                        ),
                    ]
                )
                selected_values = torch.cat(
                    [
                        selected_values,
                        torch.zeros(padding, self.width, device=row_values.device),
                    ]
                )
            finite = torch.isfinite(selected_scores)
            matches = finite & (selected_scores >= self.read_match_threshold)
            safe_scores = torch.where(
                finite, selected_scores, torch.zeros_like(selected_scores)
            )
            weights = torch.softmax(safe_scores, dim=0)
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
        index_rows = [self._record_indices(int(scope_id)) for scope_id in scope_ids]
        capacity = max((int(indices.numel()) for indices in index_rows), default=0)
        key_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        strength_rows: list[torch.Tensor] = []
        timestamp_rows: list[torch.Tensor] = []
        occupied_rows: list[torch.Tensor] = []
        for indices in index_rows:
            keys = self.keys[indices]
            values = self._values_for_indices(indices)
            strengths = self.strengths[indices]
            timestamps = self.timestamps[indices]
            padding = capacity - keys.shape[0]
            if padding:
                keys = torch.cat([keys, self.keys.new_zeros((padding, self.width))])
                values = torch.cat(
                    [values, self.keys.new_zeros((padding, self.width))]
                )
                strengths = torch.cat(
                    [strengths, self.strengths.new_zeros((padding,))]
                )
                timestamps = torch.cat(
                    [timestamps, self.timestamps.new_zeros((padding,))]
                )
            key_rows.append(keys.detach())
            value_rows.append(values.detach())
            strength_rows.append(strengths.detach())
            timestamp_rows.append(timestamps.detach())
            occupied_rows.append(
                torch.cat(
                    [
                        torch.ones(
                            indices.shape[0],
                            dtype=torch.bool,
                            device=self.keys.device,
                        ),
                        torch.zeros(
                            padding,
                            dtype=torch.bool,
                            device=self.keys.device,
                        ),
                    ]
                )
            )
        if not key_rows:
            key_rows = [self.keys.new_empty((0, self.width))]
            value_rows = [self.keys.new_empty((0, self.width))]
            strength_rows = [self.strengths.new_empty((0,))]
            timestamp_rows = [self.timestamps.new_empty((0,))]
            occupied_rows = [torch.empty((0,), dtype=torch.bool)]
        return MemoryCandidates(
            keys=torch.stack(key_rows),
            values=torch.stack(value_rows),
            strengths=torch.stack(strength_rows),
            timestamps=torch.stack(timestamp_rows),
            occupied=torch.stack(occupied_rows),
        ).validate(width=self.width, capacity=capacity, batch=batch)

    @torch.no_grad()
    def _encode_value(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(device=self.keys.device, dtype=self.keys.dtype)
        if self.basis_count:
            coefficients = value @ self.basis.transpose(0, 1)
            residual = value - coefficients @ self.basis
        else:
            coefficients = value.new_empty((0,))
            residual = value
        residual_norm = torch.linalg.vector_norm(residual)
        if self.basis_count >= self.width:
            # A square orthonormal basis spans the complete value space. Any
            # remaining residue is projection round-off, not a new direction.
            return coefficients
        if float(residual_norm) > self.basis_tolerance:
            direction = residual
            # Re-orthogonalize against the retained basis before appending.
            # The second pass prevents small projection errors from becoming
            # visible as non-orthogonality after many online growth steps.
            for _ in range(2):
                if self.basis_count:
                    direction = direction - (
                        direction @ self.basis.transpose(0, 1)
                    ) @ self.basis
            direction_norm = torch.linalg.vector_norm(direction)
            if float(direction_norm) <= self.basis_tolerance:
                return coefficients
            direction = direction / direction_norm
            new_basis = torch.cat([self.basis, direction.unsqueeze(0)], dim=0)
            if self.record_count:
                self._buffers["coefficients"] = torch.cat(
                    [
                        self.coefficients,
                        self.coefficients.new_zeros((self.record_count, 1)),
                    ],
                    dim=1,
                )
            self._buffers["basis"] = new_basis
            if not self.record_count:
                self._buffers["coefficients"] = self.coefficients.new_empty(
                    (0, new_basis.shape[0])
                )
            coefficients = value @ new_basis.transpose(0, 1)
        return coefficients

    @torch.no_grad()
    def _append_record(
        self,
        key: torch.Tensor,
        coefficients: torch.Tensor,
        strength: torch.Tensor,
        timestamp: torch.Tensor,
        scope: torch.Tensor,
    ) -> int:
        index = self.record_count
        self._buffers["keys"] = torch.cat(
            [self.keys, key.reshape(1, self.width).to(self.keys)]
        )
        self._buffers["coefficients"] = torch.cat(
            [
                self.coefficients,
                coefficients.reshape(1, self.basis_count).to(self.coefficients),
            ]
        )
        self._buffers["strengths"] = torch.cat(
            [self.strengths, strength.reshape(1).to(self.strengths)]
        )
        self._buffers["timestamps"] = torch.cat(
            [self.timestamps, timestamp.reshape(1).to(self.timestamps)]
        )
        self._buffers["scopes"] = torch.cat(
            [self.scopes, scope.reshape(1).to(self.scopes)]
        )
        self._buffers["occupied"] = torch.cat(
            [
                self.occupied,
                torch.ones(1, dtype=torch.bool, device=self.keys.device),
            ]
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
            raise ValueError("shared-basis memory key/value shapes must match")
        if key.shape[1] != self.width or not bool(torch.isfinite(key).all()) or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError("shared-basis memory key/value tensors are invalid")
        batch = key.shape[0]
        if strength.numel() != batch:
            raise ValueError("shared-basis memory strength must match the batch")
        strength = strength.reshape(batch).detach().to(self.keys)
        if not bool(torch.isfinite(strength).all()) or bool(
            torch.any(strength < 0) or torch.any(strength > 1)
        ):
            raise ValueError("shared-basis memory strength must lie in [0, 1]")
        if target_index is not None:
            raise ValueError("shared-basis memory does not support target_index")
        if timestamp is None:
            timestamp = torch.zeros(batch, device=key.device, dtype=key.dtype)
        if timestamp.numel() != batch:
            raise ValueError("shared-basis memory timestamp must match the batch")
        timestamp = timestamp.reshape(batch).detach().to(self.timestamps)
        if not bool(torch.isfinite(timestamp).all()):
            raise ValueError("shared-basis memory timestamps must be finite")
        scope_ids = self._scope_ids(scope, batch, device=self.keys.device)
        committed = strength > self.write_threshold
        indices = torch.full((batch,), -1, dtype=torch.long, device=key.device)
        previous = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }
        try:
            for row in range(batch):
                if not bool(committed[row]):
                    continue
                scope_id = int(scope_ids[row])
                normalized_key = torch.nn.functional.normalize(
                    key[row].detach().to(self.keys), dim=0
                )
                record_indices = self._record_indices(scope_id)
                if record_indices.numel():
                    scores = torch.nn.functional.normalize(
                        self.keys[record_indices], dim=-1
                    ) @ normalized_key
                    best_score, position = scores.max(dim=0)
                else:
                    best_score = torch.tensor(-torch.inf, device=self.keys.device)
                    position = torch.tensor(0, dtype=torch.long, device=self.keys.device)
                coefficients = self._encode_value(value[row])
                if bool(torch.isfinite(best_score)) and float(best_score) >= self.write_match_threshold:
                    index = int(record_indices[position])
                    self._buffers["keys"][index].copy_(key[row].to(self.keys))
                    self._buffers["coefficients"][index].copy_(coefficients)
                    self._buffers["strengths"][index].copy_(strength[row])
                    self._buffers["timestamps"][index].copy_(timestamp[row])
                else:
                    index = self._append_record(
                        key[row], coefficients, strength[row], timestamp[row], scope_ids[row]
                    )
                indices[row] = index
            if bool(committed.any()):
                self.store_version.add_(1)
            self.validate_state()
        except Exception:
            self.load_state_dict(previous)
            raise
        return MemoryWriteReceipt(
            committed=committed.to(device=key.device),
            indices=indices,
            version=int(self.store_version.item()),
        ).validate(batch=batch)

    @contextmanager
    def differentiable_transaction(self):
        """Shared-basis writes are discrete; reads remain ABI-compatible."""

        yield self

    @torch.no_grad()
    def compression_candidate(
        self, basis_rows: int
    ) -> SharedBasisContentAddressedMemory:
        """Build a copy-on-write low-rank candidate from current values."""

        if (
            not isinstance(basis_rows, int)
            or isinstance(basis_rows, bool)
            or not 1 <= basis_rows <= self.width
        ):
            raise ValueError("shared-basis candidate rank is invalid")
        if not self.record_count:
            raise ValueError("shared-basis candidate needs at least one record")
        values = self._materialized_values()
        _u, _s, right_singular = torch.linalg.svd(values, full_matrices=False)
        rank = min(basis_rows, right_singular.shape[0])
        basis = right_singular[:rank].contiguous()
        coefficients = values @ basis.transpose(0, 1)
        candidate = SharedBasisContentAddressedMemory(
            self.width,
            write_threshold=self.write_threshold,
            write_match_threshold=self.write_match_threshold,
            read_match_threshold=self.read_match_threshold,
            basis_tolerance=self.basis_tolerance,
            scope_capacity=self.scope_capacity,
        )
        state = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }
        state["basis"] = basis
        state["coefficients"] = coefficients
        candidate.load_state_dict(state)
        candidate.validate_state()
        return candidate

    def max_value_error(self, candidate: SharedBasisContentAddressedMemory) -> float:
        if not isinstance(candidate, SharedBasisContentAddressedMemory):
            raise TypeError("shared-basis candidate has the wrong type")
        if (
            not torch.equal(self.keys, candidate.keys)
            or not torch.equal(self.scopes, candidate.scopes)
            or not torch.equal(self.occupied, candidate.occupied)
            or not torch.equal(self.strengths, candidate.strengths)
            or not torch.equal(self.timestamps, candidate.timestamps)
        ):
            raise ValueError("shared-basis candidate changed logical records")
        source = self._materialized_values()
        target = candidate._materialized_values().to(source)
        return float((source - target).abs().max()) if source.numel() else 0.0

    @torch.no_grad()
    def replace_from_candidate(
        self,
        candidate: SharedBasisContentAddressedMemory,
        *,
        expected_version: int | None = None,
        retention_probe: Callable[[SharedBasisContentAddressedMemory], bool]
        | None = None,
    ) -> SharedBasisCompressionReceipt:
        """Verify and commit a factorized value representation atomically."""

        if not isinstance(candidate, SharedBasisContentAddressedMemory):
            raise TypeError("shared-basis compression candidate is invalid")
        candidate.validate_state()
        if expected_version is not None:
            if not isinstance(expected_version, int) or isinstance(expected_version, bool):
                raise TypeError("shared-basis expected version must be an integer")
            if expected_version != int(self.store_version.item()):
                raise RuntimeError("shared-basis compression candidate is stale")
        candidate_configuration = candidate.configuration()
        expected_configuration = self.configuration()
        candidate_configuration.pop("persistence", None)
        expected_configuration.pop("persistence", None)
        if candidate_configuration != expected_configuration:
            raise ValueError("shared-basis candidate configuration does not match")
        rows_before = self.record_count
        basis_before = self.basis_count
        max_error = self.max_value_error(candidate)
        accepted = retention_probe is None or bool(retention_probe(candidate))
        if not accepted:
            return SharedBasisCompressionReceipt(
                accepted=False,
                rows_before=rows_before,
                rows_after=rows_before,
                basis_rows_before=basis_before,
                basis_rows_after=basis_before,
                max_value_error=max_error,
                version=int(self.store_version.item()),
                reason="shared-basis retention verifier rejected candidate",
            ).validate()
        self._buffers["basis"] = candidate.basis.detach().clone().to(self.basis)
        self._buffers["coefficients"] = candidate.coefficients.detach().clone().to(
            self.coefficients
        )
        self.store_version.add_(1)
        self.validate_state()
        return SharedBasisCompressionReceipt(
            accepted=True,
            rows_before=rows_before,
            rows_after=self.record_count,
            basis_rows_before=basis_before,
            basis_rows_after=self.basis_count,
            max_value_error=max_error,
            version=int(self.store_version.item()),
            reason="shared-basis candidate passed retention verification",
        ).validate()

    @torch.no_grad()
    def rewrite_candidate(
        self,
        candidates: MemoryCandidates,
        *,
        basis_rows: int,
        scope: int | torch.Tensor | None = None,
    ) -> SharedBasisContentAddressedMemory:
        """Build a copy-on-write candidate with a changed logical row set.

        The candidate may remove, retain, or add rows within one external
        scope.  Other scopes are copied unchanged.  No state is mutated until
        :meth:`replace_from_rewrite_candidate` accepts an independent
        retention probe.
        """

        if not isinstance(candidates, MemoryCandidates):
            raise TypeError("shared-basis rewrite candidates are invalid")
        candidates.validate(
            width=self.width,
            capacity=candidates.keys.shape[1],
            batch=1,
        )
        if (
            not isinstance(basis_rows, int)
            or isinstance(basis_rows, bool)
            or not 1 <= basis_rows <= self.width
        ):
            raise ValueError("shared-basis rewrite basis rank is invalid")
        if scope is None:
            scope_id = 0
        elif isinstance(scope, torch.Tensor):
            if scope.numel() != 1 or scope.dtype != torch.long:
                raise ValueError("shared-basis rewrite scope must be one int64 value")
            scope_id = int(scope.reshape(()).item())
        elif isinstance(scope, int) and not isinstance(scope, bool):
            scope_id = int(scope)
        else:
            raise TypeError("shared-basis rewrite scope must be an int or int64 tensor")
        if not 0 <= scope_id < self.scope_capacity:
            raise ValueError("shared-basis rewrite scope is outside the range")

        candidate_indices = torch.nonzero(
            candidates.occupied[0], as_tuple=False
        ).reshape(-1)
        other_indices = torch.nonzero(
            self.occupied & (self.scopes != scope_id), as_tuple=False
        ).reshape(-1)
        current_values = self._materialized_values()
        keys = torch.cat(
            (
                self.keys[other_indices],
                candidates.keys[0, candidate_indices].to(self.keys),
            ),
            dim=0,
        )
        values = torch.cat(
            (
                current_values[other_indices],
                candidates.values[0, candidate_indices].to(self.keys),
            ),
            dim=0,
        )
        strengths = torch.cat(
            (
                self.strengths[other_indices],
                candidates.strengths[0, candidate_indices].to(self.strengths),
            ),
            dim=0,
        )
        timestamps = torch.cat(
            (
                self.timestamps[other_indices],
                candidates.timestamps[0, candidate_indices].to(self.timestamps),
            ),
            dim=0,
        )
        scopes = torch.cat(
            (
                self.scopes[other_indices],
                torch.full(
                    (candidate_indices.numel(),),
                    scope_id,
                    dtype=torch.long,
                    device=self.keys.device,
                ),
            ),
            dim=0,
        )
        if values.numel():
            _left, _singular, right = torch.linalg.svd(values, full_matrices=False)
            rank = min(basis_rows, right.shape[0])
            basis = right[:rank].contiguous()
            coefficients = values @ basis.transpose(0, 1)
        else:
            basis = self.keys.new_empty((0, self.width))
            coefficients = self.keys.new_empty((0, 0))
        candidate = SharedBasisContentAddressedMemory(
            self.width,
            write_threshold=self.write_threshold,
            write_match_threshold=self.write_match_threshold,
            read_match_threshold=self.read_match_threshold,
            basis_tolerance=self.basis_tolerance,
            scope_capacity=self.scope_capacity,
        )
        state = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }
        state.update(
            {
                "basis": basis,
                "keys": keys,
                "coefficients": coefficients,
                "strengths": strengths,
                "timestamps": timestamps,
                "scopes": scopes,
                "occupied": torch.ones(
                    keys.shape[0], dtype=torch.bool, device=keys.device
                ),
            }
        )
        candidate.load_state_dict(state)
        candidate.validate_state()
        return candidate

    @torch.no_grad()
    def replace_from_rewrite_candidate(
        self,
        candidate: SharedBasisContentAddressedMemory,
        *,
        expected_version: int | None = None,
        retention_probe: Callable[[SharedBasisContentAddressedMemory], bool]
        | None = None,
    ) -> SharedBasisRewriteReceipt:
        """Verify and atomically commit a changed logical row set."""

        if not isinstance(candidate, SharedBasisContentAddressedMemory):
            raise TypeError("shared-basis rewrite candidate is invalid")
        candidate.validate_state()
        if expected_version is not None:
            if not isinstance(expected_version, int) or isinstance(expected_version, bool):
                raise TypeError("shared-basis expected version must be an integer")
            if expected_version != int(self.store_version.item()):
                raise RuntimeError("shared-basis rewrite candidate is stale")
        candidate_configuration = candidate.configuration()
        expected_configuration = self.configuration()
        candidate_configuration.pop("persistence", None)
        expected_configuration.pop("persistence", None)
        if candidate_configuration != expected_configuration:
            raise ValueError("shared-basis rewrite configuration does not match")
        rows_before = self.record_count
        basis_before = self.basis_count
        accepted = retention_probe is None or bool(retention_probe(candidate))
        if not accepted:
            return SharedBasisRewriteReceipt(
                accepted=False,
                rows_before=rows_before,
                rows_after=rows_before,
                basis_rows_before=basis_before,
                basis_rows_after=basis_before,
                version=int(self.store_version.item()),
                reason="shared-basis rewrite retention verifier rejected candidate",
            ).validate()
        for name in (
            "basis",
            "keys",
            "coefficients",
            "strengths",
            "timestamps",
            "scopes",
            "occupied",
        ):
            self._buffers[name] = candidate.state_dict()[name].detach().clone().to(
                self._buffers[name]
            )
        self.store_version.add_(1)
        self.validate_state()
        return SharedBasisRewriteReceipt(
            accepted=True,
            rows_before=rows_before,
            rows_after=self.record_count,
            basis_rows_before=basis_before,
            basis_rows_after=self.basis_count,
            version=int(self.store_version.item()),
            reason="shared-basis rewrite passed retention verification",
        ).validate()

    @torch.no_grad()
    def clear(self) -> None:
        device = self.keys.device
        self._buffers["basis"] = torch.empty((0, self.width), device=device)
        self._buffers["keys"] = torch.empty((0, self.width), device=device)
        self._buffers["coefficients"] = torch.empty((0, 0), device=device)
        self._buffers["strengths"] = torch.empty((0,), device=device)
        self._buffers["timestamps"] = torch.empty((0,), device=device)
        self._buffers["scopes"] = torch.empty((0,), dtype=torch.long, device=device)
        self._buffers["occupied"] = torch.empty((0,), dtype=torch.bool, device=device)
        self.store_version.add_(1)
        self.validate_state()

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "format": self.format,
            "schema": self.schema,
            "width": self.width,
            "storage": "shared_orthonormal_value_basis",
            "write_threshold": self.write_threshold,
            "write_match_threshold": self.write_match_threshold,
            "read_match_threshold": self.read_match_threshold,
            "basis_tolerance": self.basis_tolerance,
            "scope_capacity": self.scope_capacity,
        }

    @staticmethod
    def _state_checksum(state: dict[str, torch.Tensor]) -> str:
        return ContentAddressedMemory._state_checksum(state)

    def snapshot(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            state = self.state_dict()
            torch.save(
                {
                    "format": SHARED_BASIS_MEMORY_SNAPSHOT_FORMAT,
                    "schema": self.schema,
                    "configuration": self.configuration(),
                    "state_dict": state,
                    "state_checksum": self._state_checksum(state),
                },
                temporary,
            )
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load_snapshot(
        self, path: Path, *, map_location: torch.device | str = "cpu"
    ) -> None:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("format") != SHARED_BASIS_MEMORY_SNAPSHOT_FORMAT:
            raise ValueError("unsupported shared-basis memory snapshot format")
        if payload.get("schema") != self.schema:
            raise ValueError("unsupported shared-basis memory snapshot schema")
        if payload.get("configuration") != self.configuration():
            raise ValueError("shared-basis memory configuration does not match")
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise TypeError("shared-basis memory snapshot state is invalid")
        self._validate_state(state)
        if payload.get("state_checksum") != self._state_checksum(state):
            raise ValueError("shared-basis memory snapshot checksum mismatch")
        self.load_state_dict(state)
        self.validate_state()


class PersistentSharedBasisContentAddressedMemory(SharedBasisContentAddressedMemory):
    """Atomically persisted shared-basis content memory."""

    def __init__(self, width: int, path: Path, **kwargs: Any) -> None:
        self.path = Path(path)
        super().__init__(width, **kwargs)
        if self.path.exists():
            self.load_snapshot(self.path)

    def configuration(self) -> dict[str, int | float | str]:
        configuration = super().configuration()
        configuration["persistence"] = "atomic_snapshot"
        return configuration

    def write(self, *args: Any, **kwargs: Any) -> MemoryWriteReceipt:
        previous = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }
        try:
            receipt = super().write(*args, **kwargs)
            if bool(receipt.committed.any()):
                self.snapshot(self.path)
            return receipt
        except Exception:
            self.load_state_dict(previous)
            raise

    @torch.no_grad()
    def replace_from_candidate(self, *args: Any, **kwargs: Any):
        previous = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }
        receipt = super().replace_from_candidate(*args, **kwargs)
        if receipt.accepted:
            try:
                self.snapshot(self.path)
            except Exception:
                self.load_state_dict(previous)
                raise
        return receipt

    @torch.no_grad()
    def replace_from_rewrite_candidate(self, *args: Any, **kwargs: Any):
        previous = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }
        receipt = super().replace_from_rewrite_candidate(*args, **kwargs)
        if receipt.accepted:
            try:
                self.snapshot(self.path)
            except Exception:
                self.load_state_dict(previous)
                raise
        return receipt

    @torch.no_grad()
    def clear(self) -> None:
        previous = {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }
        super().clear()
        try:
            self.snapshot(self.path)
        except Exception:
            self.load_state_dict(previous)
            raise


__all__ = [
    "SHARED_BASIS_COMPRESSION_SCHEMA",
    "SHARED_BASIS_MEMORY_BACKEND_FORMAT",
    "SHARED_BASIS_MEMORY_SCHEMA",
    "SHARED_BASIS_MEMORY_SNAPSHOT_FORMAT",
    "SHARED_BASIS_REWRITE_SCHEMA",
    "PersistentSharedBasisContentAddressedMemory",
    "SharedBasisCompressionReceipt",
    "SharedBasisContentAddressedMemory",
    "SharedBasisRewriteReceipt",
]
