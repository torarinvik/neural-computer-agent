"""Content-addressed locations for the external temporal-memory boundary.

The history store owns learned event records and relative or absolute reads.
This module owns the replaceable index that maps a learned opaque query key to
a stable history scope and absolute position.  Keeping the two stores separate
prevents physical locations, index rows, or address representations from
entering the controller's neural interface.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from .temporal_memory import ExternalTemporalHistoryMemory, ExternalTemporalHistoryRead

EXTERNAL_TEMPORAL_ADDRESS_INDEX_SCHEMA = (
    "neural-computer.external-temporal-address-index.v2"
)
EXTERNAL_TEMPORAL_ADDRESS_READ_SCHEMA = (
    "neural-computer.external-temporal-address-read.v2"
)
EXTERNAL_TEMPORAL_ADDRESS_WRITE_SCHEMA = (
    "neural-computer.external-temporal-address-write.v2"
)


@dataclass(frozen=True)
class ExternalTemporalAddressRead:
    """Opaque index candidates for one batch of learned query keys."""

    target_scopes: torch.Tensor
    target_positions: torch.Tensor
    scores: torch.Tensor
    indices: torch.Tensor
    hit: torch.Tensor
    schema: str = EXTERNAL_TEMPORAL_ADDRESS_READ_SCHEMA

    def validate(
        self,
        *,
        batch: int | None = None,
        top_k: int | None = None,
    ) -> ExternalTemporalAddressRead:
        if self.schema != EXTERNAL_TEMPORAL_ADDRESS_READ_SCHEMA:
            raise ValueError("unsupported temporal address read schema")
        if self.scores.ndim != 2:
            raise ValueError("temporal address scores must be [batch, top_k]")
        expected = self.scores.shape
        for name, value in (
            ("target_scopes", self.target_scopes),
            ("target_positions", self.target_positions),
            ("indices", self.indices),
        ):
            if value.shape != expected or value.dtype is not torch.long:
                raise ValueError(
                    f"temporal address {name} must be int64 [batch, top_k]"
                )
        if self.hit.shape != (expected[0],) or self.hit.dtype is not torch.bool:
            raise ValueError("temporal address hit must be boolean [batch]")
        if batch is not None and expected[0] != batch:
            raise ValueError("temporal address read batch does not match")
        if top_k is not None and expected[1] != top_k:
            raise ValueError("temporal address read width does not match top_k")
        valid_scores = torch.isfinite(self.scores) | torch.isneginf(self.scores)
        if not bool(valid_scores.all()):
            raise ValueError("temporal address scores must be finite or -inf")
        if bool(torch.any(self.target_scopes < -1)) or bool(
            torch.any(self.target_positions < -1)
        ):
            raise ValueError("temporal address locations cannot be below -1")
        if bool(torch.any(self.indices < -1)):
            raise ValueError("temporal address indices cannot be below -1")
        return self


@dataclass(frozen=True)
class ExternalTemporalAddressWriteReceipt:
    """Receipt for appending or updating opaque temporal locations."""

    committed: torch.Tensor
    indices: torch.Tensor
    version: int
    schema: str = EXTERNAL_TEMPORAL_ADDRESS_WRITE_SCHEMA

    def validate(
        self, *, batch: int | None = None
    ) -> ExternalTemporalAddressWriteReceipt:
        if self.schema != EXTERNAL_TEMPORAL_ADDRESS_WRITE_SCHEMA:
            raise ValueError("unsupported temporal address write schema")
        if self.committed.ndim != 1 or self.committed.dtype is not torch.bool:
            raise ValueError("temporal address committed must be boolean [batch]")
        if self.indices.shape != self.committed.shape or self.indices.dtype is not torch.long:
            raise ValueError("temporal address indices must be int64 [batch]")
        if batch is not None and self.committed.shape[0] != batch:
            raise ValueError("temporal address write batch does not match")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("temporal address version must be non-negative")
        if bool(torch.any(self.indices < -1)):
            raise ValueError("temporal address indices cannot be below -1")
        return self


@dataclass(frozen=True)
class ExternalTemporalAddressedRead:
    """An index lookup followed by the corresponding history read."""

    address: ExternalTemporalAddressRead
    history: ExternalTemporalHistoryRead

    def validate(
        self, *, width: int, batch: int | None = None
    ) -> ExternalTemporalAddressedRead:
        self.address.validate(batch=batch, top_k=1)
        self.history.validate(width=width, batch=batch, query_count=1)
        if not torch.all(self.history.present[:, 0] <= self.address.hit):
            raise ValueError("history hits cannot exist without an address hit")
        return self


class ExternalTemporalAddressIndex(nn.Module):
    """Variable-capacity content-addressed index into external history.

    The index stores no event payloads.  Each record contains a learned query
    key and an opaque `(target_scope, target_position)` location owned by a
    separate :class:`ExternalTemporalHistoryMemory`.  A miss returns explicit
    ``hit=False`` and ``-1`` locations; it never fabricates an address.  The
    target position is absolute within its scope, so appending newer history
    cannot silently retarget an older learned record.
    """

    schema = EXTERNAL_TEMPORAL_ADDRESS_INDEX_SCHEMA

    def __init__(
        self,
        key_width: int,
        *,
        scope_capacity: int = 1,
        write_match_threshold: float = 0.95,
        read_match_threshold: float = 0.75,
    ) -> None:
        super().__init__()
        if key_width < 1 or scope_capacity < 1:
            raise ValueError("temporal address dimensions must be positive")
        if not 0.0 <= write_match_threshold <= 1.0 or not 0.0 <= read_match_threshold <= 1.0:
            raise ValueError("temporal address thresholds are invalid")
        self.key_width = int(key_width)
        self.scope_capacity = int(scope_capacity)
        self.write_match_threshold = float(write_match_threshold)
        self.read_match_threshold = float(read_match_threshold)
        self.register_buffer("keys", torch.empty(0, self.key_width))
        self.register_buffer("target_scopes", torch.empty(0, dtype=torch.long))
        self.register_buffer("target_positions", torch.empty(0, dtype=torch.long))
        self.register_buffer("strengths", torch.empty(0))
        self.register_buffer("timestamps", torch.empty(0))
        self.register_buffer("scopes", torch.empty(0, dtype=torch.long))
        self.register_buffer("occupied", torch.empty(0, dtype=torch.bool))
        self.register_buffer("store_version", torch.zeros((), dtype=torch.long))

    @property
    def record_count(self) -> int:
        return int(self.keys.shape[0])

    def _scope_ids(
        self,
        scope: torch.Tensor | None,
        batch: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if scope is None:
            return torch.zeros(batch, dtype=torch.long, device=device)
        if scope.ndim == 0 or scope.numel() != batch or scope.dtype is not torch.long:
            raise ValueError("temporal address scope must be int64 [batch]")
        scope = scope.reshape(batch).to(device=device)
        if bool(torch.any(scope < 0)) or bool(torch.any(scope >= self.scope_capacity)):
            raise ValueError("temporal address scope is outside configured capacity")
        return scope

    def _record_indices(self, scope_id: int) -> torch.Tensor:
        return torch.nonzero(
            self.occupied & (self.scopes == scope_id), as_tuple=False
        ).reshape(-1)

    def _validate_state(self, state: Mapping[str, torch.Tensor] | None = None) -> None:
        values = self.state_dict() if state is None else state
        expected = {
            "keys",
            "target_scopes",
            "target_positions",
            "strengths",
            "timestamps",
            "scopes",
            "occupied",
            "store_version",
        }
        if set(values) != expected:
            raise ValueError("temporal address state has an incompatible field set")
        count = values["keys"].shape[0]
        if values["keys"].ndim != 2 or values["keys"].shape[1] != self.key_width:
            raise ValueError("temporal address keys have the wrong shape")
        if not values["keys"].is_floating_point():
            raise ValueError("temporal address keys must use a floating dtype")
        for name in (
            "target_scopes",
            "target_positions",
            "strengths",
            "timestamps",
            "scopes",
            "occupied",
        ):
            if values[name].shape != (count,):
                raise ValueError(f"temporal address {name} has the wrong shape")
        if values["target_scopes"].dtype is not torch.long:
            raise ValueError("temporal address target scopes must be int64")
        if values["target_positions"].dtype is not torch.long:
            raise ValueError("temporal address target positions must be int64")
        if values["scopes"].dtype is not torch.long:
            raise ValueError("temporal address scopes must be int64")
        if values["occupied"].dtype is not torch.bool:
            raise ValueError("temporal address occupancy must be boolean")
        for name in ("keys", "strengths", "timestamps"):
            if not bool(torch.isfinite(values[name]).all()):
                raise ValueError(f"temporal address {name} must be finite")
        if bool(torch.any(values["target_scopes"] < 0)) or bool(
            torch.any(values["target_positions"] < 0)
        ):
            raise ValueError("temporal address locations cannot be negative")
        if bool(torch.any(values["scopes"] < 0)) or bool(
            torch.any(values["scopes"] >= self.scope_capacity)
        ):
            raise ValueError("temporal address scopes are outside capacity")
        if bool(torch.any(values["strengths"] < 0)) or bool(
            torch.any(values["strengths"] > 1)
        ):
            raise ValueError("temporal address strengths must lie in [0, 1]")
        if bool(torch.any(~values["occupied"])):
            raise ValueError("temporal address storage cannot contain unoccupied rows")
        version = values["store_version"]
        if version.shape != torch.Size([]) or version.dtype is not torch.long:
            raise ValueError("temporal address version must be scalar int64")
        if int(version.item()) < 0:
            raise ValueError("temporal address version cannot be negative")

    def validate_state(self) -> None:
        self._validate_state()

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        expected = {
            "keys",
            "target_scopes",
            "target_positions",
            "strengths",
            "timestamps",
            "scopes",
            "occupied",
            "store_version",
        }
        if expected.issubset(state_dict):
            self._validate_state({name: state_dict[name] for name in expected})
            for name in expected - {"store_version"}:
                current = self._buffers[name]
                self._buffers[name] = torch.empty_like(
                    state_dict[name], device=current.device
                )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def write(
        self,
        keys: torch.Tensor,
        target_scopes: torch.Tensor,
        target_positions: torch.Tensor,
        strength: torch.Tensor,
        *,
        timestamp: torch.Tensor | None = None,
        scope: torch.Tensor | None = None,
    ) -> ExternalTemporalAddressWriteReceipt:
        if keys.ndim != 2 or keys.shape[1] != self.key_width:
            raise ValueError("temporal address keys must be [batch, key_width]")
        if not keys.is_floating_point():
            raise ValueError("temporal address keys must use a floating dtype")
        if not bool(torch.isfinite(keys).all()):
            raise ValueError("temporal address keys must be finite")
        batch = keys.shape[0]
        for name, value in (
            ("target_scopes", target_scopes),
            ("target_positions", target_positions),
            ("strength", strength),
        ):
            if value.numel() != batch:
                raise ValueError(f"temporal address {name} must have one value per row")
        if target_scopes.dtype is not torch.long or target_positions.dtype is not torch.long:
            raise ValueError("temporal address targets must be int64")
        target_scopes = target_scopes.reshape(batch)
        target_positions = target_positions.reshape(batch)
        if bool(torch.any(target_scopes < 0)) or bool(torch.any(target_positions < 0)):
            raise ValueError("temporal address targets cannot be negative")
        strength = strength.reshape(batch).to(dtype=self.keys.dtype)
        if not bool(torch.isfinite(strength).all()) or bool(
            torch.any(strength < 0) or torch.any(strength > 1)
        ):
            raise ValueError("temporal address strength must lie in [0, 1]")
        if timestamp is None:
            timestamp = torch.zeros(batch, dtype=keys.dtype, device=keys.device)
        timestamp = timestamp.reshape(batch).to(dtype=self.timestamps.dtype)
        if not bool(torch.isfinite(timestamp).all()):
            raise ValueError("temporal address timestamp must be finite")
        scope_ids = self._scope_ids(scope, batch, device=self.keys.device)
        keys = keys.to(device=self.keys.device, dtype=self.keys.dtype)
        target_scopes = target_scopes.to(device=self.keys.device)
        target_positions = target_positions.to(device=self.keys.device)
        strength = strength.to(device=self.keys.device)
        timestamp = timestamp.to(device=self.keys.device)
        committed = strength > 0.0
        indices = torch.full((batch,), -1, dtype=torch.long, device=self.keys.device)
        for row, namespace in enumerate(scope_ids.tolist()):
            if not bool(committed[row]):
                continue
            candidates = self._record_indices(namespace)
            normalized = torch.nn.functional.normalize(keys[row], dim=0)
            if candidates.numel():
                scores = torch.nn.functional.normalize(self.keys[candidates], dim=-1) @ normalized
                best_score, best_position = scores.max(dim=0)
            else:
                best_score = torch.tensor(-torch.inf, device=self.keys.device)
                best_position = torch.tensor(0, dtype=torch.long, device=self.keys.device)
            if bool(torch.isfinite(best_score)) and float(best_score) >= self.write_match_threshold:
                index = int(candidates[int(best_position.item())])
                self.keys[index] = keys[row]
                self.target_scopes[index] = target_scopes[row]
                self.target_positions[index] = target_positions[row]
                self.strengths[index] = strength[row]
                self.timestamps[index] = timestamp[row]
            else:
                index = self.record_count
                for name, value in (
                    ("keys", keys[row].reshape(1, self.key_width)),
                    ("target_scopes", target_scopes[row].reshape(1)),
                    ("target_positions", target_positions[row].reshape(1)),
                    ("strengths", strength[row].reshape(1)),
                    ("timestamps", timestamp[row].reshape(1)),
                    ("scopes", scope_ids[row].reshape(1)),
                    ("occupied", torch.ones(1, dtype=torch.bool, device=self.keys.device)),
                ):
                    self._buffers[name] = torch.cat((self._buffers[name], value), dim=0)
            indices[row] = index
        if bool(committed.any()):
            self.store_version.add_(1)
        self.validate_state()
        return ExternalTemporalAddressWriteReceipt(
            committed=committed,
            indices=indices,
            version=int(self.store_version.item()),
        ).validate(batch=batch)

    @torch.no_grad()
    def read(
        self,
        keys: torch.Tensor,
        *,
        top_k: int = 1,
        scope: torch.Tensor | None = None,
    ) -> ExternalTemporalAddressRead:
        if keys.ndim != 2 or keys.shape[1] != self.key_width:
            raise ValueError("temporal address keys must be [batch, key_width]")
        if not keys.is_floating_point():
            raise ValueError("temporal address keys must use a floating dtype")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("temporal address top_k must be positive")
        if not bool(torch.isfinite(keys).all()):
            raise ValueError("temporal address keys must be finite")
        batch = keys.shape[0]
        scope_ids = self._scope_ids(scope, batch, device=self.keys.device)
        keys = keys.to(device=self.keys.device, dtype=self.keys.dtype)
        scores = torch.full(
            (batch, top_k), -torch.inf, device=self.keys.device, dtype=self.keys.dtype
        )
        target_scopes = torch.full(
            (batch, top_k), -1, dtype=torch.long, device=self.keys.device
        )
        target_positions = torch.full_like(target_scopes, -1)
        indices = torch.full_like(target_scopes, -1)
        hit = torch.zeros(batch, dtype=torch.bool, device=self.keys.device)
        for row, namespace in enumerate(scope_ids.tolist()):
            candidates = self._record_indices(namespace)
            if not candidates.numel():
                continue
            row_scores = torch.nn.functional.normalize(self.keys[candidates], dim=-1) @ torch.nn.functional.normalize(keys[row], dim=0)
            count = min(top_k, row_scores.shape[0])
            selected_scores, selected_positions = torch.topk(row_scores, count)
            selected_indices = candidates[selected_positions]
            scores[row, :count] = selected_scores
            matches = selected_scores >= self.read_match_threshold
            indices[row, :count] = torch.where(
                matches, selected_indices, torch.full_like(selected_indices, -1)
            )
            target_scopes[row, :count] = torch.where(
                matches,
                self.target_scopes[selected_indices],
                torch.full_like(selected_indices, -1),
            )
            target_positions[row, :count] = torch.where(
                matches,
                self.target_positions[selected_indices],
                torch.full_like(selected_indices, -1),
            )
            hit[row] = bool(torch.any(matches))
        return ExternalTemporalAddressRead(
            target_scopes=target_scopes,
            target_positions=target_positions,
            scores=scores,
            indices=indices,
            hit=hit,
        ).validate(batch=batch, top_k=top_k)

    @torch.no_grad()
    def read_history(
        self,
        history: ExternalTemporalHistoryMemory,
        keys: torch.Tensor,
        *,
        scope: torch.Tensor | None = None,
    ) -> ExternalTemporalAddressedRead:
        """Resolve one opaque key and read its history record atomically."""

        address = self.read(keys, top_k=1, scope=scope)
        safe_scopes = address.target_scopes[:, 0].clamp_min(0)
        safe_positions = address.target_positions[:, 0].clamp_min(0)
        history_read = history.read_positions(safe_positions[:, None], scope=safe_scopes)
        valid = address.hit[:, None]
        values = torch.where(
            valid.unsqueeze(-1), history_read.values, torch.zeros_like(history_read.values)
        )
        present = history_read.present & valid
        positions = torch.where(
            valid, history_read.positions, torch.full_like(history_read.positions, -1)
        )
        kwargs: dict[str, torch.Tensor | None] = {
            "confidence": history_read.confidence,
            "source_key": history_read.source_key,
            "timestamp": history_read.timestamp,
            "timestamp_present": history_read.timestamp_present,
            "duration": history_read.duration,
            "duration_present": history_read.duration_present,
        }
        for name, value in tuple(kwargs.items()):
            if value is None:
                continue
            if name in {"timestamp_present", "duration_present"}:
                kwargs[name] = value & valid
            elif name == "source_key":
                kwargs[name] = torch.where(valid.unsqueeze(-1), value, torch.zeros_like(value))
            else:
                kwargs[name] = torch.where(valid, value, torch.zeros_like(value))
        return ExternalTemporalAddressedRead(
            address=address,
            history=replace(
                history_read,
                values=values,
                present=present,
                positions=positions,
                **kwargs,
            ),
        ).validate(width=history.width, batch=keys.shape[0])

    @torch.no_grad()
    def clear(self, scope: torch.Tensor | None = None) -> None:
        if scope is None:
            keep = torch.zeros(self.record_count, dtype=torch.bool, device=self.keys.device)
        else:
            selected = self._scope_ids(scope, int(scope.numel()), device=self.keys.device).unique()
            keep = torch.ones(self.record_count, dtype=torch.bool, device=self.keys.device)
            for namespace in selected.tolist():
                keep &= self.scopes != namespace
        for name in (
            "keys",
            "target_scopes",
            "target_positions",
            "strengths",
            "timestamps",
            "scopes",
            "occupied",
        ):
            self._buffers[name] = self._buffers[name][keep]
        self.store_version.add_(1)
        self.validate_state()

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "key_width": self.key_width,
            "scope_capacity": self.scope_capacity,
            "write_match_threshold": self.write_match_threshold,
            "read_match_threshold": self.read_match_threshold,
            "record_count": self.record_count,
            "storage": "append_only_opaque_temporal_locations_v2",
            "addressing": "cosine_key_to_scope_and_absolute_position_v2",
        }

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr(self.configuration()).encode("utf-8"))
        for name, value in sorted(self.state_dict().items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        return digest.hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> ExternalTemporalAddressIndex:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported temporal address index payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("temporal address index payload is incomplete")
        index = cls(
            int(configuration["key_width"]),
            scope_capacity=int(configuration["scope_capacity"]),
            write_match_threshold=float(configuration["write_match_threshold"]),
            read_match_threshold=float(configuration["read_match_threshold"]),
        )
        tensor_state = {
            name: value for name, value in state.items() if isinstance(value, torch.Tensor)
        }
        if len(tensor_state) != len(state):
            raise TypeError("temporal address index state must contain only tensors")
        index.load_state_dict(tensor_state, strict=True)
        index.validate_state()
        if payload.get("sha256") != index.digest():
            raise ValueError("temporal address index checksum mismatch")
        return index


__all__ = [
    "EXTERNAL_TEMPORAL_ADDRESS_INDEX_SCHEMA",
    "EXTERNAL_TEMPORAL_ADDRESS_READ_SCHEMA",
    "EXTERNAL_TEMPORAL_ADDRESS_WRITE_SCHEMA",
    "ExternalTemporalAddressIndex",
    "ExternalTemporalAddressRead",
    "ExternalTemporalAddressWriteReceipt",
    "ExternalTemporalAddressedRead",
]
