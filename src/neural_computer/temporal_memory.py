"""Append-only temporal history owned by external memory.

The controller and external programs exchange learned event tensors.  This
module stores those tensors as scoped, append-only records and exposes generic
relative-offset and stable-position reads.  Sequence positions, scope IDs,
and physical record indices remain memory-side state; no task name, modality,
or verifier target is represented in the contract.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .interface import AmodalEventCollection

EXTERNAL_TEMPORAL_HISTORY_SCHEMA = (
    "neural-computer.external-temporal-history.v1"
)
EXTERNAL_TEMPORAL_HISTORY_METADATA_SCHEMA = (
    "neural-computer.external-temporal-history.v2"
)
EXTERNAL_TEMPORAL_HISTORY_READ_SCHEMA = (
    "neural-computer.external-temporal-history-read.v1"
)
EXTERNAL_TEMPORAL_HISTORY_APPEND_SCHEMA = (
    "neural-computer.external-temporal-history-append.v1"
)
EXTERNAL_TEMPORAL_OFFSET_SELECTOR_SCHEMA = (
    "neural-computer.external-temporal-offset-selector.v1"
)
EXTERNAL_TEMPORAL_HISTORY_EVENT_BRIDGE_SCHEMA = (
    "neural-computer.external-temporal-history-event-bridge.v2"
)


@dataclass(frozen=True)
class ExternalTemporalHistoryEventBridgeResult:
    """Causal history augmentation plus the memory-side append receipts."""

    events: AmodalEventCollection
    read: ExternalTemporalHistoryRead
    appends: tuple[ExternalTemporalHistoryAppendReceipt, ...]
    schema: str = EXTERNAL_TEMPORAL_HISTORY_EVENT_BRIDGE_SCHEMA

    def validate(self, *, width: int, batch: int | None = None) -> ExternalTemporalHistoryEventBridgeResult:
        if self.schema != EXTERNAL_TEMPORAL_HISTORY_EVENT_BRIDGE_SCHEMA:
            raise ValueError("unsupported temporal history event bridge schema")
        self.events.validate(width=width)
        self.read.validate(width=width, batch=batch)
        if batch is not None and self.events.payload.shape[0] != batch:
            raise ValueError("temporal history bridge event batch does not match")
        for receipt in self.appends:
            receipt.validate(batch=batch)
        return self


@dataclass(frozen=True)
class ExternalTemporalHistoryAppendReceipt:
    """Opaque receipt for one append batch."""

    committed: torch.Tensor
    positions: torch.Tensor
    version: int
    schema: str = EXTERNAL_TEMPORAL_HISTORY_APPEND_SCHEMA

    def validate(self, *, batch: int | None = None) -> ExternalTemporalHistoryAppendReceipt:
        if self.schema != EXTERNAL_TEMPORAL_HISTORY_APPEND_SCHEMA:
            raise ValueError("unsupported temporal history append schema")
        if self.committed.ndim != 1 or self.committed.dtype is not torch.bool:
            raise ValueError("temporal append committed must be boolean [batch]")
        if self.positions.shape != self.committed.shape:
            raise ValueError("temporal append positions must match committed")
        if self.positions.dtype is not torch.long:
            raise ValueError("temporal append positions must be int64")
        if batch is not None and self.committed.shape[0] != batch:
            raise ValueError("temporal append batch does not match request")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("temporal history version must be non-negative")
        if bool(torch.any(self.positions < -1)):
            raise ValueError("temporal append positions cannot be below -1")
        return self


@dataclass(frozen=True)
class ExternalTemporalHistoryRead:
    """Fixed-width result of an opaque relative-offset query."""

    values: torch.Tensor
    present: torch.Tensor
    positions: torch.Tensor
    confidence: torch.Tensor | None = None
    source_key: torch.Tensor | None = None
    timestamp: torch.Tensor | None = None
    timestamp_present: torch.Tensor | None = None
    duration: torch.Tensor | None = None
    duration_present: torch.Tensor | None = None
    schema: str = EXTERNAL_TEMPORAL_HISTORY_READ_SCHEMA

    def validate(
        self,
        *,
        width: int,
        batch: int | None = None,
        query_count: int | None = None,
    ) -> ExternalTemporalHistoryRead:
        if self.schema != EXTERNAL_TEMPORAL_HISTORY_READ_SCHEMA:
            raise ValueError("unsupported temporal history read schema")
        if self.values.ndim != 3 or self.values.shape[-1] != width:
            raise ValueError("temporal history values must have shape [batch, query, width]")
        expected = self.values.shape[:2]
        if self.present.shape != expected or self.present.dtype is not torch.bool:
            raise ValueError("temporal history present has the wrong shape or dtype")
        if self.positions.shape != expected or self.positions.dtype is not torch.long:
            raise ValueError("temporal history positions have the wrong shape or dtype")
        if batch is not None and self.values.shape[0] != batch:
            raise ValueError("temporal history read batch does not match request")
        if query_count is not None and self.values.shape[1] != query_count:
            raise ValueError("temporal history read query count does not match request")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("temporal history values must be finite")
        if bool(torch.any(self.positions < -1)):
            raise ValueError("temporal history positions cannot be below -1")
        metadata = (
            ("confidence", self.confidence, False),
            ("timestamp", self.timestamp, False),
            ("duration", self.duration, False),
        )
        for name, value, _ in metadata:
            if value is not None:
                if value.shape != expected:
                    raise ValueError(f"temporal history {name} has the wrong shape")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"temporal history {name} must be finite")
        for name, value, field in (
            ("timestamp_present", self.timestamp_present, self.timestamp),
            ("duration_present", self.duration_present, self.duration),
        ):
            if value is not None:
                if field is None or value.shape != expected or value.dtype is not torch.bool:
                    raise ValueError(f"temporal history {name} has the wrong shape or dtype")
            elif field is not None:
                raise ValueError(f"temporal history {name} is required with its metadata")
        if self.confidence is not None and bool(torch.any(self.confidence < 0)):
            raise ValueError("temporal history confidence cannot be negative")
        if self.duration is not None and bool(torch.any(self.duration < 0)):
            raise ValueError("temporal history duration cannot be negative")
        if self.source_key is not None and (
            self.source_key.ndim != 3
            or self.source_key.shape[:2] != expected
            or not bool(torch.isfinite(self.source_key).all())
        ):
            raise ValueError("temporal history source_key has the wrong shape")
        return self


class ExternalTemporalOffsetSelector(nn.Module):
    """Learned opaque distribution over positive relative history offsets.

    The selector is external file state, not controller computation.  Offset
    ``1`` denotes the immediately preceding stored token; the selector never
    receives a depth, family name, target bit, or physical memory address.
    Scalar-outcome trainers may use the returned log probability for credit
    assignment while the memory itself remains a separate replaceable object.
    """

    schema = EXTERNAL_TEMPORAL_OFFSET_SELECTOR_SCHEMA

    def __init__(self, offset_count: int, *, initial_scale: float = 0.0) -> None:
        super().__init__()
        if offset_count < 1:
            raise ValueError("temporal offset count must be positive")
        if initial_scale < 0.0:
            raise ValueError("temporal offset initial scale cannot be negative")
        self.offset_count = int(offset_count)
        self.initial_scale = float(initial_scale)
        self.logits = nn.Parameter(torch.zeros(self.offset_count))
        if initial_scale:
            nn.init.normal_(self.logits, std=initial_scale)

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "offset_count": self.offset_count,
            "initial_scale": self.initial_scale,
            "offset_domain": "positive_relative_offsets_starting_at_one",
            "training": "scalar_outcome_policy_credit_external_state_v1",
        }

    def forward(
        self,
        batch_size: int,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("temporal offset selector batch must be positive")
        probabilities = self.logits.softmax(dim=-1)
        if sample:
            distribution = Categorical(
                probs=probabilities.expand(batch_size, -1)
            )
            choices = distribution.sample()
            log_probability = distribution.log_prob(choices)
        else:
            choices = probabilities.argmax().expand(batch_size)
            log_probability = torch.zeros(
                batch_size,
                device=self.logits.device,
                dtype=self.logits.dtype,
            )
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum()
        return choices + 1, log_probability, entropy


class ExternalTemporalHistoryMemory(nn.Module):
    """Variable-capacity, scoped event history for external computation.

    ``append`` assigns an internal monotonic position per scope.  After an
    append, ``read_relative(offsets)`` interprets offset zero as the newest
    present record, offset one as the preceding record, and so on, while
    ``read_positions(positions)`` addresses stable absolute positions.
    Missing history is represented by ``present=False`` rather than fabricated
    zero evidence.  The store grows by appending records and never resizes any
    controller or learned adapter.
    """

    schema = EXTERNAL_TEMPORAL_HISTORY_SCHEMA

    def __init__(
        self,
        width: int,
        *,
        scope_capacity: int = 1,
        metadata: bool = False,
        source_key_width: int = 0,
    ) -> None:
        super().__init__()
        if width < 1 or scope_capacity < 1 or source_key_width < 0:
            raise ValueError("temporal history dimensions must be positive")
        if source_key_width and not metadata:
            raise ValueError("source_key_width requires metadata history")
        self.width = int(width)
        self.scope_capacity = int(scope_capacity)
        self.metadata = bool(metadata)
        self.source_key_width = int(source_key_width)
        self.schema = (
            EXTERNAL_TEMPORAL_HISTORY_METADATA_SCHEMA
            if self.metadata
            else EXTERNAL_TEMPORAL_HISTORY_SCHEMA
        )
        self.register_buffer("values", torch.empty(0, self.width))
        self.register_buffer("scopes", torch.empty(0, dtype=torch.long))
        self.register_buffer("positions", torch.empty(0, dtype=torch.long))
        self.register_buffer("occupied", torch.empty(0, dtype=torch.bool))
        self.register_buffer(
            "next_positions",
            torch.zeros(self.scope_capacity, dtype=torch.long),
        )
        self.register_buffer("store_version", torch.zeros((), dtype=torch.long))
        if self.metadata:
            self.register_buffer("confidences", torch.empty(0))
            self.register_buffer(
                "source_keys", torch.empty(0, self.source_key_width)
            )
            self.register_buffer(
                "timestamps", torch.empty(0)
            )
            self.register_buffer(
                "timestamp_present", torch.empty(0, dtype=torch.bool)
            )
            self.register_buffer("durations", torch.empty(0))
            self.register_buffer(
                "duration_present", torch.empty(0, dtype=torch.bool)
            )

    @property
    def record_count(self) -> int:
        return int(self.values.shape[0])

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "schema": self.schema,
            "width": self.width,
            "scope_capacity": self.scope_capacity,
            "metadata": self.metadata,
            "source_key_width": self.source_key_width,
            "record_count": self.record_count,
            "store_version": int(self.store_version.item()),
            "storage": (
                "append_only_scoped_temporal_records_with_event_metadata_v2"
                if self.metadata
                else "append_only_scoped_temporal_records_v1"
            ),
            "addressing": "opaque_relative_offsets_and_absolute_positions_v1",
            "missing_history": "explicit_present_mask_v1",
        }

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
            raise ValueError("temporal history scope must be int64 with one value per row")
        scope = scope.reshape(batch).to(device=device)
        if bool(torch.any(scope < 0)) or bool(torch.any(scope >= self.scope_capacity)):
            raise ValueError("temporal history scope is outside configured capacity")
        return scope

    def _validate_state(self, state: Mapping[str, torch.Tensor] | None = None) -> None:
        values = self.state_dict() if state is None else state
        expected = {
            "values",
            "scopes",
            "positions",
            "occupied",
            "next_positions",
            "store_version",
        }
        if self.metadata:
            expected.update(
                {
                    "confidences",
                    "source_keys",
                    "timestamps",
                    "timestamp_present",
                    "durations",
                    "duration_present",
                }
            )
        if set(values) != expected:
            raise ValueError("temporal history state has an incompatible field set")
        record_count = values["values"].shape[0]
        if values["values"].ndim != 2 or values["values"].shape[1] != self.width:
            raise ValueError("temporal history values have the wrong shape")
        for name in ("scopes", "positions", "occupied"):
            if values[name].shape != (record_count,):
                raise ValueError(f"temporal history {name} has the wrong shape")
        if values["scopes"].dtype is not torch.long:
            raise ValueError("temporal history scopes must be int64")
        if values["positions"].dtype is not torch.long:
            raise ValueError("temporal history positions must be int64")
        if values["occupied"].dtype is not torch.bool:
            raise ValueError("temporal history occupancy must be boolean")
        if values["next_positions"].shape != (self.scope_capacity,):
            raise ValueError("temporal history next positions have the wrong shape")
        if values["next_positions"].dtype is not torch.long:
            raise ValueError("temporal history next positions must be int64")
        if values["store_version"].shape != torch.Size([]):
            raise ValueError("temporal history version must be scalar")
        if values["store_version"].dtype is not torch.long:
            raise ValueError("temporal history version must be int64")
        if self.metadata:
            for name in (
                "confidences",
                "timestamps",
                "durations",
            ):
                if values[name].shape != (record_count,):
                    raise ValueError(f"temporal history {name} has the wrong shape")
                if not bool(torch.isfinite(values[name]).all()):
                    raise ValueError(f"temporal history {name} must be finite")
            if values["source_keys"].shape != (
                record_count,
                self.source_key_width,
            ):
                raise ValueError("temporal history source keys have the wrong shape")
            if not bool(torch.isfinite(values["source_keys"]).all()):
                raise ValueError("temporal history source keys must be finite")
            for name in ("timestamp_present", "duration_present"):
                if (
                    values[name].shape != (record_count,)
                    or values[name].dtype is not torch.bool
                ):
                    raise ValueError(
                        f"temporal history {name} must be boolean [records]"
                    )
            if bool(torch.any(values["confidences"] < 0)):
                raise ValueError("temporal history confidences cannot be negative")
            if bool(torch.any(values["durations"] < 0)):
                raise ValueError("temporal history durations cannot be negative")
        for name in ("values",):
            if not bool(torch.isfinite(values[name]).all()):
                raise ValueError(f"temporal history {name} must be finite")
        if bool(torch.any(values["scopes"] < 0)) or bool(
            torch.any(values["scopes"] >= self.scope_capacity)
        ):
            raise ValueError("temporal history scopes are outside configured capacity")
        if bool(torch.any(values["positions"] < 0)):
            raise ValueError("temporal history positions cannot be negative")
        if bool(torch.any(values["next_positions"] < 0)):
            raise ValueError("temporal history next positions cannot be negative")
        if int(values["store_version"].item()) < 0:
            raise ValueError("temporal history version cannot be negative")
        occupied = values["occupied"]
        if bool(torch.any(~occupied)):
            raise ValueError("temporal history storage cannot contain unoccupied rows")
        if record_count:
            for scope_id in range(self.scope_capacity):
                scope_positions = values["positions"][values["scopes"] == scope_id]
                if scope_positions.numel() and (
                    scope_positions.unique().numel() != scope_positions.numel()
                ):
                    raise ValueError("temporal history positions must be unique per scope")
                if scope_positions.numel() and int(scope_positions.max()) >= int(
                    values["next_positions"][scope_id]
                ):
                    raise ValueError("temporal history next position is inconsistent")

    def validate_state(self) -> None:
        self._validate_state()

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        expected = {
            "values",
            "scopes",
            "positions",
            "occupied",
            "next_positions",
            "store_version",
        }
        if self.metadata:
            expected.update(
                {
                    "confidences",
                    "source_keys",
                    "timestamps",
                    "timestamp_present",
                    "durations",
                    "duration_present",
                }
            )
        if expected.issubset(state_dict):
            self._validate_state({name: state_dict[name] for name in expected})
            for name in expected - {"next_positions", "store_version"}:
                current = self._buffers[name]
                self._buffers[name] = torch.empty_like(
                    state_dict[name], device=current.device
                )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def append(
        self,
        values: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
        scope: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        source_key: torch.Tensor | None = None,
        timestamp: torch.Tensor | None = None,
        timestamp_present: torch.Tensor | None = None,
        duration: torch.Tensor | None = None,
        duration_present: torch.Tensor | None = None,
    ) -> ExternalTemporalHistoryAppendReceipt:
        """Append learned event tensors and return their opaque positions."""

        if values.ndim != 2 or values.shape[1] != self.width:
            raise ValueError("temporal history values must have shape [batch, width]")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("temporal history values must be finite")
        batch = values.shape[0]
        if present is None:
            present = torch.ones(batch, dtype=torch.bool, device=values.device)
        if present.shape != (batch,) or present.dtype is not torch.bool:
            raise ValueError("temporal history present must be boolean [batch]")
        if not self.metadata and any(
            value is not None
            for value in (
                confidence,
                source_key,
                timestamp,
                timestamp_present,
                duration,
                duration_present,
            )
        ):
            raise ValueError("event metadata requires the v2 temporal history ABI")
        if self.metadata:
            if confidence is None:
                confidence = torch.ones(batch, device=values.device, dtype=values.dtype)
            if confidence.shape != (batch,) or not bool(torch.isfinite(confidence).all()):
                raise ValueError("temporal history confidence must be finite [batch]")
            if bool(torch.any(confidence < 0)):
                raise ValueError("temporal history confidence cannot be negative")
            if source_key is None:
                source_key = torch.zeros(
                    batch,
                    self.source_key_width,
                    device=values.device,
                    dtype=values.dtype,
                )
            if source_key.shape != (batch, self.source_key_width):
                raise ValueError("temporal history source_key has the wrong shape")
            if not bool(torch.isfinite(source_key).all()):
                raise ValueError("temporal history source_key must be finite")
            if timestamp_present is None:
                timestamp_present = torch.full(
                    (batch,),
                    timestamp is not None,
                    dtype=torch.bool,
                    device=values.device,
                )
            elif timestamp is None:
                raise ValueError(
                    "timestamp_present requires timestamp values in temporal history"
                )
            if (
                timestamp_present.shape != (batch,)
                or timestamp_present.dtype is not torch.bool
            ):
                raise ValueError(
                    "temporal history timestamp_present must be boolean [batch]"
                )
            if timestamp is None:
                timestamp = torch.zeros(batch, device=values.device, dtype=values.dtype)
            if timestamp.shape != (batch,) or not bool(torch.isfinite(timestamp).all()):
                raise ValueError("temporal history timestamp must be finite [batch]")
            if duration_present is None:
                duration_present = torch.full(
                    (batch,),
                    duration is not None,
                    dtype=torch.bool,
                    device=values.device,
                )
            elif duration is None:
                raise ValueError(
                    "duration_present requires duration values in temporal history"
                )
            if (
                duration_present.shape != (batch,)
                or duration_present.dtype is not torch.bool
            ):
                raise ValueError(
                    "temporal history duration_present must be boolean [batch]"
                )
            if duration is None:
                duration = torch.zeros(batch, device=values.device, dtype=values.dtype)
            if duration.shape != (batch,) or not bool(torch.isfinite(duration).all()):
                raise ValueError("temporal history duration must be finite [batch]")
            if bool(torch.any(duration < 0)):
                raise ValueError("temporal history duration cannot be negative")
        scope_ids = self._scope_ids(scope, batch, device=self.values.device)
        values = values.to(device=self.values.device, dtype=self.values.dtype)
        present = present.to(device=self.values.device)
        if self.metadata:
            confidence = confidence.to(device=self.values.device, dtype=self.values.dtype)
            source_key = source_key.to(device=self.values.device, dtype=self.values.dtype)
            timestamp = timestamp.to(device=self.values.device, dtype=self.values.dtype)
            timestamp_present = timestamp_present.to(device=self.values.device)
            duration = duration.to(device=self.values.device, dtype=self.values.dtype)
            duration_present = duration_present.to(device=self.values.device)
        positions = torch.full((batch,), -1, dtype=torch.long, device=self.values.device)
        next_positions = self.next_positions.clone()
        new_values: list[torch.Tensor] = []
        new_scopes: list[int] = []
        new_positions: list[int] = []
        new_confidences: list[torch.Tensor] = []
        new_source_keys: list[torch.Tensor] = []
        new_timestamps: list[torch.Tensor] = []
        new_timestamp_present: list[bool] = []
        new_durations: list[torch.Tensor] = []
        new_duration_present: list[bool] = []
        for row, scope_id in enumerate(scope_ids.tolist()):
            if not bool(present[row]):
                continue
            position = int(next_positions[scope_id].item())
            positions[row] = position
            next_positions[scope_id] = position + 1
            new_values.append(values[row])
            new_scopes.append(scope_id)
            new_positions.append(position)
            if self.metadata:
                new_confidences.append(confidence[row])
                new_source_keys.append(source_key[row])
                new_timestamps.append(timestamp[row])
                new_timestamp_present.append(bool(timestamp_present[row]))
                new_durations.append(duration[row])
                new_duration_present.append(bool(duration_present[row]))
        if new_values:
            self._buffers["values"] = torch.cat(
                (self.values, torch.stack(new_values, dim=0)), dim=0
            )
            self._buffers["scopes"] = torch.cat(
                (
                    self.scopes,
                    torch.tensor(new_scopes, dtype=torch.long, device=self.scopes.device),
                ),
                dim=0,
            )
            self._buffers["positions"] = torch.cat(
                (
                    self.positions,
                    torch.tensor(
                        new_positions, dtype=torch.long, device=self.positions.device
                    ),
                ),
                dim=0,
            )
            self._buffers["occupied"] = torch.cat(
                (
                    self.occupied,
                    torch.ones(
                        len(new_values), dtype=torch.bool, device=self.occupied.device
                    ),
                ),
                dim=0,
            )
            if self.metadata:
                self._buffers["confidences"] = torch.cat(
                    (self.confidences, torch.stack(new_confidences)), dim=0
                )
                self._buffers["source_keys"] = torch.cat(
                    (self.source_keys, torch.stack(new_source_keys)), dim=0
                )
                self._buffers["timestamps"] = torch.cat(
                    (self.timestamps, torch.stack(new_timestamps)), dim=0
                )
                self._buffers["timestamp_present"] = torch.cat(
                    (
                        self.timestamp_present,
                        torch.tensor(
                            new_timestamp_present,
                            dtype=torch.bool,
                            device=self.timestamp_present.device,
                        ),
                    ),
                    dim=0,
                )
                self._buffers["durations"] = torch.cat(
                    (self.durations, torch.stack(new_durations)), dim=0
                )
                self._buffers["duration_present"] = torch.cat(
                    (
                        self.duration_present,
                        torch.tensor(
                            new_duration_present,
                            dtype=torch.bool,
                            device=self.duration_present.device,
                        ),
                    ),
                    dim=0,
                )
            self.store_version.add_(1)
        self._buffers["next_positions"] = next_positions
        self.validate_state()
        return ExternalTemporalHistoryAppendReceipt(
            committed=present & (positions >= 0),
            positions=positions,
            version=int(self.store_version.item()),
        ).validate(batch=batch)

    @torch.no_grad()
    def read_relative(
        self,
        offsets: torch.Tensor,
        *,
        scope: torch.Tensor | None = None,
    ) -> ExternalTemporalHistoryRead:
        """Read offsets relative to each scope's newest stored event."""

        if offsets.ndim != 2 or offsets.dtype is not torch.long:
            raise ValueError("temporal history offsets must be int64 [batch, query]")
        if bool(torch.any(offsets < 0)):
            raise ValueError("temporal history offsets cannot be negative")
        batch, query_count = offsets.shape
        scope_ids = self._scope_ids(scope, batch, device=self.values.device)
        offsets = offsets.to(device=self.values.device)
        values = torch.zeros(
            batch,
            query_count,
            self.width,
            device=self.values.device,
            dtype=self.values.dtype,
        )
        present = torch.zeros(
            batch, query_count, dtype=torch.bool, device=self.values.device
        )
        positions = torch.full(
            (batch, query_count), -1, dtype=torch.long, device=self.values.device
        )
        if self.metadata:
            confidences = torch.zeros(
                batch,
                query_count,
                device=self.values.device,
                dtype=self.values.dtype,
            )
            source_keys = torch.zeros(
                batch,
                query_count,
                self.source_key_width,
                device=self.values.device,
                dtype=self.values.dtype,
            )
            timestamps = torch.zeros_like(confidences)
            timestamp_present = torch.zeros(
                batch, query_count, dtype=torch.bool, device=self.values.device
            )
            durations = torch.zeros_like(confidences)
            duration_present = torch.zeros(
                batch, query_count, dtype=torch.bool, device=self.values.device
            )
        for row, scope_id in enumerate(scope_ids.tolist()):
            newest = int(self.next_positions[scope_id].item()) - 1
            if newest < 0:
                continue
            record_indices = torch.nonzero(
                self.occupied & (self.scopes == scope_id), as_tuple=False
            ).reshape(-1)
            if not record_indices.numel():
                continue
            row_positions = self.positions[record_indices]
            for query_index, offset in enumerate(offsets[row].tolist()):
                target = newest - int(offset)
                matches = torch.nonzero(row_positions == target, as_tuple=False).reshape(-1)
                if not matches.numel():
                    continue
                record_index = record_indices[int(matches[0].item())]
                values[row, query_index] = self.values[record_index]
                present[row, query_index] = True
                positions[row, query_index] = target
                if self.metadata:
                    confidences[row, query_index] = self.confidences[record_index]
                    source_keys[row, query_index] = self.source_keys[record_index]
                    timestamps[row, query_index] = self.timestamps[record_index]
                    timestamp_present[row, query_index] = self.timestamp_present[
                        record_index
                    ]
                    durations[row, query_index] = self.durations[record_index]
                    duration_present[row, query_index] = self.duration_present[
                        record_index
                    ]
        return ExternalTemporalHistoryRead(
            values=values,
            present=present,
            positions=positions,
            confidence=confidences if self.metadata else None,
            source_key=source_keys if self.metadata else None,
            timestamp=timestamps if self.metadata else None,
            timestamp_present=timestamp_present if self.metadata else None,
            duration=durations if self.metadata else None,
            duration_present=duration_present if self.metadata else None,
        ).validate(
            width=self.width,
            batch=batch,
            query_count=query_count,
        )

    @torch.no_grad()
    def read_positions(
        self,
        positions: torch.Tensor,
        *,
        scope: torch.Tensor | None = None,
    ) -> ExternalTemporalHistoryRead:
        """Read stable absolute positions within each opaque scope."""

        if positions.ndim != 2 or positions.dtype is not torch.long:
            raise ValueError("temporal history positions must be int64 [batch, query]")
        if bool(torch.any(positions < 0)):
            raise ValueError("temporal history positions cannot be negative")
        batch, query_count = positions.shape
        scope_ids = self._scope_ids(scope, batch, device=self.values.device)
        positions = positions.to(device=self.values.device)
        values = torch.zeros(
            batch,
            query_count,
            self.width,
            device=self.values.device,
            dtype=self.values.dtype,
        )
        present = torch.zeros(
            batch, query_count, dtype=torch.bool, device=self.values.device
        )
        resolved_positions = torch.full(
            (batch, query_count), -1, dtype=torch.long, device=self.values.device
        )
        if self.metadata:
            confidences = torch.zeros(
                batch,
                query_count,
                device=self.values.device,
                dtype=self.values.dtype,
            )
            source_keys = torch.zeros(
                batch,
                query_count,
                self.source_key_width,
                device=self.values.device,
                dtype=self.values.dtype,
            )
            timestamps = torch.zeros_like(confidences)
            timestamp_present = torch.zeros(
                batch, query_count, dtype=torch.bool, device=self.values.device
            )
            durations = torch.zeros_like(confidences)
            duration_present = torch.zeros(
                batch, query_count, dtype=torch.bool, device=self.values.device
            )
        for row, scope_id in enumerate(scope_ids.tolist()):
            record_indices = torch.nonzero(
                self.occupied & (self.scopes == scope_id), as_tuple=False
            ).reshape(-1)
            if not record_indices.numel():
                continue
            row_positions = self.positions[record_indices]
            for query_index, target in enumerate(positions[row].tolist()):
                matches = torch.nonzero(
                    row_positions == int(target), as_tuple=False
                ).reshape(-1)
                if not matches.numel():
                    continue
                record_index = record_indices[int(matches[0].item())]
                values[row, query_index] = self.values[record_index]
                present[row, query_index] = True
                resolved_positions[row, query_index] = int(target)
                if self.metadata:
                    confidences[row, query_index] = self.confidences[record_index]
                    source_keys[row, query_index] = self.source_keys[record_index]
                    timestamps[row, query_index] = self.timestamps[record_index]
                    timestamp_present[row, query_index] = self.timestamp_present[
                        record_index
                    ]
                    durations[row, query_index] = self.durations[record_index]
                    duration_present[row, query_index] = self.duration_present[
                        record_index
                    ]
        return ExternalTemporalHistoryRead(
            values=values,
            present=present,
            positions=resolved_positions,
            confidence=confidences if self.metadata else None,
            source_key=source_keys if self.metadata else None,
            timestamp=timestamps if self.metadata else None,
            timestamp_present=timestamp_present if self.metadata else None,
            duration=durations if self.metadata else None,
            duration_present=duration_present if self.metadata else None,
        ).validate(
            width=self.width,
            batch=batch,
            query_count=query_count,
        )

    @torch.no_grad()
    def clear(self, scope: torch.Tensor | None = None) -> None:
        """Clear all records or only the selected opaque scopes."""

        if scope is None:
            selected = torch.arange(
                self.scope_capacity, dtype=torch.long, device=self.values.device
            )
        else:
            if scope.dtype is not torch.long:
                raise ValueError("temporal history clear scope must be int64")
            selected = scope.reshape(-1).to(device=self.values.device)
            if not selected.numel():
                return
            if bool(torch.any(selected < 0)) or bool(
                torch.any(selected >= self.scope_capacity)
            ):
                raise ValueError("temporal history clear scope is out of range")
            selected = selected.unique()
        keep = ~torch.isin(self.scopes, selected)
        self._buffers["values"] = self.values[keep]
        self._buffers["scopes"] = self.scopes[keep]
        self._buffers["positions"] = self.positions[keep]
        self._buffers["occupied"] = self.occupied[keep]
        if self.metadata:
            self._buffers["confidences"] = self.confidences[keep]
            self._buffers["source_keys"] = self.source_keys[keep]
            self._buffers["timestamps"] = self.timestamps[keep]
            self._buffers["timestamp_present"] = self.timestamp_present[keep]
            self._buffers["durations"] = self.durations[keep]
            self._buffers["duration_present"] = self.duration_present[keep]
        next_positions = self.next_positions.clone()
        next_positions[selected] = 0
        self._buffers["next_positions"] = next_positions
        self.store_version.add_(1)
        self.validate_state()

    def _state_checksum(
        self,
        configuration: Mapping[str, object],
        state: Mapping[str, torch.Tensor],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(repr(sorted(configuration.items())).encode("utf-8"))
        for name, value in sorted(state.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def digest(self) -> str:
        state = {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}
        return self._state_checksum(self.configuration(), state)

    def payload(self) -> dict[str, object]:
        """Return an integrity-checked, controller-independent state payload."""

        state = {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}
        configuration = self.configuration()
        return {
            "schema": self.schema,
            "configuration": configuration,
            "state": state,
            "sha256": self._state_checksum(configuration, state),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> ExternalTemporalHistoryMemory:
        if not isinstance(payload, Mapping):
            raise TypeError("temporal history payload must be a mapping")
        schema = payload.get("schema")
        if schema not in (
            EXTERNAL_TEMPORAL_HISTORY_SCHEMA,
            EXTERNAL_TEMPORAL_HISTORY_METADATA_SCHEMA,
        ):
            raise ValueError("unsupported temporal history payload schema")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("temporal history payload is missing configuration or state")
        memory = cls(
            int(configuration["width"]),
            scope_capacity=int(configuration["scope_capacity"]),
            metadata=schema == EXTERNAL_TEMPORAL_HISTORY_METADATA_SCHEMA,
            source_key_width=int(configuration.get("source_key_width", 0)),
        )
        tensor_state = {
            name: value for name, value in state.items() if isinstance(value, torch.Tensor)
        }
        if len(tensor_state) != len(state):
            raise TypeError("temporal history state must contain only tensors")
        memory.load_state_dict(tensor_state, strict=True)
        expected = payload.get("sha256")
        actual = memory._state_checksum(configuration, tensor_state)
        if not isinstance(expected, str) or expected != actual:
            raise ValueError("temporal history payload checksum mismatch")
        memory.validate_state()
        return memory


class ExternalTemporalHistoryEventBridge(nn.Module):
    """Causally augment learned events with externally stored history.

    The bridge reads the requested prior records *before* appending the
    current collection. It returns separately bindable historical tokens
    followed by current tokens, with explicit presence masks for missing
    history. The runtime can process the historical prefix transiently while
    persisting only the current suffix in the controller state.

    The v1 history store contains learned payload tensors only.  To avoid
    silently discarding transport structure, collections carrying source
    keys, timestamps, or durations are rejected by v1.  The explicit v2
    metadata history ABI persists those fields and their per-token presence
    masks alongside the learned payload.
    """

    schema = EXTERNAL_TEMPORAL_HISTORY_EVENT_BRIDGE_SCHEMA

    def __init__(self, event_width: int) -> None:
        super().__init__()
        if event_width < 1:
            raise ValueError("temporal history bridge event width must be positive")
        self.event_width = int(event_width)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "event_width": self.event_width,
            "ordering": "prior_relative_tokens_then_current_tokens_v2",
            "causality": "read_before_append_v1",
            "missing_history": "explicit_present_mask_v1",
            "metadata": "v1_payload_only_or_v2_event_metadata_preserving",
            "controller_persistence": "current_tokens_only_v1",
        }

    @torch.no_grad()
    def augment(
        self,
        collection: AmodalEventCollection,
        history: ExternalTemporalHistoryMemory,
        offsets: torch.Tensor,
        *,
        scope: torch.Tensor | None = None,
        append_current: bool = True,
        _read_override: ExternalTemporalHistoryRead | None = None,
    ) -> ExternalTemporalHistoryEventBridgeResult:
        if not isinstance(collection, AmodalEventCollection):
            raise TypeError("temporal history bridge needs an event collection")
        if not isinstance(history, ExternalTemporalHistoryMemory):
            raise TypeError("temporal history bridge needs external history memory")
        collection.validate(width=self.event_width)
        if history.width != self.event_width:
            raise ValueError("temporal history width does not match event width")
        if offsets.ndim != 2 or offsets.dtype is not torch.long:
            raise ValueError("temporal history bridge offsets must be int64 [batch, query]")
        batch = collection.payload.shape[0]
        if offsets.shape[0] != batch:
            raise ValueError("temporal history bridge offsets batch does not match")
        if not history.metadata:
            if collection.source_key is not None:
                raise ValueError(
                    "temporal history bridge v1 cannot preserve event source keys"
                )
            if collection.timestamp is not None or collection.duration is not None:
                raise ValueError(
                    "temporal history bridge v1 cannot preserve event timing metadata"
                )
        elif history.source_key_width:
            if (
                collection.source_key is None
                or collection.source_key.shape[-1] != history.source_key_width
            ):
                raise ValueError(
                    "metadata history source_key width does not match current events"
                )
        elif collection.source_key is not None:
            raise ValueError(
                "metadata history was created without a source_key width"
            )

        # This read intentionally precedes every append below.  A query for
        # offset zero therefore means the most recent prior record, never the
        # current input being processed on this tick.
        read = (
            history.read_relative(offsets, scope=scope)
            if _read_override is None
            else _read_override.validate(width=self.event_width, batch=batch)
        )
        read_values = read.values.to(
            device=collection.payload.device,
            dtype=collection.payload.dtype,
        )
        read_present = read.present.to(device=collection.payload.device)
        read_confidence = (
            read.confidence.to(
                device=collection.payload.device,
                dtype=collection.confidence.dtype,
            )
            if history.metadata and read.confidence is not None
            else read_present.to(dtype=collection.confidence.dtype)
        )
        source_key = None
        if history.metadata and history.source_key_width:
            source_key = torch.cat(
                (
                    read.source_key.to(
                        device=collection.payload.device,
                        dtype=collection.payload.dtype,
                    ),
                    collection.source_key,
                ),
                dim=1,
            )
        timestamp = None
        timestamp_present = None
        if history.metadata or collection.timestamp is not None:
            current_timestamp = (
                collection.timestamp
                if collection.timestamp is not None
                else torch.zeros_like(collection.confidence)
            )
            current_timestamp_present = (
                collection.timestamp_present
                if collection.timestamp_present is not None
                else torch.zeros_like(collection.present)
            )
            timestamp = torch.cat(
                (
                    read.timestamp.to(
                        device=collection.payload.device,
                        dtype=collection.payload.dtype,
                    )
                    if read.timestamp is not None
                    else torch.zeros(
                        batch,
                        offsets.shape[1],
                        device=collection.payload.device,
                        dtype=collection.payload.dtype,
                    ),
                    current_timestamp,
                ),
                dim=1,
            )
            timestamp_present = torch.cat(
                (
                    read.timestamp_present.to(device=collection.payload.device)
                    if read.timestamp_present is not None
                    else torch.zeros(
                        batch,
                        offsets.shape[1],
                        dtype=torch.bool,
                        device=collection.payload.device,
                    ),
                    current_timestamp_present,
                ),
                dim=1,
            )
        duration = None
        duration_present = None
        if history.metadata or collection.duration is not None:
            current_duration = (
                collection.duration
                if collection.duration is not None
                else torch.zeros_like(collection.confidence)
            )
            current_duration_present = (
                collection.duration_present
                if collection.duration_present is not None
                else torch.zeros_like(collection.present)
            )
            duration = torch.cat(
                (
                    read.duration.to(
                        device=collection.payload.device,
                        dtype=collection.payload.dtype,
                    )
                    if read.duration is not None
                    else torch.zeros(
                        batch,
                        offsets.shape[1],
                        device=collection.payload.device,
                        dtype=collection.payload.dtype,
                    ),
                    current_duration,
                ),
                dim=1,
            )
            duration_present = torch.cat(
                (
                    read.duration_present.to(device=collection.payload.device)
                    if read.duration_present is not None
                    else torch.zeros(
                        batch,
                        offsets.shape[1],
                        dtype=torch.bool,
                        device=collection.payload.device,
                    ),
                    current_duration_present,
                ),
                dim=1,
            )
        events = AmodalEventCollection(
            payload=torch.cat((read_values, collection.payload), dim=1),
            present=torch.cat((read_present, collection.present), dim=1),
            confidence=torch.cat((read_confidence, collection.confidence), dim=1),
            source_key=source_key,
            timestamp=timestamp,
            timestamp_present=timestamp_present,
            duration=duration,
            duration_present=duration_present,
        ).validate(width=self.event_width)

        appends: list[ExternalTemporalHistoryAppendReceipt] = []
        if append_current:
            for index in range(collection.payload.shape[1]):
                appends.append(
                    history.append(
                        collection.payload[:, index],
                        present=collection.present[:, index],
                        scope=scope,
                        confidence=(
                            collection.confidence[:, index]
                            if history.metadata
                            else None
                        ),
                        source_key=(
                            collection.source_key[:, index]
                            if history.metadata and history.source_key_width
                            else None
                        ),
                        timestamp=(
                            collection.timestamp[:, index]
                            if history.metadata and collection.timestamp is not None
                            else None
                        ),
                        timestamp_present=(
                            collection.timestamp_present[:, index]
                            if (
                                history.metadata
                                and collection.timestamp is not None
                                and collection.timestamp_present is not None
                            )
                            else None
                        ),
                        duration=(
                            collection.duration[:, index]
                            if history.metadata and collection.duration is not None
                            else None
                        ),
                        duration_present=(
                            collection.duration_present[:, index]
                            if (
                                history.metadata
                                and collection.duration is not None
                                and collection.duration_present is not None
                            )
                            else None
                        ),
                    )
                )
        return ExternalTemporalHistoryEventBridgeResult(
            events=events,
            read=read,
            appends=tuple(appends),
        ).validate(width=self.event_width, batch=batch)

    @torch.no_grad()
    def augment_from_read(
        self,
        collection: AmodalEventCollection,
        history: ExternalTemporalHistoryMemory,
        read: ExternalTemporalHistoryRead,
        *,
        scope: torch.Tensor | None = None,
        append_current: bool = True,
    ) -> ExternalTemporalHistoryEventBridgeResult:
        """Augment from a validated external read without re-addressing it.

        Content-addressed memory can resolve a stable history position through
        a replaceable index.  This entry point preserves that read exactly and
        lets the bridge append current events without converting the stable
        position back into a shifting relative offset.
        """

        if not isinstance(read, ExternalTemporalHistoryRead):
            raise TypeError("temporal history bridge read must use the history ABI")
        if read.values.ndim != 3:
            raise ValueError("temporal history bridge read must be [batch, query, width]")
        placeholder_offsets = torch.zeros(
            read.values.shape[:2], dtype=torch.long, device=read.values.device
        )
        return self.augment(
            collection,
            history,
            placeholder_offsets,
            scope=scope,
            append_current=append_current,
            _read_override=read,
        )
