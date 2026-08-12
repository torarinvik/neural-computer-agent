"""Append-only temporal history owned by external memory.

The controller and external programs exchange learned event tensors.  This
module stores those tensors as scoped, append-only records and exposes only a
generic relative-offset read.  Sequence positions, scope IDs, and physical
record indices remain memory-side state; no task name, modality, or verifier
target is represented in the contract.
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
    present record, offset one as the preceding record, and so on.  Missing
    history is represented by ``present=False`` rather than fabricated zero
    evidence.  The store grows by appending records and never resizes any
    controller or learned adapter.
    """

    schema = EXTERNAL_TEMPORAL_HISTORY_SCHEMA

    def __init__(self, width: int, *, scope_capacity: int = 1) -> None:
        super().__init__()
        if width < 1 or scope_capacity < 1:
            raise ValueError("temporal history dimensions must be positive")
        self.width = int(width)
        self.scope_capacity = int(scope_capacity)
        self.register_buffer("values", torch.empty(0, self.width))
        self.register_buffer("scopes", torch.empty(0, dtype=torch.long))
        self.register_buffer("positions", torch.empty(0, dtype=torch.long))
        self.register_buffer("occupied", torch.empty(0, dtype=torch.bool))
        self.register_buffer(
            "next_positions",
            torch.zeros(self.scope_capacity, dtype=torch.long),
        )
        self.register_buffer("store_version", torch.zeros((), dtype=torch.long))

    @property
    def record_count(self) -> int:
        return int(self.values.shape[0])

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "width": self.width,
            "scope_capacity": self.scope_capacity,
            "record_count": self.record_count,
            "store_version": int(self.store_version.item()),
            "storage": "append_only_scoped_temporal_records_v1",
            "addressing": "opaque_relative_offset_v1",
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
        if expected.issubset(state_dict):
            self._validate_state({name: state_dict[name] for name in expected})
            for name in ("values", "scopes", "positions", "occupied"):
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
        scope_ids = self._scope_ids(scope, batch, device=self.values.device)
        values = values.to(device=self.values.device, dtype=self.values.dtype)
        present = present.to(device=self.values.device)
        positions = torch.full((batch,), -1, dtype=torch.long, device=self.values.device)
        next_positions = self.next_positions.clone()
        new_values: list[torch.Tensor] = []
        new_scopes: list[int] = []
        new_positions: list[int] = []
        for row, scope_id in enumerate(scope_ids.tolist()):
            if not bool(present[row]):
                continue
            position = int(next_positions[scope_id].item())
            positions[row] = position
            next_positions[scope_id] = position + 1
            new_values.append(values[row])
            new_scopes.append(scope_id)
            new_positions.append(position)
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
        return ExternalTemporalHistoryRead(
            values=values,
            present=present,
            positions=positions,
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
        if payload.get("schema") != EXTERNAL_TEMPORAL_HISTORY_SCHEMA:
            raise ValueError("unsupported temporal history payload schema")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("temporal history payload is missing configuration or state")
        memory = cls(
            int(configuration["width"]),
            scope_capacity=int(configuration["scope_capacity"]),
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
    keys, timestamps, or durations are rejected until a history ABI that
    persists those fields is selected explicitly.
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
            "metadata": "payload_only_with_derived_presence_confidence_v1",
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
        if collection.source_key is not None:
            raise ValueError(
                "temporal history bridge v1 cannot preserve event source keys"
            )
        if collection.timestamp is not None or collection.duration is not None:
            raise ValueError(
                "temporal history bridge v1 cannot preserve event timing metadata"
            )

        # This read intentionally precedes every append below.  A query for
        # offset zero therefore means the most recent prior record, never the
        # current input being processed on this tick.
        read = history.read_relative(offsets, scope=scope)
        read_values = read.values.to(
            device=collection.payload.device,
            dtype=collection.payload.dtype,
        )
        read_present = read.present.to(device=collection.payload.device)
        read_confidence = read_present.to(dtype=collection.confidence.dtype)
        events = AmodalEventCollection(
            payload=torch.cat((read_values, collection.payload), dim=1),
            present=torch.cat((read_present, collection.present), dim=1),
            confidence=torch.cat((read_confidence, collection.confidence), dim=1),
        ).validate(width=self.event_width)

        appends: list[ExternalTemporalHistoryAppendReceipt] = []
        if append_current:
            for index in range(collection.payload.shape[1]):
                appends.append(
                    history.append(
                        collection.payload[:, index],
                        present=collection.present[:, index],
                        scope=scope,
                    )
                )
        return ExternalTemporalHistoryEventBridgeResult(
            events=events,
            read=read,
            appends=tuple(appends),
        ).validate(width=self.event_width, batch=batch)
