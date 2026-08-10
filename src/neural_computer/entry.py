"""Versioned external memory for opaque factual value entries.

Entries are durable learned vectors, not controller activations or protocol
actions.  The repertoire can grow without resizing the controller and exposes
only a runtime-sized proposal to factual search.  New content enters through
held-out verifier admission; retained records are copy-on-write protected.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .representation import (
    DEFAULT_INTENTION_SPACE_ID,
    DEFAULT_MEMORY_VALUE_SPACE_ID,
    validate_representation_space_id,
)

EXTERNAL_ENTRY_OBSERVATION_SCHEMA = "neural-computer.external-entry-observation.v1"
EXTERNAL_ENTRY_PROPOSAL_SCHEMA = "neural-computer.external-entry-proposal.v1"
EXTERNAL_ENTRY_ADMISSION_SCHEMA = "neural-computer.external-entry-admission.v1"
EXTERNAL_ENTRY_REPERTOIRE_SCHEMA = "neural-computer.external-entry-repertoire.v1"
EXTERNAL_ENTRY_BINDING_OBSERVATION_SCHEMA = (
    "neural-computer.external-entry-binding-observation.v1"
)
EXTERNAL_ENTRY_BINDING_PROPOSAL_SCHEMA = (
    "neural-computer.external-entry-binding-proposal.v1"
)
EXTERNAL_ENTRY_BINDING_ADMISSION_SCHEMA = (
    "neural-computer.external-entry-binding-admission.v1"
)
EXTERNAL_ENTRY_BINDING_REPERTOIRE_SCHEMA = (
    "neural-computer.external-entry-binding-repertoire.v1"
)
EXTERNAL_ENTRY_BINDING_CONSOLIDATION_SCHEMA = (
    "neural-computer.external-entry-binding-consolidation.v1"
)


def _validate_tensor(
    value: torch.Tensor,
    *,
    name: str,
    ndim: int,
    width: int | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if width is not None and value.shape[-1] != width:
        raise ValueError(f"{name} has the wrong width")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class ExternalEntryObservationReceipt:
    """Auditable result of recording opaque entry experience."""

    entry_indices: tuple[int, ...]
    added: tuple[bool, ...]
    outcome_observed: bool
    version: int
    record_count: int
    content_digest: str
    schema: str = EXTERNAL_ENTRY_OBSERVATION_SCHEMA

    def validate(self) -> ExternalEntryObservationReceipt:
        if self.schema != EXTERNAL_ENTRY_OBSERVATION_SCHEMA:
            raise ValueError("unsupported external-entry observation schema")
        if not self.entry_indices or len(self.entry_indices) != len(self.added):
            raise ValueError("external-entry observation rows are misaligned")
        if any(not isinstance(index, int) or index < 0 for index in self.entry_indices):
            raise ValueError("external-entry observation index is invalid")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("external-entry observation version is invalid")
        if not isinstance(self.record_count, int) or self.record_count < 1:
            raise ValueError("external-entry observation record count is invalid")
        if any(index >= self.record_count for index in self.entry_indices):
            raise ValueError("external-entry observation index is out of range")
        if not isinstance(self.content_digest, str) or not self.content_digest:
            raise ValueError("external-entry observation digest is missing")
        return self


@dataclass(frozen=True)
class ExternalEntryProposal:
    """Runtime-sized candidate entries retrieved from external memory."""

    entries: torch.Tensor
    source_indices: tuple[int, ...]
    propensities: torch.Tensor
    version: int
    schema: str = EXTERNAL_ENTRY_PROPOSAL_SCHEMA

    def validate(self, *, width: int) -> ExternalEntryProposal:
        if self.schema != EXTERNAL_ENTRY_PROPOSAL_SCHEMA:
            raise ValueError("unsupported external-entry proposal schema")
        _validate_tensor(self.entries, name="external-entry proposal", ndim=2, width=width)
        count = self.entries.shape[0]
        if count < 1:
            raise ValueError("external-entry proposal cannot be empty")
        if len(self.source_indices) != count or len(set(self.source_indices)) != count:
            raise ValueError("external-entry proposal indices are invalid")
        if any(index < 0 for index in self.source_indices):
            raise ValueError("external-entry proposal source index is invalid")
        if self.propensities.shape != (count,):
            raise ValueError("external-entry proposal propensities are misaligned")
        if not bool(torch.isfinite(self.propensities).all()) or bool(
            torch.any(self.propensities <= 0.0) | torch.any(self.propensities > 1.0)
        ):
            raise ValueError("external-entry proposal propensities are invalid")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("external-entry proposal version is invalid")
        return self


@dataclass(frozen=True)
class ExternalEntryAdmissionReceipt:
    """Verifier-gated copy-on-write admission result."""

    accepted: bool
    entry_index: int | None
    source_record_count: int
    destination_record_count: int
    source_digest: str
    candidate_digest: str
    destination_digest: str
    reason: str
    version: int
    schema: str = EXTERNAL_ENTRY_ADMISSION_SCHEMA

    def validate(self) -> ExternalEntryAdmissionReceipt:
        if self.schema != EXTERNAL_ENTRY_ADMISSION_SCHEMA:
            raise ValueError("unsupported external-entry admission schema")
        if min(self.source_record_count, self.destination_record_count) < 0:
            raise ValueError("external-entry admission record counts are invalid")
        if self.accepted:
            if self.entry_index != self.source_record_count:
                raise ValueError("accepted external-entry admission index is invalid")
            if self.destination_record_count != self.source_record_count + 1:
                raise ValueError("accepted external-entry admission count is invalid")
        elif self.entry_index is not None:
            raise ValueError("rejected external-entry admission has an index")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("external-entry admission version is invalid")
        for name, value in (
            ("source_digest", self.source_digest),
            ("candidate_digest", self.candidate_digest),
            ("destination_digest", self.destination_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"external-entry admission {name} is missing")
        return self


class ExternalEntryRepertoire:
    """Append-only, checksummed memory for opaque factual value entries.

    The repertoire stores vectors and outcome sufficient statistics only.  It
    never ranks candidates by reward and never updates a controller or value
    model.  A separate caller-owned verifier decides whether a novel entry
    can be admitted; rejected or mutating verifiers leave the live store
    unchanged.
    """

    schema = EXTERNAL_ENTRY_REPERTOIRE_SCHEMA

    def __init__(
        self,
        width: int,
        *,
        merge_cosine: float = 0.999,
        entry_space_id: str = DEFAULT_MEMORY_VALUE_SPACE_ID,
    ) -> None:
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("external-entry repertoire width must be positive")
        if not -1.0 <= merge_cosine <= 1.0 or not math.isfinite(merge_cosine):
            raise ValueError("external-entry repertoire merge cosine is invalid")
        self.width = int(width)
        self.merge_cosine = float(merge_cosine)
        self.entry_space_id = validate_representation_space_id(
            entry_space_id,
            name="entry_space_id",
        )
        self._entries: list[torch.Tensor] = []
        self._attempts: list[int] = []
        self._outcome_counts: list[int] = []
        self._utility_sums: list[float] = []
        self._utility_square_sums: list[float] = []
        self._propensity_sums: list[float] = []
        self._inverse_propensity_utility_sums: list[float] = []
        self._last_propensities: list[float] = []
        self._last_seen: list[int] = []
        self._version = 0

    @property
    def record_count(self) -> int:
        return len(self._entries)

    @property
    def version(self) -> int:
        return self._version

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "width": self.width,
            "merge_cosine": self.merge_cosine,
            "entry_space_id": self.entry_space_id,
            "storage": "append_only_opaque_external_entry_v1",
            "learning": "outcome_sufficient_statistics_without_replay_v1",
            "proposal": "uniform_verified_external_entries_v1",
        }

    def _validate_batch(
        self,
        entries: torch.Tensor,
        utility: torch.Tensor | float | None,
        propensity: torch.Tensor | float | None,
        timestamp: torch.Tensor | int | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if entries.ndim == 1:
            entries = entries.unsqueeze(0)
        _validate_tensor(entries, name="observed external entry", ndim=2, width=self.width)
        batch = entries.shape[0]
        if batch < 1:
            raise ValueError("external-entry observation batch cannot be empty")

        def scalar_batch(
            value: torch.Tensor | float | None,
            *,
            name: str,
            default: float,
        ) -> torch.Tensor:
            if value is None:
                result = torch.full((batch,), default, dtype=torch.float64)
            elif isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    result = value.reshape(1).expand(batch)
                elif value.shape in ((batch,), (batch, 1)):
                    result = value.reshape(batch)
                else:
                    raise ValueError(f"{name} must contain one value per entry")
                result = result.detach().to(device="cpu", dtype=torch.float64)
            else:
                result = torch.full((batch,), float(value), dtype=torch.float64)
            if not bool(torch.isfinite(result).all()):
                raise ValueError(f"{name} must be finite")
            return result

        utility_values = None if utility is None else scalar_batch(
            utility,
            name="external-entry utility",
            default=0.0,
        )
        propensity_values = scalar_batch(
            propensity,
            name="external-entry logging propensity",
            default=1.0,
        )
        if bool(torch.any(propensity_values <= 0.0)) or bool(
            torch.any(propensity_values > 1.0)
        ):
            raise ValueError("external-entry logging propensities must lie in (0, 1]")

        if timestamp is None:
            timestamp_values = torch.full((batch,), self._version + 1, dtype=torch.int64)
        elif isinstance(timestamp, torch.Tensor):
            if timestamp.ndim == 0:
                timestamp_values = timestamp.reshape(1).expand(batch)
            elif timestamp.shape in ((batch,), (batch, 1)):
                timestamp_values = timestamp.reshape(batch)
            else:
                raise ValueError("external-entry timestamps must contain one value per entry")
            if timestamp_values.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("external-entry timestamps must be integer tensors")
            timestamp_values = timestamp_values.detach().to(device="cpu", dtype=torch.int64)
        else:
            timestamp_values = torch.full((batch,), int(timestamp), dtype=torch.int64)
        if bool(torch.any(timestamp_values < 0)):
            raise ValueError("external-entry timestamps cannot be negative")
        return (
            entries.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            utility_values,
            propensity_values,
            timestamp_values,
        )

    @staticmethod
    def _entry_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
        left_norm = torch.linalg.vector_norm(left)
        right_norm = torch.linalg.vector_norm(right)
        if float(left_norm) <= 1e-12 or float(right_norm) <= 1e-12:
            return 1.0 if torch.equal(left, right) else -1.0
        return float(torch.dot(left, right) / (left_norm * right_norm))

    def _find_entry(self, entry: torch.Tensor) -> int | None:
        for index, stored in enumerate(self._entries):
            if self._entry_similarity(stored, entry) >= self.merge_cosine:
                return index
        return None

    def _stack_entries(self) -> torch.Tensor:
        if not self._entries:
            return torch.empty((0, self.width), dtype=torch.float32)
        return torch.stack(self._entries).detach().clone()

    def statistics(self) -> dict[str, torch.Tensor]:
        self.validate_state()
        return {
            "entries": self._stack_entries(),
            "attempts": torch.tensor(self._attempts, dtype=torch.long),
            "outcome_counts": torch.tensor(self._outcome_counts, dtype=torch.long),
            "utility_sums": torch.tensor(self._utility_sums, dtype=torch.float64),
            "utility_square_sums": torch.tensor(
                self._utility_square_sums,
                dtype=torch.float64,
            ),
            "propensity_sums": torch.tensor(self._propensity_sums, dtype=torch.float64),
            "inverse_propensity_utility_sums": torch.tensor(
                self._inverse_propensity_utility_sums,
                dtype=torch.float64,
            ),
            "last_propensities": torch.tensor(
                self._last_propensities,
                dtype=torch.float64,
            ),
            "last_seen": torch.tensor(self._last_seen, dtype=torch.long),
        }

    def validate_state(self) -> None:
        count = self.record_count
        if not isinstance(self._version, int) or self._version < 0:
            raise ValueError("external-entry repertoire version is invalid")
        lengths = (
            len(self._attempts),
            len(self._outcome_counts),
            len(self._utility_sums),
            len(self._utility_square_sums),
            len(self._propensity_sums),
            len(self._inverse_propensity_utility_sums),
            len(self._last_propensities),
            len(self._last_seen),
        )
        if any(length != count for length in lengths):
            raise ValueError("external-entry repertoire statistics are misaligned")
        for entry in self._entries:
            _validate_tensor(entry, name="stored external entry", ndim=1, width=self.width)
        for name, values in (
            ("attempts", self._attempts),
            ("outcome counts", self._outcome_counts),
            ("last seen", self._last_seen),
        ):
            if any(not isinstance(value, int) or value < 0 for value in values):
                raise ValueError(f"external-entry repertoire {name} are invalid")
        for name, values in (
            ("utility sums", self._utility_sums),
            ("utility square sums", self._utility_square_sums),
            ("propensity sums", self._propensity_sums),
            ("inverse-propensity utility sums", self._inverse_propensity_utility_sums),
            ("last propensities", self._last_propensities),
        ):
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"external-entry repertoire {name} are not finite")
        if any(
            outcome_count > attempt
            for outcome_count, attempt in zip(
                self._outcome_counts,
                self._attempts,
                strict=True,
            )
        ):
            raise ValueError("external-entry outcome counts exceed attempts")

    @staticmethod
    def _digest_payload(payload: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        for name in sorted(payload, key=str):
            if name == "sha256":
                continue
            value = payload[name]
            digest.update(str(name).encode("utf-8"))
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
        return digest.hexdigest()

    def payload_without_digest(self) -> dict[str, Any]:
        self.validate_state()
        return {
            "schema": self.schema,
            "width": self.width,
            "merge_cosine": self.merge_cosine,
            "entry_space_id": self.entry_space_id,
            "version": self._version,
            **self.statistics(),
        }

    def payload(self) -> dict[str, Any]:
        payload = self.payload_without_digest()
        payload["sha256"] = self._digest_payload(payload)
        return payload

    def content_digest(self) -> str:
        return self._digest_payload(self.payload_without_digest())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExternalEntryRepertoire:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported external-entry repertoire payload")
        width = payload.get("width")
        merge_cosine = payload.get("merge_cosine")
        entry_space_id = payload.get("entry_space_id")
        version = payload.get("version")
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise TypeError("external-entry payload width is invalid")
        if not isinstance(merge_cosine, (int, float)) or not math.isfinite(
            float(merge_cosine)
        ):
            raise ValueError("external-entry payload merge cosine is invalid")
        if not isinstance(entry_space_id, str) or not entry_space_id:
            raise ValueError("external-entry payload space ID is missing")
        if not isinstance(version, int) or isinstance(version, bool) or version < 0:
            raise ValueError("external-entry payload version is invalid")
        if payload.get("sha256") != cls._digest_payload(payload):
            raise ValueError("external-entry repertoire checksum mismatch")
        required = (
            "entries",
            "attempts",
            "outcome_counts",
            "utility_sums",
            "utility_square_sums",
            "propensity_sums",
            "inverse_propensity_utility_sums",
            "last_propensities",
            "last_seen",
        )
        if any(name not in payload for name in required):
            raise ValueError("external-entry payload is incomplete")
        entries = payload["entries"]
        if not isinstance(entries, torch.Tensor) or entries.ndim != 2:
            raise ValueError("external-entry payload vectors are invalid")
        if entries.shape[1] != width:
            raise ValueError("external-entry payload vector width differs")
        count = entries.shape[0]
        tensors = {name: payload[name] for name in required if name != "entries"}
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor) or value.shape != (count,):
                raise ValueError(f"external-entry payload {name} is misaligned")
        repertoire = cls(
            width,
            merge_cosine=float(merge_cosine),
            entry_space_id=entry_space_id,
        )
        repertoire._entries = [
            row.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for row in entries
        ]
        repertoire._attempts = [int(value) for value in tensors["attempts"].tolist()]
        repertoire._outcome_counts = [
            int(value) for value in tensors["outcome_counts"].tolist()
        ]
        repertoire._utility_sums = [
            float(value) for value in tensors["utility_sums"].tolist()
        ]
        repertoire._utility_square_sums = [
            float(value) for value in tensors["utility_square_sums"].tolist()
        ]
        repertoire._propensity_sums = [
            float(value) for value in tensors["propensity_sums"].tolist()
        ]
        repertoire._inverse_propensity_utility_sums = [
            float(value)
            for value in tensors["inverse_propensity_utility_sums"].tolist()
        ]
        repertoire._last_propensities = [
            float(value) for value in tensors["last_propensities"].tolist()
        ]
        repertoire._last_seen = [int(value) for value in tensors["last_seen"].tolist()]
        repertoire._version = version
        repertoire.validate_state()
        return repertoire

    def _prefix_digest(self, count: int) -> str:
        if not isinstance(count, int) or count < 0 or count > self.record_count:
            raise ValueError("external-entry prefix count is invalid")
        payload = self.payload_without_digest()
        payload.pop("version")
        prefix = {
            name: value[:count] if isinstance(value, torch.Tensor) else value
            for name, value in payload.items()
        }
        return self._digest_payload(prefix)

    def _copy_from(self, other: ExternalEntryRepertoire) -> None:
        if not isinstance(other, ExternalEntryRepertoire):
            raise TypeError("external-entry replacement must use same type")
        if (
            self.width != other.width
            or self.merge_cosine != other.merge_cosine
            or self.entry_space_id != other.entry_space_id
        ):
            raise ValueError("external-entry replacement configuration differs")
        other.validate_state()
        self._entries = [entry.clone() for entry in other._entries]
        self._attempts = list(other._attempts)
        self._outcome_counts = list(other._outcome_counts)
        self._utility_sums = list(other._utility_sums)
        self._utility_square_sums = list(other._utility_square_sums)
        self._propensity_sums = list(other._propensity_sums)
        self._inverse_propensity_utility_sums = list(
            other._inverse_propensity_utility_sums
        )
        self._last_propensities = list(other._last_propensities)
        self._last_seen = list(other._last_seen)
        self._version = other._version

    def observe(
        self,
        entries: torch.Tensor,
        *,
        utility: torch.Tensor | float | None = None,
        propensity: torch.Tensor | float | None = None,
        timestamp: torch.Tensor | int | None = None,
    ) -> ExternalEntryObservationReceipt:
        """Record opaque entry experience without updating learned weights."""

        normalized, utility_values, propensity_values, timestamp_values = self._validate_batch(
            entries,
            utility,
            propensity,
            timestamp,
        )
        located: list[int] = []
        added: list[bool] = []
        for row in normalized:
            index = self._find_entry(row)
            if index is None:
                self._entries.append(row.clone())
                self._attempts.append(0)
                self._outcome_counts.append(0)
                self._utility_sums.append(0.0)
                self._utility_square_sums.append(0.0)
                self._propensity_sums.append(0.0)
                self._inverse_propensity_utility_sums.append(0.0)
                self._last_propensities.append(0.0)
                self._last_seen.append(0)
                index = len(self._entries) - 1
                added.append(True)
            else:
                added.append(False)
            located.append(index)

        self._version += 1
        for row_index, entry_index in enumerate(located):
            self._attempts[entry_index] += 1
            propensity_value = float(propensity_values[row_index])
            self._propensity_sums[entry_index] += propensity_value
            self._last_propensities[entry_index] = propensity_value
            self._last_seen[entry_index] = int(timestamp_values[row_index])
            if utility_values is not None:
                utility_value = float(utility_values[row_index])
                self._outcome_counts[entry_index] += 1
                self._utility_sums[entry_index] += utility_value
                self._utility_square_sums[entry_index] += utility_value * utility_value
                self._inverse_propensity_utility_sums[entry_index] += (
                    utility_value / propensity_value
                )
        self.validate_state()
        return ExternalEntryObservationReceipt(
            entry_indices=tuple(located),
            added=tuple(added),
            outcome_observed=utility_values is not None,
            version=self._version,
            record_count=self.record_count,
            content_digest=self.content_digest(),
        ).validate()

    def propose(
        self,
        *,
        max_candidates: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalEntryProposal:
        """Return verified entries without reward-ranked ordering."""

        if not self._entries:
            raise ValueError("external-entry repertoire cannot propose an empty set")
        if max_candidates is not None and (
            not isinstance(max_candidates, int) or max_candidates < 1
        ):
            raise ValueError("maximum external-entry candidate count is invalid")
        entries = self._stack_entries()
        if max_candidates is not None:
            entries = entries[:max_candidates]
        count = entries.shape[0]
        entries = entries.to(device=device, dtype=dtype)
        return ExternalEntryProposal(
            entries=entries,
            source_indices=tuple(range(count)),
            propensities=torch.full(
                (count,),
                1.0 / count,
                device=entries.device,
                dtype=entries.dtype,
            ),
            version=self._version,
        ).validate(width=self.width)

    def admit_verified(
        self,
        entry: torch.Tensor,
        verifier: Callable[[ExternalEntryRepertoire], bool],
        *,
        reason: str = "caller_owned_heldout_verifier",
    ) -> ExternalEntryAdmissionReceipt:
        """Stage one novel entry and admit it only after retention-safe verification."""

        if not callable(verifier):
            raise TypeError("external-entry admission verifier must be callable")
        if not isinstance(reason, str) or not reason:
            raise ValueError("external-entry admission reason is missing")
        normalized, _utility, _propensity, _timestamp = self._validate_batch(
            entry,
            None,
            None,
            None,
        )
        if normalized.shape[0] != 1:
            raise ValueError("external-entry admission accepts one vector")
        candidate_entry = normalized[0]
        source_count = self.record_count
        source_digest = self.content_digest()
        if self._find_entry(candidate_entry) is not None:
            return ExternalEntryAdmissionReceipt(
                accepted=False,
                entry_index=None,
                source_record_count=source_count,
                destination_record_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                destination_digest=source_digest,
                reason="entry already exists in verified repertoire",
                version=self._version,
            ).validate()

        candidate = ExternalEntryRepertoire.from_payload(self.payload())
        candidate.observe(candidate_entry)
        candidate_digest = candidate.content_digest()
        accepted = bool(verifier(candidate))
        prefix_unchanged = candidate._prefix_digest(source_count) == self._prefix_digest(
            source_count
        )
        shape_unchanged = candidate.record_count == source_count + 1
        staged_vector_unchanged = shape_unchanged and torch.equal(
            candidate.statistics()["entries"][source_count],
            candidate_entry,
        )
        accepted = accepted and prefix_unchanged and shape_unchanged and staged_vector_unchanged
        if accepted:
            self._copy_from(candidate)
            return ExternalEntryAdmissionReceipt(
                accepted=True,
                entry_index=source_count,
                source_record_count=source_count,
                destination_record_count=self.record_count,
                source_digest=source_digest,
                candidate_digest=candidate_digest,
                destination_digest=self.content_digest(),
                reason=reason,
                version=self._version,
            ).validate()
        return ExternalEntryAdmissionReceipt(
            accepted=False,
            entry_index=None,
            source_record_count=source_count,
            destination_record_count=source_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            destination_digest=source_digest,
            reason="heldout verifier rejected or mutated retained entry state",
            version=self._version,
        ).validate()


@dataclass(frozen=True)
class ExternalEntryBindingObservationReceipt:
    """Auditable result of recording an intention-entry pair."""

    entry_indices: tuple[int, ...]
    added: tuple[bool, ...]
    outcome_observed: bool
    version: int
    record_count: int
    content_digest: str
    schema: str = EXTERNAL_ENTRY_BINDING_OBSERVATION_SCHEMA

    def validate(self) -> ExternalEntryBindingObservationReceipt:
        if self.schema != EXTERNAL_ENTRY_BINDING_OBSERVATION_SCHEMA:
            raise ValueError("unsupported external-entry binding observation schema")
        if not self.entry_indices or len(self.entry_indices) != len(self.added):
            raise ValueError("external-entry binding observation rows are misaligned")
        if any(not isinstance(index, int) or index < 0 for index in self.entry_indices):
            raise ValueError("external-entry binding observation index is invalid")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("external-entry binding observation version is invalid")
        if not isinstance(self.record_count, int) or self.record_count < 1:
            raise ValueError("external-entry binding observation count is invalid")
        if not isinstance(self.content_digest, str) or not self.content_digest:
            raise ValueError("external-entry binding observation digest is missing")
        return self


@dataclass(frozen=True)
class ExternalEntryBindingProposal:
    """Atomic intention-entry candidates retrieved from external memory."""

    intentions: torch.Tensor
    entries: torch.Tensor
    source_indices: tuple[int, ...]
    propensities: torch.Tensor
    version: int
    schema: str = EXTERNAL_ENTRY_BINDING_PROPOSAL_SCHEMA

    def validate(
        self,
        *,
        intention_width: int,
        entry_width: int,
    ) -> ExternalEntryBindingProposal:
        if self.schema != EXTERNAL_ENTRY_BINDING_PROPOSAL_SCHEMA:
            raise ValueError("unsupported external-entry binding proposal schema")
        _validate_tensor(
            self.intentions,
            name="external-entry binding intentions",
            ndim=2,
            width=intention_width,
        )
        _validate_tensor(
            self.entries,
            name="external-entry binding entries",
            ndim=2,
            width=entry_width,
        )
        count = self.intentions.shape[0]
        if count < 1 or self.entries.shape[0] != count:
            raise ValueError("external-entry binding proposal rows are misaligned")
        if len(self.source_indices) != count or len(set(self.source_indices)) != count:
            raise ValueError("external-entry binding proposal indices are invalid")
        if any(index < 0 for index in self.source_indices):
            raise ValueError("external-entry binding proposal source index is invalid")
        if self.propensities.shape != (count,):
            raise ValueError("external-entry binding proposal propensities are misaligned")
        if not bool(torch.isfinite(self.propensities).all()) or bool(
            torch.any(self.propensities <= 0.0) | torch.any(self.propensities > 1.0)
        ):
            raise ValueError("external-entry binding proposal propensities are invalid")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("external-entry binding proposal version is invalid")
        return self


@dataclass(frozen=True)
class ExternalEntryBindingAdmissionReceipt:
    """Verifier-gated copy-on-write admission result for a pair."""

    accepted: bool
    entry_index: int | None
    source_record_count: int
    destination_record_count: int
    source_digest: str
    candidate_digest: str
    destination_digest: str
    reason: str
    version: int
    schema: str = EXTERNAL_ENTRY_BINDING_ADMISSION_SCHEMA

    def validate(self) -> ExternalEntryBindingAdmissionReceipt:
        if self.schema != EXTERNAL_ENTRY_BINDING_ADMISSION_SCHEMA:
            raise ValueError("unsupported external-entry binding admission schema")
        if min(self.source_record_count, self.destination_record_count) < 0:
            raise ValueError("external-entry binding admission counts are invalid")
        if self.accepted:
            if self.entry_index is None or self.entry_index < 0:
                raise ValueError("accepted external-entry binding index is invalid")
            if self.destination_record_count != self.source_record_count + 1:
                raise ValueError("accepted external-entry binding count is invalid")
        elif self.entry_index is not None:
            raise ValueError("rejected external-entry binding has an index")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("external-entry binding admission version is invalid")
        for name, value in (
            ("source_digest", self.source_digest),
            ("candidate_digest", self.candidate_digest),
            ("destination_digest", self.destination_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"external-entry binding admission {name} is missing")
        return self


@dataclass(frozen=True)
class ExternalEntryBindingConsolidationReceipt:
    """Retention-gated copy-on-write consolidation result."""

    accepted: bool
    retired_ids: tuple[int, ...]
    replacement_id: int | None
    source_record_count: int
    destination_record_count: int
    source_digest: str
    candidate_digest: str
    destination_digest: str
    reason: str
    version: int
    schema: str = EXTERNAL_ENTRY_BINDING_CONSOLIDATION_SCHEMA

    def validate(self) -> ExternalEntryBindingConsolidationReceipt:
        if self.schema != EXTERNAL_ENTRY_BINDING_CONSOLIDATION_SCHEMA:
            raise ValueError("unsupported external-entry binding consolidation schema")
        if not self.retired_ids or len(set(self.retired_ids)) != len(self.retired_ids):
            raise ValueError("external-entry binding retired IDs are invalid")
        if any(not isinstance(value, int) or value < 0 for value in self.retired_ids):
            raise ValueError("external-entry binding retired ID is invalid")
        if self.accepted:
            if self.replacement_id is None or self.replacement_id < 0:
                raise ValueError("accepted consolidation has no replacement ID")
            if self.destination_record_count != self.source_record_count - len(
                self.retired_ids
            ) + 1:
                raise ValueError("accepted consolidation count is invalid")
        elif self.replacement_id is not None:
            raise ValueError("rejected consolidation has a replacement ID")
        if min(self.source_record_count, self.destination_record_count) < 1:
            raise ValueError("external-entry binding consolidation counts are invalid")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("external-entry binding consolidation version is invalid")
        for name, value in (
            ("source_digest", self.source_digest),
            ("candidate_digest", self.candidate_digest),
            ("destination_digest", self.destination_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"external-entry binding consolidation {name} is missing")
        return self


class ExternalEntryBindingRepertoire:
    """Append-only memory of atomically bound opaque intention-entry pairs.

    Binding is stored outside the controller.  The proposal returns both
    tensors from one record, eliminating positional joins between independent
    repertoires.  Pair admission uses the same held-out, retention-safe
    transaction as the standalone entry repertoire.
    """

    schema = EXTERNAL_ENTRY_BINDING_REPERTOIRE_SCHEMA

    def __init__(
        self,
        intention_width: int,
        entry_width: int,
        *,
        merge_cosine: float = 0.999,
        intention_space_id: str = DEFAULT_INTENTION_SPACE_ID,
        entry_space_id: str = DEFAULT_MEMORY_VALUE_SPACE_ID,
    ) -> None:
        if not isinstance(intention_width, int) or intention_width < 1:
            raise ValueError("external-entry binding intention width must be positive")
        if not isinstance(entry_width, int) or entry_width < 1:
            raise ValueError("external-entry binding entry width must be positive")
        self.intention_width = int(intention_width)
        self.entry_width = int(entry_width)
        self.merge_cosine = float(merge_cosine)
        if not -1.0 <= self.merge_cosine <= 1.0 or not math.isfinite(
            self.merge_cosine
        ):
            raise ValueError("external-entry binding merge cosine is invalid")
        self.intention_space_id = validate_representation_space_id(
            intention_space_id,
            name="intention_space_id",
        )
        self.entry_space_id = validate_representation_space_id(
            entry_space_id,
            name="entry_space_id",
        )
        self._store = ExternalEntryRepertoire(
            self.intention_width + self.entry_width,
            merge_cosine=self.merge_cosine,
            entry_space_id=(
                f"{self.intention_space_id}+{self.entry_space_id}"
            ),
        )
        self._logical_ids: list[int] = []
        self._next_logical_id = 0
        self._aliases: dict[int, int] = {}

    @property
    def record_count(self) -> int:
        return self._store.record_count

    @property
    def version(self) -> int:
        return self._store.version

    @property
    def logical_ids(self) -> tuple[int, ...]:
        """Return stable logical IDs in current physical proposal order."""

        return tuple(self._logical_ids)

    def logical_id_at(self, index: int) -> int:
        if not 0 <= index < self.record_count:
            raise IndexError("external-entry binding physical index is out of range")
        return self._logical_ids[index]

    def resolve_logical_id(self, logical_id: int) -> int:
        """Resolve a retired logical ID to its retained replacement."""

        if not isinstance(logical_id, int) or logical_id < 0:
            raise ValueError("external-entry binding logical ID is invalid")
        seen: set[int] = set()
        current = logical_id
        while current in self._aliases:
            if current in seen:
                raise RuntimeError("external-entry binding alias cycle detected")
            seen.add(current)
            current = self._aliases[current]
        return current

    def physical_index_for_id(self, logical_id: int) -> int:
        if not isinstance(logical_id, int) or logical_id < 0:
            raise ValueError("external-entry binding logical ID is invalid")
        resolved = self.resolve_logical_id(logical_id)
        try:
            return self._logical_ids.index(resolved)
        except ValueError as error:
            raise KeyError(
                f"unknown external-entry binding logical ID: {logical_id}"
            ) from error

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "intention_width": self.intention_width,
            "entry_width": self.entry_width,
            "merge_cosine": self.merge_cosine,
            "intention_space_id": self.intention_space_id,
            "entry_space_id": self.entry_space_id,
            "storage": "append_only_atomic_opaque_intention_entry_binding_v1",
            "learning": "outcome_sufficient_statistics_without_replay_v1",
        }

    def _validate_pairs(
        self,
        intentions: torch.Tensor,
        entries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intentions.ndim == 1:
            intentions = intentions.unsqueeze(0)
        if entries.ndim == 1:
            entries = entries.unsqueeze(0)
        _validate_tensor(
            intentions,
            name="observed binding intention",
            ndim=2,
            width=self.intention_width,
        )
        _validate_tensor(
            entries,
            name="observed binding entry",
            ndim=2,
            width=self.entry_width,
        )
        if intentions.shape[0] != entries.shape[0]:
            raise ValueError("binding intentions and entries have different batches")
        if intentions.shape[0] < 1:
            raise ValueError("external-entry binding observation cannot be empty")
        return (
            intentions.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            entries.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        )

    def _joined(self, intentions: torch.Tensor, entries: torch.Tensor) -> torch.Tensor:
        return torch.cat((intentions, entries), dim=-1)

    def statistics(self) -> dict[str, torch.Tensor]:
        stats = self._store.statistics()
        joined = stats.pop("entries")
        stats["intentions"] = joined[:, : self.intention_width]
        stats["entries"] = joined[:, self.intention_width :]
        return stats

    def validate_state(self) -> None:
        self._store.validate_state()
        stats = self.statistics()
        if stats["intentions"].shape[1] != self.intention_width:
            raise ValueError("external-entry binding intention state is invalid")
        if stats["entries"].shape[1] != self.entry_width:
            raise ValueError("external-entry binding entry state is invalid")
        if len(self._logical_ids) != self.record_count:
            raise ValueError("external-entry binding logical IDs are misaligned")
        if any(not isinstance(value, int) or value < 0 for value in self._logical_ids):
            raise ValueError("external-entry binding logical IDs are invalid")
        if len(set(self._logical_ids)) != len(self._logical_ids):
            raise ValueError("external-entry binding logical IDs are duplicated")
        if not isinstance(self._next_logical_id, int) or self._next_logical_id < 0:
            raise ValueError("external-entry binding next logical ID is invalid")
        if self._next_logical_id <= max(self._logical_ids, default=-1):
            raise ValueError("external-entry binding next logical ID is stale")
        if any(
            not isinstance(source, int)
            or source < 0
            or not isinstance(destination, int)
            or destination < 0
            for source, destination in self._aliases.items()
        ):
            raise ValueError("external-entry binding aliases are invalid")
        if set(self._aliases) & set(self._logical_ids):
            raise ValueError("external-entry binding aliases shadow live IDs")
        for source, destination in self._aliases.items():
            if self.resolve_logical_id(destination) not in self._logical_ids:
                raise ValueError("external-entry binding alias target is not live")
            if source == destination:
                raise ValueError("external-entry binding alias is self-referential")

    @staticmethod
    def _digest_payload(payload: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        for name in sorted(payload, key=str):
            if name == "sha256":
                continue
            value = payload[name]
            digest.update(str(name).encode("utf-8"))
            if isinstance(value, Mapping):
                digest.update(ExternalEntryBindingRepertoire._digest_payload(value).encode())
            elif isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
        return digest.hexdigest()

    def payload_without_digest(self) -> dict[str, Any]:
        self.validate_state()
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "version": self.version,
            "logical_ids": list(self._logical_ids),
            "next_logical_id": self._next_logical_id,
            "aliases": dict(sorted(self._aliases.items())),
            "store": self._store.payload(),
        }

    def payload(self) -> dict[str, Any]:
        payload = self.payload_without_digest()
        payload["sha256"] = self._digest_payload(payload)
        return payload

    def content_digest(self) -> str:
        return self._digest_payload(self.payload_without_digest())

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalEntryBindingRepertoire:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported external-entry binding payload")
        if payload.get("sha256") != cls._digest_payload(payload):
            raise ValueError("external-entry binding checksum mismatch")
        configuration = payload.get("configuration")
        store_payload = payload.get("store")
        version = payload.get("version")
        logical_ids = payload.get("logical_ids")
        next_logical_id = payload.get("next_logical_id")
        aliases = payload.get("aliases", {})
        if not isinstance(configuration, Mapping) or not isinstance(store_payload, Mapping):
            raise TypeError("external-entry binding payload is incomplete")
        if not isinstance(version, int) or version < 0:
            raise ValueError("external-entry binding payload version is invalid")
        if not isinstance(logical_ids, list) or not all(
            isinstance(value, int) and value >= 0 for value in logical_ids
        ):
            raise ValueError("external-entry binding payload logical IDs are invalid")
        if not isinstance(next_logical_id, int) or next_logical_id < 0:
            raise ValueError("external-entry binding payload next logical ID is invalid")
        if not isinstance(aliases, Mapping):
            raise TypeError("external-entry binding payload aliases are invalid")
        normalized_aliases: dict[int, int] = {}
        for source, destination in aliases.items():
            if (
                not isinstance(source, int)
                or isinstance(source, bool)
                or not isinstance(destination, int)
                or isinstance(destination, bool)
            ):
                raise TypeError("external-entry binding payload aliases are invalid")
            normalized_aliases[int(source)] = int(destination)
        repertoire = cls(
            int(configuration["intention_width"]),
            int(configuration["entry_width"]),
            merge_cosine=float(configuration["merge_cosine"]),
            intention_space_id=str(configuration["intention_space_id"]),
            entry_space_id=str(configuration["entry_space_id"]),
        )
        repertoire._store = ExternalEntryRepertoire.from_payload(store_payload)
        if repertoire.version != version:
            raise ValueError("external-entry binding payload versions differ")
        if repertoire.configuration() != {
            key: configuration[key]
            for key in repertoire.configuration()
            if key in configuration
        }:
            raise ValueError("external-entry binding payload configuration differs")
        repertoire._logical_ids = list(logical_ids)
        repertoire._next_logical_id = next_logical_id
        repertoire._aliases = normalized_aliases
        repertoire.validate_state()
        return repertoire

    def _prefix_digest(self, count: int) -> str:
        return self._store._prefix_digest(count)

    def _copy_from(self, other: ExternalEntryBindingRepertoire) -> None:
        if not isinstance(other, ExternalEntryBindingRepertoire):
            raise TypeError("external-entry binding replacement must use same type")
        self._store = ExternalEntryRepertoire.from_payload(other._store.payload())
        self._logical_ids = list(other._logical_ids)
        self._next_logical_id = other._next_logical_id
        self._aliases = dict(other._aliases)

    def observe(
        self,
        intentions: torch.Tensor,
        entries: torch.Tensor,
        *,
        utility: torch.Tensor | float | None = None,
        propensity: torch.Tensor | float | None = None,
        timestamp: torch.Tensor | int | None = None,
    ) -> ExternalEntryBindingObservationReceipt:
        normalized_intentions, normalized_entries = self._validate_pairs(
            intentions,
            entries,
        )
        receipt = self._store.observe(
            self._joined(normalized_intentions, normalized_entries),
            utility=utility,
            propensity=propensity,
            timestamp=timestamp,
        )
        for physical_index, added in zip(
            receipt.entry_indices,
            receipt.added,
            strict=True,
        ):
            if added:
                if physical_index != len(self._logical_ids):
                    raise RuntimeError(
                        "external-entry binding storage appended out of order"
                    )
                self._logical_ids.append(self._next_logical_id)
                self._next_logical_id += 1
        logical_indices = tuple(
            self._logical_ids[physical_index] for physical_index in receipt.entry_indices
        )
        self.validate_state()
        return ExternalEntryBindingObservationReceipt(
            entry_indices=logical_indices,
            added=receipt.added,
            outcome_observed=receipt.outcome_observed,
            version=receipt.version,
            record_count=receipt.record_count,
            content_digest=self.content_digest(),
        ).validate()

    def _consolidation_candidate(
        self,
        retired_ids: tuple[int, ...],
        replacement_intention: torch.Tensor,
        replacement_entry: torch.Tensor,
    ) -> tuple[ExternalEntryBindingRepertoire, int]:
        if len(retired_ids) < 2:
            raise ValueError("external-entry binding consolidation needs two records")
        if len(set(retired_ids)) != len(retired_ids):
            raise ValueError("external-entry binding consolidation IDs are duplicated")
        if any(
            not isinstance(logical_id, int)
            or isinstance(logical_id, bool)
            or logical_id < 0
            for logical_id in retired_ids
        ):
            raise ValueError("external-entry binding consolidation ID is invalid")
        if any(logical_id not in self._logical_ids for logical_id in retired_ids):
            raise ValueError(
                "external-entry binding consolidation can retire live IDs only"
            )

        normalized_intention, normalized_entry = self._validate_pairs(
            replacement_intention,
            replacement_entry,
        )
        if normalized_intention.shape[0] != 1:
            raise ValueError("external-entry binding consolidation accepts one pair")
        replacement_pair = self._joined(
            normalized_intention,
            normalized_entry,
        )[0]
        retired_indices = {
            self._logical_ids.index(logical_id) for logical_id in retired_ids
        }
        retained_indices = [
            index
            for index in range(self.record_count)
            if index not in retired_indices
        ]
        source_store = self._store
        replacement_store = ExternalEntryRepertoire(
            self.intention_width + self.entry_width,
            merge_cosine=self.merge_cosine,
            entry_space_id=(
                f"{self.intention_space_id}+{self.entry_space_id}"
            ),
        )
        retained_entries = [
            source_store._entries[index].clone() for index in retained_indices
        ]
        replacement_store._entries = retained_entries
        if replacement_store._find_entry(replacement_pair) is not None:
            raise ValueError(
                "external-entry binding consolidation replacement duplicates a retained pair"
            )
        replacement_store._entries.append(replacement_pair.clone())

        def aggregate(values: list[int | float]) -> list[int | float]:
            retained = [values[index] for index in retained_indices]
            replacement = sum(values[index] for index in retired_indices)
            return retained + [replacement]

        replacement_store._attempts = [
            int(value)
            for value in aggregate(source_store._attempts)
        ]
        replacement_store._outcome_counts = [
            int(value)
            for value in aggregate(source_store._outcome_counts)
        ]
        replacement_store._utility_sums = [
            float(value)
            for value in aggregate(source_store._utility_sums)
        ]
        replacement_store._utility_square_sums = [
            float(value)
            for value in aggregate(source_store._utility_square_sums)
        ]
        replacement_store._propensity_sums = [
            float(value)
            for value in aggregate(source_store._propensity_sums)
        ]
        replacement_store._inverse_propensity_utility_sums = [
            float(value)
            for value in aggregate(source_store._inverse_propensity_utility_sums)
        ]
        latest_index = max(
            retired_indices,
            key=lambda index: (source_store._last_seen[index], index),
        )
        replacement_store._last_propensities = [
            source_store._last_propensities[index] for index in retained_indices
        ] + [source_store._last_propensities[latest_index]]
        replacement_store._last_seen = [
            source_store._last_seen[index] for index in retained_indices
        ] + [
            max(source_store._last_seen[index] for index in retired_indices)
        ]
        replacement_store._version = source_store.version + 1
        replacement_store.validate_state()

        candidate = ExternalEntryBindingRepertoire.from_payload(self.payload())
        candidate._store = replacement_store
        replacement_id = min(retired_ids)
        candidate._logical_ids = [
            self._logical_ids[index] for index in retained_indices
        ] + [replacement_id]
        candidate._aliases = {
            source: (
                replacement_id
                if destination in retired_ids
                else candidate.resolve_logical_id(destination)
            )
            for source, destination in self._aliases.items()
        }
        for logical_id in retired_ids:
            if logical_id != replacement_id:
                candidate._aliases[logical_id] = replacement_id
        candidate._next_logical_id = max(
            self._next_logical_id,
            replacement_id + 1,
        )
        candidate.validate_state()
        return candidate, replacement_id

    def consolidate_verified(
        self,
        retired_ids: tuple[int, ...] | list[int],
        replacement_intention: torch.Tensor,
        replacement_entry: torch.Tensor,
        retention_probe: Callable[[ExternalEntryBindingRepertoire], bool],
        *,
        reason: str = "caller_owned_heldout_retention_probe",
    ) -> ExternalEntryBindingConsolidationReceipt:
        """Consolidate records only after an isolated retention probe passes.

        The live repertoire is untouched while the candidate is compacted and
        probed.  Retired logical IDs resolve to the replacement ID, so callers
        can keep durable references while physical storage changes.
        """

        if not callable(retention_probe):
            raise TypeError("external-entry binding retention probe must be callable")
        if not isinstance(reason, str) or not reason:
            raise ValueError("external-entry binding consolidation reason is missing")
        normalized_ids = tuple(retired_ids)
        source_count = self.record_count
        source_digest = self.content_digest()
        candidate, replacement_id = self._consolidation_candidate(
            normalized_ids,
            replacement_intention,
            replacement_entry,
        )
        candidate_digest = candidate.content_digest()
        accepted = bool(retention_probe(candidate))
        probe_unchanged = candidate.content_digest() == candidate_digest
        accepted = accepted and probe_unchanged
        if accepted:
            self._copy_from(candidate)
            return ExternalEntryBindingConsolidationReceipt(
                accepted=True,
                retired_ids=normalized_ids,
                replacement_id=replacement_id,
                source_record_count=source_count,
                destination_record_count=self.record_count,
                source_digest=source_digest,
                candidate_digest=candidate_digest,
                destination_digest=self.content_digest(),
                reason=reason,
                version=self.version,
            ).validate()
        return ExternalEntryBindingConsolidationReceipt(
            accepted=False,
            retired_ids=normalized_ids,
            replacement_id=None,
            source_record_count=source_count,
            destination_record_count=source_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            destination_digest=source_digest,
            reason=(
                "heldout retention probe rejected or mutated candidate binding state"
            ),
            version=self.version,
        ).validate()

    def propose(
        self,
        *,
        max_candidates: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalEntryBindingProposal:
        proposal = self._store.propose(
            max_candidates=max_candidates,
            device=device,
            dtype=dtype,
        )
        return ExternalEntryBindingProposal(
            intentions=proposal.entries[:, : self.intention_width],
            entries=proposal.entries[:, self.intention_width :],
            source_indices=tuple(
                self._logical_ids[physical_index]
                for physical_index in proposal.source_indices
            ),
            propensities=proposal.propensities,
            version=proposal.version,
        ).validate(
            intention_width=self.intention_width,
            entry_width=self.entry_width,
        )

    def admit_verified(
        self,
        intention: torch.Tensor,
        entry: torch.Tensor,
        verifier: Callable[[ExternalEntryBindingRepertoire], bool],
        *,
        reason: str = "caller_owned_heldout_verifier",
    ) -> ExternalEntryBindingAdmissionReceipt:
        if not callable(verifier):
            raise TypeError("external-entry binding verifier must be callable")
        if not isinstance(reason, str) or not reason:
            raise ValueError("external-entry binding admission reason is missing")
        normalized_intention, normalized_entry = self._validate_pairs(intention, entry)
        if normalized_intention.shape[0] != 1:
            raise ValueError("external-entry binding admission accepts one pair")
        candidate_intention = normalized_intention[0]
        candidate_entry = normalized_entry[0]
        candidate_pair = self._joined(normalized_intention, normalized_entry)[0]
        source_count = self.record_count
        source_digest = self.content_digest()
        if self._store._find_entry(candidate_pair) is not None:
            return ExternalEntryBindingAdmissionReceipt(
                accepted=False,
                entry_index=None,
                source_record_count=source_count,
                destination_record_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                destination_digest=source_digest,
                reason="intention-entry pair already exists",
                version=self.version,
            ).validate()

        candidate = ExternalEntryBindingRepertoire.from_payload(self.payload())
        candidate.observe(candidate_intention, candidate_entry)
        candidate_digest = candidate.content_digest()
        accepted = bool(verifier(candidate))
        prefix_unchanged = candidate._prefix_digest(source_count) == self._prefix_digest(
            source_count
        )
        shape_unchanged = candidate.record_count == source_count + 1
        staged_proposal = candidate.propose()
        staged = torch.cat(
            (staged_proposal.intentions[-1], staged_proposal.entries[-1]),
            dim=0,
        )
        staged_unchanged = shape_unchanged and torch.equal(
            staged,
            candidate_pair,
        )
        accepted = accepted and prefix_unchanged and shape_unchanged and staged_unchanged
        if accepted:
            self._copy_from(candidate)
            return ExternalEntryBindingAdmissionReceipt(
                accepted=True,
                entry_index=candidate.logical_id_at(candidate.record_count - 1),
                source_record_count=source_count,
                destination_record_count=self.record_count,
                source_digest=source_digest,
                candidate_digest=candidate_digest,
                destination_digest=self.content_digest(),
                reason=reason,
                version=self.version,
            ).validate()
        return ExternalEntryBindingAdmissionReceipt(
            accepted=False,
            entry_index=None,
            source_record_count=source_count,
            destination_record_count=source_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            destination_digest=source_digest,
            reason="heldout verifier rejected or mutated retained binding state",
            version=self.version,
        ).validate()


__all__ = [
    "EXTERNAL_ENTRY_ADMISSION_SCHEMA",
    "EXTERNAL_ENTRY_BINDING_ADMISSION_SCHEMA",
    "EXTERNAL_ENTRY_BINDING_CONSOLIDATION_SCHEMA",
    "EXTERNAL_ENTRY_BINDING_OBSERVATION_SCHEMA",
    "EXTERNAL_ENTRY_BINDING_PROPOSAL_SCHEMA",
    "EXTERNAL_ENTRY_BINDING_REPERTOIRE_SCHEMA",
    "EXTERNAL_ENTRY_OBSERVATION_SCHEMA",
    "EXTERNAL_ENTRY_PROPOSAL_SCHEMA",
    "EXTERNAL_ENTRY_REPERTOIRE_SCHEMA",
    "ExternalEntryAdmissionReceipt",
    "ExternalEntryBindingAdmissionReceipt",
    "ExternalEntryBindingConsolidationReceipt",
    "ExternalEntryBindingObservationReceipt",
    "ExternalEntryBindingProposal",
    "ExternalEntryBindingRepertoire",
    "ExternalEntryObservationReceipt",
    "ExternalEntryProposal",
    "ExternalEntryRepertoire",
]
