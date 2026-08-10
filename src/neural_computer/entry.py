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
    DEFAULT_MEMORY_VALUE_SPACE_ID,
    validate_representation_space_id,
)

EXTERNAL_ENTRY_OBSERVATION_SCHEMA = "neural-computer.external-entry-observation.v1"
EXTERNAL_ENTRY_PROPOSAL_SCHEMA = "neural-computer.external-entry-proposal.v1"
EXTERNAL_ENTRY_ADMISSION_SCHEMA = "neural-computer.external-entry-admission.v1"
EXTERNAL_ENTRY_REPERTOIRE_SCHEMA = "neural-computer.external-entry-repertoire.v1"


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
        for name in sorted(payload):
            if name == "sha256":
                continue
            value = payload[name]
            digest.update(name.encode("utf-8"))
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


__all__ = [
    "EXTERNAL_ENTRY_ADMISSION_SCHEMA",
    "EXTERNAL_ENTRY_OBSERVATION_SCHEMA",
    "EXTERNAL_ENTRY_PROPOSAL_SCHEMA",
    "EXTERNAL_ENTRY_REPERTOIRE_SCHEMA",
    "ExternalEntryAdmissionReceipt",
    "ExternalEntryObservationReceipt",
    "ExternalEntryProposal",
    "ExternalEntryRepertoire",
]
