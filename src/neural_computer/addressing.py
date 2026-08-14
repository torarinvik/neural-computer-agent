"""Task-agnostic routing over opaque external-memory addresses.

The router is a memory-side component. It receives learned controller query
vectors, opaque candidate address rows, an attempted row, and a scalar
outcome during training. It does not receive task identifiers, semantic
labels, or correct unattempted actions, and it does not add a reasoning path
to the frozen controller.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .representation import (
    DEFAULT_ROUTE_QUERY_SPACE_ID,
    validate_representation_space_id,
)


def _payload_digest(payload: dict[str, object]) -> str:
    """Hash a JSON-compatible opaque-memory payload without its digest."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PersistentRouteEvidenceStatus:
    """Auditable external evidence for one append-only route bank."""

    attempts: tuple[int, ...]
    successes: tuple[float, ...]
    posterior: tuple[float, ...]
    stable_prefix_minimum: tuple[float, ...]
    protected: tuple[bool, ...]
    reversal_streak: tuple[int, ...]
    reversal_count: tuple[int, ...]
    recovery_streak: tuple[int, ...]
    preferred_slot: int | None
    last_slot: int | None
    last_outcome: float | None
    version: int


class PersistentOpaqueRouteEvidence:
    """Persist scalar route evidence outside the controller.

    The ledger stores only attempted opaque slot indices and deterministic
    scalar outcomes.  It has no task names, route labels, or correct-action
    fields.  A successful episode makes its slot the preferred starting point;
    the append-only order supplies a bounded fallback path when that prior is
    wrong.  This is external mutable state, not a trainable controller branch.
    """

    schema = "neural-computer.persistent-opaque-route-evidence.v1"

    def __init__(
        self,
        *,
        prior_strength: float = 1.0,
        mastery_threshold: float = 0.8,
        min_mastery_observations: int = 8,
        reversal_threshold: float = 0.5,
        reversal_patience: int = 4,
    ) -> None:
        if prior_strength <= 0.0:
            raise ValueError("route-evidence prior strength must be positive")
        if not 0.0 <= mastery_threshold <= 1.0:
            raise ValueError("route-evidence mastery threshold is invalid")
        if min_mastery_observations < 1:
            raise ValueError("route-evidence minimum observations must be positive")
        if not 0.0 <= reversal_threshold <= 1.0:
            raise ValueError("route-evidence reversal threshold is invalid")
        if reversal_patience < 1:
            raise ValueError("route-evidence reversal patience must be positive")
        self.prior_strength = float(prior_strength)
        self.mastery_threshold = float(mastery_threshold)
        self.min_mastery_observations = int(min_mastery_observations)
        self.reversal_threshold = float(reversal_threshold)
        self.reversal_patience = int(reversal_patience)
        self._attempts: list[int] = []
        self._successes: list[float] = []
        self._stable_prefix_minimum: list[float] = []
        self._protected: list[bool] = []
        self._reversal_streak: list[int] = []
        self._reversal_count: list[int] = []
        self._recovery_streak: list[int] = []
        self._preferred_slot: int | None = None
        self._last_slot: int | None = None
        self._last_outcome: float | None = None
        self._version = 0

    @property
    def slot_count(self) -> int:
        return len(self._attempts)

    def append_slot(self) -> int:
        """Append one opaque route row and return its stable index."""

        self._attempts.append(0)
        self._successes.append(0.0)
        self._stable_prefix_minimum.append(1.0)
        self._protected.append(False)
        self._reversal_streak.append(0)
        self._reversal_count.append(0)
        self._recovery_streak.append(0)
        self._version += 1
        return len(self._attempts) - 1

    def reset_slot(self, slot: int) -> None:
        """Clear one replaceable slot while preserving bank width and order."""

        self._validate_slot(slot)
        self._attempts[slot] = 0
        self._successes[slot] = 0.0
        self._stable_prefix_minimum[slot] = 1.0
        self._protected[slot] = False
        self._reversal_streak[slot] = 0
        self._reversal_count[slot] = 0
        self._recovery_streak[slot] = 0
        if self._preferred_slot == slot:
            self._preferred_slot = None
        if self._last_slot == slot:
            self._last_slot = None
            self._last_outcome = None
        self._version += 1

    def _validate_slot(self, slot: int) -> None:
        if not isinstance(slot, int) or not 0 <= slot < self.slot_count:
            raise IndexError("route-evidence slot is outside the bank")

    def observe(self, slot: int, outcome: float | torch.Tensor) -> None:
        """Record one attempted slot and its scalar verifier outcome."""

        self._validate_slot(slot)
        if isinstance(outcome, torch.Tensor):
            if outcome.numel() != 1:
                raise ValueError("route-evidence outcome must be scalar")
            value = float(outcome.detach().cpu().item())
        else:
            value = float(outcome)
        if not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0:
            raise ValueError("route-evidence outcome must lie in [0, 1]")
        was_protected = self._protected[slot]
        self._attempts[slot] += 1
        self._successes[slot] += value
        if value >= self.mastery_threshold:
            self._recovery_streak[slot] += 1
        else:
            self._recovery_streak[slot] = 0
        reversed_slot = False
        if was_protected:
            if value <= self.reversal_threshold:
                self._reversal_streak[slot] += 1
            else:
                self._reversal_streak[slot] = 0
            if self._reversal_streak[slot] >= self.reversal_patience:
                reversed_slot = True
                self._protected[slot] = False
                self._reversal_streak[slot] = 0
                self._reversal_count[slot] += 1
                self._attempts[slot] = 0
                self._successes[slot] = 0.0
                self._stable_prefix_minimum[slot] = 1.0
                self._recovery_streak[slot] = 0
                if self._preferred_slot == slot:
                    self._preferred_slot = None
        if (
            not reversed_slot
            and not self._protected[slot]
            and self._recovery_streak[slot] >= self.min_mastery_observations
        ):
            self._stable_prefix_minimum[slot] = 1.0
            self._protected[slot] = True
        elif (
            not reversed_slot
            and not self._protected[slot]
            and self._attempts[slot] >= self.min_mastery_observations
        ):
            prefix_mean = self._successes[slot] / self._attempts[slot]
            self._stable_prefix_minimum[slot] = min(
                self._stable_prefix_minimum[slot], prefix_mean
            )
            self._protected[slot] = (
                self._stable_prefix_minimum[slot] >= self.mastery_threshold
            )
        self._last_slot = slot
        self._last_outcome = value
        if self._protected[slot]:
            self._preferred_slot = slot
        self._version += 1

    def posterior(self, *, slot_count: int | None = None) -> torch.Tensor:
        """Return Beta-smoothed success estimates for the current bank."""

        if slot_count is not None and slot_count != self.slot_count:
            raise ValueError("route-evidence slot count does not match")
        prior = self.prior_strength
        successes = torch.tensor(self._successes, dtype=torch.float64)
        attempts = torch.tensor(self._attempts, dtype=torch.float64)
        return (successes + prior) / (attempts + 2.0 * prior)

    def preferred_order(self, *, slot_count: int | None = None) -> tuple[int, ...]:
        """Return a persistent-first, append-aware fallback order."""

        if slot_count is not None and slot_count != self.slot_count:
            raise ValueError("route-evidence slot count does not match")
        if self.slot_count < 1:
            raise ValueError("route-evidence bank has no slots")
        preferred = self._preferred_slot
        if preferred is None or preferred >= self.slot_count:
            preferred = 0
        later = range(preferred + 1, self.slot_count)
        earlier = range(preferred - 1, -1, -1)
        return (preferred, *later, *earlier)

    def status(self) -> PersistentRouteEvidenceStatus:
        posterior = tuple(float(value) for value in self.posterior().tolist())
        return PersistentRouteEvidenceStatus(
            attempts=tuple(self._attempts),
            successes=tuple(self._successes),
            posterior=posterior,
            stable_prefix_minimum=tuple(self._stable_prefix_minimum),
            protected=tuple(self._protected),
            reversal_streak=tuple(self._reversal_streak),
            reversal_count=tuple(self._reversal_count),
            recovery_streak=tuple(self._recovery_streak),
            preferred_slot=self._preferred_slot,
            last_slot=self._last_slot,
            last_outcome=self._last_outcome,
            version=self._version,
        )

    def payload(self) -> dict[str, object]:
        """Serialize the opaque evidence state without semantic metadata."""

        status = self.status()
        payload: dict[str, object] = {
            "schema": self.schema,
            "prior_strength": self.prior_strength,
            "mastery_threshold": self.mastery_threshold,
            "min_mastery_observations": self.min_mastery_observations,
            "reversal_threshold": self.reversal_threshold,
            "reversal_patience": self.reversal_patience,
            "attempts": list(status.attempts),
            "successes": list(status.successes),
            "stable_prefix_minimum": list(status.stable_prefix_minimum),
            "protected": list(status.protected),
            "reversal_streak": list(status.reversal_streak),
            "reversal_count": list(status.reversal_count),
            "recovery_streak": list(status.recovery_streak),
            "preferred_slot": status.preferred_slot,
            "last_slot": status.last_slot,
            "last_outcome": status.last_outcome,
            "version": status.version,
        }
        payload["sha256"] = _payload_digest(payload)
        return payload

    def digest(self) -> str:
        """Return the checksum of the external evidence state."""

        return str(self.payload()["sha256"])

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> PersistentOpaqueRouteEvidence:
        """Restore a validated external route-evidence snapshot."""

        unsigned = dict(payload)
        expected_digest = unsigned.pop("sha256", None)
        if expected_digest is not None and (
            not isinstance(expected_digest, str)
            or expected_digest != _payload_digest(unsigned)
        ):
            raise ValueError("route-evidence checksum mismatch")
        if unsigned.get("schema") != cls.schema:
            raise ValueError("route-evidence schema is incompatible")
        ledger = cls(
            prior_strength=float(unsigned["prior_strength"]),
            mastery_threshold=float(unsigned["mastery_threshold"]),
            min_mastery_observations=int(
                unsigned.get("min_mastery_observations", 8)
            ),
            reversal_threshold=float(unsigned.get("reversal_threshold", 0.5)),
            reversal_patience=int(unsigned.get("reversal_patience", 4)),
        )
        attempts = unsigned["attempts"]
        successes = unsigned["successes"]
        if not isinstance(attempts, list) or not isinstance(successes, list):
            raise TypeError("route-evidence rows must be lists")
        if len(attempts) != len(successes):
            raise ValueError("route-evidence rows have different lengths")
        for attempt, success in zip(attempts, successes):
            if not isinstance(attempt, int) or attempt < 0:
                raise ValueError("route-evidence attempts must be non-negative integers")
            if not isinstance(success, (int, float)) or not 0.0 <= float(success) <= attempt:
                raise ValueError("route-evidence successes are invalid")
            ledger._attempts.append(attempt)
            ledger._successes.append(float(success))
        prefix = unsigned.get("stable_prefix_minimum")
        protected = unsigned.get("protected")
        reversal_streak = unsigned.get("reversal_streak")
        reversal_count = unsigned.get("reversal_count")
        recovery_streak = unsigned.get("recovery_streak")
        if prefix is None:
            prefix = [
                1.0
                if attempt < ledger.min_mastery_observations
                else success / attempt
                for attempt, success in zip(ledger._attempts, ledger._successes)
            ]
        if protected is None:
            protected = [
                attempt >= ledger.min_mastery_observations
                and float(minimum) >= ledger.mastery_threshold
                for attempt, minimum in zip(ledger._attempts, prefix)
            ]
        if reversal_streak is None:
            reversal_streak = [0 for _ in ledger._attempts]
        if reversal_count is None:
            reversal_count = [0 for _ in ledger._attempts]
        if recovery_streak is None:
            recovery_streak = [0 for _ in ledger._attempts]
        if not isinstance(prefix, list) or not isinstance(protected, list):
            raise TypeError("route-evidence gate rows must be lists")
        if not isinstance(reversal_streak, list) or not isinstance(reversal_count, list):
            raise TypeError("route-evidence reversal rows must be lists")
        if not isinstance(recovery_streak, list):
            raise TypeError("route-evidence recovery rows must be a list")
        if any(
            len(rows) != len(ledger._attempts)
            for rows in (
                prefix,
                protected,
                reversal_streak,
                reversal_count,
                recovery_streak,
            )
        ):
            raise ValueError("route-evidence gate rows have different lengths")
        for minimum, is_protected, streak, count, recovery in zip(
            prefix,
            protected,
            reversal_streak,
            reversal_count,
            recovery_streak,
        ):
            if not isinstance(minimum, (int, float)) or not 0.0 <= float(minimum) <= 1.0:
                raise ValueError("route-evidence stable prefixes are invalid")
            if not isinstance(is_protected, bool):
                raise TypeError("route-evidence protected rows must be booleans")
            if not isinstance(streak, int) or streak < 0:
                raise ValueError("route-evidence reversal streaks are invalid")
            if not isinstance(count, int) or count < 0:
                raise ValueError("route-evidence reversal counts are invalid")
            if not isinstance(recovery, int) or recovery < 0:
                raise ValueError("route-evidence recovery streaks are invalid")
            ledger._stable_prefix_minimum.append(float(minimum))
            ledger._protected.append(is_protected)
            ledger._reversal_streak.append(streak)
            ledger._reversal_count.append(count)
            ledger._recovery_streak.append(recovery)
        preferred = unsigned.get("preferred_slot")
        if preferred is not None:
            ledger._validate_slot(int(preferred))
            ledger._preferred_slot = int(preferred)
        last_slot = unsigned.get("last_slot")
        if last_slot is not None:
            ledger._validate_slot(int(last_slot))
            ledger._last_slot = int(last_slot)
        last_outcome = unsigned.get("last_outcome")
        if last_outcome is not None:
            value = float(last_outcome)
            if not 0.0 <= value <= 1.0:
                raise ValueError("route-evidence last outcome is invalid")
            ledger._last_outcome = value
        ledger._version = int(unsigned.get("version", 0))
        if ledger._version < 0:
            raise ValueError("route-evidence version must be non-negative")
        return ledger


class PersistentOpaqueDepthEvidence:
    """Choose an external history depth from outcome-only evidence.

    Each append-only file owns an independent route-evidence ledger whose
    opaque slots are candidate query depths.  The policy exposes no preferred
    depth until one candidate passes the stable-prefix mastery gate.  Before
    that point, ``next_probe_query_count`` returns each untried candidate in
    ascending order and then fails closed with ``None``.  The policy is
    memory-side mutable state and has no trainable path into the controller.
    """

    schema = "neural-computer.persistent-opaque-depth-evidence.v1"

    def __init__(
        self,
        query_counts: Iterable[int],
        *,
        prior_strength: float = 1.0,
        mastery_threshold: float = 0.8,
        min_mastery_observations: int = 8,
        reversal_threshold: float = 0.5,
        reversal_patience: int = 4,
    ) -> None:
        counts = tuple(int(query_count) for query_count in query_counts)
        if not counts or any(query_count < 1 for query_count in counts):
            raise ValueError("depth-evidence query counts must be positive")
        if counts != tuple(sorted(set(counts))):
            raise ValueError("depth-evidence query counts must be sorted and unique")
        self.query_counts = counts
        self.prior_strength = float(prior_strength)
        self.mastery_threshold = float(mastery_threshold)
        self.min_mastery_observations = int(min_mastery_observations)
        self.reversal_threshold = float(reversal_threshold)
        self.reversal_patience = int(reversal_patience)
        self._ledgers: list[PersistentOpaqueRouteEvidence] = []
        self._version = 0

        # Reuse the route ledger's fail-closed parameter validation.
        PersistentOpaqueRouteEvidence(
            prior_strength=self.prior_strength,
            mastery_threshold=self.mastery_threshold,
            min_mastery_observations=self.min_mastery_observations,
            reversal_threshold=self.reversal_threshold,
            reversal_patience=self.reversal_patience,
        )

    @property
    def file_count(self) -> int:
        return len(self._ledgers)

    def _validate_file(self, file_slot: int) -> None:
        if not isinstance(file_slot, int) or not 0 <= file_slot < self.file_count:
            raise IndexError("depth-evidence file slot is outside the bank")

    def _validate_query_count(self, query_count: int) -> int:
        if query_count not in self.query_counts:
            raise ValueError("depth-evidence query count is outside the candidate set")
        return self.query_counts.index(query_count)

    def _ledger(self, file_slot: int) -> PersistentOpaqueRouteEvidence:
        self._validate_file(file_slot)
        return self._ledgers[file_slot]

    def _new_ledger(self) -> PersistentOpaqueRouteEvidence:
        ledger = PersistentOpaqueRouteEvidence(
            prior_strength=self.prior_strength,
            mastery_threshold=self.mastery_threshold,
            min_mastery_observations=self.min_mastery_observations,
            reversal_threshold=self.reversal_threshold,
            reversal_patience=self.reversal_patience,
        )
        for _ in self.query_counts:
            ledger.append_slot()
        return ledger

    def append_file(self) -> int:
        """Append one file with an independent depth ledger."""

        self._ledgers.append(self._new_ledger())
        self._version += 1
        return self.file_count - 1

    def preferred_query_count(self, file_slot: int) -> int | None:
        """Return a depth only after it passes the stable mastery gate."""

        preferred = self._ledger(file_slot).status().preferred_slot
        return None if preferred is None else self.query_counts[preferred]

    def probe_order(self, file_slot: int) -> tuple[int, ...]:
        """Return the persistent-first candidate order for one file."""

        order = self._ledger(file_slot).preferred_order()
        return tuple(self.query_counts[index] for index in order)

    def next_probe_query_count(self, file_slot: int) -> int | None:
        """Return the next untried depth, skipping reversed candidates.

        A candidate that has been demoted by patient failures is not retried
        immediately.  This prevents a stale preferred depth from trapping the
        policy in a reversal loop; the policy instead probes a fresh candidate
        and fails closed if no non-reversed candidate remains.
        """

        ledger = self._ledger(file_slot)
        status = ledger.status()
        if status.preferred_slot is not None:
            return self.query_counts[status.preferred_slot]
        order = ledger.preferred_order()
        for index in order:
            if status.reversal_count[index] == 0 and status.attempts[index] == 0:
                return self.query_counts[index]
        return None

    def observe(
        self,
        file_slot: int,
        query_count: int,
        outcome: float | torch.Tensor,
    ) -> None:
        """Record one attempted depth and its deterministic scalar outcome."""

        ledger = self._ledger(file_slot)
        ledger.observe(self._validate_query_count(query_count), outcome)
        self._version += 1

    def statuses(self) -> tuple[PersistentRouteEvidenceStatus, ...]:
        """Return immutable audit views for every file's depth ledger."""

        return tuple(ledger.status() for ledger in self._ledgers)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "query_counts": self.query_counts,
            "prior_strength": self.prior_strength,
            "mastery_threshold": self.mastery_threshold,
            "min_mastery_observations": self.min_mastery_observations,
            "reversal_threshold": self.reversal_threshold,
            "reversal_patience": self.reversal_patience,
            "file_count": self.file_count,
            "state": "checksummed_outcome_only_depth_evidence_v1",
        }

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            **self.configuration(),
            "query_counts": list(self.query_counts),
            "ledgers": [ledger.payload() for ledger in self._ledgers],
            "version": self._version,
        }
        payload["sha256"] = _payload_digest(payload)
        return payload

    def digest(self) -> str:
        """Return the checksum of this external depth policy."""

        return str(self.payload()["sha256"])

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> PersistentOpaqueDepthEvidence:
        """Restore a validated outcome-only depth policy snapshot."""

        unsigned = dict(payload)
        expected_digest = unsigned.pop("sha256", None)
        if expected_digest is not None and (
            not isinstance(expected_digest, str)
            or expected_digest != _payload_digest(unsigned)
        ):
            raise ValueError("depth-evidence checksum mismatch")
        if unsigned.get("schema") != cls.schema:
            raise ValueError("depth-evidence schema is incompatible")
        query_counts = unsigned.get("query_counts")
        ledgers = unsigned.get("ledgers")
        if not isinstance(query_counts, list) or not isinstance(ledgers, list):
            raise TypeError("depth-evidence payload rows must be lists")
        restored = cls(
            query_counts,
            prior_strength=float(unsigned["prior_strength"]),
            mastery_threshold=float(unsigned["mastery_threshold"]),
            min_mastery_observations=int(unsigned["min_mastery_observations"]),
            reversal_threshold=float(unsigned["reversal_threshold"]),
            reversal_patience=int(unsigned["reversal_patience"]),
        )
        restored._ledgers = [
            PersistentOpaqueRouteEvidence.from_payload(ledger)
            for ledger in ledgers
        ]
        for ledger in restored._ledgers:
            if ledger.slot_count != len(restored.query_counts):
                raise ValueError("depth-evidence ledger width does not match")
        restored._version = int(unsigned.get("version", 0))
        if restored._version < 0:
            raise ValueError("depth-evidence version must be non-negative")
        return restored


@dataclass
class _ContextRouteRecord:
    key: tuple[float, ...]
    evidence: PersistentOpaqueRouteEvidence


class PersistentOpaqueContextRouteEvidence:
    """Persist scalar route evidence indexed by learned opaque context keys.

    A context key is a learned event or controller-context vector, never a
    task identifier.  The table stores the key and one independent
    :class:`PersistentOpaqueRouteEvidence` ledger per matched context.  A
    candidate becomes preferred only through that ledger's stable-prefix gate;
    unknown contexts fall back to append order.  An opt-in generalization
    tolerance lets a protected nearest context provide a prior for a related
    key; observing that related key still creates an independent row, so the
    source memory cannot be overwritten.  The table is intentionally a
    replaceable memory-side policy and has no trainable path into the
    controller.
    """

    schema = "neural-computer.persistent-opaque-context-route-evidence.v1"

    def __init__(
        self,
        width: int,
        *,
        matching_tolerance: float = 1e-4,
        generalization_tolerance: float = 0.0,
        query_space_id: str = DEFAULT_ROUTE_QUERY_SPACE_ID,
        prior_strength: float = 1.0,
        mastery_threshold: float = 0.8,
        min_mastery_observations: int = 8,
        reversal_threshold: float = 0.5,
        reversal_patience: int = 4,
    ) -> None:
        if width < 1:
            raise ValueError("context-route width must be positive")
        if matching_tolerance < 0.0:
            raise ValueError("context-route matching tolerance must be non-negative")
        if (
            generalization_tolerance < 0.0
            or not math.isfinite(generalization_tolerance)
        ):
            raise ValueError(
                "context-route generalization tolerance must be finite and non-negative"
            )
        self.width = int(width)
        self.matching_tolerance = float(matching_tolerance)
        self.generalization_tolerance = float(generalization_tolerance)
        self.query_space_id = validate_representation_space_id(
            query_space_id,
            name="context-route query_space_id",
        )
        self.prior_strength = float(prior_strength)
        self.mastery_threshold = float(mastery_threshold)
        self.min_mastery_observations = int(min_mastery_observations)
        self.reversal_threshold = float(reversal_threshold)
        self.reversal_patience = int(reversal_patience)
        self._records: list[_ContextRouteRecord] = []
        self._slot_count = 0
        self._version = 0

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def context_count(self) -> int:
        return len(self._records)

    def configuration(self) -> dict[str, int | float | str]:
        """Return static ABI metadata without embedding mutable evidence."""

        return {
            "schema": self.schema,
            "width": self.width,
            "matching_tolerance": self.matching_tolerance,
            "generalization_tolerance": self.generalization_tolerance,
            "query_space_id": self.query_space_id,
            "prior_strength": self.prior_strength,
            "mastery_threshold": self.mastery_threshold,
            "min_mastery_observations": self.min_mastery_observations,
            "reversal_threshold": self.reversal_threshold,
            "reversal_patience": self.reversal_patience,
            "state": "checksummed_opaque_context_route_evidence_v1",
        }

    def append_slot(self) -> int:
        """Append one opaque route slot to every learned context row."""

        slot = self._slot_count
        self._slot_count += 1
        for record in self._records:
            record.evidence.append_slot()
        self._version += 1
        return slot

    def reset_slot(self, slot: int) -> None:
        """Clear one replaceable slot in every learned context ledger."""

        if not isinstance(slot, int) or not 0 <= slot < self.slot_count:
            raise IndexError("context-route slot is outside the bank")
        for record in self._records:
            record.evidence.reset_slot(slot)
        self._version += 1

    def _validate_context(self, context: torch.Tensor) -> torch.Tensor:
        if not isinstance(context, torch.Tensor):
            raise TypeError("context-route key must be a tensor")
        if context.ndim != 1 or context.shape[0] != self.width:
            raise ValueError(f"context-route key must have shape [{self.width}]")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("context-route key must contain only finite values")
        return F.normalize(
            context.detach().to(device="cpu", dtype=torch.float32), dim=0
        ).contiguous()

    def _nearest_record(
        self, context: torch.Tensor
    ) -> tuple[_ContextRouteRecord | None, float]:
        key = self._validate_context(context)
        if not self._records:
            return None, math.inf
        keys = torch.tensor([record.key for record in self._records], dtype=key.dtype)
        distances = torch.linalg.vector_norm(keys - key, dim=1)
        nearest = int(distances.argmin())
        return self._records[nearest], float(distances[nearest])

    def _find_record(
        self, context: torch.Tensor, *, create: bool
    ) -> _ContextRouteRecord | None:
        key = self._validate_context(context)
        record, distance = self._nearest_record(context)
        if record is not None and distance <= self.matching_tolerance:
            return record
        if not create:
            return None
        evidence = PersistentOpaqueRouteEvidence(
            prior_strength=self.prior_strength,
            mastery_threshold=self.mastery_threshold,
            min_mastery_observations=self.min_mastery_observations,
            reversal_threshold=self.reversal_threshold,
            reversal_patience=self.reversal_patience,
        )
        for _ in range(self._slot_count):
            evidence.append_slot()
        record = _ContextRouteRecord(
            key=tuple(float(value) for value in key.tolist()), evidence=evidence
        )
        self._records.append(record)
        self._version += 1
        return record

    def _nearest_preferred_record(
        self,
        context: torch.Tensor,
        *,
        exclude: _ContextRouteRecord | None = None,
    ) -> tuple[_ContextRouteRecord | None, float]:
        key = self._validate_context(context)
        candidates = [
            record
            for record in self._records
            if record is not exclude
            and record.evidence.status().preferred_slot is not None
        ]
        if not candidates:
            return None, math.inf
        keys = torch.tensor([record.key for record in candidates], dtype=key.dtype)
        distances = torch.linalg.vector_norm(keys - key, dim=1)
        nearest = int(distances.argmin())
        return candidates[nearest], float(distances[nearest])

    def preferred_order(self, context: torch.Tensor) -> tuple[int, ...]:
        """Return a safe exact/nearby learned order or append order."""

        if self.slot_count < 1:
            raise ValueError("context-route bank has no slots")
        record = self._find_record(context, create=False)
        if self.generalization_tolerance > self.matching_tolerance:
            allow_prior = record is None
            if record is not None:
                status = record.evidence.status()
                allow_prior = (
                    status.preferred_slot is None
                    and (
                        status.last_outcome is None
                        or status.last_outcome > self.reversal_threshold
                    )
                )
            nearest, distance = self._nearest_preferred_record(
                context,
                exclude=record,
            )
            if (
                allow_prior
                and
                nearest is not None
                and distance <= self.generalization_tolerance
            ):
                record = nearest
        if record is None:
            return tuple(range(self.slot_count))
        return record.evidence.preferred_order(slot_count=self.slot_count)

    def has_context(self, context: torch.Tensor) -> bool:
        """Return whether an opaque learned context has an evidence row."""

        return self._find_record(context, create=False) is not None

    def preferred_slots(self, contexts: torch.Tensor) -> torch.Tensor:
        """Return one preferred opaque slot for each context row."""

        if contexts.ndim != 2 or contexts.shape[1] != self.width:
            raise ValueError(
                f"contexts must have shape [batch, {self.width}]"
            )
        return torch.tensor(
            [self.preferred_order(context)[0] for context in contexts],
            dtype=torch.long,
            device=contexts.device,
        )

    def behavior_probabilities(
        self,
        contexts: torch.Tensor,
        *,
        exploration: float = 0.0,
        strategy: str = "stochastic",
    ) -> torch.Tensor:
        """Return exact exploitation/novelty probabilities for opaque routes.

        The exploitation mass follows the context's protected preferred slot.
        Exploration is weighted by ``1 / (1 + attempts)`` for each slot in that
        context's external ledger, so a newly appended file remains discoverable
        as the bank grows instead of receiving only ``epsilon / N`` forever.
        Unknown contexts have zero attempts for every slot and therefore explore
        uniformly.  This method returns behavior probabilities only; it does not
        mutate or create context rows.
        """

        if not isinstance(contexts, torch.Tensor) or contexts.ndim != 2:
            raise ValueError("context-route contexts must have shape [batch, width]")
        if contexts.shape[1] != self.width:
            raise ValueError(f"context-route contexts must have shape [batch, {self.width}]")
        if contexts.shape[0] < 1:
            raise ValueError("context-route contexts cannot be empty")
        if not 0.0 <= exploration <= 1.0 or not math.isfinite(float(exploration)):
            raise ValueError("context-route exploration must lie in [0, 1]")
        if strategy not in {"stochastic", "balanced"}:
            raise ValueError("context-route exploration strategy is unsupported")
        if self.slot_count < 1:
            raise ValueError("context-route bank has no slots")

        rows: list[torch.Tensor] = []
        for context in contexts:
            record = self._find_record(context, create=False)
            if self.generalization_tolerance > self.matching_tolerance:
                allow_prior = record is None
                if record is not None:
                    status = record.evidence.status()
                    allow_prior = (
                        status.preferred_slot is None
                        and (
                            status.last_outcome is None
                            or status.last_outcome > record.evidence.reversal_threshold
                        )
                    )
                nearest, distance = self._nearest_preferred_record(
                    context,
                    exclude=record,
                )
                if (
                    allow_prior
                    and nearest is not None
                    and distance <= self.generalization_tolerance
                ):
                    # Use the nearest protected context for behavior
                    # probabilities, just as preferred_order does. Outcome
                    # observation still creates an independent exact row for
                    # the held-out key, so generalization cannot overwrite the
                    # source route evidence.
                    record = nearest
            if record is None:
                attempts = torch.zeros(
                    self.slot_count,
                    dtype=torch.float32,
                    device=contexts.device,
                )
                preferred = 0
            else:
                status = record.evidence.status()
                attempts = torch.tensor(
                    status.attempts,
                    dtype=torch.float32,
                    device=contexts.device,
                )
                preferred = record.evidence.preferred_order(
                    slot_count=self.slot_count
                )[0]
                if strategy == "balanced" and status.preferred_slot is None:
                    recent_slot = status.last_slot
                    recent_outcome = status.last_outcome
                    if (
                        recent_slot is not None
                        and recent_outcome is not None
                        and recent_outcome >= record.evidence.mastery_threshold
                    ):
                        # Keep testing a promising replacement until its
                        # stable-prefix gate protects it. Otherwise a single
                        # successful probe can be discarded by a tie-break.
                        preferred = recent_slot
                    else:
                        least_attempts = attempts.amin()
                        least_attempted = attempts == least_attempts
                        if (
                            recent_slot is not None
                            and recent_outcome is not None
                            and recent_outcome <= record.evidence.reversal_threshold
                        ):
                            alternatives = least_attempted.clone()
                            alternatives[recent_slot] = False
                            if bool(alternatives.any()):
                                least_attempted = alternatives
                        rows.append(
                            least_attempted.to(dtype=torch.float32)
                            / least_attempted.sum().clamp_min(1.0)
                        )
                        continue
            if strategy == "balanced" and record is None:
                least_attempts = attempts.amin()
                least_attempted = attempts == least_attempts
                rows.append(
                    least_attempted.to(dtype=torch.float32)
                    / least_attempted.sum().clamp_min(1.0)
                )
                continue
            novelty = torch.reciprocal(1.0 + attempts)
            novelty = novelty / novelty.sum().clamp_min(1e-8)
            exploitation = torch.zeros(
                self.slot_count,
                dtype=novelty.dtype,
                device=contexts.device,
            )
            exploitation[preferred] = 1.0
            rows.append((1.0 - exploration) * exploitation + exploration * novelty)
        return torch.stack(rows, dim=0)

    def protected_slots(self) -> tuple[bool, ...]:
        """Return whether any learned context protects each physical slot."""

        protected = [False] * self.slot_count
        for record in self._records:
            for slot, is_protected in enumerate(record.evidence.status().protected):
                protected[slot] = protected[slot] or is_protected
        return tuple(protected)

    def observe(
        self,
        context: torch.Tensor,
        slot: int,
        outcome: float | torch.Tensor,
    ) -> None:
        """Record one attempted slot against one learned context key."""

        if not isinstance(slot, int) or not 0 <= slot < self.slot_count:
            raise IndexError("context-route slot is outside the bank")
        record = self._find_record(context, create=True)
        if record is None:
            raise RuntimeError("context-route record could not be created")
        record.evidence.observe(slot, outcome)
        self._version += 1

    def observe_batch(
        self,
        contexts: torch.Tensor,
        slots: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> None:
        """Record grouped scalar outcomes for a batch of opaque route attempts.

        Rows with the same learned context and slot are reduced to one scalar
        before they advance the corresponding route ledger.  This keeps
        reversal patience measured in rollout batches rather than raw
        per-trial verifier events, while retaining only deterministic scalar
        outcomes and opaque slot indices.
        """

        if not isinstance(contexts, torch.Tensor) or contexts.ndim != 2:
            raise ValueError("context-route batch keys must have shape [batch, width]")
        if contexts.shape[1] != self.width:
            raise ValueError(
                f"context-route batch keys must have shape [batch, {self.width}]"
            )
        if not isinstance(slots, torch.Tensor) or slots.ndim != 1:
            raise ValueError("context-route batch slots must have shape [batch]")
        if not isinstance(outcomes, torch.Tensor) or outcomes.ndim != 1:
            raise ValueError("context-route batch outcomes must have shape [batch]")
        if contexts.shape[0] != slots.shape[0] or contexts.shape[0] != outcomes.shape[0]:
            raise ValueError("context-route batch fields must have the same length")
        if slots.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("context-route batch slots must be integer tensors")
        if not bool(torch.isfinite(outcomes).all()) or not bool(
            ((outcomes >= 0.0) & (outcomes <= 1.0)).all()
        ):
            raise ValueError("context-route batch outcomes must lie in [0, 1]")
        if contexts.shape[0] == 0:
            return
        if not bool(((slots >= 0) & (slots < self.slot_count)).all()):
            raise IndexError("context-route batch slot is outside the bank")

        grouped: dict[tuple[int, int], list[float]] = {}
        records: dict[int, _ContextRouteRecord] = {}
        for context, slot_tensor, outcome in zip(contexts, slots, outcomes):
            record = self._find_record(context, create=True)
            if record is None:
                raise RuntimeError("context-route record could not be created")
            record_id = id(record)
            records[record_id] = record
            key = (record_id, int(slot_tensor.detach().cpu().item()))
            grouped.setdefault(key, []).append(float(outcome.detach().cpu().item()))
        for (record_id, slot), values in grouped.items():
            record = records[record_id]
            record.evidence.observe(slot, sum(values) / len(values))
        self._version += len(grouped)

    def payload(self) -> dict[str, object]:
        """Serialize context keys and opaque scalar route evidence."""

        payload: dict[str, object] = {
            "schema": self.schema,
            "width": self.width,
            "matching_tolerance": self.matching_tolerance,
            "generalization_tolerance": self.generalization_tolerance,
            "query_space_id": self.query_space_id,
            "prior_strength": self.prior_strength,
            "mastery_threshold": self.mastery_threshold,
            "min_mastery_observations": self.min_mastery_observations,
            "reversal_threshold": self.reversal_threshold,
            "reversal_patience": self.reversal_patience,
            "slot_count": self.slot_count,
            "version": self._version,
            "contexts": [
                {"key": list(record.key), "evidence": record.evidence.payload()}
                for record in self._records
            ],
        }
        payload["sha256"] = _payload_digest(payload)
        return payload

    def digest(self) -> str:
        """Return the checksum of the context-conditioned evidence state."""

        return str(self.payload()["sha256"])

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> PersistentOpaqueContextRouteEvidence:
        """Restore a validated context-conditioned route table."""

        unsigned = dict(payload)
        expected_digest = unsigned.pop("sha256", None)
        if expected_digest is not None and (
            not isinstance(expected_digest, str)
            or expected_digest != _payload_digest(unsigned)
        ):
            raise ValueError("context-route checksum mismatch")
        if unsigned.get("schema") != cls.schema:
            raise ValueError("context-route schema is incompatible")
        table = cls(
            int(unsigned["width"]),
            matching_tolerance=float(unsigned["matching_tolerance"]),
            generalization_tolerance=float(
                unsigned.get("generalization_tolerance", 0.0)
            ),
            query_space_id=str(
                unsigned.get("query_space_id", DEFAULT_ROUTE_QUERY_SPACE_ID)
            ),
            prior_strength=float(unsigned["prior_strength"]),
            mastery_threshold=float(unsigned["mastery_threshold"]),
            min_mastery_observations=int(unsigned["min_mastery_observations"]),
            reversal_threshold=float(unsigned.get("reversal_threshold", 0.5)),
            reversal_patience=int(unsigned.get("reversal_patience", 4)),
        )
        slot_count = unsigned["slot_count"]
        contexts = unsigned["contexts"]
        if not isinstance(slot_count, int) or slot_count < 0:
            raise ValueError("context-route slot count is invalid")
        if not isinstance(contexts, list):
            raise TypeError("context-route contexts must be a list")
        for _ in range(slot_count):
            table.append_slot()
        for item in contexts:
            if not isinstance(item, dict):
                raise TypeError("context-route row must be a dictionary")
            key = item.get("key")
            evidence_payload = item.get("evidence")
            if not isinstance(key, list) or not isinstance(evidence_payload, dict):
                raise TypeError("context-route row has invalid fields")
            key_tensor = torch.tensor(key, dtype=torch.float32)
            if key_tensor.ndim != 1 or key_tensor.shape[0] != table.width:
                raise ValueError("context-route serialized key has the wrong shape")
            if not bool(torch.isfinite(key_tensor).all()):
                raise ValueError("context-route serialized key is not finite")
            norm = torch.linalg.vector_norm(key_tensor)
            if float(norm) <= 1e-8:
                raise ValueError("context-route serialized key cannot be zero")
            # Payloads emitted by this class already contain normalized
            # float32 keys. Preserve those exact values so a save/load cycle
            # is byte-stable; normalize only legacy or externally authored
            # rows that are materially off the expected unit sphere.
            if not torch.isclose(norm, torch.ones_like(norm), atol=1e-5, rtol=1e-5):
                key_tensor = F.normalize(key_tensor, dim=0)
            key_tensor = key_tensor.contiguous()
            evidence = PersistentOpaqueRouteEvidence.from_payload(evidence_payload)
            if evidence.slot_count != table.slot_count:
                raise ValueError("context-route evidence has the wrong slot count")
            table._records.append(
                _ContextRouteRecord(
                    key=tuple(float(value) for value in key_tensor.tolist()),
                    evidence=evidence,
                )
            )
        version = unsigned.get("version", 0)
        if not isinstance(version, int) or version < 0:
            raise ValueError("context-route version is invalid")
        table._version = version
        return table


class OpaqueAddressRouter(nn.Module):
    """Permutation-equivariant scorer for a variable set of memory rows."""

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError("width and hidden must be positive")
        self.width = int(width)
        self.net = nn.Sequential(
            nn.Linear(width * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        """Return one route score per candidate row."""
        if query.ndim != 2 or query.shape[1] != self.width:
            raise ValueError("query must have shape [batch, width]")
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        if (
            keys.ndim != 3
            or keys.shape[0] != query.shape[0]
            or keys.shape[2] != self.width
            or keys.shape[1] < 1
        ):
            raise ValueError(
                "keys must have shape [batch, rows, width] or [rows, width]"
            )
        query_rows = query.unsqueeze(1).expand(-1, keys.shape[1], -1)
        pair = torch.cat(
            (query_rows, keys, (query_rows - keys).abs(), query_rows * keys),
            dim=-1,
        )
        return self.net(pair).squeeze(-1)


class FactorizedOpaqueAddressRouter(nn.Module):
    """Learned query/key addressing with a permutation-equivariant score.

    The query and each opaque memory key are encoded independently into a
    shared latent space, then matched by a scaled dot product.  This gives a
    memory-side learner a direct way to discover a reusable address relation
    from scalar attempted-row outcomes without assigning meaning to key
    coordinates or adding a candidate-specific reasoning branch.
    """

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError("width and hidden must be positive")
        self.width = int(width)
        self.hidden = int(hidden)
        self.query_encoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        """Return one learned compatibility score per candidate row."""
        if query.ndim != 2 or query.shape[1] != self.width:
            raise ValueError("query must have shape [batch, width]")
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        if (
            keys.ndim != 3
            or keys.shape[0] != query.shape[0]
            or keys.shape[2] != self.width
            or keys.shape[1] < 1
        ):
            raise ValueError(
                "keys must have shape [batch, rows, width] or [rows, width]"
            )
        query_latent = self.query_encoder(query)
        key_latent = self.key_encoder(keys)
        return torch.einsum("bh,brh->br", query_latent, key_latent) / self.hidden**0.5


class OpaqueCandidateGrowthRouter(nn.Module):
    """Shared zero-impact router for a variable bank of new candidates.

    Query and candidate keys are encoded independently, then a shared
    permutation-equivariant pair scorer returns one residual activation score
    per candidate.  The final score layer is zero-initialized, so adding this
    external router cannot preempt an established route before new outcomes
    are observed.  It replaces one scalar extension module per capability with
    one reusable candidate-conditioned growth boundary.
    """

    schema = "neural-computer.opaque-candidate-growth-router.v1"

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError("width and hidden must be positive")
        self.width = int(width)
        self.hidden = int(hidden)
        self.query_encoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Return one residual activation score per candidate key."""

        if query.ndim != 2 or query.shape[1] != self.width:
            raise ValueError("query must have shape [batch, width]")
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        if (
            keys.ndim != 3
            or keys.shape[0] != query.shape[0]
            or keys.shape[1] < 1
            or keys.shape[2] != self.width
        ):
            raise ValueError(
                "keys must have shape [batch, rows, width] or [rows, width]"
            )
        query_latent = self.query_encoder(query).unsqueeze(1)
        key_latent = self.key_encoder(keys)
        pair = torch.cat(
            (
                query_latent.expand_as(key_latent),
                key_latent,
                query_latent * key_latent,
                (query_latent - key_latent).abs(),
            ),
            dim=-1,
        )
        return self.score(pair).squeeze(-1)

    def configuration(self) -> dict[str, object]:
        """Return the replaceable scorer contract for external persistence."""

        return {
            "schema": self.schema,
            "width": self.width,
            "hidden": self.hidden,
            "behavior": "permutation_equivariant_pair_score_v1",
            "initialization": "zero_impact_final_score_v1",
        }


class OpaqueViewRouteExtension(nn.Module):
    """Memory-side score for one newly appended opaque executable view.

    The base router can be frozen while a new view is acquired.  This small
    external state maps the controller's learned query to one scalar score;
    callers use a neutral zero threshold to fall back to the frozen router
    until the extension has learned evidence for the new view.  Its identity
    remains in the artifact-memory manifest, not in a semantic coordinate or
    a controller branch.

    The final head is zero-initialized deliberately: constructing an extension
    cannot change routing behavior before any new-view outcomes are observed.
    """

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError("width and hidden must be positive")
        self.width = int(width)
        self.hidden = int(hidden)
        self.encoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.score = nn.Linear(hidden, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Return one scalar new-view score per controller query."""
        if query.ndim != 2 or query.shape[1] != self.width:
            raise ValueError("query must have shape [batch, width]")
        return self.score(self.encoder(query)).squeeze(-1)


def failure_gated_view_scores(
    old_scores: torch.Tensor,
    extension_scores: torch.Tensor,
    failed_old: torch.Tensor | bool,
) -> torch.Tensor:
    """Append a new-view score without preempting a route that has not failed.

    The returned matrix contains the frozen old scores followed by one new
    view.  A scalar verifier failure is the only event that may activate the
    extension.  This gives external capability growth a safe cold-start
    contract: existing behavior is preserved until an opaque old attempt has
    produced evidence that it was insufficient.
    """
    if old_scores.ndim != 2 or old_scores.shape[1] < 1:
        raise ValueError("old_scores must have shape [batch, at least one row]")
    if extension_scores.ndim != 1 or extension_scores.shape[0] != old_scores.shape[0]:
        raise ValueError("extension_scores must align with old_scores")
    if not bool(torch.isfinite(old_scores).all()) or not bool(
        torch.isfinite(extension_scores).all()
    ):
        raise ValueError("route scores must be finite")
    if isinstance(failed_old, bool):
        failure = torch.full_like(extension_scores, failed_old, dtype=torch.bool)
    else:
        if failed_old.shape != extension_scores.shape:
            raise ValueError("failed_old must align with route scores")
        failure = failed_old.to(dtype=torch.bool)
    old_best = old_scores.max(dim=-1).values
    new_score = torch.where(
        failure,
        old_best + extension_scores,
        old_best - torch.finfo(old_best.dtype).eps,
    )
    return torch.cat((old_scores, new_score.unsqueeze(1)), dim=1)


class OpaqueAppendOnlyRouteChain(nn.Module):
    """Compose append-only route extensions behind one frozen route bank.

    Each extension is an independently replaceable memory-side component.  A
    stage can only add its row after the caller supplies scalar failure
    evidence for that stage; with no failures, the original bank remains the
    argmax.  This keeps cold-start growth behavior-preserving while allowing
    an arbitrary number of sequential appends instead of a hand-wired
    two-extension experiment.
    """

    schema = "neural-computer.opaque-append-only-route-chain.v1"

    def __init__(
        self,
        base_router: nn.Module,
        *,
        width: int,
        extensions: Iterable[OpaqueViewRouteExtension] = (),
    ) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("route-chain width must be positive")
        self.width = int(width)
        self.base_router = base_router
        self.extensions = nn.ModuleList()
        for extension in extensions:
            self.append(extension)

    def append(self, extension: OpaqueViewRouteExtension) -> None:
        """Attach one new route extension without changing the base router."""

        if not isinstance(extension, OpaqueViewRouteExtension):
            raise TypeError("route-chain extensions must be opaque view routes")
        if extension.width != self.width:
            raise ValueError("route-chain extension width must match the chain")
        self.extensions.append(extension)

    def configuration(self) -> dict[str, object]:
        """Return the versioned route-chain contract."""

        return {
            "schema": self.schema,
            "width": self.width,
            "extension_count": len(self.extensions),
            "failure_signal": "cumulative_stage_scalar_verifier_failure_v1",
            "cold_start": "base_routes_preserved_until_failure",
        }

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        failed_stages: torch.Tensor | bool | None = None,
    ) -> torch.Tensor:
        """Return base rows plus all append rows for the current route state.

        ``failed_stages`` is either a scalar applied to every extension or a
        boolean tensor of shape ``[batch, extension_count]``.  A stage may
        compete only when it and every earlier stage have failed; otherwise
        it is forced below the best established route.
        """

        if query.ndim != 2 or query.shape[1] != self.width:
            raise ValueError("query must have shape [batch, route width]")
        batch_size = query.shape[0]
        extension_count = len(self.extensions)
        if failed_stages is None:
            failures = torch.zeros(
                batch_size,
                extension_count,
                dtype=torch.bool,
                device=query.device,
            )
        elif isinstance(failed_stages, bool):
            failures = torch.full(
                (batch_size, extension_count),
                failed_stages,
                dtype=torch.bool,
                device=query.device,
            )
        else:
            if failed_stages.shape != (batch_size, extension_count):
                raise ValueError(
                    "failed_stages must have shape [batch, extension_count]"
                )
            failures = failed_stages.to(device=query.device, dtype=torch.bool)

        scores = self.base_router(query, keys)
        for index, extension in enumerate(self.extensions):
            stage_failed = failures[:, : index + 1].all(dim=-1)
            scores = failure_gated_view_scores(
                scores,
                extension(query),
                stage_failed,
            )
        return scores


def failure_gated_candidate_scores(
    old_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
    failed_old: torch.Tensor | bool,
) -> torch.Tensor:
    """Append a shared candidate bank behind an opaque failure gate.

    Candidate residuals are added to the best established score only after
    the established route has failed.  Before failure every candidate is
    forced below the old bank, so a newly allocated growth router has no
    opportunity to change mastered behavior during cold start.
    """

    if old_scores.ndim != 2 or old_scores.shape[1] < 1:
        raise ValueError("old_scores must have shape [batch, at least one row]")
    if (
        candidate_scores.ndim != 2
        or candidate_scores.shape[0] != old_scores.shape[0]
        or candidate_scores.shape[1] < 1
    ):
        raise ValueError(
            "candidate_scores must have shape [batch, at least one candidate]"
        )
    if not bool(torch.isfinite(old_scores).all()) or not bool(
        torch.isfinite(candidate_scores).all()
    ):
        raise ValueError("route scores must be finite")
    if isinstance(failed_old, bool):
        failure = torch.full(
            (old_scores.shape[0],), failed_old, dtype=torch.bool, device=old_scores.device
        )
    else:
        if failed_old.shape != (old_scores.shape[0],):
            raise ValueError("failed_old must have shape [batch]")
        failure = failed_old.to(dtype=torch.bool)
    old_best = old_scores.max(dim=-1).values.unsqueeze(-1)
    gated = torch.where(
        failure.unsqueeze(-1),
        old_best + candidate_scores,
        old_best - torch.finfo(old_scores.dtype).eps,
    )
    return torch.cat((old_scores, gated), dim=-1)


def attempted_outcome_loss(
    logits: torch.Tensor,
    attempted: torch.Tensor,
    outcomes: torch.Tensor,
) -> torch.Tensor:
    """Train from only the attempted row and its scalar binary outcome."""
    if logits.ndim != 2 or attempted.ndim != 1 or outcomes.ndim != 1:
        raise ValueError("invalid router transition shapes")
    if logits.shape[0] != attempted.shape[0] or outcomes.shape != attempted.shape:
        raise ValueError("router transition batch lengths differ")
    if not bool(((attempted >= 0) & (attempted < logits.shape[1])).all()):
        raise ValueError("attempted row is out of range")
    if not bool(((outcomes == 0) | (outcomes == 1)).all()):
        raise ValueError("outcomes must be binary scalar rewards")
    selected = logits.gather(1, attempted[:, None]).squeeze(1)
    return F.binary_cross_entropy_with_logits(selected, outcomes)


def selector_distillation_loss(
    current_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Preserve old opaque route behavior during memory-side updates."""
    if current_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and current router shapes must match")
    return F.mse_loss(current_logits, teacher_logits.detach())
