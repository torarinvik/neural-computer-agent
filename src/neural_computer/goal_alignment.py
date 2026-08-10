"""Persistent external alignment slots for replaceable goal frontends.

The controller and verifier remain frozen while this bank manages the external
compatibility layer between opaque frontend spaces and one verifier space.  It
is deliberately a memory-management boundary, not a semantic classifier:
callers provide opaque space IDs and held-out verifier pairs, while the bank
only admits, routes, quarantines, grows, and evicts alignment state.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .online_transition import (
    ExternalGoalRepresentationAlignmentReceipt,
    ExternalGoalRepresentationAlignmentStatistics,
    ExternalGoalRepresentationRandomFeatureAlignmentStatistics,
)

EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_SCHEMA = (
    "neural-computer.external-goal-representation-alignment-bank.v1"
)
EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_ADMISSION_SCHEMA = (
    "neural-computer.external-goal-representation-alignment-bank-admission.v1"
)
EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_EVICTION_SCHEMA = (
    "neural-computer.external-goal-representation-alignment-bank-eviction.v1"
)
EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_GROWTH_SCHEMA = (
    "neural-computer.external-goal-representation-alignment-bank-growth.v1"
)

_ALIGNMENT_TYPES = (
    ExternalGoalRepresentationAlignmentStatistics,
    ExternalGoalRepresentationRandomFeatureAlignmentStatistics,
)


def _adapter_from_payload(payload: Mapping[str, Any]) -> nn.Module:
    schema = payload.get("schema")
    if schema == ExternalGoalRepresentationAlignmentStatistics.schema:
        return ExternalGoalRepresentationAlignmentStatistics.from_payload(payload)
    if schema == ExternalGoalRepresentationRandomFeatureAlignmentStatistics.schema:
        return ExternalGoalRepresentationRandomFeatureAlignmentStatistics.from_payload(
            payload
        )
    raise ValueError("unsupported goal alignment adapter schema")


def _digest_text(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ExternalGoalRepresentationAlignmentBankAdmissionReceipt:
    """Auditable result of an isolated adapter admission attempt."""

    accepted: bool
    frontend_space_id: str
    slot_id: int | None
    active_count: int
    capacity: int
    candidate_digest: str
    heldout: ExternalGoalRepresentationAlignmentReceipt
    quarantined: bool
    reason: str
    schema: str = EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_ADMISSION_SCHEMA

    def validate(self) -> ExternalGoalRepresentationAlignmentBankAdmissionReceipt:
        if self.schema != EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_ADMISSION_SCHEMA:
            raise ValueError("unsupported goal alignment bank admission schema")
        if not isinstance(self.frontend_space_id, str) or not self.frontend_space_id:
            raise ValueError("goal alignment admission space ID is missing")
        if self.slot_id is not None and (not isinstance(self.slot_id, int) or self.slot_id < 0):
            raise ValueError("goal alignment admission slot ID is invalid")
        if min(self.active_count, self.capacity) < 0 or self.active_count > self.capacity:
            raise ValueError("goal alignment admission capacity accounting is invalid")
        if not isinstance(self.candidate_digest, str) or not self.candidate_digest:
            raise ValueError("goal alignment admission candidate digest is missing")
        if not isinstance(self.quarantined, bool) or not isinstance(self.reason, str):
            raise TypeError("goal alignment admission receipt is malformed")
        self.heldout.validate()
        return self


@dataclass(frozen=True)
class ExternalGoalRepresentationAlignmentBankEvictionReceipt:
    """Auditable retention-gated logical-slot eviction."""

    accepted: bool
    evicted_slot_id: int
    evicted_frontend_space_id: str
    source_active_count: int
    destination_active_count: int
    source_digest: str
    destination_digest: str
    reason: str
    schema: str = EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_EVICTION_SCHEMA

    def validate(self) -> ExternalGoalRepresentationAlignmentBankEvictionReceipt:
        if self.schema != EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_EVICTION_SCHEMA:
            raise ValueError("unsupported goal alignment bank eviction schema")
        if self.evicted_slot_id < 0 or min(
            self.source_active_count, self.destination_active_count
        ) < 0:
            raise ValueError("goal alignment eviction accounting is invalid")
        if self.destination_active_count > self.source_active_count:
            raise ValueError("goal alignment eviction increased active count")
        for name, value in (
            ("evicted_frontend_space_id", self.evicted_frontend_space_id),
            ("source_digest", self.source_digest),
            ("destination_digest", self.destination_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"goal alignment eviction {name} is missing")
        return self


@dataclass(frozen=True)
class ExternalGoalRepresentationAlignmentBankGrowthReceipt:
    """Auditable capacity growth that leaves active adapters byte-stable."""

    accepted: bool
    source_capacity: int
    destination_capacity: int
    active_count: int
    content_digest_before: str
    content_digest_after: str
    reason: str
    schema: str = EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_GROWTH_SCHEMA

    def validate(self) -> ExternalGoalRepresentationAlignmentBankGrowthReceipt:
        if self.schema != EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_GROWTH_SCHEMA:
            raise ValueError("unsupported goal alignment bank growth schema")
        if min(self.source_capacity, self.destination_capacity, self.active_count) < 1:
            raise ValueError("goal alignment growth dimensions are invalid")
        if self.accepted and self.destination_capacity <= self.source_capacity:
            raise ValueError("accepted goal alignment growth did not grow")
        if self.active_count > self.source_capacity:
            raise ValueError("goal alignment growth active count exceeds source capacity")
        for name, value in (
            ("content_digest_before", self.content_digest_before),
            ("content_digest_after", self.content_digest_after),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"goal alignment growth {name} is missing")
        return self


class ExternalGoalRepresentationAlignmentBank(nn.Module):
    """Bounded external bank of independently replaceable goal alignments.

    A frontend space is never inferred from tensor similarity.  The opaque
    space ID selects an existing slot, while admission is authorized only by
    a held-out alignment gate.  Failed or capacity-blocked candidates may be
    retained in quarantine without touching active adapters and can later be
    promoted after capacity is made available.
    """

    schema = EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_SCHEMA

    def __init__(
        self,
        target_width: int,
        *,
        capacity: int,
        quarantine_capacity: int = 2,
    ) -> None:
        super().__init__()
        if target_width < 1:
            raise ValueError("goal alignment bank target width must be positive")
        if capacity < 1:
            raise ValueError("goal alignment bank capacity must be positive")
        if quarantine_capacity < 0:
            raise ValueError("goal alignment bank quarantine capacity cannot be negative")
        self.target_width = int(target_width)
        self.capacity = int(capacity)
        self.quarantine_capacity = int(quarantine_capacity)
        self.adapters = nn.ModuleList()
        self._frontend_space_ids: list[str] = []
        self._slot_ids: list[int] = []
        self._next_slot_id = 0
        self._quarantine: list[dict[str, Any]] = []

    @property
    def active_count(self) -> int:
        return len(self.adapters)

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(self._slot_ids)

    @property
    def frontend_space_ids(self) -> tuple[str, ...]:
        return tuple(self._frontend_space_ids)

    @property
    def quarantined_space_ids(self) -> tuple[str, ...]:
        return tuple(item["frontend_space_id"] for item in self._quarantine)

    def _validate_space_id(self, frontend_space_id: str) -> str:
        if not isinstance(frontend_space_id, str) or not frontend_space_id.strip():
            raise ValueError("goal alignment frontend space ID must be non-empty")
        return frontend_space_id.strip()

    def _validate_adapter(self, adapter: nn.Module) -> None:
        if not isinstance(adapter, _ALIGNMENT_TYPES):
            raise TypeError("goal alignment bank adapter type is unsupported")
        if adapter.target_width != self.target_width:
            raise ValueError("goal alignment adapter target space width differs")

    @staticmethod
    def _clone_adapter(adapter: nn.Module) -> nn.Module:
        restored = _adapter_from_payload(adapter.state_payload())
        restored.eval()
        return restored

    def adapter_for_space(self, frontend_space_id: str) -> nn.Module:
        space_id = self._validate_space_id(frontend_space_id)
        try:
            index = self._frontend_space_ids.index(space_id)
        except ValueError as error:
            raise KeyError(f"unknown goal frontend space: {space_id}") from error
        return self.adapters[index]

    def route(self, frontend_space_id: str, source: torch.Tensor) -> torch.Tensor:
        """Read one active alignment without mutating bank state."""

        return self.adapter_for_space(frontend_space_id)(source)

    def _store_quarantine(self, space_id: str, adapter: nn.Module, reason: str) -> bool:
        if self.quarantine_capacity == 0 or len(self._quarantine) >= self.quarantine_capacity:
            return False
        if space_id in self.quarantined_space_ids:
            return False
        self._quarantine.append(
            {
                "frontend_space_id": space_id,
                "adapter": self._clone_adapter(adapter).state_payload(),
                "reason": reason,
            }
        )
        return True

    def admit_verified(
        self,
        frontend_space_id: str,
        adapter: nn.Module,
        heldout_source: torch.Tensor,
        heldout_target: torch.Tensor,
        *,
        prediction_tolerance: float,
    ) -> ExternalGoalRepresentationAlignmentBankAdmissionReceipt:
        """Stage and commit an adapter only after held-out verification."""

        space_id = self._validate_space_id(frontend_space_id)
        self._validate_adapter(adapter)
        if space_id in self.frontend_space_ids:
            raise ValueError("goal frontend space is already active")
        if space_id in self.quarantined_space_ids:
            raise ValueError("goal frontend space is already quarantined")
        heldout = adapter.verify_heldout(
            heldout_source,
            heldout_target,
            prediction_tolerance=prediction_tolerance,
        )
        quarantined = False
        reason = heldout.reason
        if not heldout.accepted:
            quarantined = self._store_quarantine(space_id, adapter, "held-out alignment rejected")
            reason = (
                "held-out alignment rejected and candidate quarantined"
                if quarantined
                else "held-out alignment rejected and quarantine was full"
            )
            return ExternalGoalRepresentationAlignmentBankAdmissionReceipt(
                accepted=False,
                frontend_space_id=space_id,
                slot_id=None,
                active_count=self.active_count,
                capacity=self.capacity,
                candidate_digest=adapter.digest(),
                heldout=heldout,
                quarantined=quarantined,
                reason=reason,
            ).validate()
        if self.active_count >= self.capacity:
            quarantined = self._store_quarantine(space_id, adapter, "active capacity is full")
            reason = (
                "active capacity is full; candidate quarantined"
                if quarantined
                else "active capacity and quarantine are full"
            )
            return ExternalGoalRepresentationAlignmentBankAdmissionReceipt(
                accepted=False,
                frontend_space_id=space_id,
                slot_id=None,
                active_count=self.active_count,
                capacity=self.capacity,
                candidate_digest=adapter.digest(),
                heldout=heldout,
                quarantined=quarantined,
                reason=reason,
            ).validate()
        committed = self._clone_adapter(adapter)
        self.adapters.append(committed)
        self._frontend_space_ids.append(space_id)
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        self._slot_ids.append(slot_id)
        return ExternalGoalRepresentationAlignmentBankAdmissionReceipt(
            accepted=True,
            frontend_space_id=space_id,
            slot_id=slot_id,
            active_count=self.active_count,
            capacity=self.capacity,
            candidate_digest=committed.digest(),
            heldout=heldout,
            quarantined=False,
            reason="held-out alignment passed and active slot was committed",
        ).validate()

    def promote_quarantined_verified(
        self,
        frontend_space_id: str,
        heldout_source: torch.Tensor,
        heldout_target: torch.Tensor,
        *,
        prediction_tolerance: float,
    ) -> ExternalGoalRepresentationAlignmentBankAdmissionReceipt:
        """Retry one quarantined candidate without replaying old training rows."""

        space_id = self._validate_space_id(frontend_space_id)
        try:
            index = self.quarantined_space_ids.index(space_id)
        except ValueError as error:
            raise KeyError(f"unknown quarantined goal frontend space: {space_id}") from error
        item = self._quarantine[index]
        adapter = _adapter_from_payload(item["adapter"])
        heldout = adapter.verify_heldout(
            heldout_source,
            heldout_target,
            prediction_tolerance=prediction_tolerance,
        )
        if not heldout.accepted or self.active_count >= self.capacity:
            return ExternalGoalRepresentationAlignmentBankAdmissionReceipt(
                accepted=False,
                frontend_space_id=space_id,
                slot_id=None,
                active_count=self.active_count,
                capacity=self.capacity,
                candidate_digest=adapter.digest(),
                heldout=heldout,
                quarantined=True,
                reason=(
                    "quarantined candidate still fails held-out gate"
                    if not heldout.accepted
                    else "quarantined candidate is verified but active capacity is full"
                ),
            ).validate()
        committed = self._clone_adapter(adapter)
        del self._quarantine[index]
        self.adapters.append(committed)
        self._frontend_space_ids.append(space_id)
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        self._slot_ids.append(slot_id)
        return ExternalGoalRepresentationAlignmentBankAdmissionReceipt(
            accepted=True,
            frontend_space_id=space_id,
            slot_id=slot_id,
            active_count=self.active_count,
            capacity=self.capacity,
            candidate_digest=committed.digest(),
            heldout=heldout,
            quarantined=False,
            reason="quarantined alignment passed after capacity became available",
        ).validate()

    def grow_verified(
        self,
        destination_capacity: int,
        retention_probe: Callable[[ExternalGoalRepresentationAlignmentBank], bool],
    ) -> ExternalGoalRepresentationAlignmentBankGrowthReceipt:
        if not isinstance(destination_capacity, int) or destination_capacity <= self.capacity:
            raise ValueError("goal alignment destination capacity must grow")
        if not callable(retention_probe):
            raise TypeError("goal alignment growth retention probe is invalid")
        before = self.active_digest()
        source = self.capacity
        if not bool(retention_probe(self)):
            return ExternalGoalRepresentationAlignmentBankGrowthReceipt(
                False, source, source, self.active_count, before, before,
                "pre-growth retention probe failed"
            ).validate()
        self.capacity = destination_capacity
        after = self.active_digest()
        if after != before or not bool(retention_probe(self)):
            self.capacity = source
            return ExternalGoalRepresentationAlignmentBankGrowthReceipt(
                False, source, source, self.active_count, before, self.active_digest(),
                "post-growth retention or content-integrity probe failed"
            ).validate()
        return ExternalGoalRepresentationAlignmentBankGrowthReceipt(
            True, source, destination_capacity, self.active_count, before, after,
            "retention-verified alignment capacity growth committed"
        ).validate()

    def evict_verified(
        self,
        slot_id: int,
        retention_probe: Callable[[ExternalGoalRepresentationAlignmentBank], bool],
    ) -> ExternalGoalRepresentationAlignmentBankEvictionReceipt:
        """Evict any physical slot by stable logical ID after a proof."""

        if not isinstance(slot_id, int) or isinstance(slot_id, bool) or slot_id < 0:
            raise ValueError("goal alignment eviction slot ID is invalid")
        if not callable(retention_probe):
            raise TypeError("goal alignment eviction retention probe is invalid")
        try:
            index = self._slot_ids.index(slot_id)
        except ValueError as error:
            raise KeyError(f"unknown goal alignment slot ID: {slot_id}") from error
        source_digest = self.active_digest()
        space_id = self._frontend_space_ids[index]
        candidate = self.from_payload(self.state_payload())
        del candidate.adapters[index]
        del candidate._frontend_space_ids[index]
        del candidate._slot_ids[index]
        if not bool(retention_probe(candidate)):
            return ExternalGoalRepresentationAlignmentBankEvictionReceipt(
                False, slot_id, space_id, self.active_count, self.active_count,
                source_digest, source_digest, "post-eviction retention probe failed"
            ).validate()
        self.adapters = candidate.adapters
        self._frontend_space_ids = candidate._frontend_space_ids
        self._slot_ids = candidate._slot_ids
        destination_digest = self.active_digest()
        return ExternalGoalRepresentationAlignmentBankEvictionReceipt(
            True, slot_id, space_id, self.active_count + 1, self.active_count,
            source_digest, destination_digest,
            "stable-slot eviction passed retained-alignment probe"
        ).validate()

    def active_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        digest.update(str(self.target_width).encode("utf-8"))
        for space_id, slot_id, adapter in zip(
            self._frontend_space_ids, self._slot_ids, self.adapters, strict=True
        ):
            digest.update(space_id.encode("utf-8"))
            digest.update(str(slot_id).encode("utf-8"))
            digest.update(adapter.digest().encode("utf-8"))
        return digest.hexdigest()

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.active_digest().encode("utf-8"))
        digest.update(str(self.capacity).encode("utf-8"))
        digest.update(str(self.quarantine_capacity).encode("utf-8"))
        for item in self._quarantine:
            digest.update(str(item["frontend_space_id"]).encode("utf-8"))
            digest.update(_adapter_from_payload(item["adapter"]).digest().encode("utf-8"))
            digest.update(str(item["reason"]).encode("utf-8"))
        return digest.hexdigest()

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target_width": self.target_width,
            "capacity": self.capacity,
            "quarantine_capacity": self.quarantine_capacity,
            "frontend_space_ids": list(self._frontend_space_ids),
            "slot_ids": list(self._slot_ids),
            "next_slot_id": self._next_slot_id,
            "behavior": "opaque_frontend_slots_with_verified_lifecycle_v1",
        }

    def state_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "active": [
                {
                    "frontend_space_id": space_id,
                    "slot_id": slot_id,
                    "adapter": adapter.state_payload(),
                }
                for space_id, slot_id, adapter in zip(
                    self._frontend_space_ids, self._slot_ids, self.adapters, strict=True
                )
            ],
            "quarantine": [
                {
                    "frontend_space_id": item["frontend_space_id"],
                    "adapter": item["adapter"],
                    "reason": item["reason"],
                }
                for item in self._quarantine
            ],
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> ExternalGoalRepresentationAlignmentBank:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported goal alignment bank payload")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("goal alignment bank configuration is missing")
        bank = cls(
            int(configuration["target_width"]),
            capacity=int(configuration["capacity"]),
            quarantine_capacity=int(configuration.get("quarantine_capacity", 0)),
        )
        active = payload.get("active")
        if not isinstance(active, list):
            raise TypeError("goal alignment bank active state is invalid")
        for item in active:
            if not isinstance(item, Mapping):
                raise TypeError("goal alignment bank active slot is invalid")
            space_id = bank._validate_space_id(str(item["frontend_space_id"]))
            adapter = _adapter_from_payload(item["adapter"])
            bank._validate_adapter(adapter)
            bank.adapters.append(adapter)
            bank._frontend_space_ids.append(space_id)
            bank._slot_ids.append(int(item["slot_id"]))
        if len(bank.adapters) > bank.capacity or len(set(bank._slot_ids)) != len(bank._slot_ids):
            raise ValueError("goal alignment bank active slots are invalid")
        bank._next_slot_id = int(configuration.get("next_slot_id", max(bank._slot_ids, default=-1) + 1))
        bank._quarantine = []
        quarantine = payload.get("quarantine", [])
        if not isinstance(quarantine, list) or len(quarantine) > bank.quarantine_capacity:
            raise ValueError("goal alignment bank quarantine is invalid")
        for item in quarantine:
            if not isinstance(item, Mapping):
                raise TypeError("goal alignment bank quarantine item is invalid")
            bank._quarantine.append(
                {
                    "frontend_space_id": bank._validate_space_id(str(item["frontend_space_id"])),
                    "adapter": item["adapter"],
                    "reason": str(item["reason"]),
                }
            )
            _adapter_from_payload(item["adapter"])
        if payload.get("sha256") != bank.digest():
            raise ValueError("goal alignment bank checksum mismatch")
        return bank
