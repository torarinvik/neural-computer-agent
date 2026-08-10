"""Persistent external alignment slots for replaceable goal frontends.

The controller and verifier remain frozen while this bank manages the external
compatibility layer between opaque frontend spaces and one verifier space.  It
is deliberately a memory-management boundary, not a semantic classifier:
callers provide opaque space IDs and held-out verifier pairs, while the bank
only admits, routes, quarantines, grows, and evicts alignment state.
"""

from __future__ import annotations

import hashlib
import math
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
from .world_model import (
    ExternalTransitionRouteMemory,
    ExternalTransitionRouteQueryProposal,
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
EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_IDENTITY_SCHEMA = (
    "neural-computer.external-goal-representation-alignment-bank-identity.v1"
)
EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_IDENTITY_QUARANTINE_SCHEMA = (
    "neural-computer.external-goal-representation-alignment-identity-quarantine.v1"
)
EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_IDENTITY_RESOLUTION_SCHEMA = (
    "neural-computer.external-goal-representation-alignment-identity-resolution.v1"
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


@dataclass(frozen=True)
class ExternalGoalRepresentationAlignmentRouteResult:
    """A safe signature route or an explicit refusal to guess."""

    selected_slot_id: int | None
    eligible_slot_ids: tuple[int, ...]
    scores: torch.Tensor
    margin: float | None
    aligned: torch.Tensor | None
    reason: str
    schema: str = EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_IDENTITY_SCHEMA

    def validate(self) -> ExternalGoalRepresentationAlignmentRouteResult:
        if self.schema != EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_IDENTITY_SCHEMA:
            raise ValueError("unsupported goal alignment identity schema")
        if self.scores.ndim != 1 or self.scores.shape[0] != len(self.eligible_slot_ids):
            raise ValueError("goal alignment identity scores are misaligned")
        if not bool(torch.isfinite(self.scores).all()):
            raise ValueError("goal alignment identity scores are not finite")
        if len(set(self.eligible_slot_ids)) != len(self.eligible_slot_ids):
            raise ValueError("goal alignment identity slot IDs are duplicated")
        if self.selected_slot_id is not None and self.selected_slot_id not in self.eligible_slot_ids:
            raise ValueError("goal alignment identity selected slot is ineligible")
        if self.margin is not None and (self.margin < 0.0 or not math.isfinite(self.margin)):
            raise ValueError("goal alignment identity margin is invalid")
        if self.selected_slot_id is None and self.aligned is not None:
            raise ValueError("refused goal alignment route cannot have output")
        if self.selected_slot_id is not None and self.aligned is None:
            raise ValueError("accepted goal alignment route must have output")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("goal alignment identity reason is missing")
        return self


@dataclass(frozen=True)
class ExternalGoalRepresentationIdentityQuarantineReceipt:
    """Auditable storage decision for one unresolved identity signature."""

    accepted: bool
    candidate_slot_ids: tuple[int, ...]
    quarantined_count: int
    capacity: int
    signature_digest: str
    reason: str
    schema: str = EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_IDENTITY_QUARANTINE_SCHEMA

    def validate(self) -> ExternalGoalRepresentationIdentityQuarantineReceipt:
        if self.schema != EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_IDENTITY_QUARANTINE_SCHEMA:
            raise ValueError("unsupported goal alignment identity-quarantine schema")
        if min(self.quarantined_count, self.capacity) < 0:
            raise ValueError("goal alignment identity-quarantine counts are invalid")
        if self.quarantined_count > self.capacity:
            raise ValueError("goal alignment identity quarantine exceeds capacity")
        if len(self.candidate_slot_ids) < 1 or len(set(self.candidate_slot_ids)) != len(
            self.candidate_slot_ids
        ):
            raise ValueError("goal alignment identity quarantine candidates are invalid")
        if any(
            not isinstance(slot_id, int) or isinstance(slot_id, bool) or slot_id < 0
            for slot_id in self.candidate_slot_ids
        ):
            raise ValueError("goal alignment identity quarantine slot ID is invalid")
        for name, value in (("signature_digest", self.signature_digest), ("reason", self.reason)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"goal alignment identity quarantine {name} is missing")
        return self


@dataclass(frozen=True)
class ExternalGoalRepresentationIdentityResolutionReceipt:
    """Auditable verifier-gated consumption of deferred identity evidence."""

    accepted: bool
    anchor_slot_id: int
    attempted_count: int
    resolved_count: int
    remaining_count: int
    verifier_accepted: bool
    source_digest: str
    destination_digest: str
    reason: str
    schema: str = EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_IDENTITY_RESOLUTION_SCHEMA

    def validate(self) -> ExternalGoalRepresentationIdentityResolutionReceipt:
        if self.schema != EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_IDENTITY_RESOLUTION_SCHEMA:
            raise ValueError("unsupported goal alignment identity-resolution schema")
        if self.anchor_slot_id < 0 or min(
            self.attempted_count, self.resolved_count, self.remaining_count
        ) < 0:
            raise ValueError("goal alignment identity resolution counts are invalid")
        if self.resolved_count > self.attempted_count:
            raise ValueError("goal alignment identity resolution over-consumed evidence")
        if not isinstance(self.verifier_accepted, bool):
            raise TypeError("goal alignment identity verifier decision is invalid")
        for name, value in (
            ("source_digest", self.source_digest),
            ("destination_digest", self.destination_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"goal alignment identity resolution {name} is missing")
        return self


def _signature_digest(signature: torch.Tensor) -> str:
    digest = hashlib.sha256()
    detached = signature.detach().cpu().contiguous()
    digest.update(str(detached.dtype).encode("utf-8"))
    digest.update(repr(tuple(detached.shape)).encode("utf-8"))
    digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


class ExternalGoalRepresentationAlignmentBank(nn.Module):
    """Bounded external bank of independently replaceable goal alignments.

    Frontend space IDs are admission-only metadata. At runtime, a generic
    learned-event signature proposes an opaque slot and the bank refuses to
    serve it when score or margin evidence is insufficient. Admission is
    authorized only by a held-out alignment gate. Failed or capacity-blocked
    candidates may be retained in quarantine without touching active adapters
    and can later be promoted after capacity is made available.
    """

    schema = EXTERNAL_GOAL_REPRESENTATION_ALIGNMENT_BANK_SCHEMA

    def __init__(
        self,
        target_width: int,
        *,
        capacity: int,
        quarantine_capacity: int = 2,
        identity_width: int | None = None,
        identity_min_score: float = 0.85,
        identity_min_margin: float = 0.05,
        identity_max_prototypes_per_slot: int = 4,
        identity_merge_cosine: float = 0.98,
        identity_quarantine_capacity: int = 0,
    ) -> None:
        super().__init__()
        if target_width < 1:
            raise ValueError("goal alignment bank target width must be positive")
        if capacity < 1:
            raise ValueError("goal alignment bank capacity must be positive")
        if quarantine_capacity < 0:
            raise ValueError("goal alignment bank quarantine capacity cannot be negative")
        if identity_width is not None and identity_width < 1:
            raise ValueError("goal alignment identity width must be positive")
        if not -1.0 <= identity_min_score <= 1.0 or not math.isfinite(identity_min_score):
            raise ValueError("goal alignment identity score floor is invalid")
        if identity_min_margin < 0.0 or not math.isfinite(identity_min_margin):
            raise ValueError("goal alignment identity margin floor is invalid")
        if identity_quarantine_capacity < 0:
            raise ValueError("goal alignment identity quarantine capacity cannot be negative")
        self.target_width = int(target_width)
        self.capacity = int(capacity)
        self.quarantine_capacity = int(quarantine_capacity)
        self.identity_width = None if identity_width is None else int(identity_width)
        self.identity_min_score = float(identity_min_score)
        self.identity_min_margin = float(identity_min_margin)
        self.identity_quarantine_capacity = int(identity_quarantine_capacity)
        self.adapters = nn.ModuleList()
        self._frontend_space_ids: list[str] = []
        self._slot_ids: list[int] = []
        self._next_slot_id = 0
        self._quarantine: list[dict[str, Any]] = []
        self._identity_quarantine: list[dict[str, Any]] = []
        self.identity_memory = (
            None
            if self.identity_width is None
            else ExternalTransitionRouteMemory(
                self.identity_width,
                max_prototypes_per_slot=identity_max_prototypes_per_slot,
                merge_cosine=identity_merge_cosine,
            )
        )

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

    @property
    def identity_enabled(self) -> bool:
        return self.identity_memory is not None

    @property
    def identity_quarantined_count(self) -> int:
        return len(self._identity_quarantine)

    def _validate_space_id(self, frontend_space_id: str) -> str:
        if not isinstance(frontend_space_id, str) or not frontend_space_id.strip():
            raise ValueError("goal alignment frontend space ID must be non-empty")
        return frontend_space_id.strip()

    def _validate_adapter(self, adapter: nn.Module) -> None:
        if not isinstance(adapter, _ALIGNMENT_TYPES):
            raise TypeError("goal alignment bank adapter type is unsupported")
        if adapter.target_width != self.target_width:
            raise ValueError("goal alignment adapter target space width differs")

    def _validate_identity_signature(self, signature: torch.Tensor | None) -> None:
        if self.identity_memory is None:
            if signature is not None:
                raise ValueError("identity signature supplied to a disabled identity bank")
            return
        if signature is None:
            raise ValueError("identity-enabled goal alignment admission needs a signature")
        if signature.ndim != 1 or signature.shape[0] != self.identity_width:
            raise ValueError("goal alignment identity signature has the wrong shape")
        if not bool(torch.isfinite(signature).all()):
            raise ValueError("goal alignment identity signature must be finite")
        if float(torch.linalg.vector_norm(signature)) <= 1e-12:
            raise ValueError("goal alignment identity signature must be non-zero")

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

    def _store_quarantine(
        self,
        space_id: str,
        adapter: nn.Module,
        reason: str,
        identity_signature: torch.Tensor | None,
    ) -> bool:
        if self.quarantine_capacity == 0 or len(self._quarantine) >= self.quarantine_capacity:
            return False
        if space_id in self.quarantined_space_ids:
            return False
        self._quarantine.append(
            {
                "frontend_space_id": space_id,
                "adapter": self._clone_adapter(adapter).state_payload(),
                "reason": reason,
                "identity_signature": (
                    None
                    if identity_signature is None
                    else identity_signature.detach().cpu().tolist()
                ),
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
        identity_signature: torch.Tensor | None = None,
    ) -> ExternalGoalRepresentationAlignmentBankAdmissionReceipt:
        """Stage and commit an adapter only after held-out verification."""

        space_id = self._validate_space_id(frontend_space_id)
        self._validate_adapter(adapter)
        self._validate_identity_signature(identity_signature)
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
            quarantined = self._store_quarantine(
                space_id,
                adapter,
                "held-out alignment rejected",
                identity_signature,
            )
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
            quarantined = self._store_quarantine(
                space_id,
                adapter,
                "active capacity is full",
                identity_signature,
            )
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
        if self.identity_memory is not None:
            self.identity_memory.register_slot(slot_id)
            self.identity_memory.observe(slot_id, identity_signature)
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
        identity_signature = (
            None
            if item.get("identity_signature") is None
            else torch.tensor(item["identity_signature"], dtype=torch.float32)
        )
        self._validate_identity_signature(identity_signature)
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
        if self.identity_memory is not None:
            self.identity_memory.register_slot(slot_id)
            self.identity_memory.observe(slot_id, identity_signature)
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
        if any(
            slot_id in item["candidate_slot_ids"]
            for item in self._identity_quarantine
        ):
            return ExternalGoalRepresentationAlignmentBankEvictionReceipt(
                False,
                slot_id,
                self._frontend_space_ids[index],
                self.active_count,
                self.active_count,
                self.active_digest(),
                self.active_digest(),
                "eviction blocked while deferred identity evidence references slot",
            ).validate()
        source_digest = self.active_digest()
        space_id = self._frontend_space_ids[index]
        candidate = self.from_payload(self.state_payload())
        del candidate.adapters[index]
        del candidate._frontend_space_ids[index]
        del candidate._slot_ids[index]
        if candidate.identity_memory is not None:
            candidate.identity_memory.unregister_slot(slot_id)
        if not bool(retention_probe(candidate)):
            return ExternalGoalRepresentationAlignmentBankEvictionReceipt(
                False, slot_id, space_id, self.active_count, self.active_count,
                source_digest, source_digest, "post-eviction retention probe failed"
            ).validate()
        self.adapters = candidate.adapters
        self._frontend_space_ids = candidate._frontend_space_ids
        self._slot_ids = candidate._slot_ids
        self.identity_memory = candidate.identity_memory
        destination_digest = self.active_digest()
        return ExternalGoalRepresentationAlignmentBankEvictionReceipt(
            True, slot_id, space_id, self.active_count + 1, self.active_count,
            source_digest, destination_digest,
            "stable-slot eviction passed retained-alignment probe"
        ).validate()

    def route_by_signature(
        self,
        signature: torch.Tensor,
        source: torch.Tensor,
        *,
        minimum_score: float | None = None,
        minimum_margin: float | None = None,
    ) -> ExternalGoalRepresentationAlignmentRouteResult:
        """Route without a frontend ID, refusing ambiguous signatures."""

        if self.identity_memory is None:
            raise RuntimeError("signature routing is disabled for this alignment bank")
        self._validate_identity_signature(signature)
        score_floor = (
            self.identity_min_score if minimum_score is None else float(minimum_score)
        )
        margin_floor = (
            self.identity_min_margin if minimum_margin is None else float(minimum_margin)
        )
        if not -1.0 <= score_floor <= 1.0 or not math.isfinite(score_floor):
            raise ValueError("goal alignment route score floor is invalid")
        if margin_floor < 0.0 or not math.isfinite(margin_floor):
            raise ValueError("goal alignment route margin floor is invalid")
        proposal: ExternalTransitionRouteQueryProposal = self.identity_memory.propose(
            signature,
            self.slot_ids,
            minimum_score=score_floor,
        )
        selected = proposal.selected_slot_id
        reason = proposal.reason
        if selected is not None and proposal.margin is not None and proposal.margin < margin_floor:
            selected = None
            reason = "signature route margin was below the ambiguity floor"
        aligned = None
        if selected is not None:
            index = self._slot_ids.index(selected)
            aligned = self.adapters[index](source)
            reason = "slot-local signature route passed score and margin floors"
        return ExternalGoalRepresentationAlignmentRouteResult(
            selected_slot_id=selected,
            eligible_slot_ids=proposal.eligible_slot_ids,
            scores=proposal.scores,
            margin=proposal.margin,
            aligned=aligned,
            reason=reason,
        ).validate()

    def route_slot(self, slot_id: int, source: torch.Tensor) -> torch.Tensor:
        """Apply a slot returned by ``route_by_signature``."""

        if not isinstance(slot_id, int) or isinstance(slot_id, bool) or slot_id < 0:
            raise ValueError("goal alignment route slot ID is invalid")
        try:
            index = self._slot_ids.index(slot_id)
        except ValueError as error:
            raise KeyError(f"unknown goal alignment slot ID: {slot_id}") from error
        return self.adapters[index](source)

    def observe_identity_verified(self, slot_id: int, signature: torch.Tensor) -> bool:
        """Update one slot's identity prototypes only after external verification."""

        if self.identity_memory is None:
            raise RuntimeError("signature routing is disabled for this alignment bank")
        self._validate_identity_signature(signature)
        if slot_id not in self._slot_ids:
            raise KeyError(f"unknown goal alignment slot ID: {slot_id}")
        return self.identity_memory.observe(slot_id, signature)

    def defer_identity_signature(
        self,
        signature: torch.Tensor,
        *,
        candidate_slot_ids: tuple[int, ...] | None = None,
    ) -> ExternalGoalRepresentationIdentityQuarantineReceipt:
        """Retain an ambiguous signature without changing active identity state."""

        if self.identity_memory is None:
            raise RuntimeError("identity quarantine requires signature routing")
        self._validate_identity_signature(signature)
        candidates = self.slot_ids if candidate_slot_ids is None else tuple(candidate_slot_ids)
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("identity quarantine candidate slots are invalid")
        if any(slot_id not in self._slot_ids for slot_id in candidates):
            raise KeyError("identity quarantine candidate slot is not active")
        signature_digest = _signature_digest(signature)
        if len(self._identity_quarantine) >= self.identity_quarantine_capacity:
            return ExternalGoalRepresentationIdentityQuarantineReceipt(
                accepted=False,
                candidate_slot_ids=candidates,
                quarantined_count=len(self._identity_quarantine),
                capacity=self.identity_quarantine_capacity,
                signature_digest=signature_digest,
                reason="identity quarantine capacity rejected unresolved signature",
            ).validate()
        self._identity_quarantine.append(
            {
                "signature": signature.detach().cpu().tolist(),
                "candidate_slot_ids": list(candidates),
            }
        )
        return ExternalGoalRepresentationIdentityQuarantineReceipt(
            accepted=True,
            candidate_slot_ids=candidates,
            quarantined_count=len(self._identity_quarantine),
            capacity=self.identity_quarantine_capacity,
            signature_digest=signature_digest,
            reason="ambiguous identity signature retained outside active prototypes",
        ).validate()

    def resolve_identity_quarantine(
        self,
        anchor_slot_id: int,
        *,
        verifier_accepted: bool,
    ) -> ExternalGoalRepresentationIdentityResolutionReceipt:
        """Resolve deferred signatures only after an external verifier accepts an anchor."""

        if self.identity_memory is None:
            raise RuntimeError("identity quarantine requires signature routing")
        if not isinstance(anchor_slot_id, int) or isinstance(anchor_slot_id, bool) or anchor_slot_id < 0:
            raise ValueError("identity resolution anchor slot ID is invalid")
        if anchor_slot_id not in self._slot_ids:
            raise KeyError(f"unknown identity resolution anchor slot: {anchor_slot_id}")
        if not isinstance(verifier_accepted, bool):
            raise TypeError("identity resolution verifier decision must be boolean")
        source_digest = self.digest()
        attempted = sum(
            anchor_slot_id in item["candidate_slot_ids"]
            for item in self._identity_quarantine
        )
        if not verifier_accepted:
            return ExternalGoalRepresentationIdentityResolutionReceipt(
                accepted=False,
                anchor_slot_id=anchor_slot_id,
                attempted_count=attempted,
                resolved_count=0,
                remaining_count=len(self._identity_quarantine),
                verifier_accepted=False,
                source_digest=source_digest,
                destination_digest=source_digest,
                reason="verifier rejected identity anchor; deferred signatures retained",
            ).validate()
        unresolved: list[dict[str, Any]] = []
        resolved = 0
        for item in self._identity_quarantine:
            if anchor_slot_id not in item["candidate_slot_ids"]:
                unresolved.append(item)
                continue
            signature = torch.tensor(item["signature"], dtype=torch.float32)
            if self.identity_memory.observe(anchor_slot_id, signature):
                resolved += 1
            else:
                unresolved.append(item)
        self._identity_quarantine = unresolved
        destination_digest = self.digest()
        return ExternalGoalRepresentationIdentityResolutionReceipt(
            accepted=resolved > 0,
            anchor_slot_id=anchor_slot_id,
            attempted_count=attempted,
            resolved_count=resolved,
            remaining_count=len(unresolved),
            verifier_accepted=True,
            source_digest=source_digest,
            destination_digest=destination_digest,
            reason=(
                "verifier-accepted anchor resolved deferred identity signatures"
                if resolved > 0
                else "no deferred identity signatures could be resolved for anchor"
            ),
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
        if self.identity_memory is not None:
            digest.update(self.identity_memory.digest().encode("utf-8"))
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
            digest.update(repr(item.get("identity_signature")).encode("utf-8"))
        for item in self._identity_quarantine:
            digest.update(repr(item["candidate_slot_ids"]).encode("utf-8"))
            digest.update(
                _signature_digest(torch.tensor(item["signature"])).encode("utf-8")
            )
        return digest.hexdigest()

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target_width": self.target_width,
            "capacity": self.capacity,
            "quarantine_capacity": self.quarantine_capacity,
            "identity_width": self.identity_width,
            "identity_min_score": self.identity_min_score,
            "identity_min_margin": self.identity_min_margin,
            "identity_quarantine_capacity": self.identity_quarantine_capacity,
            "identity_memory": (
                None
                if self.identity_memory is None
                else self.identity_memory.configuration()
            ),
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
            "identity_memory": (
                None
                if self.identity_memory is None
                else self.identity_memory.state_payload()
            ),
            "identity_quarantine": [
                {
                    "signature": item["signature"],
                    "candidate_slot_ids": item["candidate_slot_ids"],
                }
                for item in self._identity_quarantine
            ],
            "quarantine": [
                {
                    "frontend_space_id": item["frontend_space_id"],
                    "adapter": item["adapter"],
                    "reason": item["reason"],
                    "identity_signature": item.get("identity_signature"),
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
            identity_width=(
                None
                if configuration.get("identity_width") is None
                else int(configuration["identity_width"])
            ),
            identity_min_score=float(configuration.get("identity_min_score", 0.85)),
            identity_min_margin=float(configuration.get("identity_min_margin", 0.05)),
            identity_max_prototypes_per_slot=int(
                (configuration.get("identity_memory") or {}).get(
                    "max_prototypes_per_slot", 4
                )
            ),
            identity_merge_cosine=float(
                (configuration.get("identity_memory") or {}).get("merge_cosine", 0.98)
            ),
            identity_quarantine_capacity=int(
                configuration.get("identity_quarantine_capacity", 0)
            ),
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
        if (
            len(bank.adapters) > bank.capacity
            or len(set(bank._slot_ids)) != len(bank._slot_ids)
            or len(set(bank._frontend_space_ids)) != len(bank._frontend_space_ids)
        ):
            raise ValueError("goal alignment bank active slots are invalid")
        if bank.identity_memory is not None:
            identity_payload = payload.get("identity_memory")
            if not isinstance(identity_payload, Mapping):
                raise TypeError("goal alignment identity memory payload is missing")
            bank.identity_memory = ExternalTransitionRouteMemory.from_payload(
                identity_payload
            )
            if bank.identity_memory.slot_ids != bank.slot_ids:
                raise ValueError("goal alignment identity slots do not match active slots")
        bank._next_slot_id = int(configuration.get("next_slot_id", max(bank._slot_ids, default=-1) + 1))
        identity_quarantine = payload.get("identity_quarantine", [])
        if not isinstance(identity_quarantine, list) or len(identity_quarantine) > bank.identity_quarantine_capacity:
            raise ValueError("goal alignment identity quarantine payload is invalid")
        bank._identity_quarantine = []
        for item in identity_quarantine:
            if not isinstance(item, Mapping):
                raise TypeError("goal alignment identity quarantine item is invalid")
            signature = torch.tensor(item.get("signature"), dtype=torch.float32)
            bank._validate_identity_signature(signature)
            candidates = item.get("candidate_slot_ids")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("goal alignment identity quarantine candidates are invalid")
            normalized_candidates = tuple(int(slot_id) for slot_id in candidates)
            if len(set(normalized_candidates)) != len(normalized_candidates) or any(
                slot_id not in bank._slot_ids for slot_id in normalized_candidates
            ):
                raise ValueError("goal alignment identity quarantine references unknown slot")
            bank._identity_quarantine.append(
                {
                    "signature": signature.tolist(),
                    "candidate_slot_ids": list(normalized_candidates),
                }
            )
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
                    "identity_signature": item.get("identity_signature"),
                }
            )
            _adapter_from_payload(item["adapter"])
            if bank.identity_memory is not None:
                signature = item.get("identity_signature")
                if signature is None:
                    raise ValueError("identity-enabled quarantine item lacks signature")
                bank._validate_identity_signature(
                    torch.tensor(signature, dtype=torch.float32)
                )
        active_ids = set(bank._frontend_space_ids)
        quarantine_ids = set(bank.quarantined_space_ids)
        if active_ids & quarantine_ids:
            raise ValueError("goal alignment active and quarantined IDs overlap")
        if payload.get("sha256") != bank.digest():
            raise ValueError("goal alignment bank checksum mismatch")
        return bank
