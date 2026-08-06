"""Canonical lifecycle coordination for external executable capabilities.

This module composes storage, retention, capacity planning, and verified
transactions without interpreting an artifact or adding controller logic.
The controller still emits learned keys and intentions; callers still own the
behavior verifier and the code that executes a promoted artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .artifact_memory import ArtifactConsolidationReceipt, ExecutableArtifactMemory
from .capacity import CapacityPlan, OpaqueCapacityPlanner
from .retention import (
    CapabilityRetentionLedger,
    CapabilityRetentionProbe,
    CapabilityRetentionStatus,
    RetentionPolicyConfig,
)

CAPABILITY_LIFECYCLE_SCHEMA = "neural-computer.external-capability-lifecycle.v1"
CAPABILITY_STAGING_SCHEMA = "neural-computer.external-capability-staging.v1"


@dataclass(frozen=True)
class CapabilityAdmissionReceipt:
    """Auditable result of one memory-side capability admission attempt."""

    accepted: bool
    action: str
    index: int | None
    source_capacity: int
    destination_capacity: int
    rows_before: int
    rows_after: int
    reason: str = ""


@dataclass(frozen=True)
class StagedCapabilityReceipt:
    """Result of one verifier update for a staged opaque capability."""

    accepted: bool
    pending: bool
    action: str
    index: int | None
    key_digest: str
    observations: int
    stable_prefix_minimum: float
    reason: str = ""


@dataclass
class _StagedCapability:
    key: torch.Tensor
    artifact: dict[str, torch.Tensor]
    strength: float
    evidence: CapabilityRetentionLedger


def _artifact_summary(
    artifact: Mapping[str, torch.Tensor], *, width: int
) -> torch.Tensor:
    """Create the fixed-width opaque value summary used by capacity policies."""

    if not artifact:
        raise ValueError("capability artifact must be nonempty")
    tensors: list[torch.Tensor] = []
    for name, value in sorted(artifact.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("capability artifacts must map names to tensors")
        if value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError("capability artifact tensors must be finite and nonempty")
        tensors.append(
            value.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
        )
    flat = torch.cat(tensors)
    positions = torch.linspace(0, flat.numel() - 1, width).round().long()
    return flat[positions]


class ExternalCapabilityLifecycle:
    """Coordinate safe admission and verified rewrites of executable artifacts.

    The lifecycle is deliberately not an executor. It only decides how a
    replaceable artifact bank changes shape and delegates all semantic
    evidence to caller-supplied verifier callbacks. A planner is optional:
    without one, the fallback policy admits into free capacity, evicts only an
    unprotected row, or grows when no safe eviction exists.
    """

    schema = CAPABILITY_LIFECYCLE_SCHEMA

    def __init__(
        self,
        memory: ExecutableArtifactMemory,
        *,
        planner: OpaqueCapacityPlanner | None = None,
    ) -> None:
        if not isinstance(memory, ExecutableArtifactMemory):
            raise TypeError("lifecycle memory must be ExecutableArtifactMemory")
        if planner is not None and planner.width != memory.width:
            raise ValueError("capacity planner width must match artifact memory")
        self.memory = memory
        self.planner = planner

    @property
    def width(self) -> int:
        return self.memory.width

    def configuration(self) -> dict[str, object]:
        """Return the versioned coordination contract without artifact data."""

        return {
            "schema": self.schema,
            "memory": {
                "width": self.memory.width,
                "capacity": self.memory.capacity,
            },
            "planner": None
            if self.planner is None
            else self.planner.configuration(),
            "execution": "caller_owned_verified_artifact_loader_v1",
        }

    def protection_mask(self) -> torch.Tensor:
        """Expose opaque row protection without exposing retention internals."""

        return self.memory.protection_mask()

    def _plan_fallback(self) -> CapacityPlan:
        occupied = self.memory.occupied
        if len(occupied) < self.memory.capacity:
            return CapacityPlan(
                action="admit",
                action_index=0,
                eviction_index=None,
                pair=None,
                score=torch.tensor(0.0),
            )
        protected = self.protection_mask()
        for index in occupied:
            if not bool(protected[index]):
                return CapacityPlan(
                    action="evict",
                    action_index=1,
                    eviction_index=index,
                    pair=None,
                    score=torch.tensor(0.0),
                )
        return CapacityPlan(
            action="grow",
            action_index=3,
            eviction_index=None,
            pair=None,
            score=torch.tensor(0.0),
        )

    def plan_admission(
        self,
        key: torch.Tensor,
        artifact: Mapping[str, torch.Tensor],
        *,
        consolidation_available: bool = False,
    ) -> CapacityPlan:
        """Plan one append from opaque key/value evidence and safety facts."""

        if key.shape != (self.width,):
            raise ValueError(f"capability key must have shape [{self.width}]")
        if not bool(torch.isfinite(key).all()):
            raise ValueError("capability key must be finite")
        incoming_value = _artifact_summary(artifact, width=self.width)
        if self.planner is None:
            return self._plan_fallback()
        bank = self.memory.planner_candidates()
        protected = self.protection_mask().unsqueeze(0)
        plan = self.planner.propose(
            bank,
            key.detach().cpu().reshape(1, -1),
            incoming_value.unsqueeze(0),
            protected,
            consolidation_available=torch.tensor(
                [consolidation_available], dtype=torch.bool
            ),
        )
        if not isinstance(plan, CapacityPlan):
            raise TypeError("capacity planner returned multiple plans for one bank")
        return plan

    def admit(
        self,
        key: torch.Tensor,
        artifact: Mapping[str, torch.Tensor],
        *,
        plan: CapacityPlan | None = None,
        grow_destination: Path | None = None,
        strength: float = 1.0,
    ) -> CapabilityAdmissionReceipt:
        """Execute an admission plan while preserving protected rows."""

        selected = plan or self.plan_admission(key, artifact)
        source_capacity = self.memory.capacity
        rows_before = len(self.memory.occupied)
        if selected.action == "consolidate":
            return CapabilityAdmissionReceipt(
                accepted=False,
                action=selected.action,
                index=None,
                source_capacity=source_capacity,
                destination_capacity=source_capacity,
                rows_before=rows_before,
                rows_after=rows_before,
                reason="consolidation requires an explicit verified rewrite transaction",
            )
        if selected.action == "grow":
            if grow_destination is None:
                return CapabilityAdmissionReceipt(
                    accepted=False,
                    action=selected.action,
                    index=None,
                    source_capacity=source_capacity,
                    destination_capacity=source_capacity,
                    rows_before=rows_before,
                    rows_after=rows_before,
                    reason="growth requires a new destination path",
                )
            destination_capacity = max(source_capacity + 1, rows_before + 1)
            grown = self.memory.grow(grow_destination, destination_capacity)
            index = grown.put(key, artifact, strength=strength)
            grown.validate()
            self.memory = grown
            return CapabilityAdmissionReceipt(
                accepted=True,
                action=selected.action,
                index=index,
                source_capacity=source_capacity,
                destination_capacity=grown.capacity,
                rows_before=rows_before,
                rows_after=len(grown.occupied),
                reason="grown store validated and admitted",
            )
        if selected.action == "evict":
            index = selected.eviction_index
            if index is None or index not in self.memory.occupied:
                return CapabilityAdmissionReceipt(
                    accepted=False,
                    action=selected.action,
                    index=None,
                    source_capacity=source_capacity,
                    destination_capacity=source_capacity,
                    rows_before=rows_before,
                    rows_after=rows_before,
                    reason="eviction plan did not identify an occupied row",
                )
            if bool(self.protection_mask()[index]):
                return CapabilityAdmissionReceipt(
                    accepted=False,
                    action=selected.action,
                    index=index,
                    source_capacity=source_capacity,
                    destination_capacity=source_capacity,
                    rows_before=rows_before,
                    rows_after=rows_before,
                    reason="eviction plan targeted a protected capability",
                )
            scores = torch.zeros(self.memory.capacity)
            scores[index] = 1.0
            committed = self.memory.put(
                key,
                artifact,
                strength=strength,
                eviction_scores=scores,
            )
            self.memory.validate()
            return CapabilityAdmissionReceipt(
                accepted=True,
                action=selected.action,
                index=committed,
                source_capacity=source_capacity,
                destination_capacity=source_capacity,
                rows_before=rows_before,
                rows_after=len(self.memory.occupied),
                reason="unprotected row evicted and replacement validated",
            )
        if selected.action != "admit":
            raise ValueError(f"unsupported admission action {selected.action!r}")
        try:
            index = self.memory.put(key, artifact, strength=strength)
        except (MemoryError, RuntimeError) as error:
            return CapabilityAdmissionReceipt(
                accepted=False,
                action=selected.action,
                index=None,
                source_capacity=source_capacity,
                destination_capacity=source_capacity,
                rows_before=rows_before,
                rows_after=len(self.memory.occupied),
                reason=str(error),
            )
        self.memory.validate()
        return CapabilityAdmissionReceipt(
            accepted=True,
            action=selected.action,
            index=index,
            source_capacity=source_capacity,
            destination_capacity=source_capacity,
            rows_before=rows_before,
            rows_after=len(self.memory.occupied),
            reason="artifact admitted into available capacity",
        )

    def consolidate(
        self,
        source_indices: Sequence[int],
        replacement_key: torch.Tensor,
        replacement_artifact: Mapping[str, torch.Tensor],
        destination: Path,
        *,
        verifier: Callable[[ExecutableArtifactMemory], bool],
        strength: float = 1.0,
        replacement_aliases: Sequence[torch.Tensor] = (),
        replacement_alias_views: Sequence[str | None] = (),
        candidate_outcomes: (
            Sequence[float]
            | torch.Tensor
            | Sequence[CapabilityRetentionProbe]
            | None
        ) = None,
        candidate_outcome_probe: Callable[[ExecutableArtifactMemory], object]
        | None = None,
        retained_scores: Sequence[float] | torch.Tensor | None = None,
        candidate_threshold: float = 0.8,
        retention_floor: float = 0.8,
        min_candidate_observations: int = 8,
    ) -> ArtifactConsolidationReceipt:
        """Run one immutable verified rewrite and adopt it only on success."""

        candidate, receipt = self.memory.consolidate_verified(
            source_indices,
            replacement_key,
            replacement_artifact,
            destination,
            verifier=verifier,
            strength=strength,
            replacement_aliases=replacement_aliases,
            replacement_alias_views=replacement_alias_views,
            candidate_outcomes=candidate_outcomes,
            candidate_outcome_probe=candidate_outcome_probe,
            retained_scores=retained_scores,
            candidate_threshold=candidate_threshold,
            retention_floor=retention_floor,
            min_candidate_observations=min_candidate_observations,
        )
        if receipt.accepted and candidate is not None:
            candidate.validate()
            self.memory = candidate
        return receipt


class ConfidenceAwareCapabilityStaging:
    """Stage opaque growth until verifier evidence earns executable admission.

    A staged artifact is external mutable state and is deliberately absent from
    the executable bank.  The controller remains frozen while a caller records
    deterministic scalar verifier outcomes.  Once the candidate's stable
    prefix clears the configured threshold, this coordinator delegates the
    normal protected admission transaction and transfers the accumulated
    evidence directly into the destination ledger.  No old episode or raw
    trajectory is replayed.

    Staging is intentionally an in-process queue in this first contract.  A
    durable artifact spool can replace it later without changing the evidence
    or lifecycle interfaces.
    """

    schema = CAPABILITY_STAGING_SCHEMA

    def __init__(
        self,
        lifecycle: ExternalCapabilityLifecycle,
        *,
        candidate_threshold: float | None = None,
        min_candidate_observations: int | None = None,
        reversal_threshold: float | None = None,
        reversal_patience: int | None = None,
        recent_window: int | None = None,
    ) -> None:
        if not isinstance(lifecycle, ExternalCapabilityLifecycle):
            raise TypeError("staging requires an external capability lifecycle")
        destination_policy = lifecycle.memory.retention.config
        resolved_threshold = (
            destination_policy.mastery_threshold
            if candidate_threshold is None
            else float(candidate_threshold)
        )
        resolved_observations = (
            destination_policy.min_mastery_observations
            if min_candidate_observations is None
            else int(min_candidate_observations)
        )
        resolved_reversal_threshold = (
            destination_policy.reversal_threshold
            if reversal_threshold is None
            else float(reversal_threshold)
        )
        resolved_reversal_patience = (
            destination_policy.reversal_patience
            if reversal_patience is None
            else int(reversal_patience)
        )
        resolved_recent_window = (
            destination_policy.recent_window
            if recent_window is None
            else int(recent_window)
        )
        if not 0.0 <= resolved_threshold <= 1.0:
            raise ValueError("candidate threshold must lie in [0, 1]")
        if resolved_observations < 1:
            raise ValueError("minimum candidate observations must be positive")
        self.lifecycle = lifecycle
        self.candidate_threshold = resolved_threshold
        self.min_candidate_observations = resolved_observations
        self._policy = RetentionPolicyConfig(
            mastery_threshold=self.candidate_threshold,
            min_mastery_observations=self.min_candidate_observations,
            reversal_threshold=resolved_reversal_threshold,
            reversal_patience=resolved_reversal_patience,
            recent_window=resolved_recent_window,
        ).validate()
        if self._policy.as_dict() != destination_policy.as_dict():
            raise ValueError(
                "staging policy must match the destination retention policy"
            )
        self._staged: dict[str, _StagedCapability] = {}

    @property
    def pending_count(self) -> int:
        """Return the number of candidates not yet admitted or discarded."""

        return len(self._staged)

    def configuration(self) -> dict[str, object]:
        """Return the versioned staging contract without artifact contents."""

        return {
            "schema": self.schema,
            "width": self.lifecycle.width,
            "candidate_threshold": self.candidate_threshold,
            "min_candidate_observations": self.min_candidate_observations,
            "policy": self._policy.as_dict(),
            "storage": "external_in_process_staging_v1",
        }

    def _validate_key(self, key: torch.Tensor) -> torch.Tensor:
        if not isinstance(key, torch.Tensor):
            raise TypeError("staged capability key must be a tensor")
        if key.shape != (self.lifecycle.width,):
            raise ValueError(
                f"staged capability key must have shape [{self.lifecycle.width}]"
            )
        if not bool(torch.isfinite(key).all()):
            raise ValueError("staged capability key must be finite")
        return key.detach().to(device="cpu", dtype=torch.float32).clone()

    def _digest(self, key: torch.Tensor) -> str:
        ledger = CapabilityRetentionLedger(
            self.lifecycle.width,
            config=self._policy,
        )
        return ledger.status(key).key_digest

    def stage(
        self,
        key: torch.Tensor,
        artifact: Mapping[str, torch.Tensor],
        *,
        strength: float = 1.0,
    ) -> CapabilityRetentionStatus:
        """Place one unverified opaque artifact outside the executable bank."""

        normalized_key = self._validate_key(key)
        _artifact_summary(artifact, width=self.lifecycle.width)
        if not 0.0 < strength <= 1.0:
            raise ValueError("staged capability strength must lie in (0, 1]")
        digest = self._digest(normalized_key)
        if digest in self._staged:
            raise ValueError("staged capability key already exists")
        evidence = CapabilityRetentionLedger(
            self.lifecycle.width,
            config=self._policy,
        )
        status = evidence.status(normalized_key)
        self._staged[digest] = _StagedCapability(
            key=normalized_key,
            artifact={
                name: value.detach().to(device="cpu").clone()
                for name, value in artifact.items()
            },
            strength=float(strength),
            evidence=evidence,
        )
        return status

    def status(self, key: torch.Tensor) -> CapabilityRetentionStatus:
        """Return the current evidence for one pending candidate."""

        normalized_key = self._validate_key(key)
        digest = self._digest(normalized_key)
        candidate = self._staged.get(digest)
        if candidate is None:
            raise KeyError("capability key is not staged")
        return candidate.evidence.status(normalized_key)

    def pending_statuses(self) -> tuple[CapabilityRetentionStatus, ...]:
        """Return pending evidence in deterministic opaque-key order."""

        return tuple(
            candidate.evidence.status(candidate.key)
            for _digest, candidate in sorted(self._staged.items())
        )

    def observe(
        self,
        key: torch.Tensor,
        outcome: float | torch.Tensor,
        *,
        plan: CapacityPlan | None = None,
        grow_destination: Path | None = None,
    ) -> StagedCapabilityReceipt:
        """Record one outcome and admit only after stable candidate mastery."""

        normalized_key = self._validate_key(key)
        digest = self._digest(normalized_key)
        candidate = self._staged.get(digest)
        if candidate is None:
            raise KeyError("capability key is not staged")
        status = candidate.evidence.observe(normalized_key, outcome)
        if not status.protected:
            return StagedCapabilityReceipt(
                accepted=False,
                pending=True,
                action="stage",
                index=None,
                key_digest=status.key_digest,
                observations=status.observations,
                stable_prefix_minimum=status.stable_prefix_minimum,
                reason="candidate remains staged until stable mastery is verified",
            )

        admission = self.lifecycle.admit(
            normalized_key,
            candidate.artifact,
            plan=plan,
            grow_destination=grow_destination,
            strength=candidate.strength,
        )
        if not admission.accepted:
            return StagedCapabilityReceipt(
                accepted=False,
                pending=True,
                action=admission.action,
                index=admission.index,
                key_digest=status.key_digest,
                observations=status.observations,
                stable_prefix_minimum=status.stable_prefix_minimum,
                reason=f"stable candidate remains staged: {admission.reason}",
            )
        self.lifecycle.memory.retention.adopt(candidate.evidence, normalized_key)
        self.lifecycle.memory.save()
        self._staged.pop(digest)
        adopted = self.lifecycle.memory.retention.status(normalized_key)
        return StagedCapabilityReceipt(
            accepted=True,
            pending=False,
            action=admission.action,
            index=admission.index,
            key_digest=adopted.key_digest,
            observations=adopted.observations,
            stable_prefix_minimum=adopted.stable_prefix_minimum,
            reason="stable candidate admitted with transferred evidence",
        )

    def discard(self, key: torch.Tensor) -> bool:
        """Discard one pending candidate without touching executable memory."""

        normalized_key = self._validate_key(key)
        return self._staged.pop(self._digest(normalized_key), None) is not None


__all__ = [
    "CAPABILITY_LIFECYCLE_SCHEMA",
    "CAPABILITY_STAGING_SCHEMA",
    "CapabilityAdmissionReceipt",
    "ConfidenceAwareCapabilityStaging",
    "ExternalCapabilityLifecycle",
    "StagedCapabilityReceipt",
]
