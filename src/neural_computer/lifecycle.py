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
from .retention import CapabilityRetentionProbe

CAPABILITY_LIFECYCLE_SCHEMA = "neural-computer.external-capability-lifecycle.v1"


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


__all__ = [
    "CAPABILITY_LIFECYCLE_SCHEMA",
    "CapabilityAdmissionReceipt",
    "ExternalCapabilityLifecycle",
]
