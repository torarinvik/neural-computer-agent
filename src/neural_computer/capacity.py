"""Opaque planning for external-memory admission under capacity pressure.

The planner is deliberately a memory-side component.  It consumes learned
keys/values and generic storage metadata, not task labels, raw modalities, or
device protocols.  Its output is only a proposal.  Protection masks and the
behavior verifier remain authoritative at the transaction boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .memory import MemoryCandidates

CAPACITY_PLANNER_SCHEMA = "neural-computer.opaque-capacity-planner.v1"
ADMISSION_ACTIONS = ("admit", "evict", "consolidate", "grow")


@dataclass(frozen=True)
class CapacityPlannerOutput:
    """Permutation-equivariant scores for one admission decision."""

    action_logits: torch.Tensor
    eviction_scores: torch.Tensor
    pair_scores: torch.Tensor
    valid_evictions: torch.Tensor
    valid_pairs: torch.Tensor
    available_actions: torch.Tensor


@dataclass(frozen=True)
class CapacityPlan:
    """One opaque admission proposal for an external-memory transaction."""

    action: str
    action_index: int
    eviction_index: int | None
    pair: tuple[int, int] | None
    score: torch.Tensor


class OpaqueCapacityPlanner(nn.Module):
    """Learn a generic admission action without physical-slot semantics.

    The planner scores the action set ``admit``, ``evict``, ``consolidate``,
    and ``grow`` from an incoming learned key/value plus an unordered bank of
    learned rows.  Candidate rows and pair scores are equivariant to a joint
    permutation of the bank.  The caller supplies ``protected`` and whether
    verified consolidation is currently available; those are safety facts,
    not learned substitutes for retention or behavior verification.
    """

    schema = CAPACITY_PLANNER_SCHEMA

    def __init__(
        self,
        *,
        width: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(width, hidden) < 1:
            raise ValueError("capacity planner widths must be positive")
        self.width = int(width)
        self.hidden = int(hidden)
        row_width = 2 * self.width + 2
        incoming_width = 2 * self.width
        self.row_width = row_width
        self.incoming_width = incoming_width
        global_width = incoming_width + row_width + 5
        self.global_width = global_width
        self.action_network = nn.Sequential(
            nn.Linear(global_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(ADMISSION_ACTIONS)),
        )
        self.eviction_network = nn.Sequential(
            nn.Linear(global_width + row_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        pair_width = 2 * row_width + incoming_width
        self.pair_network = nn.Sequential(
            nn.Linear(pair_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def configuration(self) -> dict[str, int | str | tuple[str, ...]]:
        return {
            "schema": self.schema,
            "width": self.width,
            "hidden": self.hidden,
            "actions": ADMISSION_ACTIONS,
            "row_features": "key_value_strength_relative_age",
        }

    def _validate_inputs(
        self,
        bank: MemoryCandidates,
        incoming_key: torch.Tensor,
        incoming_value: torch.Tensor,
        protected: torch.Tensor,
        *,
        consolidation_available: torch.Tensor,
    ) -> None:
        bank.validate(width=self.width, capacity=bank.keys.shape[1])
        batch, capacity, _ = bank.keys.shape
        if incoming_key.shape != (batch, self.width):
            raise ValueError(
                f"incoming_key must have shape [{batch}, {self.width}]"
            )
        if incoming_value.shape != (batch, self.width):
            raise ValueError(
                f"incoming_value must have shape [{batch}, {self.width}]"
            )
        if protected.shape != (batch, capacity) or protected.dtype != torch.bool:
            raise ValueError(
                f"protected must be bool [{batch}, {capacity}]"
            )
        if consolidation_available.shape != (batch,) or consolidation_available.dtype != torch.bool:
            raise ValueError("consolidation_available must be bool [batch]")
        tensors = (incoming_key, incoming_value)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("incoming planner tensors must be finite")

    def _row_features(self, bank: MemoryCandidates) -> torch.Tensor:
        timestamps = bank.timestamps
        occupied = bank.occupied
        occupied_timestamps = timestamps.masked_fill(~occupied, 0.0)
        latest = occupied_timestamps.amax(dim=-1, keepdim=True)
        earliest = torch.where(occupied, timestamps, latest).amin(
            dim=-1, keepdim=True
        )
        relative_age = (latest - timestamps) / (latest - earliest).clamp_min(1.0)
        return torch.cat(
            (
                bank.keys,
                bank.values,
                bank.strengths.unsqueeze(-1),
                relative_age.unsqueeze(-1),
            ),
            dim=-1,
        )

    def forward(
        self,
        bank: MemoryCandidates,
        incoming_key: torch.Tensor,
        incoming_value: torch.Tensor,
        protected: torch.Tensor,
        *,
        consolidation_available: torch.Tensor | None = None,
    ) -> CapacityPlannerOutput:
        """Score an admission state without assigning meaning to rows."""

        batch, capacity, _ = bank.keys.shape
        if consolidation_available is None:
            consolidation_available = torch.ones(
                batch, dtype=torch.bool, device=bank.keys.device
            )
        self._validate_inputs(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=consolidation_available,
        )
        row_features = self._row_features(bank)
        occupied = bank.occupied
        counts = occupied.sum(dim=-1, keepdim=True).to(row_features.dtype)
        pooled = (row_features * occupied.unsqueeze(-1)).sum(dim=1) / counts.clamp_min(1.0)
        incoming = torch.cat((incoming_key, incoming_value), dim=-1)
        occupancy_fraction = counts / float(capacity)
        protected_fraction = (
            (protected & occupied).sum(dim=-1, keepdim=True).to(row_features.dtype)
            / counts.clamp_min(1.0)
        )
        normalized_values = torch.nn.functional.normalize(bank.values, dim=-1)
        similarity = normalized_values @ normalized_values.transpose(-1, -2)
        pair_mask = bank.occupied[:, :, None] & bank.occupied[:, None, :]
        pair_mask = pair_mask & ~torch.eye(
            capacity, dtype=torch.bool, device=bank.keys.device
        )
        pair_similarity = similarity.masked_fill(~pair_mask, -torch.inf).amax(
            dim=(-1, -2)
        ).unsqueeze(-1)
        pair_similarity = torch.where(
            torch.isfinite(pair_similarity), pair_similarity, torch.zeros_like(pair_similarity)
        )
        global_features = torch.cat(
            (
                incoming,
                pooled,
                occupancy_fraction,
                protected_fraction,
                pair_similarity,
                occupied.any(dim=-1, keepdim=True).to(row_features.dtype),
                consolidation_available.unsqueeze(-1).to(row_features.dtype),
            ),
            dim=-1,
        )
        action_logits = self.action_network(global_features)
        repeated_global = global_features[:, None, :].expand(-1, capacity, -1)
        eviction_scores = self.eviction_network(
            torch.cat((repeated_global, row_features), dim=-1)
        ).squeeze(-1)
        left = row_features[:, :, None, :].expand(-1, -1, capacity, -1)
        right = row_features[:, None, :, :].expand(-1, capacity, -1, -1)
        incoming_pairs = incoming[:, None, None, :].expand(-1, capacity, capacity, -1)
        pair_scores = self.pair_network(
            torch.cat((left + right, (left - right).abs(), incoming_pairs), dim=-1)
        ).squeeze(-1)
        valid_evictions = occupied & ~protected
        valid_pairs = occupied[:, :, None] & occupied[:, None, :]
        diagonal = torch.eye(capacity, dtype=torch.bool, device=bank.keys.device)
        valid_pairs = valid_pairs & ~diagonal
        available_actions = torch.stack(
            (
                (counts < float(capacity)).squeeze(-1),
                valid_evictions.any(dim=-1),
                valid_pairs.any(dim=(-1, -2)) & consolidation_available,
                torch.ones(batch, dtype=torch.bool, device=bank.keys.device),
            ),
            dim=-1,
        )
        return CapacityPlannerOutput(
            action_logits=action_logits,
            eviction_scores=eviction_scores.masked_fill(~valid_evictions, -torch.inf),
            pair_scores=pair_scores.masked_fill(~valid_pairs, -torch.inf),
            valid_evictions=valid_evictions,
            valid_pairs=valid_pairs,
            available_actions=available_actions,
        )

    @torch.no_grad()
    def propose(
        self,
        bank: MemoryCandidates,
        incoming_key: torch.Tensor,
        incoming_value: torch.Tensor,
        protected: torch.Tensor,
        *,
        consolidation_available: torch.Tensor | None = None,
    ) -> CapacityPlan | tuple[CapacityPlan, ...]:
        """Return the highest-scoring safe action for one or more banks."""

        if bank.keys.shape[0] < 1:
            raise ValueError("capacity planner requires at least one bank")
        output = self(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=consolidation_available,
        )
        plans: list[CapacityPlan] = []
        for batch_index in range(bank.keys.shape[0]):
            available = output.available_actions[batch_index]
            logits = output.action_logits[batch_index].masked_fill(~available, -torch.inf)
            action_index = int(logits.argmax())
            eviction_index: int | None = None
            pair: tuple[int, int] | None = None
            score = logits[action_index]
            if action_index == 1:
                eviction_index = int(output.eviction_scores[batch_index].argmax())
                score = output.eviction_scores[batch_index, eviction_index]
            elif action_index == 2:
                pair_scores = output.pair_scores[batch_index].masked_fill(
                    ~torch.triu(
                        torch.ones_like(output.valid_pairs[batch_index]), diagonal=1
                    ),
                    -torch.inf,
                )
                flat_index = int(pair_scores.reshape(-1).argmax())
                first, second = divmod(flat_index, pair_scores.shape[1])
                pair = (first, second)
                score = pair_scores[first, second]
            plans.append(
                CapacityPlan(
                    action=ADMISSION_ACTIONS[action_index],
                    action_index=action_index,
                    eviction_index=eviction_index,
                    pair=pair,
                    score=score.detach().clone(),
                )
            )
        return plans[0] if len(plans) == 1 else tuple(plans)


__all__ = [
    "ADMISSION_ACTIONS",
    "CAPACITY_PLANNER_SCHEMA",
    "CapacityPlan",
    "CapacityPlannerOutput",
    "OpaqueCapacityPlanner",
]
