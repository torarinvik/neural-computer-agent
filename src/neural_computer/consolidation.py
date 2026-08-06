"""Opaque, verifier-gated consolidation for controller-native memory rows.

The policy in this module is a replaceable memory-side component.  It sees
only learned keys, learned values, scalar strength, and relative age.  It does
not receive task names, modality fields, physical row identities as features,
correct actions, or raw sensory data.  A proposal is only a candidate rewrite:
the caller's verifier must accept the rewritten snapshot before adoption.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .memory import MemoryCandidates
from .retention import RetentionGateDecision, evaluate_retention_gate

CONSOLIDATION_POLICY_SCHEMA = "neural-computer.opaque-consolidation-policy.v1"
CONSOLIDATION_OPERATION_COUNT = 3


@dataclass(frozen=True)
class ConsolidationPolicyOutput:
    """Scores for unordered row pairs and their mechanical rewrite choices."""

    pair_scores: torch.Tensor
    operation_logits: torch.Tensor
    merge_logits: torch.Tensor
    row_preferences: torch.Tensor
    valid_pairs: torch.Tensor


@dataclass(frozen=True)
class ConsolidationProposal:
    """One opaque two-row-to-one-row candidate rewrite."""

    first: int
    second: int
    operation: int
    key: torch.Tensor
    value: torch.Tensor
    strength: torch.Tensor
    score: torch.Tensor
    operation_logits: torch.Tensor


@dataclass(frozen=True)
class MemoryConsolidationReceipt:
    """Auditable result of a verifier-gated snapshot rewrite."""

    accepted: bool
    source_indices: tuple[int, ...]
    rows_before: int
    rows_after: int
    rows_saved: int
    retention_checked: bool
    retention_accepted: bool | None
    reason: str = ""


def _as_scalar(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return value.reshape(()).detach()


class OpaqueConsolidationPolicy(nn.Module):
    """Learn pair selection and mechanical rewrite choice outside the core.

    Operation ``0`` merges the two rows with learned convex gates.  Operation
    ``1`` keeps the more preferred row and operation ``2`` keeps the less
    preferred row.  These are storage operations, not semantic or
    modality-specific reasoning branches.  Pair features are symmetric, so
    candidate permutation cannot change which unordered pair is selected.
    Training can use scalar verifier utility on the output scores; deployment
    still requires an independent transaction verifier.
    """

    schema = CONSOLIDATION_POLICY_SCHEMA

    def __init__(self, width: int, *, hidden: int = 128) -> None:
        super().__init__()
        if min(width, hidden) < 1:
            raise ValueError("consolidation policy widths must be positive")
        self.width = int(width)
        self.hidden = int(hidden)
        row_width = 2 * self.width + 2
        pair_width = row_width * 2
        self.row_width = row_width
        self.pair_width = pair_width
        self.pair_score = nn.Sequential(
            nn.Linear(pair_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.operation = nn.Sequential(
            nn.Linear(pair_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, CONSOLIDATION_OPERATION_COUNT),
        )
        self.merge = nn.Sequential(
            nn.Linear(pair_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * self.width),
        )
        self.row_preference = nn.Sequential(
            nn.Linear(row_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # A new policy starts neutral: pair selection is not allowed to invent
        # a deterministic physical-slot preference before verifier feedback.
        nn.init.zeros_(self.pair_score[-1].weight)
        nn.init.zeros_(self.pair_score[-1].bias)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "width": self.width,
            "hidden": self.hidden,
            "operations": CONSOLIDATION_OPERATION_COUNT,
            "row_features": "key_value_strength_relative_age",
        }

    def _row_features(self, bank: MemoryCandidates) -> torch.Tensor:
        timestamps = bank.timestamps
        occupied = bank.occupied
        occupied_timestamps = timestamps.masked_fill(~occupied, 0.0)
        latest = occupied_timestamps.amax(dim=-1, keepdim=True)
        earliest = torch.where(
            occupied,
            timestamps,
            latest,
        ).amin(dim=-1, keepdim=True)
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

    def forward(self, bank: MemoryCandidates) -> ConsolidationPolicyOutput:
        bank.validate(width=self.width, capacity=bank.keys.shape[1])
        row_features = self._row_features(bank)
        left = row_features[:, :, None, :].expand(-1, -1, row_features.shape[1], -1)
        right = row_features[:, None, :, :].expand(-1, row_features.shape[1], -1, -1)
        pair_features = torch.cat((left + right, (left - right).abs()), dim=-1)
        pair_scores = self.pair_score(pair_features).squeeze(-1)
        operation_logits = self.operation(pair_features)
        merge_logits = self.merge(pair_features)
        row_preferences = self.row_preference(row_features).squeeze(-1)
        valid_pairs = bank.occupied[:, :, None] & bank.occupied[:, None, :]
        diagonal = torch.eye(
            bank.keys.shape[1], dtype=torch.bool, device=bank.keys.device
        ).unsqueeze(0)
        valid_pairs = valid_pairs & ~diagonal
        pair_scores = pair_scores.masked_fill(~valid_pairs, -torch.inf)
        return ConsolidationPolicyOutput(
            pair_scores=pair_scores,
            operation_logits=operation_logits,
            merge_logits=merge_logits,
            row_preferences=row_preferences,
            valid_pairs=valid_pairs,
        )

    @torch.no_grad()
    def propose(self, bank: MemoryCandidates) -> ConsolidationProposal | None:
        """Return the highest-scoring unordered pair for a single bank."""

        if bank.keys.shape[0] != 1:
            raise ValueError("propose requires exactly one memory bank")
        output = self(bank)
        upper = torch.triu(
            torch.ones_like(output.valid_pairs[0], dtype=torch.bool), diagonal=1
        )
        scores = output.pair_scores[0].masked_fill(~upper, -torch.inf)
        if not bool(torch.isfinite(scores).any()):
            return None
        flat_index = int(scores.reshape(-1).argmax())
        first, second = divmod(flat_index, scores.shape[1])
        operation_logits = output.operation_logits[0, first, second]
        operation = int(operation_logits.argmax())
        pair_merge_logits = output.merge_logits[0, first, second]
        first_key = bank.keys[0, first]
        second_key = bank.keys[0, second]
        first_value = bank.values[0, first]
        second_value = bank.values[0, second]
        first_strength = bank.strengths[0, first]
        second_strength = bank.strengths[0, second]
        if operation == 0:
            gates = torch.sigmoid(pair_merge_logits)
            key_gate, value_gate = gates[: self.width], gates[self.width :]
            key = key_gate * first_key + (1.0 - key_gate) * second_key
            value = value_gate * first_value + (1.0 - value_gate) * second_value
            strength = torch.maximum(first_strength, second_strength)
        elif operation == 1:
            chosen = first if output.row_preferences[0, first] >= output.row_preferences[0, second] else second
            key = bank.keys[0, chosen]
            value = bank.values[0, chosen]
            strength = bank.strengths[0, chosen]
        else:
            chosen = second if output.row_preferences[0, first] >= output.row_preferences[0, second] else first
            key = bank.keys[0, chosen]
            value = bank.values[0, chosen]
            strength = bank.strengths[0, chosen]
        return ConsolidationProposal(
            first=first,
            second=second,
            operation=operation,
            key=key.detach().clone(),
            value=value.detach().clone(),
            strength=_as_scalar(strength, name="proposal strength"),
            score=_as_scalar(scores[first, second], name="proposal score"),
            operation_logits=operation_logits.detach().clone(),
        )


def apply_consolidation_proposal(
    bank: MemoryCandidates,
    proposal: ConsolidationProposal,
) -> MemoryCandidates:
    """Apply a proposal to an immutable tensor snapshot."""

    bank.validate(width=bank.keys.shape[-1], capacity=bank.keys.shape[1])
    if bank.keys.shape[0] != 1:
        raise ValueError("consolidation transactions require one memory bank")
    if proposal.first == proposal.second:
        raise ValueError("consolidation proposal rows must be distinct")
    capacity = bank.keys.shape[1]
    if not 0 <= proposal.first < capacity or not 0 <= proposal.second < capacity:
        raise ValueError("consolidation proposal row is outside the bank")
    if not bool(bank.occupied[0, proposal.first]) or not bool(
        bank.occupied[0, proposal.second]
    ):
        raise ValueError("consolidation proposal rows must be occupied")
    width = bank.keys.shape[-1]
    if proposal.key.shape != (width,) or proposal.value.shape != (width,):
        raise ValueError("consolidation replacement tensors have the wrong shape")
    if not bool(torch.isfinite(proposal.key).all()) or not bool(
        torch.isfinite(proposal.value).all()
    ):
        raise ValueError("consolidation replacement tensors must be finite")
    strength = float(proposal.strength.item())
    if not 0.0 <= strength <= 1.0:
        raise ValueError("consolidation replacement strength must lie in [0, 1]")
    candidate_keys = bank.keys.detach().clone()
    candidate_values = bank.values.detach().clone()
    candidate_strengths = bank.strengths.detach().clone()
    candidate_timestamps = bank.timestamps.detach().clone()
    candidate_occupied = bank.occupied.detach().clone()
    candidate_keys[0, proposal.first] = proposal.key
    candidate_values[0, proposal.first] = proposal.value
    candidate_strengths[0, proposal.first] = proposal.strength
    candidate_timestamps[0, proposal.first] = torch.maximum(
        bank.timestamps[0, proposal.first], bank.timestamps[0, proposal.second]
    )
    candidate_keys[0, proposal.second].zero_()
    candidate_values[0, proposal.second].zero_()
    candidate_strengths[0, proposal.second] = 0.0
    candidate_timestamps[0, proposal.second] = 0.0
    candidate_occupied[0, proposal.second] = False
    return MemoryCandidates(
        keys=candidate_keys,
        values=candidate_values,
        strengths=candidate_strengths,
        timestamps=candidate_timestamps,
        occupied=candidate_occupied,
    )


def verify_consolidation_proposal(
    bank: MemoryCandidates,
    proposal: ConsolidationProposal,
    verifier: Callable[[MemoryCandidates], bool],
    *,
    candidate_outcomes: Sequence[float] | torch.Tensor | None = None,
    retained_scores: Sequence[float] | torch.Tensor | None = None,
    candidate_threshold: float = 0.8,
    retention_floor: float = 0.8,
    min_candidate_observations: int = 8,
) -> tuple[MemoryCandidates | None, MemoryConsolidationReceipt]:
    """Build and verify a rewrite without mutating the source snapshot."""

    if not callable(verifier):
        raise TypeError("consolidation verifier must be callable")
    if (candidate_outcomes is None) != (retained_scores is None):
        raise ValueError(
            "candidate outcomes and retained scores must be supplied together"
        )
    source_indices = (proposal.first, proposal.second)
    rows_before = int(bank.occupied.sum().item())
    retention_decision: RetentionGateDecision | None = None
    if candidate_outcomes is not None and retained_scores is not None:
        retention_decision = evaluate_retention_gate(
            candidate_outcomes,
            retained_scores,
            candidate_threshold=candidate_threshold,
            retention_floor=retention_floor,
            min_candidate_observations=min_candidate_observations,
        )
        if not retention_decision.accepted:
            return None, MemoryConsolidationReceipt(
                accepted=False,
                source_indices=source_indices,
                rows_before=rows_before,
                rows_after=rows_before,
                rows_saved=0,
                retention_checked=True,
                retention_accepted=False,
                reason=retention_decision.reason,
            )
    candidate = apply_consolidation_proposal(bank, proposal)
    accepted = bool(verifier(candidate))
    rows_after = int(candidate.occupied.sum().item())
    return (
        candidate if accepted else None,
        MemoryConsolidationReceipt(
            accepted=accepted,
            source_indices=source_indices,
            rows_before=rows_before,
            rows_after=rows_after if accepted else rows_before,
            rows_saved=rows_before - rows_after if accepted else 0,
            retention_checked=retention_decision is not None,
            retention_accepted=(
                retention_decision.accepted if retention_decision is not None else None
            ),
            reason=(
                "behavior verifier passed"
                if accepted
                else "behavior verifier rejected candidate bank"
            ),
        ),
    )


__all__ = [
    "CONSOLIDATION_OPERATION_COUNT",
    "CONSOLIDATION_POLICY_SCHEMA",
    "ConsolidationPolicyOutput",
    "ConsolidationProposal",
    "MemoryConsolidationReceipt",
    "OpaqueConsolidationPolicy",
    "apply_consolidation_proposal",
    "verify_consolidation_proposal",
]
