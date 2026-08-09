"""Goal-conditioned addressing over opaque external-memory keys.

The controller or event system learns the query representation.  This module
is intentionally stateless: it provides the replaceable memory-side contract
that compares that learned query with learned opaque keys, without introducing
a random cold-start adapter or a protocol-specific reasoning branch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

GOAL_CONDITIONED_MEMORY_RELEVANCE_SCHEMA = (
    "neural-computer.goal-conditioned-memory-relevance.v1"
)


def _validate_batch(value: torch.Tensor, *, name: str, width: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must have shape [rows, {width}]")
    if value.shape[1] != width or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} has the wrong shape or non-finite values")


@dataclass(frozen=True)
class GoalConditionedMemoryProposal:
    """One verifier-independent proposal over stable logical memory IDs."""

    selected_slot_id: int | None
    scores: torch.Tensor
    candidate_slot_ids: tuple[int, ...]
    reason: str
    schema: str = GOAL_CONDITIONED_MEMORY_RELEVANCE_SCHEMA

    def validate(self) -> GoalConditionedMemoryProposal:
        if self.schema != GOAL_CONDITIONED_MEMORY_RELEVANCE_SCHEMA:
            raise ValueError("unsupported goal-conditioned relevance schema")
        if self.scores.ndim != 1 or self.scores.shape[0] != len(
            self.candidate_slot_ids
        ):
            raise ValueError("goal-conditioned relevance scores are misaligned")
        if not bool(torch.isfinite(self.scores).all()):
            raise ValueError("goal-conditioned relevance scores are not finite")
        if len(set(self.candidate_slot_ids)) != len(self.candidate_slot_ids):
            raise ValueError("goal-conditioned relevance slot IDs are duplicated")
        if any(
            not isinstance(slot_id, int) or isinstance(slot_id, bool) or slot_id < 0
            for slot_id in self.candidate_slot_ids
        ):
            raise ValueError("goal-conditioned relevance slot ID is invalid")
        if self.selected_slot_id is not None and self.selected_slot_id not in (
            self.candidate_slot_ids
        ):
            raise ValueError("goal-conditioned relevance selection is not a candidate")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("goal-conditioned relevance proposal reason is missing")
        return self


class ExternalGoalConditionedMemoryRelevance:
    """Stateless shared learned-space address resolver."""

    schema = GOAL_CONDITIONED_MEMORY_RELEVANCE_SCHEMA

    def __init__(self, query_width: int, key_width: int) -> None:
        if min(query_width, key_width) < 1:
            raise ValueError("goal-conditioned relevance widths must be positive")
        self.query_width = int(query_width)
        self.key_width = int(key_width)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "query_width": self.query_width,
            "key_width": self.key_width,
            "addressing": "normalized_shared_learned_space_dot_product_v1",
            "state": "stateless_upstream_query_owned_v1",
        }

    def similarity_scores(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Return one normalized alignment score per opaque key."""

        if query.ndim == 1:
            query = query.unsqueeze(0)
        _validate_batch(query, name="relevance query", width=self.query_width)
        _validate_batch(keys, name="relevance keys", width=self.key_width)
        if query.shape[0] != 1:
            raise ValueError("relevance query must contain one row")
        if self.query_width != self.key_width:
            raise ValueError("shared-space addressing requires equal query and key widths")
        normalized_query = torch.nn.functional.normalize(query, dim=-1)
        normalized_keys = torch.nn.functional.normalize(keys, dim=-1)
        return torch.einsum("bw,kw->k", normalized_query, normalized_keys)

    @torch.no_grad()
    def propose(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        slot_ids: Sequence[int],
        *,
        protected: torch.Tensor | None = None,
    ) -> GoalConditionedMemoryProposal:
        """Propose the most relevant unprotected stable logical address."""

        if keys.ndim != 2:
            raise ValueError("relevance keys must have shape [candidates, width]")
        if len(slot_ids) != keys.shape[0]:
            raise ValueError("relevance slot IDs do not match candidate keys")
        normalized_ids = tuple(int(slot_id) for slot_id in slot_ids)
        if len(set(normalized_ids)) != len(normalized_ids) or any(
            slot_id < 0 for slot_id in normalized_ids
        ):
            raise ValueError("relevance slot IDs are invalid")
        if protected is None:
            protected = torch.zeros(keys.shape[0], dtype=torch.bool)
        if protected.shape != (keys.shape[0],) or protected.dtype != torch.bool:
            raise ValueError("relevance protected mask is misaligned")
        scores = self.similarity_scores(query, keys)
        eligible = [
            index for index, is_protected in enumerate(protected.tolist()) if not is_protected
        ]
        if not eligible:
            return GoalConditionedMemoryProposal(
                selected_slot_id=None,
                scores=scores.new_empty((0,)),
                candidate_slot_ids=(),
                reason="all relevance candidates are protected",
            ).validate()
        selected_index = max(eligible, key=lambda index: (float(scores[index]), -index))
        eligible_ids = tuple(normalized_ids[index] for index in eligible)
        return GoalConditionedMemoryProposal(
            selected_slot_id=normalized_ids[selected_index],
            scores=scores[eligible].detach().clone(),
            candidate_slot_ids=eligible_ids,
            reason="shared learned-space similarity selected an opaque slot",
        ).validate()


__all__ = [
    "GOAL_CONDITIONED_MEMORY_RELEVANCE_SCHEMA",
    "ExternalGoalConditionedMemoryRelevance",
    "GoalConditionedMemoryProposal",
]
