"""External evidence-trained proposal routing for compositional search.

The controller never sees this artifact.  It ranks opaque library slots and
boolean combiners from learned event/prediction agreement, then leaves the
existing verifier-backed confirmation gate in charge.  A bounded exhaustive
fallback keeps the development path behavior-preserving while the proposer is
being measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

COMPOSITION_PROPOSER_SCHEMA = "neural-computer.external-composition-proposer.v1"


@dataclass(frozen=True)
class ProposalBudget:
    """Accounting for one fast proposal pass."""

    slots_considered: int
    hypotheses: int
    fallback: bool

    def payload(self) -> dict[str, int | bool]:
        return {
            "slots_considered": self.slots_considered,
            "hypotheses": self.hypotheses,
            "fallback": self.fallback,
        }


class LearnedCompositionProposer:
    """Learn a cheap proposal order from opaque confirmation outcomes.

    The proposer does not assign meaning to a slot or combiner.  It only keeps
    Beta-smoothed success rates for the allowed opaque combiners and uses the
    current learned prediction vectors to shortlist slots whose marginal
    behavior is compatible with the observed scalar outcomes.  Confirmation
    remains mandatory; this object can reduce search work, never admit a
    candidate by itself.
    """

    schema = COMPOSITION_PROPOSER_SCHEMA

    def __init__(
        self,
        *,
        slot_budget: int = 8,
        candidate_budget: int = 96,
        fallback_on_miss: bool = True,
    ) -> None:
        if slot_budget < 2:
            raise ValueError("composition proposer slot budget must be at least two")
        if candidate_budget < 1:
            raise ValueError("composition proposer candidate budget must be positive")
        self.slot_budget = int(slot_budget)
        self.candidate_budget = int(candidate_budget)
        self.fallback_on_miss = bool(fallback_on_miss)
        self._attempts: dict[str, int] = {}
        self._successes: dict[str, int] = {}
        self._last_budget = ProposalBudget(0, 0, False)

    @property
    def last_budget(self) -> ProposalBudget:
        return self._last_budget

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "behavior": "opaque-marginal-shortlist_beta-combiner-prior_v1",
            "slot_budget": self.slot_budget,
            "candidate_budget": self.candidate_budget,
            "fallback_on_miss": self.fallback_on_miss,
        }

    def observe(self, combiner: str, *, accepted: bool) -> None:
        """Update the prior only after verifier-backed confirmation."""

        if not isinstance(combiner, str) or not combiner:
            raise ValueError("a proposer update needs an opaque combiner label")
        self._attempts[combiner] = self._attempts.get(combiner, 0) + 1
        if accepted:
            self._successes[combiner] = self._successes.get(combiner, 0) + 1

    def _combiner_prior(self, combiner: str) -> float:
        return (self._successes.get(combiner, 0) + 1.0) / (
            self._attempts.get(combiner, 0) + 2.0
        )

    @staticmethod
    def _slot_score(prediction: tuple[int, ...], targets: tuple[int, ...]) -> float:
        if not prediction or len(prediction) != len(targets):
            return float("-inf")
        agreement = sum(a == b for a, b in zip(prediction, targets, strict=True))
        predicted_positive = sum(prediction)
        target_positive = sum(targets)
        # Marginal agreement plus a weak prevalence tie-break.  The latter is
        # useful for OR/AND but never replaces pair confirmation.
        prevalence = 1.0 - abs(predicted_positive - target_positive) / max(
            1, len(targets)
        )
        return agreement / len(targets) + 0.1 * prevalence

    def propose(
        self,
        predictions: dict[int, tuple[int, ...]],
        targets: tuple[int, ...],
        usable: tuple[int, ...],
        combiners: tuple[str, ...],
    ) -> tuple[tuple[int, ...], tuple[str, ...], ProposalBudget]:
        """Return a bounded slot/combiner shortlist.

        Singles remain available for every usable slot.  Pair proposals use a
        learned shortlist, so the expensive elementwise merges scale with the
        configured slot budget rather than the whole library.
        """

        ranked = sorted(
            (slot for slot in usable if slot in predictions),
            key=lambda slot: self._slot_score(predictions[slot], targets),
            reverse=True,
        )
        selected = tuple(ranked[: self.slot_budget])
        ordered_combiners = tuple(
            sorted(combiners, key=self._combiner_prior, reverse=True)
        )
        proposals: list[tuple[int, ...]] = [(slot,) for slot in ranked]
        for left_index, left in enumerate(selected):
            for right in selected[left_index + 1 :]:
                for combiner in ordered_combiners:
                    proposals.append((left, right, combiner))
        bounded = tuple(proposals[: self.candidate_budget])
        self._last_budget = ProposalBudget(len(selected), len(bounded), False)
        return bounded, ordered_combiners, self._last_budget

    def mark_fallback(self) -> ProposalBudget:
        """Record that the safe exhaustive path was needed after a miss."""

        self._last_budget = ProposalBudget(
            self._last_budget.slots_considered,
            self._last_budget.hypotheses,
            True,
        )
        return self._last_budget


__all__ = ["COMPOSITION_PROPOSER_SCHEMA", "LearnedCompositionProposer", "ProposalBudget"]
