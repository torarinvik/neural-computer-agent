import pytest
import torch

from neural_computer import (
    ExternalGoalConditionedMemoryRelevance,
)


def test_goal_conditioned_relevance_is_permutation_equivariant_and_masks_protected() -> None:
    relevance = ExternalGoalConditionedMemoryRelevance(3, 3)
    query = torch.randn(3)
    keys = torch.randn(3, 3)
    slot_ids = (10, 20, 30)
    protected = torch.tensor([False, True, False])
    proposal = relevance.propose(query, keys, slot_ids, protected=protected)
    permutation = torch.tensor([2, 0, 1])
    permuted = relevance.propose(
        query,
        keys.index_select(0, permutation),
        tuple(slot_ids[index] for index in permutation.tolist()),
        protected=protected.index_select(0, permutation),
    )
    original = dict(zip(proposal.candidate_slot_ids, proposal.scores.tolist()))
    reordered = dict(zip(permuted.candidate_slot_ids, permuted.scores.tolist()))
    assert original.keys() == reordered.keys()
    for slot_id, score in original.items():
        assert score == pytest.approx(reordered[slot_id])
    assert proposal.selected_slot_id != 20
    assert relevance.propose(
        query,
        keys,
        slot_ids,
        protected=torch.ones(3, dtype=torch.bool),
    ).selected_slot_id is None


def test_similarity_addressing_selects_matching_learned_key_without_random_adapter() -> None:
    relevance = ExternalGoalConditionedMemoryRelevance(3, 3)
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    proposal = relevance.propose(
        keys[1],
        keys,
        (10, 20, 30),
        protected=torch.tensor([False, False, True]),
    )
    assert proposal.selected_slot_id == 20
    assert proposal.reason.startswith("shared learned-space")
