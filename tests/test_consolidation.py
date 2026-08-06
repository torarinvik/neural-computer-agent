from __future__ import annotations

import torch

from neural_computer import (
    MemoryCandidates,
    OpaqueConsolidationPolicy,
    apply_consolidation_proposal,
    verify_consolidation_proposal,
)


def _bank() -> MemoryCandidates:
    return MemoryCandidates(
        keys=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
        ),
        values=torch.tensor(
            [[[0.1, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0]]]
        ),
        strengths=torch.tensor([[0.8, 0.7, 0.6]]),
        timestamps=torch.tensor([[3.0, 2.0, 1.0]]),
        occupied=torch.tensor([[True, True, True]]),
    )


def test_consolidation_pair_scores_are_permutation_equivariant() -> None:
    policy = OpaqueConsolidationPolicy(4, hidden=8)
    bank = _bank()
    permutation = torch.tensor([2, 0, 1])
    permuted = MemoryCandidates(
        keys=bank.keys[:, permutation],
        values=bank.values[:, permutation],
        strengths=bank.strengths[:, permutation],
        timestamps=bank.timestamps[:, permutation],
        occupied=bank.occupied[:, permutation],
    )

    original = policy(bank)
    reordered = policy(permuted)
    for new_row, old_row in enumerate(permutation.tolist()):
        assert torch.allclose(
            reordered.pair_scores[0, new_row],
            original.pair_scores[0, old_row][permutation],
        )
        assert torch.allclose(
            reordered.operation_logits[0, new_row],
            original.operation_logits[0, old_row][permutation],
        )
    assert torch.allclose(
        reordered.row_preferences[0], original.row_preferences[0][permutation]
    )


def test_consolidation_transaction_is_immutable_and_verifier_gated() -> None:
    policy = OpaqueConsolidationPolicy(4, hidden=8)
    source = _bank()
    proposal = policy.propose(source)
    assert proposal is not None

    candidate = apply_consolidation_proposal(source, proposal)
    assert int(source.occupied.sum()) == 3
    assert int(candidate.occupied.sum()) == 2

    accepted, receipt = verify_consolidation_proposal(
        source,
        proposal,
        verifier=lambda rewritten: int(rewritten.occupied.sum()) == 2,
        candidate_outcomes=[1.0, 1.0],
        retained_scores=[1.0],
        min_candidate_observations=2,
    )
    assert accepted is not None
    assert receipt.accepted
    assert receipt.rows_saved == 1
    assert receipt.retention_checked
    assert receipt.retention_accepted

    rejected, rejected_receipt = verify_consolidation_proposal(
        source,
        proposal,
        verifier=lambda _rewritten: True,
        candidate_outcomes=[1.0, 0.0],
        retained_scores=[1.0],
        min_candidate_observations=2,
    )
    assert rejected is None
    assert not rejected_receipt.accepted
    assert "stable" in rejected_receipt.reason
