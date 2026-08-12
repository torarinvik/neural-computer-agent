from __future__ import annotations

import pytest
import torch

from neural_computer import (
    OpaqueSharedBasisCompressionPolicy,
    SharedBasisCompressionPlan,
)


def _features() -> torch.Tensor:
    return torch.tensor(
        [
            [0.25, 0.20, 0.30, 0.70],
            [0.50, 0.02, 0.45, 0.55],
            [1.00, 0.01, 0.80, 0.20],
        ]
    ).unsqueeze(0)


def test_shared_basis_policy_scores_runtime_candidate_sets() -> None:
    policy = OpaqueSharedBasisCompressionPolicy(feature_width=4, hidden=12)

    output = policy(_features())
    plan = policy.propose(_features())

    assert output.logits.shape == (1, 3)
    assert isinstance(plan, SharedBasisCompressionPlan)
    assert 0 <= plan.candidate_index < 3
    assert policy.configuration()["proposal"] == "candidate_index_only_v1"


def test_shared_basis_policy_is_candidate_order_equivariant() -> None:
    policy = OpaqueSharedBasisCompressionPolicy(feature_width=4, hidden=12).eval()
    features = _features()
    permutation = torch.tensor([2, 0, 1])

    original = policy.propose(features)
    permuted = policy.propose(features[:, permutation])

    assert permuted.candidate_index == int(
        (permutation == original.candidate_index).nonzero()[0]
    )


def test_shared_basis_policy_accepts_one_scalar_utility_update() -> None:
    policy = OpaqueSharedBasisCompressionPolicy(feature_width=4, hidden=12)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    plan = policy.propose(_features(), explore=True, generator=torch.Generator().manual_seed(4))

    loss = policy.adaptation_step(_features(), plan, 1.0, optimizer=optimizer)

    assert torch.isfinite(torch.tensor(loss))


def test_shared_basis_policy_rejects_invalid_feature_contracts() -> None:
    policy = OpaqueSharedBasisCompressionPolicy(feature_width=4, hidden=12)

    with pytest.raises(ValueError, match="shape"):
        policy(torch.zeros(1, 3, 5))
    with pytest.raises(ValueError, match="finite"):
        policy(torch.full((1, 3, 4), float("nan")))
    with pytest.raises(ValueError, match="utility"):
        policy.adaptation_step(
            _features(),
            SharedBasisCompressionPlan(candidate_index=0, score=torch.tensor(0.0)),
            1.1,
        )
