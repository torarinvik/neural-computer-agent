from __future__ import annotations

import pytest
import torch

from neural_computer import (
    factorized_counterfactual_policy_loss,
    paired_counterfactual_advantage,
    paired_counterfactual_policy_loss,
    paired_counterfactual_ranking_loss,
)


def test_paired_advantage_is_detached_and_uses_only_arm_difference() -> None:
    utilities = torch.tensor([[1.0, 0.0], [0.25, 0.75]], requires_grad=True)

    advantage = paired_counterfactual_advantage(utilities)

    assert torch.equal(advantage, torch.tensor([1.0, -0.5]))
    assert not advantage.requires_grad


def test_policy_loss_uses_common_random_counterfactual_difference() -> None:
    score = torch.tensor([2.0, 4.0], requires_grad=True)
    utilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss, advantage = paired_counterfactual_policy_loss(score, utilities)
    loss.backward()

    assert advantage.tolist() == [1.0, -1.0]
    assert loss.item() == pytest.approx(1.0)
    assert score.grad is not None
    assert score.grad.tolist() == [-0.5, 0.5]


def test_ranking_loss_scores_only_attempted_rows() -> None:
    scores = torch.tensor([[3.0, 1.0, -2.0]], requires_grad=True)
    attempted = torch.tensor([[0, 2]])
    utilities = torch.tensor([[1.0, 0.0]])

    loss, advantage = paired_counterfactual_ranking_loss(
        scores,
        attempted,
        utilities,
    )
    loss.backward()

    assert advantage.item() == pytest.approx(1.0)
    assert loss.item() == pytest.approx(torch.nn.functional.softplus(torch.tensor(-5.0)).item())
    assert scores.grad is not None
    assert scores.grad[0, 0].item() == pytest.approx(-0.00669285, abs=1e-6)
    assert scores.grad[0, 1].item() == pytest.approx(0.0)
    assert scores.grad[0, 2].item() == pytest.approx(0.00669285, abs=1e-6)


def test_factorized_credit_keeps_intervention_factors_separate() -> None:
    scores = torch.tensor([[2.0, 3.0]], requires_grad=True)
    utilities = torch.tensor([[[1.0, 0.0], [0.25, 0.75]]])

    loss, advantage = factorized_counterfactual_policy_loss(scores, utilities)
    loss.backward()

    assert advantage.tolist() == [[1.0, -0.5]]
    assert loss.item() == pytest.approx(-0.25)
    assert scores.grad is not None
    assert scores.grad.tolist() == [[-0.5, 0.25]]


def test_counterfactual_credit_rejects_mismatched_pairs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_counterfactual_policy_loss(
            torch.zeros(2),
            torch.zeros(3, 2),
        )

    with pytest.raises(ValueError, match="distinct"):
        paired_counterfactual_advantage(
            torch.zeros(2, 2),
            positive_arm=0,
            negative_arm=0,
        )
