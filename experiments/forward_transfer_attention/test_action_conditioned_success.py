import torch

from .train_action_conditioned_success import (
    expected_calibration_error,
    exploration_probabilities,
    selected_success_loss,
)


def test_selected_success_loss_ignores_unattempted_action() -> None:
    actions = torch.tensor([0, 1])
    rewards = torch.tensor([1.0, 0.0])
    first = torch.tensor(
        [[0.3, -100.0], [100.0, -0.7]], requires_grad=True)
    second = torch.tensor(
        [[0.3, 100.0], [-100.0, -0.7]], requires_grad=True)
    loss_first = selected_success_loss(first, actions, rewards)
    loss_second = selected_success_loss(second, actions, rewards)
    assert torch.allclose(loss_first, loss_second)
    loss_first.backward()
    assert first.grad is not None
    assert first.grad[0, 1] == 0
    assert first.grad[1, 0] == 0


def test_exploration_probabilities_have_known_floor() -> None:
    logits = torch.tensor([[100.0, -100.0], [-20.0, 20.0]])
    probabilities = exploration_probabilities(logits, epsilon=0.20)
    assert torch.allclose(probabilities.sum(-1), torch.ones(2))
    assert bool((probabilities >= 0.10).all())


def test_calibration_error_is_zero_for_perfect_binary_predictions() -> None:
    probabilities = torch.tensor([0.0, 1.0, 1.0, 0.0])
    outcomes = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert expected_calibration_error(probabilities, outcomes) == 0.0


def test_calibration_error_detects_confidently_wrong_predictions() -> None:
    probabilities = torch.tensor([1.0, 0.0])
    outcomes = torch.tensor([0.0, 1.0])
    assert expected_calibration_error(probabilities, outcomes) > 0.9
