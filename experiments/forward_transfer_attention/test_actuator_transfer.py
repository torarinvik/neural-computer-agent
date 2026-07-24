import torch

from .train_actuator_transfer import (
    correct_protocol_actions,
    opposite_rule_permutation,
    protocol_for_seed,
    uniform_logged_protocol_buffer,
)


def test_protocol_is_deterministic_and_uses_distinct_commands() -> None:
    first = protocol_for_seed(17, 4)
    second = protocol_for_seed(17, 4)
    assert torch.equal(first, second)
    assert first.shape == (2,)
    assert first.unique().numel() == 2
    assert int(first.min()) >= 0
    assert int(first.max()) < 4


def test_correct_protocol_actions_apply_private_mapping() -> None:
    rules = torch.tensor([0, 1, 1, 0])
    protocol = torch.tensor([3, 1])
    assert torch.equal(
        correct_protocol_actions(rules, protocol),
        torch.tensor([3, 1, 1, 3]))


def test_opposite_rule_permutation_mismatches_every_balanced_rule() -> None:
    rules = torch.tensor([0, 0, 1, 1, 0, 1])
    permutation = opposite_rule_permutation(rules)
    assert torch.all(rules[permutation] != rules)
    assert torch.equal(
        permutation.sort().values, torch.arange(rules.numel()))


def test_uniform_protocol_logging_is_rule_independent() -> None:
    states = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    protocol = torch.tensor([3, 1])
    zeros = uniform_logged_protocol_buffer(
        states, torch.zeros(8, dtype=torch.long), protocol,
        actions=4, seed=29)
    ones = uniform_logged_protocol_buffer(
        states, torch.ones(8, dtype=torch.long), protocol,
        actions=4, seed=29)
    assert torch.equal(zeros[0], ones[0])
    assert torch.equal(zeros[2], ones[2])
    assert torch.bincount(zeros[2], minlength=4).tolist() == [2, 2, 2, 2]
    assert torch.all(zeros[4] == 0.25)
    assert torch.all(ones[4] == 0.25)


def test_unattempted_commands_do_not_affect_selected_loss() -> None:
    from .train_action_conditioned_success import selected_success_loss

    logits = torch.tensor(
        [[0.2, 20.0, -20.0, 7.0], [-4.0, 0.5, 11.0, -9.0]],
        requires_grad=True)
    actions = torch.tensor([0, 1])
    rewards = torch.tensor([1.0, 0.0])
    selected_success_loss(logits, actions, rewards).backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] != 0
    assert logits.grad[1, 1] != 0
    assert torch.count_nonzero(logits.grad).item() == 2
