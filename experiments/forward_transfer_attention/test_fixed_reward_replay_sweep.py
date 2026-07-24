import torch

from .train_fixed_reward_replay_sweep import (
    select_policy_input,
    uniform_logged_buffer,
)


def test_uniform_buffer_balances_actions_without_rule_dependence() -> None:
    states = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    rules_a = torch.zeros(8, dtype=torch.long)
    rules_b = torch.ones(8, dtype=torch.long)
    output_a = uniform_logged_buffer(states, rules_a, seed=17)
    output_b = uniform_logged_buffer(states, rules_b, seed=17)
    states_a, _, actions_a, rewards_a, propensities_a = output_a
    states_b, _, actions_b, rewards_b, propensities_b = output_b
    assert torch.equal(states_a, states_b)
    assert torch.equal(actions_a, actions_b)
    assert torch.bincount(actions_a).tolist() == [4, 4]
    assert torch.all(propensities_a == 0.5)
    assert torch.equal(propensities_a, propensities_b)
    assert torch.equal(rewards_a, 1.0 - rewards_b)


def test_uniform_buffer_is_deterministic_for_seed() -> None:
    states = torch.randn(10, 4)
    rules = torch.tensor([0, 1] * 5)
    first = uniform_logged_buffer(states, rules, seed=29)
    second = uniform_logged_buffer(states, rules, seed=29)
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_support_only_input_removes_query_frames() -> None:
    frames = torch.randn(7, 5, 3, 8, 8)
    support = select_policy_input(frames, "support-only")
    assert support.shape == (7, 3, 3, 8, 8)
    assert torch.equal(support, frames[:, :3])
    assert torch.equal(select_policy_input(frames, "full"), frames)
