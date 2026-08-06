import torch

from experiments.games_amodal.environments import SnakeVerifier
from experiments.games_amodal.snake_acquisition import (
    SnakePolicy,
    capability_key,
    evaluate,
    rollout,
    train_reward_only,
)


def _policy() -> SnakePolicy:
    torch.manual_seed(0)
    return SnakePolicy(height=8, width=8, event_width=16, intent_width=8, hidden=16)


def test_rollout_shapes_and_masking() -> None:
    policy = _policy()
    verifier = SnakeVerifier(batch_size=4, seed=2)
    summary = rollout(policy, verifier, steps=16, seed=2, sample=True, gamma=0.9)
    assert summary.total_reward.shape == (4,)
    assert summary.advantage is not None
    assert summary.log_propensity is not None
    assert summary.mask is not None
    assert summary.advantage.shape == summary.log_propensity.shape
    assert bool((summary.survival_steps <= 16).all())
    assert bool(torch.isfinite(summary.log_propensity).all())


def test_greedy_rollout_has_no_gradient_terms() -> None:
    policy = _policy()
    verifier = SnakeVerifier(batch_size=2, seed=3)
    summary = rollout(policy, verifier, steps=8, seed=3, sample=False, gamma=0.9)
    assert summary.advantage is None
    assert summary.log_propensity is None
    assert summary.mask is None


def test_training_changes_parameters_and_replays_nothing() -> None:
    policy = _policy()
    before = [parameter.clone() for parameter in policy.parameters()]
    history = train_reward_only(
        policy,
        updates=2,
        batch_size=8,
        steps=8,
        seed=1,
        gamma=0.9,
        learning_rate=1e-3,
        shuffle_rewards=False,
    )
    assert len(history) == 2
    assert all(entry["replayed_examples"] == 0.0 for entry in history)
    changed = any(
        not torch.equal(old, new)
        for old, new in zip(before, policy.parameters(), strict=True)
    )
    assert changed


def test_shuffled_training_runs_with_same_interface() -> None:
    policy = _policy()
    history = train_reward_only(
        policy,
        updates=1,
        batch_size=8,
        steps=8,
        seed=4,
        gamma=0.9,
        learning_rate=1e-3,
        shuffle_rewards=True,
    )
    assert len(history) == 1


def test_evaluate_and_capability_key_are_deterministic() -> None:
    policy = _policy()
    first = evaluate(policy, batch_size=4, steps=8, seeds=(5, 6), gamma=0.9)
    second = evaluate(policy, batch_size=4, steps=8, seeds=(5, 6), gamma=0.9)
    assert first == second
    key_one = capability_key(policy, batch_size=4, steps=8, seed=7)
    key_two = capability_key(policy, batch_size=4, steps=8, seed=7)
    assert torch.allclose(key_one, key_two)
    assert key_one.shape == (16,)
