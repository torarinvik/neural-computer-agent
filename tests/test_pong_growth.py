import torch

from experiments.games_amodal.environments import PongVerifier
from experiments.games_amodal.pong_growth import parameter_digest
from experiments.games_amodal.snake_acquisition import (
    SnakePolicy,
    evaluate,
    train_reward_only,
)


def test_parameter_digest_detects_any_change() -> None:
    torch.manual_seed(0)
    policy = SnakePolicy(height=8, width=8, event_width=16, intent_width=8, hidden=16)
    before = parameter_digest(policy)
    assert before == parameter_digest(policy)
    with torch.no_grad():
        next(policy.parameters()).add_(1e-6)
    assert parameter_digest(policy) != before


def test_frozen_snake_is_untouched_by_pong_training() -> None:
    torch.manual_seed(1)
    snake = SnakePolicy(height=8, width=8, event_width=16, intent_width=8, hidden=16)
    for parameter in snake.parameters():
        parameter.requires_grad_(False)
    digest = parameter_digest(snake)
    pong = SnakePolicy(
        height=8, width=8, event_width=16, intent_width=8, hidden=16,
        channels=2, action_count=3,
    )
    train_reward_only(
        pong,
        updates=2,
        batch_size=8,
        steps=8,
        seed=2,
        gamma=0.9,
        learning_rate=1e-3,
        shuffle_rewards=False,
        verifier_factory=PongVerifier,
    )
    assert parameter_digest(snake) == digest


def test_evaluate_works_with_pong_verifier() -> None:
    torch.manual_seed(3)
    pong = SnakePolicy(
        height=8, width=8, event_width=16, intent_width=8, hidden=16,
        channels=2, action_count=3,
    )
    result = evaluate(
        pong,
        batch_size=4,
        steps=8,
        seeds=(4,),
        gamma=0.9,
        verifier_factory=PongVerifier,
    )
    assert 0.0 <= float(result["mastery"]) <= 1.0
