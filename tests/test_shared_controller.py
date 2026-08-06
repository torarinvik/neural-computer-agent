import torch

from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    acquire_pong_through_core,
    controller_digest,
    evaluate_game,
    rollout,
    train_game,
    trainable_parameters,
)


def _agent() -> SharedControllerAgent:
    torch.manual_seed(0)
    return SharedControllerAgent(
        event_width=32, intention_width=16, feedback_width=8, hidden=16
    )


def test_rollout_runs_both_games_through_one_controller() -> None:
    agent = _agent()
    for game in ("snake", "pong"):
        summary = rollout(
            agent, game, batch_size=4, steps=8, seed=1, sample=True, gamma=0.9
        )
        assert summary["total_reward"].shape == (4,)
        assert summary["advantage"] is not None


def test_snake_training_updates_controller() -> None:
    agent = _agent()
    digest = controller_digest(agent)
    train_game(
        agent,
        "snake",
        trainable=trainable_parameters(
            [agent.controller, *agent.game_modules("snake")]
        ),
        updates=2,
        batch_size=4,
        steps=8,
        seed=2,
        gamma=0.9,
        learning_rate=1e-3,
        shuffle_rewards=False,
    )
    assert controller_digest(agent) != digest


def test_pong_acquisition_leaves_frozen_core_untouched() -> None:
    import argparse

    agent = _agent()
    digest = controller_digest(agent)
    args = argparse.Namespace(
        updates=2,
        batch_size=4,
        steps=8,
        seed=3,
        gamma=0.9,
        learning_rate=1e-3,
    )
    history = acquire_pong_through_core(agent, args=args, shuffle_rewards=False)
    assert len(history) == 2
    assert controller_digest(agent) == digest
    assert all(entry["replayed_examples"] == 0.0 for entry in history)


def test_evaluate_game_is_deterministic() -> None:
    agent = _agent()
    first = evaluate_game(
        agent, "snake", batch_size=4, steps=8, seeds=(5, 6), gamma=0.9
    )
    second = evaluate_game(
        agent, "snake", batch_size=4, steps=8, seeds=(5, 6), gamma=0.9
    )
    assert first == second
