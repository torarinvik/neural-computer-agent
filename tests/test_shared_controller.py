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


def test_detach_interval_backward_succeeds_across_windows() -> None:
    agent = _agent()
    for interval in (1, 4):
        torch.manual_seed(9)
        summary = rollout(
            agent, "snake", batch_size=2, steps=8, seed=9, sample=True,
            gamma=0.9, detach_interval=interval,
        )
        loss = -(
            summary["advantage"] * summary["log_propensity"] * summary["mask"]
        ).sum()
        agent.zero_grad(set_to_none=True)
        loss.backward()
        assert bool(torch.isfinite(loss))
        grads = [
            p.grad
            for p in agent.controller.parameters()
            if p.grad is not None
        ]
        assert grads
        assert all(bool(torch.isfinite(g).all()) for g in grads)
    agent.zero_grad(set_to_none=True)


def test_shared_drivers_unify_encoder_decoder_across_games() -> None:
    torch.manual_seed(0)
    agent = SharedControllerAgent(
        event_width=32, intention_width=16, feedback_width=8, hidden=16,
        games=("snake", "pong", "breakout"), shared_drivers=True,
    )
    assert set(agent.runtime.encoders.keys()) == {"screen"}
    assert set(agent.runtime.output_bus.decoders.keys()) == {"keypress"}
    assert agent.runtime.output_bus.decoders["keypress"].key_count == 4
    modules_snake = agent.game_modules("snake")
    modules_pong = agent.game_modules("pong")
    for a, b in zip(modules_snake, modules_pong, strict=True):
        assert a is b
    for game in ("snake", "pong", "breakout"):
        summary = rollout(
            agent, game, batch_size=2, steps=6, seed=3, sample=True, gamma=0.9
        )
        assert summary["total_reward"].shape == (2,)


def test_conv_screen_driver_is_translation_equivariant() -> None:
    """A pattern learned at one position must transfer to another.

    The linear frontend gives every pixel its own weight, which is the
    measured cause of the motor-game wall (F22). Convolution shares
    weights across positions: shifting the input must shift the features,
    not produce something unrelated.
    """

    from experiments.games_amodal.train import ConvGridEventEncoder

    torch.manual_seed(0)
    encoder = ConvGridEventEncoder(
        channels=3, height=8, width=8, event_width=16, hidden=4
    )
    grid = torch.zeros(1, 3, 8, 8)
    grid[0, 1, 2, 2] = 1.0
    shifted = torch.roll(grid, shifts=(1, 1), dims=(2, 3))
    features = encoder.features(grid)
    shifted_features = encoder.features(shifted)
    # Interior features shift with the content (padding affects edges).
    assert torch.allclose(
        features[0, :, 1:5, 1:5],
        shifted_features[0, :, 2:6, 2:6],
        atol=1e-6,
    )
    event = encoder(grid)
    assert event.payload.shape == (1, 16)


def test_agent_can_select_the_conv_screen_driver() -> None:
    from experiments.games_amodal.train import (
        ConvGridEventEncoder,
        GridEventEncoder,
    )

    torch.manual_seed(0)
    linear = SharedControllerAgent(
        event_width=16, intention_width=8, feedback_width=8, hidden=8,
        shared_drivers=True,
    )
    conv = SharedControllerAgent(
        event_width=16, intention_width=8, feedback_width=8, hidden=8,
        shared_drivers=True, conv_screen=True,
    )
    assert isinstance(linear.runtime.encoders["screen"], GridEventEncoder)
    assert isinstance(conv.runtime.encoders["screen"], ConvGridEventEncoder)
    observation = torch.zeros(2, 3, 8, 8)
    observation[:, 0, 4, 4] = 1.0
    assert conv.runtime.encoders["screen"](observation).payload.shape == (2, 16)
