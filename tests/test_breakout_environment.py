import torch

from experiments.games_amodal.environments import BreakoutVerifier
from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    rollout,
)


def test_observation_planes_render_ball_paddle_bricks() -> None:
    verifier = BreakoutVerifier(batch_size=2, seed=1)
    verifier.reset(seed=1)
    grid = verifier.observation()
    assert grid.shape == (2, 3, 8, 8)
    assert float(grid[:, 0].sum()) == 2.0
    assert float(grid[:, 1].sum()) == 2.0 * verifier.paddle_width
    assert float(grid[:, 2].sum()) == 2.0 * verifier.brick_rows * verifier.width


def test_brick_break_rewards_and_reflects() -> None:
    verifier = BreakoutVerifier(batch_size=1, seed=2)
    verifier.reset(seed=2)
    verifier._ball[0, 0] = 2
    verifier._ball[0, 1] = 3
    verifier._velocity[0, 0] = -1
    verifier._velocity[0, 1] = 1
    outcome = verifier.step(torch.tensor([1]))
    assert float(outcome.reward[0]) == 1.0
    assert int(verifier._velocity[0, 0]) == 1
    assert not bool(verifier._bricks[0, 1, 4])


def test_clearing_final_brick_ends_episode() -> None:
    verifier = BreakoutVerifier(batch_size=1, seed=3)
    verifier.reset(seed=3)
    verifier._bricks.fill_(False)
    verifier._bricks[0, 0, 2] = True
    verifier._ball[0, 0] = 1
    verifier._ball[0, 1] = 1
    verifier._velocity[0, 0] = -1
    verifier._velocity[0, 1] = 1
    outcome = verifier.step(torch.tensor([1]))
    assert float(outcome.reward[0]) == 1.0
    assert not bool(outcome.alive[0])


def test_miss_ends_episode_with_penalty() -> None:
    verifier = BreakoutVerifier(batch_size=1, seed=4)
    verifier.reset(seed=4)
    verifier._ball[0, 0] = verifier.height - 2
    verifier._ball[0, 1] = 0
    verifier._velocity[0, 0] = 1
    verifier._velocity[0, 1] = -1
    verifier._paddle[0] = verifier.width - verifier.paddle_width
    outcome = verifier.step(torch.tensor([1]))
    assert float(outcome.reward[0]) == -1.0
    assert not bool(outcome.alive[0])


def test_ball_stays_in_bounds_under_random_play() -> None:
    verifier = BreakoutVerifier(batch_size=4, seed=5)
    verifier.reset(seed=5)
    generator = torch.Generator().manual_seed(5)
    for _ in range(64):
        actions = torch.randint(0, verifier.action_count, (4,), generator=generator)
        verifier.step(actions)
        assert bool((verifier._ball[:, 0] >= 0).all())
        assert bool((verifier._ball[:, 0] < verifier.height).all())
        assert bool((verifier._ball[:, 1] >= 0).all())
        assert bool((verifier._ball[:, 1] < verifier.width).all())


def test_breakout_runs_through_shared_controller() -> None:
    torch.manual_seed(0)
    agent = SharedControllerAgent(
        event_width=32,
        intention_width=16,
        feedback_width=8,
        hidden=16,
        games=("snake", "pong", "breakout"),
    )
    summary = rollout(
        agent, "breakout", batch_size=2, steps=8, seed=6, sample=True, gamma=0.9
    )
    assert summary["total_reward"].shape == (2,)
