import torch

from experiments.games_amodal.environments import PongVerifier, SnakeVerifier
from experiments.games_amodal.train import GridEventEncoder


def test_snake_observation_planes_are_disjoint_and_valid() -> None:
    verifier = SnakeVerifier(batch_size=3, seed=11)
    verifier.reset(seed=11)
    grid = verifier.observation()
    assert grid.shape == (3, 3, 8, 8)
    body, head, food = grid[:, 0], grid[:, 1], grid[:, 2]
    assert bool((head <= body).all())
    assert float(head.sum()) == 3.0
    assert float(food.sum()) == 3.0
    assert bool(((body * food) == 0).all())


def test_snake_eats_food_and_grows() -> None:
    verifier = SnakeVerifier(batch_size=1, seed=5)
    verifier.reset(seed=5)
    head = verifier._bodies[0][0]
    verifier._food[0] = (head[0], head[1] + 1)
    before = len(verifier._bodies[0])
    outcome = verifier.step(torch.tensor([1]))
    assert float(outcome.reward[0]) == 1.0
    assert bool(outcome.alive[0])
    assert len(verifier._bodies[0]) == before + 1


def test_snake_dies_on_wall_and_stays_dead() -> None:
    verifier = SnakeVerifier(batch_size=1, seed=3)
    verifier.reset(seed=3)
    dead = False
    for _ in range(20):
        outcome = verifier.step(torch.tensor([1]))
        if not bool(outcome.alive[0]):
            assert float(outcome.reward[0]) == -1.0
            dead = True
            break
    assert dead
    after = verifier.step(torch.tensor([1]))
    assert not bool(after.alive[0])
    assert float(after.reward[0]) == 0.0
    assert float(verifier.observation().sum()) == 0.0


def test_snake_reversal_continues_straight() -> None:
    verifier = SnakeVerifier(batch_size=1, seed=7)
    verifier.reset(seed=7)
    head_before = verifier._bodies[0][0]
    outcome = verifier.step(torch.tensor([3]))
    assert bool(outcome.alive[0])
    head_after = verifier._bodies[0][0]
    assert head_after == (head_before[0], head_before[1] + 1)


def test_pong_paddle_hit_reflects_and_miss_ends_row() -> None:
    verifier = PongVerifier(batch_size=1, seed=9)
    verifier.reset(seed=9)
    verifier._ball[0, 0] = verifier.height - 2
    verifier._ball[0, 1] = 3
    verifier._velocity[0, 0] = 1
    verifier._velocity[0, 1] = 1
    verifier._paddle[0] = 4
    outcome = verifier.step(torch.tensor([1]))
    assert float(outcome.reward[0]) == 1.0
    assert int(verifier._velocity[0, 0]) == -1

    verifier.reset(seed=9)
    verifier._ball[0, 0] = verifier.height - 2
    verifier._ball[0, 1] = 0
    verifier._velocity[0, 0] = 1
    verifier._velocity[0, 1] = -1
    verifier._paddle[0] = verifier.width - verifier.paddle_width
    outcome = verifier.step(torch.tensor([1]))
    assert float(outcome.reward[0]) == -1.0
    assert not bool(outcome.alive[0])


def test_pong_ball_stays_in_bounds_under_random_play() -> None:
    verifier = PongVerifier(batch_size=4, seed=13)
    verifier.reset(seed=13)
    generator = torch.Generator().manual_seed(13)
    for _ in range(64):
        actions = torch.randint(0, verifier.action_count, (4,), generator=generator)
        verifier.step(actions)
        assert bool((verifier._ball[:, 0] >= 0).all())
        assert bool((verifier._ball[:, 0] < verifier.height).all())
        assert bool((verifier._ball[:, 1] >= 0).all())
        assert bool((verifier._ball[:, 1] < verifier.width).all())


def test_grid_event_encoder_emits_validated_opaque_events() -> None:
    verifier = SnakeVerifier(batch_size=2, seed=1)
    verifier.reset(seed=1)
    encoder = GridEventEncoder(channels=3, height=8, width=8, event_width=16)
    event = encoder(verifier.observation())
    assert event.payload.shape == (2, 16)
    assert bool(torch.isfinite(event.payload).all())
