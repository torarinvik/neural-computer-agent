import torch

from experiments.games_amodal.game_family import (
    COMPONENTS,
    FamilyConfig,
    FamilyVerifier,
    compositional_split,
    family_variants,
)


def test_variant_enumeration_covers_all_components() -> None:
    variants = family_variants()
    active = {component for v in variants for component in v.active()}
    assert active == set(COMPONENTS)
    names = [v.name for v in variants]
    assert len(names) == len(set(names))
    assert len(variants) > 12


def test_compositional_split_keeps_support() -> None:
    variants = family_variants()
    train, holdout = compositional_split(variants, seed=1)
    assert holdout and len(train) + len(holdout) == len(variants)
    support = {
        (component, getattr(v, component))
        for v in train
        for component in COMPONENTS
        if getattr(v, component)
    }
    for v in holdout:
        assert len(v.active()) > 1
        needed = {
            (component, getattr(v, component))
            for component in COMPONENTS
            if getattr(v, component)
        }
        assert needed <= support


def test_collect_component_rewards_and_respawns() -> None:
    verifier = FamilyVerifier(
        FamilyConfig(collect=1), batch_size=1, seed=3
    )
    verifier.reset(seed=3)
    food = verifier._food[0][0]
    verifier._avatar[0] = (food[0], max(food[1] - 1, 0))
    if verifier._avatar[0] == food:
        verifier._avatar[0] = (food[0], food[1] + 1)
    action = 1 if verifier._avatar[0][1] < food[1] else 3
    outcome = verifier.step(torch.tensor([action]))
    assert float(outcome.reward[0]) == 1.0
    assert len(verifier._food[0]) == 1
    assert verifier._food[0][0] != food or verifier._avatar[0] != food


def test_hazard_contact_ends_episode() -> None:
    verifier = FamilyVerifier(FamilyConfig(avoid=1), batch_size=1, seed=4)
    verifier.reset(seed=4)
    hazard = verifier._hazards[0][0]
    verifier._avatar[0] = (hazard[0], hazard[1] - hazard[2])
    stay_target = verifier._avatar[0]
    verifier._walls[0] = set()
    action = 0 if stay_target[0] > 0 else 2
    outcome = verifier.step(torch.tensor([action]))
    if not bool(outcome.alive[0]):
        assert float(outcome.reward[0]) <= -1.0
    else:
        assert bool(outcome.alive[0])


def test_navigate_walls_block_and_goal_rewards() -> None:
    verifier = FamilyVerifier(
        FamilyConfig(navigate=True), batch_size=1, seed=5
    )
    verifier.reset(seed=5)
    assert verifier._walls[0]
    wall = next(iter(verifier._walls[0]))
    verifier._avatar[0] = (wall[0], wall[1] - 1)
    outcome = verifier.step(torch.tensor([1]))
    assert verifier._avatar[0] == (wall[0], wall[1] - 1)
    assert bool(outcome.alive[0])
    goal = verifier._goal[0]
    verifier._avatar[0] = (goal[0], goal[1] - 1) if goal[1] > 0 else (
        goal[0], goal[1] + 1
    )
    action = 1 if verifier._avatar[0][1] < goal[1] else 3
    verifier._walls[0] = set()
    outcome = verifier.step(torch.tensor([action]))
    assert float(outcome.reward[0]) == 1.0


def test_intercept_catch_and_miss() -> None:
    verifier = FamilyVerifier(
        FamilyConfig(intercept=1), batch_size=1, seed=6
    )
    verifier.reset(seed=6)
    verifier._fallers[0] = [(verifier.height - 1, 2)]
    verifier._avatar[0] = (verifier.height - 1, 1)
    outcome = verifier.step(torch.tensor([1]))
    assert float(outcome.reward[0]) == 1.0
    assert bool(outcome.alive[0])
    verifier._fallers[0] = [(verifier.height - 1, 0)]
    verifier._avatar[0] = (verifier.height - 1, verifier.width - 1)
    outcome = verifier.step(torch.tensor([0]))
    assert float(outcome.reward[0]) <= -1.0
    assert not bool(outcome.alive[0])


def test_observation_planes_and_random_play_bounds() -> None:
    config = FamilyConfig(collect=2, intercept=1, avoid=1, navigate=True)
    verifier = FamilyVerifier(config, batch_size=4, seed=7)
    verifier.reset(seed=7)
    grid = verifier.observation()
    assert grid.shape == (4, 3, 8, 8)
    assert float(grid[:, 0].sum()) == 4.0
    generator = torch.Generator().manual_seed(7)
    for _ in range(50):
        actions = torch.randint(0, 4, (4,), generator=generator)
        verifier.step(actions)
        for row in range(4):
            avatar = verifier._avatar[row]
            assert 0 <= avatar[0] < 8 and 0 <= avatar[1] < 8
            assert avatar not in verifier._walls[row]
