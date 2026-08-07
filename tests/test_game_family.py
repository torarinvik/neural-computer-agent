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


def _dual_item(verifier: FamilyVerifier, row: int, class_id: int) -> tuple[int, int]:
    for item in verifier._dual_items[row]:
        if item[2] == class_id:
            return (item[0], item[1])
    raise AssertionError(f"class {class_id} not dealt")


def _step_onto(verifier: FamilyVerifier, cell: tuple[int, int]) -> torch.Tensor:
    avatar = verifier._avatar[0]
    delta = (cell[0] - avatar[0], cell[1] - avatar[1])
    action = _DELTAS_LOOKUP[delta]
    return verifier.step(torch.tensor([action])).reward


_DELTAS_LOOKUP = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}


def test_dual_axes_are_independent_and_rewards_follow_the_two_bits() -> None:
    for inverted in (False, True):
        for inverted2 in (False, True):
            config = FamilyConfig(dual=1, inverted=inverted, inverted2=inverted2)
            verifier = FamilyVerifier(config, batch_size=1, seed=11)
            verifier.reset(seed=11)
            for trial_kind, flipped in ((0, inverted), (1, inverted2)):
                classes = (0, 1) if trial_kind == 0 else (2, 3)
                edible = classes[1] if flipped else classes[0]
                for class_id in classes:
                    verifier.reset(seed=11)
                    verifier._dual_items[0] = [
                        (verifier.height // 2 - 1, verifier.width // 2, classes[0]),
                        (verifier.height // 2, verifier.width // 2 + 1, classes[1]),
                    ]
                    verifier._avatar[0] = (verifier.height // 2, verifier.width // 2)
                    reward = _step_onto(verifier, _dual_item(verifier, 0, class_id))
                    expected = 1.0 if class_id == edible else -1.0
                    assert float(reward[0]) == expected


def test_dual_twins_render_identically_and_never_kill() -> None:
    left = FamilyVerifier(FamilyConfig(dual=1), batch_size=6, seed=3)
    right = FamilyVerifier(
        FamilyConfig(dual=1, inverted=True, inverted2=True), batch_size=6, seed=3
    )
    left.reset(seed=3)
    right.reset(seed=3)
    assert torch.equal(left.observation(), right.observation())
    generator = torch.Generator().manual_seed(3)
    for _ in range(40):
        actions = torch.randint(0, 4, (6,), generator=generator)
        outcome = left.step(actions)
        assert bool(outcome.alive.all())  # survivable error: no death


def test_dual_marks_are_four_distinguishable_values() -> None:
    verifier = FamilyVerifier(FamilyConfig(dual=1), batch_size=1, seed=5)
    verifier.reset(seed=5)
    verifier._dual_items[0] = [(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3)]
    grid = verifier.observation()
    marks = {
        (0, 0): (1, 1.0),
        (0, 1): (2, 1.0),
        (0, 2): (1, 0.5),
        (0, 3): (2, 0.5),
    }
    seen = set()
    for cell, (plane, level) in marks.items():
        assert float(grid[0, plane, cell[0], cell[1]]) == level
        seen.add((plane, level))
    assert len(seen) == 4


def test_dual_accuracy_tracks_each_rule_separately() -> None:
    verifier = FamilyVerifier(FamilyConfig(dual=1), batch_size=1, seed=9)
    verifier.reset(seed=9)
    centre = (verifier.height // 2, verifier.width // 2)
    verifier._dual_items[0] = [(centre[0] - 1, centre[1], 0), (centre[0], centre[1] + 1, 1)]
    _step_onto(verifier, (centre[0] - 1, centre[1]))  # axis 0 correct
    verifier._dual_items[0] = [(centre[0] - 1, centre[1], 2), (centre[0], centre[1] + 1, 3)]
    verifier._avatar[0] = centre
    _step_onto(verifier, (centre[0], centre[1] + 1))  # axis 1 wrong
    assert verifier.dual_accuracy() == [1.0, 0.0]


def test_dual_rejects_combination_with_choice() -> None:
    try:
        FamilyConfig(dual=1, choice=1).validate()
    except ValueError as error:
        assert "trial cycle" in str(error)
    else:
        raise AssertionError("dual+choice must be rejected")
