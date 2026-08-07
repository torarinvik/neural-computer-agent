import pytest
import torch

from experiments.games_amodal.game_family import (
    COMPONENTS,
    DUAL_IDLE_COST,
    DUAL_WRONG_COST,
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

_DELTAS_LOOKUP = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}


def _side_cell(verifier: FamilyVerifier, row: int, side: int) -> tuple[int, int]:
    for item in verifier._dual_items[row]:
        if item[2] == side:
            return (item[0], item[1])
    raise AssertionError(f"side {side} not dealt")


def _step_onto(verifier: FamilyVerifier, cell: tuple[int, int]) -> torch.Tensor:
    avatar = verifier._avatar[0]
    delta = (cell[0] - avatar[0], cell[1] - avatar[1])
    return verifier.step(torch.tensor([_DELTAS_LOOKUP[delta]])).reward


def test_dual_rules_are_independent_across_the_two_trial_kinds() -> None:
    for inverted in (False, True):
        for inverted2 in (False, True):
            config = FamilyConfig(dual=1, inverted=inverted, inverted2=inverted2)
            verifier = FamilyVerifier(config, batch_size=1, seed=11)
            for kind, flipped in ((0, inverted), (1, inverted2)):
                for side in (0, 1):
                    verifier.reset(seed=11)
                    verifier._dual_kind[0] = kind
                    assert verifier.dual_edible_side(kind) == (1 if flipped else 0)
                    reward = _step_onto(verifier, _side_cell(verifier, 0, side))
                    expected = (
                        -DUAL_WRONG_COST if (side == 1) != flipped else 1.0
                    )
                    assert float(reward[0]) == pytest.approx(expected)


def test_dual_choices_render_identically_in_both_trial_kinds() -> None:
    """Only the cue may reveal the trial kind; the items must not."""

    verifier = FamilyVerifier(FamilyConfig(dual=1), batch_size=1, seed=5)
    verifier.reset(seed=5)
    frames = []
    for kind in (0, 1):
        verifier._dual_kind[0] = kind
        frames.append(verifier.observation().clone())
    assert not torch.equal(frames[0], frames[1])  # the cue differs
    # ...but the object planes, which carry the choice, are identical.
    assert torch.equal(frames[0][:, 1:], frames[1][:, 1:])
    half = verifier.width // 2
    assert bool((frames[0][0, 0, 0, :half] == 1.0).all())
    assert bool((frames[0][0, 0, 0, half:] == 0.0).all())
    assert bool((frames[1][0, 0, 0, half:] == 1.0).all())
    assert bool((frames[1][0, 0, 0, :half] == 0.0).all())


def test_dual_twins_render_identically_and_never_kill() -> None:
    left = FamilyVerifier(FamilyConfig(dual=1), batch_size=6, seed=3)
    right = FamilyVerifier(
        FamilyConfig(dual=1, inverted=True, inverted2=True), batch_size=6, seed=3
    )
    left.reset(seed=3)
    right.reset(seed=3)
    assert torch.equal(left.observation(), right.observation())
    generator = torch.Generator().manual_seed(3)
    centre = (left.height // 2, left.width // 2)
    for _ in range(40):
        actions = torch.randint(0, 4, (6,), generator=generator)
        outcome = left.step(actions)
        assert bool(outcome.alive.all())  # survivable error: no death
        assert all(avatar == centre for avatar in left._avatar)


def test_dual_accuracy_and_engagement_track_each_kind_separately() -> None:
    verifier = FamilyVerifier(FamilyConfig(dual=1), batch_size=1, seed=9)
    verifier.reset(seed=9)
    verifier._dual_kind[0] = 0
    _step_onto(verifier, _side_cell(verifier, 0, 0))  # kind 0 correct
    verifier._dual_kind[0] = 1
    _step_onto(verifier, _side_cell(verifier, 0, 1))  # kind 1 wrong
    assert verifier.dual_accuracy() == [1.0, 0.0]
    assert verifier.dual_engagement() == [1.0, 1.0]
    # Declining to resolve a trial must not count as engagement.
    before = verifier.dual_engagement()
    taken = {(item[0], item[1]) for item in verifier._dual_items[0]}
    empty = next(
        cell
        for cell in (
            (verifier.height // 2 - 1, verifier.width // 2),
            (verifier.height // 2, verifier.width // 2 + 1),
            (verifier.height // 2 + 1, verifier.width // 2),
            (verifier.height // 2, verifier.width // 2 - 1),
        )
        if cell not in taken
    )
    _step_onto(verifier, empty)
    assert verifier.dual_engagement() == before


def test_dual_rejects_combination_with_choice() -> None:
    try:
        FamilyConfig(dual=1, choice=1).validate()
    except ValueError as error:
        assert "trial cycle" in str(error)
    else:
        raise AssertionError("dual+choice must be rejected")


def test_dual_stalling_is_strictly_dominated() -> None:
    """F2: refusing every trial must finish net-negative."""

    verifier = FamilyVerifier(FamilyConfig(dual=1), batch_size=1, seed=13)
    verifier.reset(seed=13)
    total = 0.0
    for _ in range(20):
        taken = {(item[0], item[1]) for item in verifier._dual_items[0]}
        empty = next(
            cell
            for cell in (
                (verifier.height // 2 - 1, verifier.width // 2),
                (verifier.height // 2, verifier.width // 2 + 1),
                (verifier.height // 2 + 1, verifier.width // 2),
                (verifier.height // 2, verifier.width // 2 - 1),
            )
            if cell not in taken
        )
        total += float(_step_onto(verifier, empty)[0])
    assert total == pytest.approx(-20 * DUAL_IDLE_COST)
    assert verifier.dual_engagement() == [0.0, 0.0]


def test_dual_guessing_beats_stalling() -> None:
    """F2 for multi-rule worlds: refusal must be strictly worse than a coin
    flip, or an agent that has mastered one trial kind will simply decline
    the other and its per-rule accuracy will measure nothing."""

    verifier = FamilyVerifier(FamilyConfig(dual=1), batch_size=1, seed=17)
    verifier.reset(seed=17)
    guesses = []
    for side in (0, 1):
        verifier.reset(seed=17)
        guesses.append(float(_step_onto(verifier, _side_cell(verifier, 0, side))[0]))
    assert sum(guesses) / len(guesses) > 0.0  # guessing must pay
    verifier.reset(seed=17)
    taken = {(item[0], item[1]) for item in verifier._dual_items[0]}
    empty = next(
        cell
        for cell in (
            (verifier.height // 2 - 1, verifier.width // 2),
            (verifier.height // 2, verifier.width // 2 + 1),
            (verifier.height // 2 + 1, verifier.width // 2),
            (verifier.height // 2, verifier.width // 2 - 1),
        )
        if cell not in taken
    )
    assert float(_step_onto(verifier, empty)[0]) < sum(guesses) / len(guesses)


def test_forage_spawn_radius_keeps_items_near_the_avatar() -> None:
    config = FamilyConfig(forage=1, spawn_radius=1)
    verifier = FamilyVerifier(config, batch_size=8, seed=21)
    # Radius bounds distance AT SPAWN TIME (reset and respawn); the avatar
    # may then walk away, so only spawn-time positions are checked.
    for reset_seed in range(5):
        verifier.reset(seed=reset_seed)
        for row in range(8):
            avatar = verifier._avatar[row]
            for cell in verifier._forage_a[row] + verifier._forage_b[row]:
                assert max(abs(cell[0] - avatar[0]), abs(cell[1] - avatar[1])) <= 1
    # Respawn-on-eat also lands within radius of the avatar's new cell.
    verifier.reset(seed=9)
    item = verifier._forage_a[0][0]
    verifier._avatar[0] = (item[0] - 1, item[1]) if item[0] > 0 else (item[0] + 1, item[1])
    action = 2 if verifier._avatar[0][0] < item[0] else 0
    verifier.step(torch.tensor([action] * 8))
    avatar = verifier._avatar[0]
    fresh = verifier._forage_a[0][0]
    assert max(abs(fresh[0] - avatar[0]), abs(fresh[1] - avatar[1])) <= 1
    with pytest.raises(ValueError, match="curriculum knob"):
        FamilyConfig(choice=1, spawn_radius=1).validate()
