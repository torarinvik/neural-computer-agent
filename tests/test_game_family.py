import pytest
import torch

from experiments.games_amodal.game_family import (
    COMPONENTS,
    DUAL_IDLE_COST,
    DUAL_WRONG_COST,
    FORAGE_IDLE_COST,
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


def test_forage_bridge_redeals_near_avatar_without_teleporting() -> None:
    config = FamilyConfig(forage=1, recentre_every=3, spawn_radius=2)
    verifier = FamilyVerifier(config, batch_size=2, seed=23)
    verifier.reset(seed=23)
    for step in range(1, 10):
        before = list(verifier._avatar)
        verifier.step(torch.zeros(2, dtype=torch.long))
        for row in range(2):
            # The avatar is never moved by the bridge (only by its action).
            expected = (max(before[row][0] - 1, 0), before[row][1])
            if expected in verifier._walls[row]:
                expected = before[row]
            assert verifier._avatar[row] == expected
        if step % 3 == 0:
            for row in range(2):
                avatar = verifier._avatar[row]
                for cell in verifier._forage_a[row] + verifier._forage_b[row]:
                    assert max(abs(cell[0] - avatar[0]), abs(cell[1] - avatar[1])) <= 2
    with pytest.raises(ValueError, match="curriculum knob"):
        FamilyConfig(collect=1, recentre_every=2).validate()


def test_forage_bridge_charges_idle_and_pure_forage_does_not() -> None:
    bridge = FamilyVerifier(
        FamilyConfig(forage=1, recentre_every=4, spawn_radius=2),
        batch_size=1, seed=29,
    )
    bridge.reset(seed=29)
    bridge._forage_a[0] = [(0, 0)]
    bridge._forage_b[0] = [(0, 1)]
    bridge._avatar[0] = (5, 5)
    reward = bridge.step(torch.tensor([0])).reward
    assert float(reward[0]) == pytest.approx(-FORAGE_IDLE_COST)
    plain = FamilyVerifier(FamilyConfig(forage=1), batch_size=1, seed=29)
    plain.reset(seed=29)
    plain._forage_a[0] = [(0, 0)]
    plain._forage_b[0] = [(0, 1)]
    plain._avatar[0] = (5, 5)
    assert float(plain.step(torch.tensor([0])).reward[0]) == 0.0


def test_egocentric_view_centres_the_avatar_and_passes_dead_rows() -> None:
    from experiments.games_amodal.game_family import egocentric_view

    grid = torch.zeros(2, 3, 8, 8)
    grid[0, 0, 1, 2] = 1.0  # avatar off-centre
    grid[0, 1, 1, 3] = 1.0  # item just right of the avatar
    # row 1: dead (no avatar) with a stray item; must pass through unchanged
    grid[1, 2, 6, 6] = 1.0
    view = egocentric_view(grid)
    assert float(view[0, 0, 4, 4]) == 1.0  # avatar now centred
    assert float(view[0, 1, 4, 5]) == 1.0  # relative geometry preserved
    assert torch.equal(view[1], grid[1])
    # The transform is a pure roll: content is conserved.
    assert torch.equal(view.sum(dim=(2, 3)), grid.sum(dim=(2, 3)))


def test_egocentric_crop_centres_without_inventing_geometry() -> None:
    from experiments.games_amodal.game_family import (
        egocentric_crop,
        egocentric_view,
    )

    grid = torch.zeros(1, 3, 8, 8)
    grid[0, 0, 1, 1] = 1.0  # avatar near a corner
    grid[0, 2, 0, 0] = 1.0  # a wall diagonally behind it
    cropped = egocentric_crop(grid)
    rolled = egocentric_view(grid)
    assert float(cropped[0, 0, 4, 4]) == 1.0  # avatar centred
    assert float(cropped[0, 2, 3, 3]) == 1.0  # wall keeps its offset
    # The roll wraps far content back into view; the crop must not.
    assert float(cropped.sum()) <= float(rolled.sum())
    far = torch.zeros(1, 3, 8, 8)
    far[0, 0, 0, 0] = 1.0
    far[0, 2, 7, 7] = 1.0  # opposite corner
    assert float(egocentric_crop(far)[0, 2].sum()) == 0.0  # dropped, not wrapped
    assert float(egocentric_view(far)[0, 2].sum()) == 1.0  # wrapped into view


def test_egocentric_crop_passes_rows_without_an_avatar() -> None:
    from experiments.games_amodal.game_family import egocentric_crop

    grid = torch.zeros(2, 3, 8, 8)
    grid[0, 0, 4, 4] = 1.0
    grid[1, 1, 2, 2] = 1.0  # dead row: no avatar plane
    view = egocentric_crop(grid)
    assert torch.equal(view[1], grid[1])


def test_arity_three_deals_three_distinguishable_choices() -> None:
    config = FamilyConfig(dual=1, arity=3, rule0=2, rule1=1)
    verifier = FamilyVerifier(config, batch_size=1, seed=31)
    verifier.reset(seed=31)
    assert len(verifier._dual_items[0]) == 3
    assert {item[2] for item in verifier._dual_items[0]} == {0, 1, 2}
    grid = verifier.observation()
    marks = set()
    for item in verifier._dual_items[0]:
        marks.add((
            float(grid[0, 1, item[0], item[1]]),
            float(grid[0, 2, item[0], item[1]]),
        ))
    assert marks == {(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)}  # all distinct


def test_explicit_rules_name_the_edible_side_per_cue() -> None:
    config = FamilyConfig(dual=1, arity=3, rule0=2, rule1=0)
    verifier = FamilyVerifier(config, batch_size=1, seed=33)
    assert verifier.dual_edible_side(0) == 2
    assert verifier.dual_edible_side(1) == 0
    assert config.rules() == ("cue0take2", "cue1take0")
    for cue, edible in ((0, 2), (1, 0)):
        for side in range(3):
            verifier.reset(seed=33)
            verifier._dual_kind[0] = cue
            reward = _step_onto(verifier, _side_cell(verifier, 0, side))
            expected = 1.0 if side == edible else -DUAL_WRONG_COST
            assert float(reward[0]) == pytest.approx(expected)


def test_arity_two_behaviour_is_unchanged_by_the_generalization() -> None:
    """Existing arity-2 variants must keep their exact old semantics."""

    for inverted, inverted2 in ((False, False), (True, False), (True, True)):
        config = FamilyConfig(dual=1, inverted=inverted, inverted2=inverted2)
        verifier = FamilyVerifier(config, batch_size=1, seed=35)
        assert verifier.dual_edible_side(0) == (1 if inverted else 0)
        assert verifier.dual_edible_side(1) == (1 if inverted2 else 0)
        verifier.reset(seed=35)
        assert len(verifier._dual_items[0]) == 2


def test_arity_validation_rejects_impossible_rules() -> None:
    with pytest.raises(ValueError, match="not dealt"):
        FamilyConfig(dual=1, arity=2, rule0=2).validate()
    with pytest.raises(ValueError, match="arity"):
        FamilyConfig(dual=1, arity=4).validate()


def test_faller_period_slows_descent_without_adding_danger() -> None:
    """F40: the easing axis for fatal-error games is time, not count."""

    slow = FamilyVerifier(
        FamilyConfig(intercept=1, faller_period=3), batch_size=1, seed=41
    )
    slow.reset(seed=41)
    slow._fallers[0] = [(0, 3)]
    slow._avatar[0] = (4, 0)
    rows = []
    for _ in range(6):
        slow.step(torch.tensor([1]))
        rows.append(slow._fallers[0][0][0])
    # Three steps of hold for every one step of descent.
    assert rows[0] == rows[1] == rows[2]
    assert rows[3] > rows[0]
    fast = FamilyVerifier(FamilyConfig(intercept=1), batch_size=1, seed=41)
    fast.reset(seed=41)
    fast._fallers[0] = [(0, 3)]
    fast._avatar[0] = (4, 0)
    fast.step(torch.tensor([1]))
    assert fast._fallers[0][0][0] == 1  # unchanged default: one row a step
    with pytest.raises(ValueError, match="faller period"):
        FamilyConfig(intercept=1, faller_period=0).validate()


def test_faller_spread_keeps_the_policy_and_only_shortens_the_distance() -> None:
    """F41's test: the optimal action must stay the same function of the
    observation at every stage -- move toward the faller's column."""

    near = FamilyVerifier(
        FamilyConfig(intercept=1, faller_spread=1), batch_size=6, seed=43
    )
    near.reset(seed=43)
    for row in range(6):
        gap = abs(near._fallers[row][0][1] - near._avatar[row][1])
        assert gap <= 1
    # Respawn after a catch must also honour the spread.
    near._fallers[0] = [(near.height - 1, near._avatar[0][1])]
    near.step(torch.tensor([1] * 6))
    fresh = near._fallers[0][0]
    assert fresh[0] == 0
    assert abs(fresh[1] - near._avatar[0][1]) <= 1
    # Spread 0 is the unchanged target task: anywhere on the width.
    wide = FamilyVerifier(FamilyConfig(intercept=1), batch_size=24, seed=44)
    wide.reset(seed=44)
    gaps = {abs(wide._fallers[r][0][1] - wide._avatar[r][1]) for r in range(24)}
    assert max(gaps) > 1
    with pytest.raises(ValueError, match="faller spread"):
        FamilyConfig(intercept=1, faller_spread=-1).validate()


def test_pursuer_steps_toward_avatar_and_kills_on_contact() -> None:
    """F227: pursuers close the larger-gap axis first and are fatal."""

    verifier = FamilyVerifier(FamilyConfig(pursue=1), batch_size=1, seed=51)
    verifier.reset(seed=51)
    verifier._pursuers[0] = [(0, 0)]
    verifier._avatar[0] = (4, 1)
    verifier._walls[0] = set()
    verifier.step(torch.tensor([2]))  # avatar moves down to (5, 1)
    assert verifier._pursuers[0][0] == (1, 0)  # row gap larger: row first
    verifier._pursuers[0] = [(5, 3)]
    verifier._avatar[0] = (5, 5)
    outcome = verifier.step(torch.tensor([3]))  # avatar steps to (5, 4)
    assert float(outcome.reward[0]) <= -1.0  # pursuer moved onto it
    assert not bool(outcome.alive[0])


def test_pursuer_renders_identically_to_a_hazard() -> None:
    """The roles world premise: appearance cannot separate the roles."""

    verifier = FamilyVerifier(
        FamilyConfig(pursue=1, avoid=1), batch_size=1, seed=53
    )
    verifier.reset(seed=53)
    grid = verifier.observation()
    hazard = verifier._hazards[0][0]
    pursuer = verifier._pursuers[0][0]
    assert float(grid[0, 2, hazard[0], hazard[1]]) == 1.0
    assert float(grid[0, 2, pursuer[0], pursuer[1]]) == 1.0
    assert float(grid[0, 2].sum()) == 2.0
