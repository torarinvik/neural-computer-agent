from __future__ import annotations

import pytest

from experiments.brainworkshop_canonical.slot_alignment import (
    MAXIMUM_TRACKS,
    Tracker,
    TrackHistory,
    assign,
    beam_track,
    displacement_costs,
    identify_roles,
    persistence_costs,
    self_posterior,
)

# --- assignment -------------------------------------------------------------


def test_every_part_must_receive_a_track() -> None:
    """A part with no track would be a marker that arrived from nowhere."""

    # Both tracks would rather be part 0, but part 1 has to be explained.
    costs = [[0.0, 5.0], [0.0, 1.0]]
    assert assign(costs) == (0, 1)


def test_two_tracks_may_share_one_part() -> None:
    """Objects here merge, so the matching is a function, not a permutation."""

    costs = [[1.0], [2.0]]
    assert assign(costs) == (0, 0)


def test_more_parts_than_tracks_is_refused() -> None:
    with pytest.raises(ValueError, match="appeared from nowhere"):
        assign([[0.0, 1.0, 2.0]])


def test_exhaustive_assignment_is_bounded() -> None:
    with pytest.raises(ValueError, match="too many tracks"):
        assign([[0.0]] * (MAXIMUM_TRACKS + 1))


def test_displacement_prefers_the_nearest_part() -> None:
    costs = displacement_costs([(0.0, 0.0), (10.0, 10.0)], [(0.0, 1.0), (10.0, 9.0)])
    assert assign(costs) == (0, 1)


def test_minimum_change_beats_displacement_when_things_teleport() -> None:
    """The measured failure of AlignNet's prior in a world with no continuity."""

    # Track 0 sits at symbol 7 and has not moved; track 1 is the agent, which
    # jumped clear across the grid. By centroid alone the agent's new position
    # is nearer track 0.
    track_centroids = [(0.0, 0.0), (30.0, 30.0)]
    track_symbols = [7, 3]
    part_centroids = [(1.0, 1.0), (2.0, 2.0)]
    part_symbols = [7, 5]
    assert assign(displacement_costs(track_centroids, part_centroids)) == (0, 1)
    costs = persistence_costs(
        track_centroids, track_symbols, part_centroids, part_symbols
    )
    # The static track keeps its symbol; the mover takes the changed one.
    assert assign(costs) == (0, 1)


def test_memory_breaks_the_tie_after_two_markers_separate() -> None:
    """Both tracks last saw the same symbol, so change alone cannot decide."""

    track_centroids = [(0.0, 0.0), (0.0, 0.0)]
    track_symbols = [4, 4]
    # Track 0 has spent the episode at place 4; track 1 has wandered.
    memories = [{4: 9}, {0: 3, 1: 3, 4: 1}]
    part_symbols = [4, 6]
    part_centroids = [(0.0, 0.0), (20.0, 20.0)]
    costs = persistence_costs(
        track_centroids, track_symbols, part_centroids, part_symbols, memories
    )
    assert assign(costs) == (0, 1)


# --- the tracker ------------------------------------------------------------


def test_a_tracker_grows_when_something_separates() -> None:
    tracker = Tracker.started((((0.0, 0.0), 3),))
    assert tracker.count == 1
    tracker.update((((0.0, 0.0), 3), ((9.0, 9.0), 5)))
    assert tracker.count == 2
    assert set(tracker.reading()) == {3, 5}


def test_a_tracker_keeps_a_static_marker_on_its_place() -> None:
    parts = [((0.0, 0.0), 2), ((9.0, 9.0), 6)]
    tracker = Tracker.started(parts)
    goal = tracker.symbols.index(2)
    for symbol in (5, 1, 7, 2, 4):
        tracker.update((((0.0, 0.0), 2), ((9.0, 9.0), symbol)), action=0)
        assert tracker.symbols[goal] == 2


# --- identity ---------------------------------------------------------------


def _history(steps) -> TrackHistory:
    history = TrackHistory()
    for symbol, action, following in steps:
        history.observe(symbol, action, following)
    return history


def test_controllability_separates_the_agent_from_everything_else() -> None:
    """Accuracy and description length both fail here; a matched contrast does.

    Four tracks over one short stream: the agent, a marker walking at random, a
    marker on a fixed circuit, and a marker that never moves.
    """

    table = {(s, a): (s * 3 + a * 5) % 8 for s in range(8) for a in range(4)}
    walk = [3, 7, 1, 1, 6, 0, 4, 4, 2, 5, 5, 0, 3, 6, 6, 2, 7, 1, 0]
    agent_steps, walker_steps, cycler_steps, static_steps = [], [], [], []
    place = 0
    walker = 0
    cycler = 0
    for index in range(18):
        action = index % 4 if index % 5 else (index + 1) % 4
        agent_steps.append((place, action, table[(place, action)]))
        place = table[(place, action)]
        walker_steps.append((walker, action, walk[index]))
        walker = walk[index]
        cycler_steps.append((cycler, action, (cycler + 1) % 8))
        cycler = (cycler + 1) % 8
        static_steps.append((3, action, 3))

    agent = _history(agent_steps).evidence(alphabet=8)
    walker_evidence = _history(walker_steps).evidence(alphabet=8)
    cycler_evidence = _history(cycler_steps).evidence(alphabet=8)
    static = _history(static_steps).evidence(alphabet=8)

    assert agent.controllability > 0.5
    # A random walker is not predictable, and a circuit is -- but neither is
    # *responsive*, which is the property being measured.
    assert walker_evidence.controllability < 0.1
    assert cycler_evidence.controllability < 0.1
    assert static.controllability < 0.1
    # The plausible wrong measure would have chosen the circuit over the agent.
    assert cycler_evidence.with_action >= agent.with_action


def test_a_track_that_never_moved_is_the_target() -> None:
    table = {(s, a): (s * 3 + a * 5) % 8 for s in range(8) for a in range(4)}
    place = 0
    agent_steps = []
    static_steps = []
    for index in range(16):
        action = index % 4
        agent_steps.append((place, action, table[(place, action)]))
        place = table[(place, action)]
        static_steps.append((6, action, 6))
    roles = identify_roles(
        [_history(agent_steps), _history(static_steps)], alphabet=8
    )
    assert roles.own == 0
    assert roles.target == 1
    assert roles.loose == ()


def test_roles_refuse_rather_than_guess_without_evidence() -> None:
    roles = identify_roles([], alphabet=8)
    assert roles.own is None and roles.targets == ()


def test_a_self_model_refuses_when_it_explains_no_track() -> None:
    histories = [
        _history([(0, 0, 1), (1, 0, 2), (2, 0, 3)]),
        _history([(4, 1, 5), (5, 1, 6), (6, 1, 7)]),
    ]

    def incompatible(_symbol: int, _action: int):
        return {0: 100.0}

    assert self_posterior(histories, incompatible, alphabet=8) == pytest.approx(
        [0.5, 0.5]
    )


def test_self_posterior_is_a_normalized_nonnegative_belief() -> None:
    histories = [
        _history([(0, 0, 1), (1, 1, 2), (2, 0, 3)]),
        _history([(4, 0, 4), (4, 1, 4), (4, 0, 4)]),
    ]

    def successor(symbol: int, action: int):
        return {(0, 0): {1: 5.0}, (1, 1): {2: 5.0}}.get((symbol, action))

    posterior = self_posterior(histories, successor, alphabet=8)
    assert all(value >= 0.0 for value in posterior)
    assert sum(posterior) == pytest.approx(1.0)


def test_identical_dynamics_produce_abstention_not_a_tie_broken_name() -> None:
    history = _history([(0, 0, 1), (1, 1, 3), (3, 0, 2), (2, 1, 0)])
    counts = {
        (0, 0): {1: 10.0},
        (1, 1): {3: 10.0},
        (3, 0): {2: 10.0},
        (2, 1): {0: 10.0},
    }

    def successor(symbol: int, action: int):
        return counts.get((symbol, action))

    assert self_posterior([history, history], successor, alphabet=8) == pytest.approx(
        [0.5, 0.5]
    )


def test_controllability_breaks_a_likelihood_tie() -> None:
    table = {(s, a): (s * 3 + a * 5) % 8 for s in range(8) for a in range(4)}
    agent_steps = []
    circuit_steps = []
    place = circuit = 0
    counts: dict[tuple[int, int], dict[int, float]] = {}
    for index in range(20):
        action = index % 4 if index % 5 else (index + 1) % 4
        after = table[(place, action)]
        circuit_after = (circuit + 1) % 8
        agent_steps.append((place, action, after))
        circuit_steps.append((circuit, action, circuit_after))
        for symbol, following in ((place, after), (circuit, circuit_after)):
            cell = counts.setdefault((symbol, action), {})
            cell[following] = cell.get(following, 0.0) + 20.0
        place, circuit = after, circuit_after

    histories = [_history(agent_steps), _history(circuit_steps)]

    def successor(symbol: int, action: int):
        return counts.get((symbol, action))

    likelihood = self_posterior(
        histories, successor, alphabet=8, controllability_weight=0.0
    )
    guarded = self_posterior(histories, successor, alphabet=8)
    assert guarded[0] > likelihood[0]
    assert guarded[0] > 0.75


# --- search -----------------------------------------------------------------


def test_search_recovers_correspondence_when_two_things_move() -> None:
    """Minimum change cannot do this; scoring correspondences by cost can.

    The agent follows a fixed table; a distractor walks its own circuit. Both
    move at every step, so "fewest things moved" has nothing to choose between
    the correct pairing and the swapped one.
    """

    table = {(s, a): (s * 5 + a * 3) % 8 for s in range(8) for a in range(4)}
    place, other = 0, 4
    readings = []
    actions = []
    for index in range(24):
        readings.append((((0.0, float(place)), place), ((1.0, float(other)), other)))
        action = (index * 3 + index // 4) % 4
        actions.append(action)
        place = table[(place, action)]
        other = (other + 3) % 8
    readings.append((((0.0, float(place)), place), ((1.0, float(other)), other)))

    trace = beam_track(readings, actions, alphabet=8)
    assert len(trace) == len(readings)
    # Some track follows the agent exactly the whole way through.
    truth = []
    place, other = 0, 4
    for index in range(24):
        truth.append(place)
        place = table[(place, actions[index])]
        other = (other + 3) % 8
    truth.append(place)
    followed = max(
        sum(1 for step, row in enumerate(trace) if row[track] == truth[step])
        for track in range(len(trace[0]))
    )
    assert followed == len(truth)
