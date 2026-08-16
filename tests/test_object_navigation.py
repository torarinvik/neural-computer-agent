from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.navigation_environment import (
    NavigationTask,
    sample_navigation_task,
)
from experiments.brainworkshop_canonical.object_navigation import (
    SceneVerifier,
    best_return,
    identify_goal,
    run_object_navigation,
    shortest,
)
from experiments.brainworkshop_canonical.object_scene import (
    PLACE_COUNT,
    encode_slots,
    render_scene,
    scene_slots,
)
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


def _encoders():
    from experiments.brainworkshop_canonical.controller_pretraining import (
        load_temporal_controller_artifact,
    )
    from experiments.brainworkshop_canonical.current_symbol_acquire import (
        FRONTEND_SEED,
        _machine,
        curated_frontend,
    )

    payload = load_temporal_controller_artifact(CONTROLLER)
    return curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=FRONTEND
    )


# --- the scene has parts ---------------------------------------------------


def test_a_scene_separates_into_one_slot_per_marker() -> None:
    counts: dict[int, int] = {}
    for agent in range(PLACE_COUNT):
        for goal in range(PLACE_COUNT):
            slots = scene_slots(render_scene(agent, goal))
            counts[len(slots)] = counts.get(len(slots), 0) + 1
    # Two markers apart, one where they coincide.
    assert counts == {2: PLACE_COUNT * (PLACE_COUNT - 1), 1: PLACE_COUNT}


def test_slot_order_carries_no_identity() -> None:
    """Ordering is by position, so the index flips as the agent moves past."""

    first = scene_slots(render_scene(0, 5))
    second = scene_slots(render_scene(7, 5))
    assert len(first) == len(second) == 2
    # The goal is the same place in both, but not at the same slot index.
    assert not torch.equal(first[1], second[1])


def test_the_whole_scene_is_ambiguous_and_the_parts_are_not() -> None:
    """The finding that makes atomic reading lossy rather than merely large.

    With identical markers, "agent at a, goal at g" is the same picture as
    "agent at g, goal at a" -- so sixty-four configurations are thirty-six
    pictures, and a reactive reader of pictures cannot tell them apart.
    """

    aliased = sum(
        1
        for agent in range(PLACE_COUNT)
        for goal in range(PLACE_COUNT)
        if agent != goal
        and torch.equal(render_scene(agent, goal), render_scene(goal, agent))
    )
    assert aliased == PLACE_COUNT * (PLACE_COUNT - 1)


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_a_slot_names_a_place_whatever_role_it_plays() -> None:
    """Both markers share a colour on purpose.

    Drawn differently, "goal at place three" and "self at place three" encode
    to different events and the agent could never tell it had arrived.
    """

    from experiments.brainworkshop_canonical.counter_state_programs import (
        nearest_cluster,
    )
    from experiments.brainworkshop_canonical.prototype_templates import (
        cluster_events,
        estimated_tolerance,
    )

    encoders = _encoders()
    events = torch.cat(
        [
            encode_slots(encoders, render_scene(agent, goal))
            for agent in range(PLACE_COUNT)
            for goal in range(PLACE_COUNT)
        ]
    )
    tolerance = estimated_tolerance(events)
    assert tolerance is not None
    clusters = cluster_events(events, tolerance=tolerance, maximum_clusters=256)
    # Eight places, not sixty-four scenes.
    assert int(clusters.shape[0]) == PLACE_COUNT

    for agent in range(PLACE_COUNT):
        for goal in range(PLACE_COUNT):
            read = {
                int(index)
                for index in nearest_cluster(
                    encode_slots(encoders, render_scene(agent, goal)), clusters
                )
            }
            occupied = {
                int(
                    nearest_cluster(
                        encode_slots(encoders, render_scene(place, place)), clusters
                    )[0]
                )
                for place in {agent, goal}
            }
            assert read == occupied, (agent, goal)


# --- working out which object you are --------------------------------------


def test_the_goal_is_the_thing_that_never_moved() -> None:
    assert identify_goal([(1, 5), (2, 5), (5,), (3, 5)]) == 5
    # Never disambiguated: the agent stayed put the whole time.
    assert identify_goal([(1, 5), (1, 5)]) is None
    # Nothing in common at all.
    assert identify_goal([(1, 5), (2, 3)]) is None
    assert identify_goal([]) is None


# --- the ceiling -----------------------------------------------------------


def test_the_best_return_does_not_assume_the_agent_can_stand_still() -> None:
    """The bug that made the ceiling unreachable and read as agent failure."""

    # Two actions, and neither holds at the goal: 1 <-> 2 forever.
    task = NavigationTask(
        transitions=((1, 2, 1), (2, 2, 0)), goal=2, start=0, place_count=3
    ).validate()
    assert shortest(task, 0, 2) == 1
    # Arrive on step one, but cannot stay, so half the steps at best.
    scored = best_return(task, 0, 2, 10)
    assert 0.0 < scored < 1.0
    # Where a hold does exist, arriving and staying is achievable.
    holds = NavigationTask(
        transitions=((1, 2, 2), (0, 0, 2)), goal=2, start=0, place_count=3
    ).validate()
    assert best_return(holds, 0, 2, 10) == pytest.approx(0.9)


def test_the_verifier_only_accepts_actions_in_the_protocol() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    verifier = SceneVerifier(task, start=0, goal=3, steps=4)
    verifier.observation()
    with pytest.raises(ValueError, match="outside the protocol"):
        verifier.score(torch.tensor([99], dtype=torch.long))


# --- end to end ------------------------------------------------------------


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_a_goal_can_be_given_and_transfers_to_goals_never_trained_on(
    tmp_path,
) -> None:
    """The claim, with the controls that can refute it."""

    before = sha256_file(BANK)
    report = run_object_navigation(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        tasks=3,
        explore_episodes=24,
    )

    # The scene decomposes into places, not pictures.
    assert report["mean_slot_alphabet"] == PLACE_COUNT
    assert report["mean_scene_alphabet"] > PLACE_COUNT

    # On goals it trained on, both agents work.
    assert report["trained_object"] > 0.7 * report["trained_optimal"]
    assert report["trained_scene"] > 0.5 * report["trained_optimal"]

    # On goals it was never trained on, only the object agent transfers.
    assert report["held_out_object"] > 0.7 * report["held_out_optimal"]
    assert report["held_out_scene"] < 0.4 * report["held_out_optimal"]

    # It is really using the goal it was shown: pointed at some other place it
    # does no better than acting blindly.
    assert report["held_out_wrong_goal"] < 0.5 * report["held_out_object"]
    assert report["held_out_random"] < 0.5 * report["held_out_object"]

    # Working out which marker is the instruction is cheap, not free.
    assert report["held_out_told"] >= report["held_out_object"]
    assert report["held_out_told"] - report["held_out_object"] < 0.15

    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before
