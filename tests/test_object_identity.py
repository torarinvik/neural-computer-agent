from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.navigation_environment import (
    sample_navigation_task,
)
from experiments.brainworkshop_canonical.object_identity import (
    ARMS,
    CONDITIONS,
    DistractorVerifier,
    persistence_identify,
    run_object_identity,
)
from experiments.brainworkshop_canonical.object_scene import scene_parts
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


# --- what the old rule could and could not do ------------------------------


def test_elimination_is_exact_with_two_markers() -> None:
    observations = [(1, 5), (2, 5), (5,), (3, 5)]
    own, goal = persistence_identify(observations, observations[-1])
    assert goal == 5
    assert own == 3


def test_elimination_becomes_a_coin_flip_with_three() -> None:
    """The goal half survives a distractor; the self half does not.

    Five is still the only place in every frame, so intersection finds it. But
    the last frame now holds two non-goal markers and the rule takes whichever
    sorted first, which is not the agent except by luck.
    """

    observations = [(1, 4, 5), (2, 5, 7), (0, 3, 5)]
    own, goal = persistence_identify(observations, observations[-1])
    assert goal == 5
    assert own in (0, 3)


def test_elimination_refuses_when_two_things_stayed_put() -> None:
    observations = [(1, 4, 5), (2, 4, 5)]
    assert persistence_identify(observations, observations[-1]) == (None, None)


# --- the environment --------------------------------------------------------


def test_the_distractor_moves_without_being_asked() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    for condition in ("random_walk", "cycling"):
        verifier = DistractorVerifier(
            task, start=0, goal=3, steps=8, condition=condition, seed=7
        )
        places = []
        while not verifier.done:
            places.append(verifier.distractor)
            verifier.score(torch.tensor([0], dtype=torch.long))
        assert len(set(places)) > 1


def test_the_distractor_changes_neither_the_dynamics_nor_the_reward() -> None:
    """It is there to break identification, not to make the task harder."""

    task = sample_navigation_task(seed=9000)
    assert task is not None
    scores = {}
    for condition in CONDITIONS:
        verifier = DistractorVerifier(
            task, start=0, goal=3, steps=10, condition=condition, seed=7
        )
        total = 0.0
        step = 0
        while not verifier.done:
            total += verifier.score(
                torch.tensor([step % task.action_count], dtype=torch.long)
            )
            step += 1
        scores[condition] = (total, verifier.place)
    assert len(set(scores.values())) == 1


def test_a_cycling_distractor_is_a_permutation_of_every_place() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    verifier = DistractorVerifier(
        task, start=0, goal=3, steps=24, condition="cycling", seed=7
    )
    places = set()
    while not verifier.done:
        places.add(verifier.distractor)
        verifier.score(torch.tensor([0], dtype=torch.long))
    assert len(places) == 8


def test_three_markers_render_as_three_parts() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    verifier = DistractorVerifier(
        task, start=0, goal=3, steps=4, condition="cycling", seed=7
    )
    assert len(scene_parts(verifier.observation())) <= 3
    plain = DistractorVerifier(
        task, start=0, goal=3, steps=4, condition="none", seed=7
    )
    assert len(scene_parts(plain.observation())) == 2


def test_the_verifier_only_accepts_actions_in_the_protocol() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    verifier = DistractorVerifier(
        task, start=0, goal=3, steps=4, condition="none", seed=7
    )
    with pytest.raises(ValueError, match="outside the protocol"):
        verifier.score(torch.tensor([99], dtype=torch.long))


def test_an_unknown_condition_is_refused() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    with pytest.raises(ValueError, match="unknown distractor condition"):
        DistractorVerifier(
            task, start=0, goal=3, steps=4, condition="teleporting", seed=7
        )


# --- end to end -------------------------------------------------------------


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_search_finds_the_agent_where_persistence_cannot(tmp_path) -> None:
    before = sha256_file(BANK)
    report = run_object_identity(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        tasks=2,
        explore_episodes=12,
    )
    conditions = report["conditions"]

    # The control: with two markers nothing has been broken in passing.
    plain = conditions["none"]
    for arm in ("hybrid", "search", "alignment", "persistence"):
        assert plain[f"{arm}_own_accuracy"] == 1.0
    assert plain["search_track_fidelity"] > plain["alignment_track_fidelity"]

    for condition in ("random_walk", "cycling"):
        block = conditions[condition]
        # A third moving marker breaks elimination, exactly as predicted.
        assert block["persistence_own_accuracy"] < 0.6
        # Searching over correspondences recovers a good part of it.
        assert block["search_own_accuracy"] > block["persistence_own_accuracy"]
        assert block["search_track_fidelity"] > block["alignment_track_fidelity"]
        # Intersection still finds the goal, which was never the fragile half.
        assert block["persistence_goal_accuracy"] == 1.0
        assert block["hybrid_goal_accuracy"] == 1.0
        # Picking the track you can predict best chooses the distractor.
        assert block["predictability_own_accuracy"] < block["search_own_accuracy"]
        # A wrong identification poisons the model, which is how it does damage.
        assert block["search_model_accuracy"] > block["predictability_model_accuracy"]

    assert set(ARMS) == {
        "hybrid",
        "search",
        "alignment",
        "predictability",
        "persistence",
    }
    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before
