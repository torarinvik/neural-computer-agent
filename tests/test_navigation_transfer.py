from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.navigation_environment import (
    NavigationTask,
    NavigationVerifier,
    sample_navigation_task,
)
from experiments.brainworkshop_canonical.navigation_transfer import (
    cluster_of_place,
    discover_places,
    run_navigation,
)
from experiments.brainworkshop_canonical.world_model import (
    WorldModel,
    plan_to,
    policy_from_model,
)
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


def _task(seed: int = 9000):
    task = sample_navigation_task(seed=seed)
    assert task is not None
    return task


# --- the world -------------------------------------------------------------


def test_an_action_changes_what_is_seen_next() -> None:
    """The property no earlier environment in this repository has."""

    task = _task()
    frames = []
    for action in range(task.action_count):
        verifier = NavigationVerifier(task, steps=2, seed=5)
        verifier.observation()
        verifier.score(torch.tensor([action], dtype=torch.long))
        frames.append(verifier.observation())
    # At least two actions lead somewhere different.
    assert any(not torch.equal(frames[0], frame) for frame in frames[1:])


def test_the_verifier_says_only_whether_the_step_paid() -> None:
    task = _task()
    verifier = NavigationVerifier(task, steps=6, seed=5)
    verifier.observation()
    step = verifier.score(torch.tensor([0], dtype=torch.long))
    assert step.reward.shape == (1,)
    assert float(step.reward.item()) in (0.0, 1.0)
    with pytest.raises(ValueError, match="outside the protocol"):
        verifier.score(torch.tensor([99], dtype=torch.long))


def test_a_sampled_task_is_reachable_worth_solving_and_holdable() -> None:
    for seed in range(9000, 9010):
        task = sample_navigation_task(seed=seed)
        if task is None:
            continue
        distance = task.distances()[task.start]
        assert 2 <= distance <= task.place_count
        # Some action holds at the goal, so staying is possible.
        assert any(int(row[task.goal]) == task.goal for row in task.transitions)


def test_the_best_available_return_counts_the_arrival_step() -> None:
    """The off-by-one that let the agent 'beat' the optimum."""

    task = NavigationTask(
        transitions=((1, 2, 2), (0, 0, 2)), goal=2, start=0, place_count=3
    ).validate()
    assert task.place_count == 3
    assert task.distances()[0] == 2
    # Arriving on step 2 pays on steps 2..10: nine of ten.
    assert task.optimal_return(10) == pytest.approx(0.9)
    assert task.optimal_return(1) == 0.0


# --- the model and the planner --------------------------------------------


def test_the_planner_will_not_route_through_what_was_never_tried() -> None:
    """An untried action is not an edge, so a plan cannot invent one."""

    model = WorldModel(place_count=4, action_count=2)
    model.observe(0, 0, 1, 0)
    model.observe(1, 0, 2, 1)
    assert model.coverage == pytest.approx(2 / 8)
    assert model.goals() == (2,)
    route = plan_to(model, 0, model.goals())
    assert route is not None
    assert route.actions == (0, 0)
    # Place 3 was never seen leaving anywhere, so nothing reaches it.
    assert plan_to(model, 3, model.goals()) is None


def test_a_model_outvotes_a_single_bad_reading() -> None:
    """A mis-clustered place must not overwrite what was seen repeatedly."""

    model = WorldModel(place_count=4, action_count=1)
    for _ in range(5):
        model.observe(0, 0, 1, 0)
    model.observe(0, 0, 3, 0)
    assert model.successor(0, 0) == 1


def test_a_policy_with_no_goal_yet_still_acts() -> None:
    model = WorldModel(place_count=3, action_count=2)
    assert model.goals() == ()
    assert policy_from_model(model)(0) == 0


def test_places_cannot_be_clustered_from_one_look_each() -> None:
    """Why discovery walks the world instead of rendering a catalogue.

    With one observation per place there is no within-place mode, so there is
    no boundary between 'same place' and 'different place' to estimate. The
    estimator refusing is the correct answer, not a defect.
    """

    from experiments.brainworkshop_canonical.prototype_templates import (
        estimated_tolerance,
    )
    from experiments.brainworkshop_canonical.rendered_environment import (
        render_position,
    )

    frames = torch.stack([render_position(place, size=36) for place in range(8)])
    flattened = frames.reshape(8, -1)
    assert estimated_tolerance(flattened) is None


# --- end to end ------------------------------------------------------------


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_the_agent_models_a_world_and_plans_the_shortest_route(tmp_path) -> None:
    """The claim, with the controls that can refute it."""

    before = sha256_file(BANK)
    report = run_navigation(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        tasks=6,
        explore_episodes=12,
    )

    # It finds the goal by stumbling on it, and finds only the goal.
    assert report["found_the_goal"] == report["tasks"]
    assert all(row["goals_are_only_the_goal"] for row in report["rows"])

    # It plans the true shortest route and achieves the best return available.
    assert report["plans_optimal"] == report["tasks"]
    assert report["mean_planned_return"] == pytest.approx(
        report["mean_optimal_return"], abs=1e-9
    )

    # It beats guessing policies on the same experience, and wandering.
    assert report["mean_planned_return"] > report["mean_model_free_return"] + 0.2
    assert report["mean_planned_return"] > report["mean_random_return"] + 0.5

    # Destroying where the reward was leaves the planner nothing worth seeking.
    assert report["mean_shuffled_return"] < 0.5 * report["mean_planned_return"]

    # And it works from starts it never began an episode from, which a
    # memorised trajectory could not do.
    assert report["mean_held_out_return"] >= 0.9 * report["mean_held_out_optimal"]

    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_the_places_are_read_through_the_same_frontend_and_it_has_a_limit() -> None:
    """Stimulus noise stops this exactly where it stops everything else."""

    from experiments.brainworkshop_canonical.controller_pretraining import (
        load_temporal_controller_artifact,
    )
    from experiments.brainworkshop_canonical.current_symbol_acquire import (
        FRONTEND_SEED,
        _machine,
        curated_frontend,
    )

    payload = load_temporal_controller_artifact(CONTROLLER)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=FRONTEND
    )
    task = _task()
    clusters = discover_places(
        encoders, task, episodes=4, steps=24, seed=41, frame_noise=0.10
    )
    assert int(clusters.shape[0]) == task.place_count
    # And the map from place to cluster is a permutation, which is why the
    # record needs it and the agent does not.
    mapping = cluster_of_place(encoders, clusters, place_count=task.place_count)
    assert sorted(mapping) == list(range(task.place_count))

    with pytest.raises(ValueError, match="do not separate"):
        discover_places(
            encoders, task, episodes=4, steps=24, seed=41, frame_noise=0.30
        )
