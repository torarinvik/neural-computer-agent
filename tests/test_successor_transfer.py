from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.navigation_environment import (
    NavigationTask,
    sample_navigation_task,
)
from experiments.brainworkshop_canonical.successor_transfer import (
    WeightedVerifier,
    best_weighted_return,
    run_successor_transfer,
    task_families,
    weights_for,
    worst_weighted_return,
)
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


# --- a task is now a vector -------------------------------------------------


def test_the_goal_language_widens_without_new_machinery() -> None:
    single = weights_for(8, (5,), None)
    disjunction = weights_for(8, (5, 6), None)
    avoiding = weights_for(8, (5,), 2)
    assert float(single.sum()) == 1.0
    assert float(disjunction.sum()) == 2.0
    # The case no goal-reaching policy is built for.
    assert float(avoiding[2]) < 0.0
    assert float(avoiding[5]) == 1.0


def test_the_families_cover_places_never_trained_on() -> None:
    generator = torch.Generator().manual_seed(0)
    families = task_families(8, [4, 5, 6, 7], generator)
    kinds = {row["family"] for row in families}
    assert kinds == {"single", "disjunction", "avoid"}
    for row in families:
        assert all(place >= 4 for place in row["shown"])
        if row["hazard"] is not None:
            assert row["hazard"] not in row["shown"]


# --- the ceiling ------------------------------------------------------------


def test_the_ceiling_is_backward_induction_not_a_shortest_path() -> None:
    """With a negative weight the quickest route is not the best one."""

    # Two places. Action 0 holds, action 1 swaps.
    task = NavigationTask(
        transitions=((0, 1), (1, 0)), goal=1, start=0, place_count=2
    ).validate()
    weights = torch.tensor([0.0, -1.0], dtype=torch.float64)
    # Staying put avoids the penalty entirely.
    assert best_weighted_return(task, 0, weights, 6) == pytest.approx(0.0)
    assert worst_weighted_return(task, 0, weights, 6) == pytest.approx(-1.0)
    # And the sign flips when the same place pays.
    reward = torch.tensor([0.0, 1.0], dtype=torch.float64)
    assert best_weighted_return(task, 0, reward, 6) == pytest.approx(1.0)


def test_the_floor_is_below_the_ceiling_everywhere() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    weights = weights_for(task.place_count, (5,), 2)
    for start in range(task.place_count):
        floor = worst_weighted_return(task, start, weights, 12)
        ceiling = best_weighted_return(task, start, weights, 12)
        assert floor <= ceiling


# --- the environment --------------------------------------------------------


def test_the_verifier_pays_the_weight_of_where_it_lands() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    weights = torch.zeros(task.place_count, dtype=torch.float64)
    landing = int(task.transitions[0][0])
    weights[landing] = -0.25
    verifier = WeightedVerifier(
        task,
        start=0,
        shown=(landing,),
        weights_by_place=weights,
        steps=3,
    )
    assert verifier.score(torch.tensor([0], dtype=torch.long)) == pytest.approx(-0.25)


def test_the_verifier_refuses_weights_of_the_wrong_width() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    with pytest.raises(ValueError, match="defined at every place"):
        WeightedVerifier(
            task,
            start=0,
            shown=(1,),
            weights_by_place=torch.zeros(3, dtype=torch.float64),
            steps=3,
        )


def test_the_verifier_only_accepts_actions_in_the_protocol() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    verifier = WeightedVerifier(
        task,
        start=0,
        shown=(1,),
        weights_by_place=torch.zeros(task.place_count, dtype=torch.float64),
        steps=3,
    )
    with pytest.raises(ValueError, match="outside the protocol"):
        verifier.score(torch.tensor([99], dtype=torch.long))


# --- end to end -------------------------------------------------------------


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_a_task_never_seen_is_solved_without_planning(tmp_path) -> None:
    before = sha256_file(BANK)
    report = run_successor_transfer(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        tasks=2,
        explore_episodes=16,
        library_directory=tmp_path / "libraries",
    )

    # Storing more policies gets closer to optimal on tasks never seen, with no
    # further experience at all. That is the accumulation claim.
    curve = report["accumulation"]
    assert len(curve) >= 2
    assert curve[-1]["gpi_over_optimal"] >= curve[0]["gpi_over_optimal"]
    assert curve[-1]["gpi_over_optimal"] > 0.8
    # And stitching is worth far more than the best single stored policy.
    assert curve[-1]["gpi_over_optimal"] > 3 * curve[-1]["single_over_optimal"]

    for family in ("single", "disjunction", "avoid"):
        # Reaching without planning is close to re-solving from scratch.
        assert report[f"{family}_gpi_told_fraction"] > 0.8
        assert (
            report[f"{family}_replan_told_fraction"]
            - report[f"{family}_gpi_told_fraction"]
            < 0.2
        )
        # Both beat following the best stored policy, and beat acting blindly.
        assert report[f"{family}_gpi_fraction"] > report[f"{family}_random_fraction"]
        assert (
            report[f"{family}_gpi_fraction"]
            > report[f"{family}_best_single_fraction"]
        )

    assert (tmp_path / "libraries" / "task0.successors").is_file()
    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before
