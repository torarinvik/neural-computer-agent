from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.navigation_environment import (
    NavigationTask,
    sample_navigation_task,
)
from experiments.brainworkshop_canonical.relation_cumulants import (
    CONSTANT_PLACE_LIMIT,
    PLACE_COUNT,
    RELATIONS,
    constant_place_rate,
    joint_state,
    marginal_place_weights,
    relation_cumulants,
    relation_weights,
    satisfying_places,
    split_state,
)
from experiments.brainworkshop_canonical.relational_transfer import (
    ARMS,
    HELD_OUT_RELATIONS,
    TRAINED_RELATIONS,
    RelationVerifier,
    joint_ceiling,
    run_relational_transfer,
    target_circuit,
)
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


# --- the cumulants ----------------------------------------------------------


def test_no_relation_is_solvable_by_standing_still() -> None:
    """The guard that removed three earlier candidates.

    `above`, `left_of` and `far` are each satisfied 0.625 of the time by one
    fixed place, so an agent that never looks at the marker scores 1.000 on
    them and the whole comparison measures nothing.
    """

    for relation in RELATIONS:
        assert constant_place_rate(relation) <= CONSTANT_PLACE_LIMIT, relation
    for rejected in ("above", "left_of"):
        assert rejected not in RELATIONS


def test_the_relations_are_independent_features() -> None:
    features = relation_cumulants()
    assert features.shape == (PLACE_COUNT * PLACE_COUNT, len(RELATIONS))
    assert int(torch.linalg.matrix_rank(features)) == len(RELATIONS)


def test_the_satisfying_set_moves_which_is_the_whole_point() -> None:
    """A place-vector can name a fixed set of places. This is not one."""

    first = satisfying_places("adjacent", 0)
    second = satisfying_places("adjacent", 7)
    assert first and second
    assert set(first) != set(second)
    # And every relation here has at least one target for which the set moves.
    for relation in RELATIONS:
        sets = {satisfying_places(relation, theirs) for theirs in range(PLACE_COUNT)}
        assert len(sets) > 1, relation


def test_configurations_index_and_unindex() -> None:
    for mine in range(PLACE_COUNT):
        for theirs in range(PLACE_COUNT):
            assert split_state(joint_state(mine, theirs)) == (mine, theirs)
    with pytest.raises(ValueError, match="leaves the place set"):
        joint_state(0, PLACE_COUNT)


def test_a_task_is_one_relation_being_worth_having() -> None:
    weights = relation_weights("adjacent")
    assert weights.shape == (len(RELATIONS),)
    assert float(weights.sum()) == 1.0
    with pytest.raises(ValueError, match="unknown relation"):
        relation_weights("beside")


def test_the_place_control_gets_the_marginal_not_a_strawman() -> None:
    weights = marginal_place_weights("same")
    # "Be on the marker" holds one time in eight from anywhere, so a place
    # vector genuinely cannot prefer one place over another for it.
    assert float(weights.max()) == pytest.approx(1.0 / PLACE_COUNT)
    adjacent = marginal_place_weights("adjacent")
    assert float(adjacent.max()) == pytest.approx(constant_place_rate("adjacent"))


# --- the environment --------------------------------------------------------


def test_the_target_walks_a_cycle_through_every_place() -> None:
    circuit = target_circuit(3)
    assert sorted(circuit) == list(range(PLACE_COUNT))
    seen = {0}
    place = 0
    for _ in range(PLACE_COUNT - 1):
        place = circuit[place]
        seen.add(place)
    assert len(seen) == PLACE_COUNT


def test_the_target_moves_whatever_the_agent_does() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    circuit = target_circuit(3)
    moves = []
    for action in range(task.action_count):
        verifier = RelationVerifier(
            task, circuit, start=0, target=2, relation="same", steps=3
        )
        verifier.score(torch.tensor([action], dtype=torch.long))
        moves.append(verifier.target)
    assert len(set(moves)) == 1
    assert moves[0] != 2


def test_the_verifier_pays_the_relation() -> None:
    # Two places in a row; action 0 holds, action 1 swaps. Target holds still
    # by cycling between the same two.
    task = NavigationTask(
        transitions=((0, 1), (1, 0)), goal=1, start=0, place_count=2
    ).validate()
    verifier = RelationVerifier(
        task, (1, 0), start=0, target=1, relation="same", steps=4
    )
    # Agent stays at 0, target moves 1 -> 0, so they coincide.
    assert verifier.score(torch.tensor([0], dtype=torch.long)) == 1.0


def test_the_verifier_only_accepts_actions_and_relations_it_knows() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    circuit = target_circuit(3)
    with pytest.raises(ValueError, match="unknown relation"):
        RelationVerifier(
            task, circuit, start=0, target=1, relation="beside", steps=3
        )
    verifier = RelationVerifier(
        task, circuit, start=0, target=1, relation="same", steps=3
    )
    with pytest.raises(ValueError, match="outside the protocol"):
        verifier.score(torch.tensor([99], dtype=torch.long))


def test_the_ceiling_brackets_what_is_achievable() -> None:
    task = sample_navigation_task(seed=9000)
    assert task is not None
    circuit = target_circuit(3)
    for relation in RELATIONS:
        for start in range(0, PLACE_COUNT, 3):
            best = joint_ceiling(
                task,
                circuit,
                start=start,
                target=1,
                relation=relation,
                steps=8,
                best=True,
            )
            worst = joint_ceiling(
                task,
                circuit,
                start=start,
                target=1,
                relation=relation,
                steps=8,
                best=False,
            )
            assert 0.0 <= worst <= best <= 1.0


# --- end to end -------------------------------------------------------------


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_a_relation_never_rewarded_transfers_but_not_for_free(tmp_path) -> None:
    before = sha256_file(BANK)
    report = run_relational_transfer(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        tasks=4,
        explore_episodes=30,
    )
    trained = report["trained"]
    held_out = report["held_out"]

    # Both representations saw the same experience.
    assert report["mean_joint_coverage"] > 0.9
    assert report["mean_place_coverage"] > 0.9

    # Pairs are necessary: a place vector cannot follow a marker that moves,
    # and lands close to acting blindly.
    assert trained["place_gpi_fraction"] < 0.5 * trained["gpi_fraction"]
    assert held_out["place_gpi_fraction"] < 0.5 * held_out["gpi_fraction"]

    # A relation that was never a reward still transfers, well past the best
    # single stored policy.
    assert held_out["gpi_fraction"] > 1.4 * held_out["best_single_fraction"]
    assert held_out["gpi_fraction"] > 2.0 * held_out["random_fraction"]

    # But held-out *relations* are harder than held-out goals were: on the
    # trained relations stitching matches re-solving, and on the held-out ones
    # it does not. Replicated at two seeds and two task counts; the gap does
    # not survive at two tasks, which is why this runs at four.
    assert trained["replan_fraction"] - trained["gpi_fraction"] < 0.05
    assert held_out["replan_fraction"] - held_out["gpi_fraction"] > 0.05
    assert held_out["gpi_fraction"] < trained["gpi_fraction"] - 0.05

    assert set(TRAINED_RELATIONS) & set(HELD_OUT_RELATIONS) == set()
    assert set(ARMS) == {"gpi", "best_single", "replan", "place_gpi", "random"}
    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before
