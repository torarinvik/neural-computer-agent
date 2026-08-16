from __future__ import annotations

import pytest
import torch

from experiments.brainworkshop_canonical.successor_features import (
    SUCCESSOR_LIBRARY_EXTENSION,
    SuccessorFeatureLibrary,
    SuccessorFeatureRecord,
    generalised_policy_improvement,
    gpi_policy,
    greedy_policy,
    known_successor,
    policy_values,
    reach,
    reach_any,
    reach_avoiding,
    stitching_gain,
    successor_features,
    transition_matrix,
)
from experiments.brainworkshop_canonical.world_model import WorldModel

DISCOUNT = 0.95


def _world(seed: int, places: int = 8, actions: int = 4) -> WorldModel:
    generator = torch.Generator().manual_seed(seed)
    table = torch.randint(0, places, (actions, places), generator=generator).tolist()
    model = WorldModel(place_count=places, action_count=actions)
    for action in range(actions):
        for place in range(places):
            model.observe(place, action, int(table[action][place]), 0)
    return model


# --- the arithmetic is exact ------------------------------------------------


def test_psi_dotted_with_the_task_is_the_discounted_return() -> None:
    """The claim that makes this worth doing without a network.

    Every published version approximates psi. Here it is a linear solve, so
    the number it reports is the return itself and can be checked against a
    rollout to whatever precision the discount allows.
    """

    model = _world(0)
    table = transition_matrix(model)
    weights = reach(model.place_count, 3)
    policy = greedy_policy(model, weights, discount=DISCOUNT)
    psi = successor_features(model, policy, discount=DISCOUNT)

    for place in range(model.place_count):
        for action in range(model.action_count):
            total = 0.0
            discount = 1.0
            cursor = table[action][place]
            total += discount * float(weights[cursor])
            for _ in range(700):
                discount *= DISCOUNT
                cursor = table[policy[cursor]][cursor]
                total += discount * float(weights[cursor])
            assert float(psi[place, action] @ weights) == pytest.approx(
                total, abs=1e-7
            )


def test_an_untried_action_is_held_in_place_rather_than_invented() -> None:
    model = WorldModel(place_count=3, action_count=2)
    model.observe(0, 0, 1, 0)
    assert known_successor(model, 0, 0) == 1
    # Never tried, so the policy stays put rather than claiming an edge.
    assert known_successor(model, 0, 1) == 0
    assert known_successor(model, 2, 0) == 2


def test_a_one_hot_task_is_the_goal_reaching_task() -> None:
    model = _world(1)
    policy = greedy_policy(model, reach(model.place_count, 5), discount=DISCOUNT)
    table = transition_matrix(model)
    # From anywhere that can reach place five, following the policy gets there.
    place = 0
    for _ in range(model.place_count * 2):
        place = table[policy[place]][place]
    assert place == 5


# --- generalised policy improvement ----------------------------------------


def test_stitching_is_never_worse_than_the_best_stored_policy() -> None:
    """The guarantee. Checked across task families, including ones no stored
    policy was built for."""

    for seed in range(6):
        model = _world(seed)
        places = model.place_count
        policies = [
            greedy_policy(model, reach(places, goal), discount=DISCOUNT)
            for goal in range(4)
        ]
        psis = [
            successor_features(model, policy, discount=DISCOUNT)
            for policy in policies
        ]
        for weights in (
            reach(places, 6),
            reach_any(places, (5, 6)),
            reach_avoiding(places, 6, 1),
        ):
            report = stitching_gain(
                model, psis, policies, weights, discount=DISCOUNT
            )
            assert min(report["gains"]) >= -1e-9


def test_the_gain_is_measured_on_the_stitched_policy_not_a_tautology() -> None:
    """A first version compared two maxima that are the same number.

    `max over actions of (max over policies)` and `max over policies of (max
    over actions)` are both the maximum of the whole matrix, so their
    difference was identically zero and the metric said nothing.
    """

    model = _world(2)
    places = model.place_count
    policies = [
        greedy_policy(model, reach(places, goal), discount=DISCOUNT)
        for goal in range(4)
    ]
    psis = [successor_features(model, policy, discount=DISCOUNT) for policy in policies]
    weights = reach(places, 7)
    scores = torch.stack([psi[0] @ weights.double() for psi in psis])
    assert float(scores.max(dim=0).values.max()) == pytest.approx(
        float(scores.max(dim=1).values.max())
    )
    report = stitching_gain(model, psis, policies, weights, discount=DISCOUNT)
    assert report["mean_gain"] > 0.0


def test_a_negative_weight_makes_the_stored_policies_wrong() -> None:
    """The case that is worth measuring, because no stored policy suits it."""

    model = _world(3)
    places = model.place_count
    policies = [
        greedy_policy(model, reach(places, goal), discount=DISCOUNT)
        for goal in range(4)
    ]
    psis = [successor_features(model, policy, discount=DISCOUNT) for policy in policies]
    hazard = reach_avoiding(places, 6, 1)
    assert float(hazard[1]) < 0.0
    stitched = gpi_policy(psis, places, hazard)
    stitched_values = policy_values(
        successor_features(model, stitched, discount=DISCOUNT), stitched, hazard
    )
    best_single = [
        max(policy_values(psi, policy, hazard)[place] for psi, policy in zip(psis, policies))
        for place in range(places)
    ]
    assert sum(stitched_values) > sum(best_single)


def test_improvement_needs_something_stored() -> None:
    with pytest.raises(ValueError, match="needs a stored policy"):
        generalised_policy_improvement([], 0, reach(4, 1))


# --- the library ------------------------------------------------------------


def _record(model: WorldModel, goal: int) -> SuccessorFeatureRecord:
    policy = greedy_policy(model, reach(model.place_count, goal), discount=DISCOUNT)
    return SuccessorFeatureRecord(
        policy=policy,
        psi=successor_features(model, policy, discount=DISCOUNT),
        discount=DISCOUNT,
        provenance={"goal": goal},
    )


def test_a_library_round_trips_and_checksums(tmp_path) -> None:
    model = _world(4)
    library = SuccessorFeatureLibrary(
        place_count=model.place_count,
        action_count=model.action_count,
        cumulant_dimension=model.place_count,
    )
    for goal in range(3):
        library.append(_record(model, goal))
    path = tmp_path / f"policies{SUCCESSOR_LIBRARY_EXTENSION}"
    library.save(path)
    loaded = SuccessorFeatureLibrary.load(path)
    assert loaded.digest() == library.digest()
    assert loaded.record_count == 3


def test_a_tampered_library_is_refused(tmp_path) -> None:
    model = _world(5)
    library = SuccessorFeatureLibrary(
        place_count=model.place_count,
        action_count=model.action_count,
        cumulant_dimension=model.place_count,
    )
    library.append(_record(model, 1))
    path = tmp_path / f"policies{SUCCESSOR_LIBRARY_EXTENSION}"
    library.save(path)
    path.write_text(path.read_text().replace("0.9", "0.8", 1))
    with pytest.raises(ValueError, match="checksum mismatch"):
        SuccessorFeatureLibrary.load(path)


def test_a_policy_that_acts_identically_is_a_duplicate() -> None:
    model = _world(6)
    library = SuccessorFeatureLibrary(
        place_count=model.place_count,
        action_count=model.action_count,
        cumulant_dimension=model.place_count,
    )
    record = _record(model, 2)
    library.append(record)
    assert library.duplicate_of(record.signature) == 0
    assert library.duplicate_of(tuple(reversed(record.signature))) in (0, None)


def test_a_library_refuses_a_psi_of_the_wrong_shape() -> None:
    model = _world(7)
    library = SuccessorFeatureLibrary(
        place_count=model.place_count,
        action_count=model.action_count,
        cumulant_dimension=model.place_count,
    )
    record = _record(model, 0)
    trimmed = SuccessorFeatureRecord(
        policy=record.policy[:-1],
        psi=record.psi[:-1],
        discount=DISCOUNT,
    )
    with pytest.raises(ValueError, match="does not match the library"):
        library.append(trimmed)


def test_libraries_must_use_their_own_extension(tmp_path) -> None:
    model = _world(8)
    library = SuccessorFeatureLibrary(
        place_count=model.place_count,
        action_count=model.action_count,
        cumulant_dimension=model.place_count,
    )
    with pytest.raises(ValueError, match="extension"):
        library.save(tmp_path / "policies.json")
