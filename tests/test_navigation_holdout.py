from __future__ import annotations

import inspect

import pytest

from experiments.brainworkshop_canonical import (
    curious_exploration,
    learned_decomposition,
    object_identity,
    relational_transfer,
    successor_transfer,
)
from experiments.brainworkshop_canonical.navigation_environment import (
    sample_navigation_task,
)
from experiments.brainworkshop_canonical.navigation_holdout import (
    BLOCK_NAME,
    decomposition_claims,
    exploration_claims,
    identity_claims,
    relational_claims,
    successor_claims,
)
from experiments.brainworkshop_canonical.seed_ledger import (
    BLOCKS,
    INTEGRATED_SESSIONS_PER_REPLICATE,
    assert_unused_block,
    block,
)

ENTRY_POINTS = (
    (successor_transfer, "run_successor_transfer"),
    (learned_decomposition, "run_learned_decomposition"),
    (object_identity, "run_object_identity"),
    (curious_exploration, "run_curious_exploration"),
    (relational_transfer, "run_relational_transfer"),
)


# --- the thing that made the earlier reruns not holdouts --------------------


def test_every_experiment_can_be_given_unseen_worlds() -> None:
    """The bug this whole record exists to fix.

    All five drew their worlds from a hard-coded seed, so changing the run seed
    varied only the exploration randomness and the sampled worlds were the same
    ones every time. A rerun would have looked like a holdout and been nothing
    of the kind.
    """

    for module, name in ENTRY_POINTS:
        signature = inspect.signature(getattr(module, name))
        assert "world_seed" in signature.parameters, name
        assert (
            signature.parameters["world_seed"].default
            == module.DEVELOPMENT_WORLD_SEED
        ), name


def test_the_development_default_is_unchanged() -> None:
    """Every recorded diagnostic has to keep reproducing exactly."""

    for module, _ in ENTRY_POINTS:
        assert module.DEVELOPMENT_WORLD_SEED == 9000
        assert module.WORLD_SEED_STRIDE == 37


def test_a_different_world_seed_really_is_a_different_world() -> None:
    development = [
        sample_navigation_task(seed=9000 + 37 * index) for index in range(4)
    ]
    held_out = [
        sample_navigation_task(seed=block(BLOCK_NAME)[0] + 37 * index)
        for index in range(4)
    ]
    assert all(task is not None for task in (*development, *held_out))
    for left, right in zip(development, held_out):
        assert left.transitions != right.transitions


# --- the block --------------------------------------------------------------


def test_the_holdout_block_is_registered_and_clear() -> None:
    seeds = block(BLOCK_NAME)
    assert len(seeds) == 3
    assert_unused_block(
        BLOCK_NAME, seeds, sessions=INTEGRATED_SESSIONS_PER_REPLICATE
    )


def test_the_holdout_block_is_nowhere_near_the_development_worlds() -> None:
    """The worlds are drawn by striding from the seed, so the span matters."""

    for seed in block(BLOCK_NAME):
        assert seed > 9000 + 37 * 1000


def test_claiming_a_used_block_fails_closed() -> None:
    taken = BLOCKS["compositional_transfer_holdout"]
    with pytest.raises(ValueError, match="collides with recorded lifetimes"):
        assert_unused_block(
            "a_new_campaign", taken, sessions=INTEGRATED_SESSIONS_PER_REPLICATE
        )


# --- the claims are read off the reports, not re-derived --------------------


def test_successor_claims_read_the_orderings_that_were_published() -> None:
    passing = {
        "accumulation": [
            {"gpi_over_optimal": 0.72, "single_over_optimal": 0.07},
            {"gpi_over_optimal": 0.84, "single_over_optimal": 0.14},
        ],
    }
    for family in ("single", "disjunction", "avoid"):
        passing[f"{family}_gpi_fraction"] = 0.80
        passing[f"{family}_best_single_fraction"] = 0.40
        passing[f"{family}_random_fraction"] = 0.20
        passing[f"{family}_gpi_told_fraction"] = 0.85
        passing[f"{family}_replan_told_fraction"] = 0.99
    assert all(successor_claims(passing).values())

    # Flip the one that matters and only that one should fail.
    failing = dict(passing)
    failing["single_best_single_fraction"] = 0.95
    claims = successor_claims(failing)
    assert claims["single_beats_best_stored"] is False
    assert claims["disjunction_beats_best_stored"] is True


def test_decomposition_claims_require_components_to_win() -> None:
    report = {
        "tasks": 4,
        "components_chosen": 4,
        "cuts": {
            "components": {"total_bits": 134.0, "error_bits": 0.0},
            "whole": {"total_bits": 360.0, "error_bits": 44.0},
            "scatter": {"total_bits": 878.0, "error_bits": 108.0},
            "cells": {"total_bits": 709.0, "error_bits": 375.0},
        },
    }
    assert all(decomposition_claims(report).values())
    report["components_chosen"] = 3
    assert decomposition_claims(report)["components_chosen_everywhere"] is False


def _identity_report(search_own: float, persistence_own: float) -> dict:
    quiet = {f"{arm}_own_accuracy": 1.0 for arm in ("hybrid", "search", "alignment", "persistence")}
    quiet.update({"search_track_fidelity": 0.99, "alignment_track_fidelity": 0.88})
    condition = {
        "persistence_own_accuracy": persistence_own,
        "search_own_accuracy": search_own,
        "search_track_fidelity": 0.70,
        "alignment_track_fidelity": 0.57,
        "hybrid_goal_accuracy": 1.0,
        "predictability_own_accuracy": 0.18,
        # The claim that actually survived the holdout: a coherent track feeds
        # a coherent model even when the final naming is wrong.
        "hybrid_model_accuracy": 0.68,
        "persistence_model_accuracy": 0.49,
    }
    return {
        "conditions": {
            "none": quiet,
            "random_walk": dict(condition),
            "cycling": dict(condition),
        }
    }


def test_identity_claims_track_the_published_orderings() -> None:
    assert all(identity_claims(_identity_report(0.62, 0.45)).values())
    # Persistence not actually broken by the distractor -- which is what the
    # holdout found on two worlds of three.
    claims = identity_claims(_identity_report(0.62, 0.80))
    assert claims["random_walk_persistence_breaks"] is False
    assert claims["random_walk_search_beats_persistence"] is False
    # The model claim is independent of the naming claim, and is the one that
    # held everywhere. It must not be dragged down with the withdrawn one.
    assert claims["random_walk_search_builds_a_better_model"] is True


def test_exploration_claims_include_the_two_negative_results() -> None:
    quiet = {
        "uniform": {"coverage": 0.80, "downstream_fraction": 0.84},
        "optimistic": {"coverage": 0.92, "downstream_fraction": 0.90},
        "curious": {"coverage": 0.98, "downstream_fraction": 0.92},
        "curious_ungated": {"coverage": 0.97, "downstream_fraction": 0.92},
        "curious_g50": {"coverage": 0.98},
        "curious_g99": {"coverage": 0.98},
    }
    noisy = {
        "curious": {"coverage": 0.98},
        "curious_ungated": {"coverage": 0.94},
    }
    claims = exploration_claims({"conditions": {"none": quiet, "random_walk": noisy}})
    assert all(claims.values())
    # The degeneracy of the horizon family is a claim in its own right.
    assert claims["horizon_family_is_degenerate"] is True


def test_relational_claims_include_the_cost_of_a_held_out_relation() -> None:
    report = {
        "trained": {
            "gpi_fraction": 0.96,
            "replan_fraction": 0.96,
            "place_gpi_fraction": 0.39,
            "best_single_fraction": 0.96,
            "random_fraction": 0.31,
        },
        "held_out": {
            "gpi_fraction": 0.81,
            "replan_fraction": 0.96,
            "place_gpi_fraction": 0.30,
            "best_single_fraction": 0.47,
            "random_fraction": 0.27,
        },
    }
    assert all(relational_claims(report).values())
    # If stitching matched re-solving on held-out relations too, the PGM
    # finding would not be there and the claim should say so.
    report["held_out"]["replan_fraction"] = 0.82
    assert (
        relational_claims(report)["held_out_costs_more_than_interpolation"] is False
    )
