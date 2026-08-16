from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.curious_exploration import (
    ARMS,
    CONDITIONS,
    DISCOUNT_FAMILY,
    base_policies,
    run_curious_exploration,
    steps_to_coverage,
    untried_actions,
)
from experiments.brainworkshop_canonical.novelty import NoveltyCounts
from experiments.brainworkshop_canonical.successor_features import (
    generalised_policy_improvement,
)
from experiments.brainworkshop_canonical.world_model import WorldModel
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


def _world(seed: int, places: int = 8, actions: int = 4) -> WorldModel:
    generator = torch.Generator().manual_seed(seed)
    table = torch.randint(0, places, (actions, places), generator=generator).tolist()
    model = WorldModel(place_count=places, action_count=actions)
    for action in range(actions):
        for place in range(places):
            model.observe(place, action, int(table[action][place]), 0)
    return model


# --- optimism, which is the part that cannot be left out --------------------


def test_untried_actions_are_the_ones_never_taken_here() -> None:
    model = WorldModel(place_count=4, action_count=3)
    assert untried_actions(model, 0) == [0, 1, 2]
    model.observe(0, 1, 2, 0)
    assert untried_actions(model, 0) == [0, 2]
    assert untried_actions(model, 1) == [0, 1, 2]


def test_pure_novelty_seeking_would_never_try_a_new_action() -> None:
    """The trap that makes optimism load-bearing rather than an optimisation.

    An untried cell is treated as a self-loop everywhere downstream, so it
    looks like standing still -- and standing still is the one thing novelty
    scores as worthless, because the agent has just been here. Without
    preferring untried cells outright, the agent sits where it is.
    """

    model = WorldModel(place_count=4, action_count=2)
    # Action 0 is known and leads somewhere; action 1 has never been tried.
    for _ in range(3):
        model.observe(0, 0, 1, 0)
    counts = NoveltyCounts(alphabet=4)
    for _ in range(5):
        counts.observe(0, (0,))
    psis = base_policies(model, discount=0.95)
    weights = counts.weights()
    # Value alone picks the *known* action, so the untried one is never seen.
    assert generalised_policy_improvement(psis, 0, weights) == 0
    assert untried_actions(model, 0) == [1]


def test_base_policies_cover_every_place() -> None:
    model = _world(1)
    psis = base_policies(model, discount=0.95)
    assert len(psis) == model.place_count
    for psi in psis:
        assert psi.shape == (model.place_count, model.action_count, model.place_count)


# --- accounting -------------------------------------------------------------


def test_steps_to_coverage_reports_experience_not_endpoint() -> None:
    assert steps_to_coverage([0.2, 0.5, 0.95, 1.0], threshold=0.9, steps=10) == 30
    assert steps_to_coverage([0.2, 0.5], threshold=0.9, steps=10) is None


# --- end to end -------------------------------------------------------------


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_curiosity_explores_faster_and_gating_makes_it_immune(tmp_path) -> None:
    before = sha256_file(BANK)
    report = run_curious_exploration(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        tasks=3,
        episodes=6,
    )
    quiet = report["conditions"]["none"]
    noisy = report["conditions"]["random_walk"]

    # Directed novelty beats uniform wandering, on coverage and downstream.
    assert quiet["curious"]["coverage"] > quiet["uniform"]["coverage"]
    assert (
        quiet["curious"]["steps_to_threshold"] < quiet["uniform"]["steps_to_threshold"]
    )
    assert (
        quiet["curious"]["downstream_fraction"]
        > quiet["uniform"]["downstream_fraction"]
    )

    # But optimism alone is most of that gain, and is not allowed to be hidden.
    assert quiet["optimistic"]["coverage"] > quiet["uniform"]["coverage"]
    assert quiet["curious"]["coverage"] > quiet["optimistic"]["coverage"]

    # Gating: the distractor is invisible to an agent counting only what it
    # controls, so both conditions come out identical.
    assert quiet["curious"]["coverage"] == pytest.approx(
        noisy["curious"]["coverage"]
    )
    # And it is not invisible to one that counts whole readings.
    assert noisy["curious_ungated"]["coverage"] < quiet["curious_ungated"]["coverage"]
    assert (
        noisy["curious_ungated"]["steps_to_threshold"]
        > noisy["curious"]["steps_to_threshold"]
    )

    # The horizon family is degenerate here, and the record says so rather
    # than claiming the meta-controller earned something.
    assert quiet["curious_g50"]["coverage"] == pytest.approx(
        quiet["curious_g99"]["coverage"]
    )
    assert quiet["curious_bandit"]["coverage"] == pytest.approx(
        quiet["curious"]["coverage"]
    )

    assert set(CONDITIONS) == set(report["conditions"])
    assert len(DISCOUNT_FAMILY) == 4
    assert set(ARMS) <= set(quiet)
    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before
