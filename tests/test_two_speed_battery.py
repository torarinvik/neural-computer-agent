import argparse

import pytest
import torch

from experiments.games_amodal.fragment_bank import FragmentBank, battery_suite
from experiments.games_amodal.game_family import FamilyConfig
from experiments.games_amodal.shared_controller import SharedControllerAgent
from experiments.games_amodal.two_speed_battery import (
    SOLO_CEILINGS,
    acquire_game,
    family_fisher,
    plant_named_parameters,
    run,
)


def _agent() -> SharedControllerAgent:
    torch.manual_seed(0)
    return SharedControllerAgent(
        event_width=16,
        intention_width=8,
        feedback_width=8,
        hidden=8,
        event_window_capacity=8,
        shared_drivers=True,
    )


def _args(**overrides) -> argparse.Namespace:
    base = {
        "seed": 0,
        "updates_per_game": 2,
        "batch_size": 2,
        "steps": 6,
        "gamma": 0.9,
        "learning_rate": 1e-3,
        "fragments_per_variant": 2,
        "ewc_lambda": 1.0,
        "arbitration_mu": 3.0,
        "arbitration_decay": 0.99,
        "fisher_batches": 2,
        "ignorance_weight": 0.5,
        "ignorance_every": 3,
        "egocentric": False,
        "eval_seeds": 1,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_solo_ceilings_cover_every_battery_game() -> None:
    train, _ = battery_suite()
    assert {config.name for config in train} <= set(SOLO_CEILINGS)
    # Ceilings are measurements, not aspirations: none may be a placeholder.
    assert all(0.0 < value <= 1.0 for value in SOLO_CEILINGS.values())


def test_protected_set_is_the_whole_persistent_plant() -> None:
    agent = _agent()
    named = dict(plant_named_parameters(agent))
    assert any(name.startswith("controller.") for name in named)
    # The shared drivers persist across games, so they are infrastructure
    # and must be protected too -- not just the controller.
    for prefix in ("encoder.", "decoder.", "feedback."):
        assert any(name.startswith(prefix) for name in named), prefix
    assert all(isinstance(t, torch.nn.Parameter) for t in named.values())


def test_fisher_is_unit_mean_and_finite() -> None:
    agent = _agent()
    bank = FragmentBank(
        fragments=4, tokens_per_fragment=2, width=16, variants=["choiceA"]
    )
    named = plant_named_parameters(agent)
    fisher = family_fisher(
        agent,
        FamilyConfig(choice=1, name="choiceA"),
        bank.fetch([0, 1]).detach(),
        named,
        args=_args(),
        seed=0,
    )
    assert set(fisher) == {name for name, _ in named}
    total = sum(t.sum() for t in fisher.values())
    count = sum(t.numel() for t in fisher.values())
    assert float(total / count) == pytest.approx(1.0, rel=1e-5)
    assert all(bool(torch.isfinite(t).all()) for t in fisher.values())
    assert all(bool((t >= 0).all()) for t in fisher.values())


def test_acquisition_reports_release_only_once_penalties_exist() -> None:
    agent = _agent()
    bank = FragmentBank(
        fragments=4, tokens_per_fragment=2, width=16, variants=["choiceA"]
    )
    config = FamilyConfig(choice=1, name="choiceA")
    first = acquire_game(agent, bank, config, [], args=_args(), seed_offset=0)
    # The first game has nothing to protect, so nothing is released.
    assert all(entry["release_fraction"] == 0.0 for entry in first)
    named = plant_named_parameters(agent)
    fisher = family_fisher(
        agent, config, bank.fetch([0, 1]).detach(), named, args=_args(), seed=1
    )
    anchor = {name: p.detach().clone() for name, p in named}
    second = acquire_game(
        agent, bank, config, [(fisher, anchor)], args=_args(), seed_offset=99
    )
    assert any(entry["release_fraction"] > 0.0 for entry in second)
    assert all(0.0 <= entry["release_fraction"] <= 1.0 for entry in second)
    assert all(entry["replayed_examples"] == 0.0 for entry in first + second)


def test_run_reports_forgetting_against_post_acquisition_scores() -> None:
    report = run(
        _args(
            games="choiceA,choiceB",
            updates_per_game=2,
            event_width=16,
            intent_width=8,
            feedback_width=8,
            hidden=8,
            fragments=8,
            tokens_per_fragment=2,
            report_out=None,
        )
    )
    assert report["order"] == ["choiceA", "choiceB"]
    assert report["no_replay"] is True
    for name in report["order"]:
        expected = (
            report["acquisition_mastery"][name]
            - report["final_mastery"][name]["mastery"]
        )
        assert report["forgetting"][name] == expected
    assert report["worst_forgetting"] == max(report["forgetting"].values())


def test_audit_fetches_the_same_fragments_acquisition_trained_with() -> None:
    """A game must be scored with the fragments it actually learned with.

    Without this the audit silently falls back to untrained selection
    logits and every game reads as a failure to learn.
    """

    args = _args(
        games="choiceA",
        updates_per_game=1,
        event_width=16,
        intent_width=8,
        feedback_width=8,
        hidden=8,
        fragments=8,
        tokens_per_fragment=2,
        report_out=None,
    )
    assert not hasattr(args, "oracle_selection")
    run(args)
    assert args.oracle_selection is True
