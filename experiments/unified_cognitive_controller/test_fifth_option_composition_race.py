import json

import pytest
import torch

from .audit_fifth_option_composition import (
    load_fifth_router,
    load_generation_router,
)
from .train_fifth_option_composition_race import (
    FlatFiveActionValueHead,
    five_action_hierarchy,
    four_action_hierarchy,
    replay_updates_for_step,
    target_bits,
)
from .train_option_composition_race import OptionValueHead
from .train_shadow_compute_advantage import ComputeAdvantageHead
from .replay_stopping_probe import (
    ReplayBenefitProbe,
    FEATURES,
    load_trace,
    load_probe,
    predict_replay_benefit,
    save_probe,
    trace_features,
    trace_targets,
)


def test_default_third_generation_router_reuses_four_action_hierarchy() -> None:
    torch.manual_seed(15)
    champion = ComputeAdvantageHead(8)
    option3 = OptionValueHead(7, 8)
    router4 = OptionValueHead(7, 8)
    router5 = OptionValueHead(7, 8)
    features = torch.randn(47, 9)
    assert torch.equal(
        five_action_hierarchy(
            router5, router4, option3, champion, features),
        four_action_hierarchy(
            router4, option3, champion, features))


def test_fifth_router_checkpoint_round_trip(tmp_path) -> None:
    router = OptionValueHead(11, 13)
    path = tmp_path / "router.pt"
    torch.save({
        "schema": "fifth-option-router-v1",
        "input_width": 11,
        "hidden": 13,
        "state_dict": router.state_dict(),
    }, path)
    restored = load_fifth_router(path, torch.device("cpu"))
    features = torch.randn(19, 11)
    assert torch.equal(router(features), restored(features))


def test_stable_target_ignores_isolated_crossing() -> None:
    rows = [
        {"verifier_bits": 0, "reaches_target": False},
        {"verifier_bits": 120, "reaches_target": True},
        {"verifier_bits": 240, "reaches_target": False},
        {"verifier_bits": 360, "reaches_target": True},
        {"verifier_bits": 480, "reaches_target": True},
    ]
    assert target_bits(rows, stable=False) == 120
    assert target_bits(rows, stable=True) == 360


def test_replay_schedule_keeps_initial_then_switches_to_late_budget() -> None:
    assert replay_updates_for_step(1, 56, 48, 16) == 56
    assert replay_updates_for_step(16, 56, 48, 16) == 56
    assert replay_updates_for_step(17, 56, 48, 16) == 48
    assert replay_updates_for_step(17, 48, None, None) == 48


def test_flat_head_supports_six_action_control() -> None:
    head = FlatFiveActionValueHead(13, 17, actions=6)
    assert head.q_values(torch.randn(23, 13)).shape == (23, 6)


def test_generation_router_rejects_wrong_schema(tmp_path) -> None:
    router = OptionValueHead(11, 13)
    path = tmp_path / "router.pt"
    torch.save({
        "schema": "sixth-option-router-v1",
        "input_width": 11,
        "hidden": 13,
        "state_dict": router.state_dict(),
    }, path)
    restored = load_generation_router(
        path, torch.device("cpu"), schema="sixth-option-router-v1")
    features = torch.randn(7, 11)
    assert torch.equal(router(features), restored(features))
    try:
        load_fifth_router(path, torch.device("cpu"))
    except ValueError:
        pass
    else:
        raise AssertionError("schema mismatch must be rejected")


def test_replay_trace_features_exclude_future_outcome() -> None:
    rows = [{
        "loss_before": 0.3,
        "previous_loss_reduction": 0.02,
        "previous_gradient_norm": 0.4,
        "observed_examples": 60,
        "replay_index": 3,
        "loss_reduction": 0.7,
    }]
    features = trace_features(rows, replay_updates=16)
    assert features.shape == (1, len(FEATURES))
    assert float(features[0, 0]) == pytest.approx(0.3)
    assert float(features[0, 4]) == pytest.approx(3 / 15)
    assert float(trace_targets(rows)[0]) == pytest.approx(0.7)
    changed = [{**rows[0], "loss_reduction": -9.0}]
    assert torch.equal(
        features, trace_features(changed, replay_updates=16))


def test_replay_benefit_probe_round_trip(tmp_path) -> None:
    model = ReplayBenefitProbe(hidden=7)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.1)
    path = tmp_path / "probe.pt"
    mean = torch.zeros(1, len(FEATURES))
    scale = torch.ones(1, len(FEATURES))
    target_mean = torch.tensor([[0.2]])
    target_scale = torch.tensor([[0.5]])
    save_probe(
        path, model, feature_mean=mean, feature_scale=scale,
        target_mean=target_mean, target_scale=target_scale,
        hidden=7, target_horizon=4, target_kind="future-utility")
    restored, normalization = load_probe(path, torch.device("cpu"))
    expected = predict_replay_benefit(
        model, {
            "feature_mean": mean,
            "feature_scale": scale,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "target_horizon": 4,
            "target_kind": "future-utility",
        },
        loss_before=0.3,
        previous_loss_reduction=0.02,
        previous_gradient_norm=0.4,
        observed_examples=60,
        replay_index=3,
        replay_updates=16,
        device=torch.device("cpu"),
    )
    actual = predict_replay_benefit(
        restored, normalization,
        loss_before=0.3,
        previous_loss_reduction=0.02,
        previous_gradient_norm=0.4,
        observed_examples=60,
        replay_index=3,
        replay_updates=16,
        device=torch.device("cpu"),
    )
    assert actual == pytest.approx(expected)
    assert normalization["target_horizon"] == 4
    assert normalization["target_kind"] == "future-utility"


def test_future_utility_trace_uses_only_decision_state_features(
        tmp_path) -> None:
    path = tmp_path / "behavior.json"
    path.write_text(json.dumps({
        "configuration": {"replay_updates": 16},
        "behavioral_replay_trace": [{
            "arm": "composition",
            "loss_before": 0.2,
            "previous_loss_reduction": 0.01,
            "previous_gradient_norm": 0.3,
            "observed_examples": 120,
            "replay_index": 8,
            "replay_horizon": 8,
            "future_utility_gain": 0.04,
        }, {
            "arm": "flat",
            "loss_before": 9.0,
            "previous_loss_reduction": 9.0,
            "previous_gradient_norm": 9.0,
            "observed_examples": 120,
            "replay_index": 8,
            "replay_horizon": 8,
            "future_utility_gain": 9.0,
        }],
    }))
    features, targets, updates = load_trace(
        path, "composition", 8, "future-utility")
    assert features.shape == (1, len(FEATURES))
    assert targets.shape == (1, 1)
    assert float(targets[0]) == pytest.approx(0.04)
    assert updates == 16
    changed = json.loads(path.read_text())
    changed["behavioral_replay_trace"][0]["future_utility_gain"] = -9
    path.write_text(json.dumps(changed))
    changed_features, _, _ = load_trace(
        path, "composition", 8, "future-utility")
    assert torch.equal(features, changed_features)
