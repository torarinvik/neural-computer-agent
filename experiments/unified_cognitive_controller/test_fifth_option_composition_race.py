import torch

from .audit_fifth_option_composition import load_fifth_router
from .train_fifth_option_composition_race import (
    five_action_hierarchy,
    four_action_hierarchy,
    target_bits,
)
from .train_option_composition_race import OptionValueHead
from .train_shadow_compute_advantage import ComputeAdvantageHead


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
