from pathlib import Path

import torch

from .audit_option_composition import (
    load_option,
    option_from_skill_payload,
    option_skill_payload,
)
from .train_option_composition_race import OptionValueHead


def test_option_checkpoint_round_trip(tmp_path: Path) -> None:
    source = OptionValueHead(7, 8)
    path = tmp_path / "option.pt"
    torch.save({
        "schema": "option-composition-head-v1",
        "input_width": 7,
        "hidden": 8,
        "state_dict": source.state_dict(),
    }, path)
    restored = load_option(path, torch.device("cpu"))
    features = torch.randn(13, 7)
    assert torch.equal(source(features), restored(features))


def test_option_skill_payload_round_trip() -> None:
    source = OptionValueHead(7, 8)
    restored = option_from_skill_payload(
        option_skill_payload(source), "cpu")
    features = torch.randn(13, 7)
    assert torch.equal(source(features), restored(features))
