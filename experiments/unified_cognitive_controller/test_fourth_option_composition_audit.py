from pathlib import Path

import torch

from .audit_fourth_option_composition import (
    load_router,
    router_from_skill_payload,
    router_skill_payload,
)
from .train_option_composition_race import OptionValueHead


def test_fourth_router_checkpoint_and_skill_round_trip(
        tmp_path: Path) -> None:
    source = OptionValueHead(9, 8)
    checkpoint = tmp_path / "router.pt"
    torch.save({
        "schema": "fourth-option-router-v1",
        "input_width": 9,
        "hidden": 8,
        "state_dict": source.state_dict(),
    }, checkpoint)
    from_checkpoint = load_router(checkpoint, torch.device("cpu"))
    from_skill = router_from_skill_payload(
        router_skill_payload(source), "cpu")
    features = torch.randn(17, 9)
    assert torch.equal(source(features), from_checkpoint(features))
    assert torch.equal(source(features), from_skill(features))
