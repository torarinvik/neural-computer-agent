import torch

from .train_fourth_option_composition_race import (
    composed_physical_actions,
    previous_option_actions,
)
from .train_option_composition_race import OptionValueHead
from .train_shadow_compute_advantage import ComputeAdvantageHead


def test_default_second_generation_router_reuses_previous_option() -> None:
    torch.manual_seed(14)
    champion = ComputeAdvantageHead(8)
    previous = OptionValueHead(7, 8)
    router = OptionValueHead(9, 8)
    features = torch.randn(43, 9)
    assert torch.equal(
        composed_physical_actions(
            router, previous, champion, features),
        previous_option_actions(previous, champion, features))
