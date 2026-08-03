import torch

from .train_option_composition_race import (
    OptionValueHead,
    champion_actions,
    option_physical_actions,
)
from .train_shadow_compute_advantage import ComputeAdvantageHead


def test_default_option_reuses_champion_exactly() -> None:
    torch.manual_seed(13)
    champion = ComputeAdvantageHead(8)
    option = OptionValueHead(input_width=7, hidden=8)
    features = torch.randn(41, 7)
    assert torch.equal(
        option_physical_actions(option, champion, features),
        champion_actions(champion, features))
