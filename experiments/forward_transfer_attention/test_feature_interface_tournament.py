import torch

from .train_feature_interface_tournament import (
    BLIND_START,
    CANDIDATES,
    REFINED_CANDIDATES,
    feature_interface,
)
from .train_identify_then_act import TEST_START


def test_population_has_eight_unique_clones() -> None:
    assert len(CANDIDATES) == 8
    assert len({candidate.name for candidate in CANDIDATES}) == 8
    assert len(REFINED_CANDIDATES) == 8
    assert len({
        candidate.name for candidate in REFINED_CANDIDATES}) == 8


def test_feature_interfaces_have_expected_dimensions() -> None:
    features = torch.randn(5, 192)
    expected = {
        "concat": 192,
        "state": 64,
        "delta": 64,
        "state_delta": 128,
        "state_delta_product": 192,
        "consequence_relation": 256,
        "state_pairwise": 320,
    }
    for interface, width in expected.items():
        assert feature_interface(features, interface).shape == (5, width)


def test_blind_generator_range_is_disjoint_from_selection() -> None:
    assert BLIND_START >= TEST_START + 1_000_000
