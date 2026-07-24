import torch

from .race_variance_decomposition import (
    ANCHOR_SEED,
    HORSES,
    causal_floor,
    state_is_bit_identical,
)


def test_horse_population_changes_exactly_one_factor() -> None:
    anchor = HORSES[0]
    fields = (
        "core_init_seed", "pretrain_sampling_seed",
        "readout_init_seed", "readout_sampling_seed")
    assert len(HORSES) == 9
    assert len({horse.name for horse in HORSES}) == len(HORSES)
    for horse in HORSES[1:]:
        changed = [
            field for field in fields
            if getattr(horse, field) != getattr(anchor, field)]
        assert len(changed) == 1
    assert anchor.level == ANCHOR_SEED


def test_causal_floor_uses_weakest_required_behavior() -> None:
    audit = {
        "normal_accuracy": 0.9,
        "protocol_swap_accuracy": 0.8,
        "protocol_swap_prediction_flip": 0.7,
        "target_reverse_accuracy": 0.85,
        "target_reverse_prediction_flip": 0.75,
    }
    assert causal_floor(audit) == 0.7


def test_retention_comparison_requires_exact_tensor_equality() -> None:
    before = {"weight": torch.tensor([1.0, 2.0])}
    assert state_is_bit_identical(
        before, {"weight": torch.tensor([1.0, 2.0])})
    assert not state_is_bit_identical(
        before, {"weight": torch.tensor([1.0, 2.001])})
