import torch

from .train_third_primitive_compounding import same_different_sequences
from .train_zero_label_predictive_state import TEST_PALETTES


def test_same_different_is_exactly_balanced() -> None:
    frames, rules = same_different_sequences(
        79_000_000, 60, heldout=True, palettes=TEST_PALETTES)
    assert frames.shape == (60, 2, 3, 96, 160)
    assert torch.bincount(rules, minlength=2).tolist() == [30, 30]


def test_identity_counterfactual_flips_only_second_frame_and_rule() -> None:
    frames, rules = same_different_sequences(
        79_000_000, 6, heldout=True, palettes=TEST_PALETTES)
    changed, changed_rules = same_different_sequences(
        79_000_000, 6, heldout=True, palettes=TEST_PALETTES,
        counterfactual=True)
    assert torch.equal(frames[:, 0], changed[:, 0])
    assert torch.all(
        (frames[:, 1] != changed[:, 1]).flatten(1).any(1))
    assert torch.equal(changed_rules, 1 - rules)


def test_missing_identity_frames_preserve_other_evidence() -> None:
    full, rules = same_different_sequences(
        79_000_000, 6, heldout=True, palettes=TEST_PALETTES)
    no_first, first_rules = same_different_sequences(
        79_000_000, 6, heldout=True, palettes=TEST_PALETTES,
        omit_first=True)
    no_second, second_rules = same_different_sequences(
        79_000_000, 6, heldout=True, palettes=TEST_PALETTES,
        omit_second=True)
    assert torch.equal(no_first[:, 0], full[:, 1])
    assert torch.equal(no_second[:, 0], full[:, 0])
    assert torch.equal(rules, first_rules)
    assert torch.equal(rules, second_rules)
