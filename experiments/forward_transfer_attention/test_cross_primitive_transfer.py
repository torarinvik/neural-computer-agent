import torch

from .train_cross_primitive_transfer import spatial_policy_sequences
from .train_zero_label_predictive_state import TEST_PALETTES


def test_spatial_mirror_flips_only_private_relation() -> None:
    frames, rules = spatial_policy_sequences(
        73_000_000, 6, heldout=True, palettes=TEST_PALETTES)
    mirrored, mirrored_rules = spatial_policy_sequences(
        73_000_000, 6, heldout=True, palettes=TEST_PALETTES,
        mirror=True)
    assert frames.shape == mirrored.shape == (6, 2, 3, 96, 160)
    assert torch.equal(mirrored_rules, 1 - rules)
    # Feedback is selected identity and must stay bit-identical.
    assert torch.equal(frames[:, 1], mirrored[:, 1])
    # The simultaneous object frame must actually change.
    assert torch.all(
        (frames[:, 0] != mirrored[:, 0]).flatten(1).any(1))


def test_missing_feedback_removes_exactly_one_frame() -> None:
    full, rules = spatial_policy_sequences(
        73_000_000, 6, heldout=True, palettes=TEST_PALETTES)
    missing, missing_rules = spatial_policy_sequences(
        73_000_000, 6, heldout=True, palettes=TEST_PALETTES,
        omit_feedback=True)
    assert missing.shape == (6, 1, 3, 96, 160)
    assert torch.equal(full[:, :1], missing)
    assert torch.equal(rules, missing_rules)


def test_spatial_generator_is_balanced() -> None:
    _, rules = spatial_policy_sequences(
        73_000_000, 60, heldout=True, palettes=TEST_PALETTES)
    assert torch.bincount(rules, minlength=2).tolist() == [30, 30]
