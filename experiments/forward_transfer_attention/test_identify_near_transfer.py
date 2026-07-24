from __future__ import annotations

import torch

from .audit_identify_near_transfer import (
    _balanced_indices,
    _subset_data,
    private_label,
    transfer_features,
)
from .train_identify_then_act import (
    ActionHistoryCore,
    identify_batch,
    make_readout,
)


def test_near_transfer_private_labels_are_balanced() -> None:
    pool = identify_batch(177_000_000, 128, heldout=True)
    for task in ("target_side", "effect_side", "effect_target_match"):
        indices = _balanced_indices(
            private_label(pool, task), 64, seed=31)
        data = _subset_data(pool, indices)
        labels = private_label(data, task)
        assert torch.bincount(labels, minlength=2).tolist() == [32, 32]


def test_valid_counterfactuals_flip_corresponding_labels() -> None:
    normal = identify_batch(179_000_000, 64, heldout=True)
    target_reversed = identify_batch(
        179_000_000, 64, heldout=True, reverse_target=True)
    protocol_swapped = identify_batch(
        179_000_000, 64, heldout=True, swap_protocol=True)
    assert torch.equal(
        private_label(target_reversed, "target_side"),
        1 - private_label(normal, "target_side"))
    for task in ("effect_side", "effect_target_match"):
        assert torch.equal(
            private_label(protocol_swapped, task),
            1 - private_label(normal, task))


def test_appearance_bridge_changes_only_rendered_pixels() -> None:
    baseline = identify_batch(
        181_000_000, 64, heldout=True, appearance_style="baseline")
    for style in ("palette", "shape", "combined"):
        shifted = identify_batch(
            181_000_000, 64, heldout=True, appearance_style=style)
        assert not torch.equal(shifted["frames"], baseline["frames"])
        for key in (
                "transition_actions", "previous_actions",
                "attempted_actions", "rewards", "correct_actions",
                "probe_actions", "private_protocol_ids"):
            assert torch.equal(shifted[key], baseline[key]), (style, key)


def test_appearance_bridge_preserves_counterfactual_pairing() -> None:
    for style in ("palette", "shape", "combined"):
        normal = identify_batch(
            183_000_000, 64, heldout=True, appearance_style=style)
        swapped = identify_batch(
            183_000_000, 64, heldout=True, appearance_style=style,
            swap_protocol=True)
        assert torch.equal(
            private_label(swapped, "effect_target_match"),
            1 - private_label(normal, "effect_target_match"))


def test_color_identity_bridge_changes_only_public_pixels() -> None:
    position = identify_batch(
        185_000_000, 64, heldout=True, relation_axis="position")
    for axis in ("color_salient", "color_fixed", "color_varied"):
        colors = identify_batch(
            185_000_000, 64, heldout=True, relation_axis=axis)
        assert not torch.equal(colors["frames"], position["frames"])
        for key in (
                "transition_actions", "previous_actions",
                "attempted_actions", "rewards", "correct_actions",
                "probe_actions", "private_protocol_ids"):
            assert torch.equal(colors[key], position[key]), (axis, key)


def test_color_identity_bridge_preserves_true_counterfactuals() -> None:
    for axis in ("color_salient", "color_fixed", "color_varied"):
        normal = identify_batch(
            187_000_000, 64, heldout=True, relation_axis=axis)
        swapped = identify_batch(
            187_000_000, 64, heldout=True, relation_axis=axis,
            swap_protocol=True)
        reversed_target = identify_batch(
            187_000_000, 64, heldout=True, relation_axis=axis,
            reverse_target=True)
        for changed in (swapped, reversed_target):
            assert torch.equal(
                private_label(changed, "effect_target_match"),
                1 - private_label(normal, "effect_target_match"))


def test_missing_target_removes_pixels_without_changing_private_problem() -> None:
    normal = identify_batch(
        188_000_000, 64, heldout=True, relation_axis="color_salient")
    missing = identify_batch(
        188_000_000, 64, heldout=True, relation_axis="color_salient",
        missing_target=True)
    assert torch.equal(
        private_label(normal, "effect_target_match"),
        private_label(missing, "effect_target_match"))
    for key in (
            "transition_actions", "previous_actions", "attempted_actions",
            "rewards", "correct_actions", "probe_actions",
            "private_protocol_ids"):
        assert torch.equal(normal[key], missing[key]), key
    assert torch.equal(normal["frames"][:, :2], missing["frames"][:, :2])
    assert not torch.equal(normal["frames"][:, 2:], missing["frames"][:, 2:])


def test_antisymmetric_readout_has_one_exclusive_preference_axis() -> None:
    model = make_readout("antisymmetric", hidden=11, intention_width=7)
    logits = model(torch.randn(5, 11))
    assert logits.shape == (5, 2)
    assert torch.equal(logits[:, 0], -logits[:, 1])


def test_event_vision_interface_preserves_every_public_event_embedding() -> None:
    data = identify_batch(
        189_000_000, 8, heldout=True, relation_axis="color_fixed")
    core = ActionHistoryCore(64)
    event = transfer_features(
        core, data, interface="event_vision", device=torch.device("cpu"))
    decision_event = transfer_features(
        core, data, interface="decision_event_vision",
        device=torch.device("cpu"))
    assert event.shape == (8, 64 * 3)
    assert decision_event.shape == (8, 64 * 6)
