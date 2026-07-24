from __future__ import annotations

import torch

from .audit_identify_near_transfer import (
    _balanced_indices,
    _subset_data,
    private_label,
)
from .train_identify_then_act import identify_batch


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
