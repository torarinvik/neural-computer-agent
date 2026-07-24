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
