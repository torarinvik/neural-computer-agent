from __future__ import annotations

import torch

from .environment import generate_lifetimes


def test_blended_magnitude_counterfactual_is_pixel_valid() -> None:
    normal = generate_lifetimes(
        64, 6, seed=22501, heldout=True,
        task="visible_pair_magnitude", appearance="bars",
        appearance_blend=0.15625)
    reversed_order = generate_lifetimes(
        64, 6, seed=22501, heldout=True,
        task="visible_pair_magnitude", appearance="bars",
        appearance_blend=0.15625, reverse_contexts=True)
    assert torch.equal(normal.rule_bits, reversed_order.rule_bits)
    assert torch.equal(
        normal.stimulus_identities,
        reversed_order.stimulus_identities)
    assert torch.equal(
        reversed_order.correct_actions, 1 - normal.correct_actions)
    assert not torch.equal(normal.frames, reversed_order.frames)
