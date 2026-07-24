from __future__ import annotations

from pathlib import Path

import torch

from .audit_color_primitive_compounding import (
    _balanced_data,
    _decorrelated_control_rewards,
    _logged_outcomes,
    load_color_compounder_checkpoint,
)
from .audit_identify_near_transfer import private_label


def test_color_compounding_data_is_exactly_balanced() -> None:
    for task in ("target_side", "effect_side", "effect_target_match"):
        data = _balanced_data(
            231_000_000, 64, task=task, heldout=True,
            seed=17, relation_axis="color_salient")
        assert torch.bincount(
            private_label(data, task), minlength=2).tolist() == [32, 32]


def test_logged_atom_outcomes_expose_only_attempted_answer() -> None:
    labels = torch.tensor([0, 1] * 16)
    attempted, rewards, order = _logged_outcomes(labels, seed=23)
    assert torch.equal(
        rewards, (attempted == labels[order]).float())
    assert sorted(order.tolist()) == list(range(labels.shape[0]))


def test_shuffled_control_is_exactly_decorrelated_from_private_labels() -> None:
    target = torch.tensor([0, 1, 0, 1] * 16)
    effect = torch.tensor([0, 0, 1, 1] * 16)
    attempted = torch.randint(
        0, 2, (64,), generator=torch.Generator().manual_seed(29))
    rewards = _decorrelated_control_rewards(
        target, effect, attempted, seed=31)
    implied = torch.where(rewards.bool(), attempted, 1 - attempted)
    assert torch.bincount(implied, minlength=2).tolist() == [32, 32]
    joint = effect * 2 + target
    for value in range(4):
        local = implied[joint == value]
        assert abs(int(local.sum()) * 2 - local.numel()) <= 1


def test_curated_color_compounder_checkpoint_is_loadable() -> None:
    path = (
        Path(__file__).parents[2]
        / "artifacts/checkpoints"
        / "color_primitive_compounder_bits16_seed1901.pt")
    loaded = load_color_compounder_checkpoint(
        path, device=torch.device("cpu"))
    assert loaded["source"]["stable_relation_reward_bits"] == 16
    with torch.no_grad():
        relation = loaded["relation_head"](torch.randn(3, 4))
    assert relation.shape == (3, 2)
    assert torch.equal(relation[:, 0], -relation[:, 1])
