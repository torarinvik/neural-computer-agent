"""Contracts for the first relative-magnitude primitive."""
from __future__ import annotations

import torch

from .environment import (
    _MAGNITUDE_MASK_BANKS, _MASK_BANKS, _magnitude_level_indices,
    generate_lifetimes)
from .model import UnifiedCognitiveController
from .train import evaluate
from .train_fourth_primitive_transfer import (
    _operation_cue_ablation_accuracy)


def test_pair_magnitude_is_balanced_deterministic_and_hidden_rule_bound() -> None:
    first = generate_lifetimes(
        32, 6, seed=21101, task="pair_magnitude")
    second = generate_lifetimes(
        32, 6, seed=21101, task="pair_magnitude")
    assert torch.equal(first.frames, second.frames)
    assert torch.equal(first.correct_actions, second.correct_actions)
    assert first.context_ids is not None
    assert all(set(row.tolist()) == {0, 1} for row in first.context_ids)
    assert torch.equal(
        first.correct_actions,
        first.context_ids ^ first.rule_bits.unsqueeze(1))


def test_magnitude_positions_do_not_change_legacy_renderer_banks() -> None:
    assert all(bank.shape[1] == 4 for bank in _MASK_BANKS.values())
    assert all(
        bank.shape[1] == 8 for bank in _MAGNITUDE_MASK_BANKS.values())
    for appearance in _MASK_BANKS:
        assert torch.equal(
            _MASK_BANKS[appearance],
            _MAGNITUDE_MASK_BANKS[appearance][:, :4])


def test_pair_magnitude_blend_has_exact_distinct_endpoints() -> None:
    bars = generate_lifetimes(
        32, 6, seed=21108, task="visible_pair_magnitude",
        appearance="bars")
    zero = generate_lifetimes(
        32, 6, seed=21108, task="visible_pair_magnitude",
        appearance="diamonds", appearance_blend=0.0)
    diamonds = generate_lifetimes(
        32, 6, seed=21108, task="visible_pair_magnitude",
        appearance="diamonds")
    one = generate_lifetimes(
        32, 6, seed=21108, task="visible_pair_magnitude",
        appearance="bars", appearance_blend=1.0)
    middle = generate_lifetimes(
        32, 6, seed=21108, task="visible_pair_magnitude",
        appearance="bars", appearance_blend=0.5)
    assert torch.equal(zero.frames, bars.frames)
    assert torch.equal(one.frames, diamonds.frames)
    assert not torch.equal(middle.frames, bars.frames)
    assert not torch.equal(middle.frames, diamonds.frames)
    assert torch.equal(middle.correct_actions, bars.correct_actions)


def test_pair_magnitude_counterfactual_swaps_order_and_every_answer() -> None:
    normal = generate_lifetimes(
        32, 6, seed=21102, heldout=True, task="pair_magnitude")
    swapped = generate_lifetimes(
        32, 6, seed=21102, heldout=True, task="pair_magnitude",
        reverse_contexts=True)
    assert torch.equal(
        normal.stimulus_identities, swapped.stimulus_identities)
    assert torch.equal(normal.rule_bits, swapped.rule_bits)
    assert normal.context_ids is not None
    assert swapped.context_ids is not None
    assert torch.equal(swapped.context_ids, 1 - normal.context_ids)
    assert torch.equal(
        swapped.correct_actions, 1 - normal.correct_actions)
    assert not torch.equal(normal.frames, swapped.frames)


def test_pair_magnitude_heldout_positions_preserve_logic() -> None:
    train = generate_lifetimes(
        32, 6, seed=21103, task="pair_magnitude")
    heldout = generate_lifetimes(
        32, 6, seed=21103, heldout=True, task="pair_magnitude")
    assert torch.equal(train.correct_actions, heldout.correct_actions)
    assert torch.equal(train.context_ids, heldout.context_ids)
    assert not torch.equal(train.frames, heldout.frames)


def test_position_holdout_is_independent_of_palette_holdout() -> None:
    train_position = generate_lifetimes(
        32, 6, seed=21106, task="pair_magnitude",
        position_holdout=False)
    alternate_position = generate_lifetimes(
        32, 6, seed=21106, task="pair_magnitude",
        position_holdout=True)
    assert torch.equal(
        train_position.correct_actions, alternate_position.correct_actions)
    assert torch.equal(
        train_position.context_ids, alternate_position.context_ids)
    assert not torch.equal(train_position.frames, alternate_position.frames)


def test_pair_magnitude_uses_hidden_rule_audit_contract() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    report = evaluate(
        model, count=32, trials=6, seed=21104,
        device=torch.device("cpu"), task="pair_magnitude",
        feedback_trials=1)
    assert "overall_accuracy" not in report
    assert "zero_shot_near_chance" in report["gate"]
    assert "counterfactual_flip_at_least_80" in report["gate"]


def test_pair_magnitude_ablation_removes_second_object() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    accuracy = _operation_cue_ablation_accuracy(
        model, count=32, seed=21105, device=torch.device("cpu"),
        support_trials=1, new_task="pair_magnitude",
        appearance="bars")
    assert 0.0 <= accuracy <= 1.0


def test_visible_pair_magnitude_is_the_direct_atom() -> None:
    normal = generate_lifetimes(
        32, 6, seed=21107, task="visible_pair_magnitude")
    swapped = generate_lifetimes(
        32, 6, seed=21107, task="visible_pair_magnitude",
        reverse_contexts=True)
    assert normal.context_ids is not None
    assert torch.equal(normal.correct_actions, normal.context_ids)
    assert torch.equal(swapped.correct_actions, 1 - normal.correct_actions)
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    report = evaluate(
        model, count=32, trials=6, seed=21108,
        device=torch.device("cpu"), task="visible_pair_magnitude",
        feedback_trials=1)
    assert "overall_accuracy" in report
    assert "pixel_counterfactual_flip_at_least_80" in report["gate"]


def test_single_object_absolute_size_is_bounded_to_62_point_5_percent() -> None:
    intervals = torch.arange(4).repeat_interleave(2)
    relations = torch.tensor([0, 1]).repeat(4)
    first_levels, second_levels = _magnitude_level_indices(
        intervals, relations)
    assert torch.equal(
        (first_levels > second_levels).long(), relations)
    # The Bayes-optimal lookup using either absolute size alone can exploit
    # only the two endpoint levels.  Three interior levels remain balanced.
    for levels in (first_levels, second_levels):
        correct = 0
        for level in range(5):
            labels = relations[levels == level]
            correct += max(
                int((labels == 0).sum()), int((labels == 1).sum()))
        assert correct / len(relations) == 0.625
