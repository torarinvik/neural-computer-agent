"""Contracts for the adjacent discrete-numerosity primitive."""
from __future__ import annotations

import torch

from .environment import (
    _NUMEROSITY_MASK_BANK, _numerosity_count_indices,
    _numerosity_mass_scale, generate_lifetimes)
from .model import UnifiedCognitiveController
from .train import evaluate
from .train_fourth_primitive_transfer import (
    _operation_cue_ablation_accuracy)


def test_numerosity_is_deterministic_balanced_and_counterfactual() -> None:
    normal = generate_lifetimes(
        32, 6, seed=23201, task="visible_pair_numerosity")
    duplicate = generate_lifetimes(
        32, 6, seed=23201, task="visible_pair_numerosity")
    reversed_batch = generate_lifetimes(
        32, 6, seed=23201, task="visible_pair_numerosity",
        reverse_contexts=True)
    assert torch.equal(normal.frames, duplicate.frames)
    assert torch.equal(normal.correct_actions, duplicate.correct_actions)
    assert normal.context_ids is not None
    assert all(set(row.tolist()) == {0, 1} for row in normal.context_ids)
    assert torch.equal(normal.correct_actions, normal.context_ids)
    assert torch.equal(
        reversed_batch.correct_actions, 1 - normal.correct_actions)
    assert torch.equal(
        reversed_batch.context_ids, 1 - normal.context_ids)
    assert not torch.equal(normal.frames, reversed_batch.frames)


def test_numerosity_heldout_layouts_preserve_answers() -> None:
    training = generate_lifetimes(
        32, 6, seed=23202, task="visible_pair_numerosity",
        position_holdout=False)
    heldout = generate_lifetimes(
        32, 6, seed=23202, task="visible_pair_numerosity",
        position_holdout=True)
    assert torch.equal(training.correct_actions, heldout.correct_actions)
    assert torch.equal(training.context_ids, heldout.context_ids)
    assert not torch.equal(training.frames, heldout.frames)


def test_full_mass_control_equalizes_integrated_dot_opacity() -> None:
    for side in range(2):
        totals = []
        for count_index in range(5):
            mask = _NUMEROSITY_MASK_BANK[side, count_index, 0]
            scale = _numerosity_mass_scale(
                torch.tensor(count_index), 1.0)
            totals.append(float((mask * scale).sum()))
        assert max(totals) - min(totals) < 1e-6


def test_one_count_alone_is_bounded_to_62_point_5_percent() -> None:
    intervals = torch.arange(4).repeat_interleave(2)
    relations = torch.tensor([0, 1]).repeat(4)
    first_counts, second_counts = _numerosity_count_indices(
        intervals, relations)
    assert torch.equal(
        (first_counts < second_counts).long(), relations)
    for counts in (first_counts, second_counts):
        correct = 0
        for count in range(5):
            labels = relations[counts == count]
            correct += max(
                int((labels == 0).sum()),
                int((labels == 1).sum()))
        assert correct / len(relations) == 0.625


def test_numerosity_uses_direct_visual_gate_and_second_field() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    report = evaluate(
        model, count=32, trials=6, seed=23203,
        device=torch.device("cpu"), task="visible_pair_numerosity",
        feedback_trials=1)
    assert "overall_accuracy" in report
    assert "pixel_counterfactual_flip_at_least_80" in report["gate"]
    accuracy = _operation_cue_ablation_accuracy(
        model, count=32, seed=23204, device=torch.device("cpu"),
        support_trials=1, new_task="visible_pair_numerosity")
    assert 0.0 <= accuracy <= 1.0


def test_mass_control_is_private_difficulty_not_answer_information() -> None:
    opaque = generate_lifetimes(
        32, 6, seed=23205, task="visible_pair_numerosity",
        numerosity_mass_control=0.0)
    normalized = generate_lifetimes(
        32, 6, seed=23205, task="visible_pair_numerosity",
        numerosity_mass_control=1.0)
    assert torch.equal(opaque.correct_actions, normalized.correct_actions)
    assert torch.equal(opaque.context_ids, normalized.context_ids)
    assert not torch.equal(opaque.frames, normalized.frames)


def test_appearance_bridge_preserves_answers_while_changing_pixels() -> None:
    magnitude_endpoint = generate_lifetimes(
        32, 6, seed=23206, task="visible_pair_numerosity",
        numerosity_appearance_blend=0.0)
    count_endpoint = generate_lifetimes(
        32, 6, seed=23206, task="visible_pair_numerosity",
        numerosity_appearance_blend=1.0)
    assert torch.equal(
        magnitude_endpoint.correct_actions, count_endpoint.correct_actions)
    assert torch.equal(
        magnitude_endpoint.context_ids, count_endpoint.context_ids)
    assert not torch.equal(
        magnitude_endpoint.frames, count_endpoint.frames)


def test_numerosity_equality_is_balanced_and_counterfactual() -> None:
    normal = generate_lifetimes(
        32, 6, seed=23207, task="visible_numerosity_equality")
    duplicate = generate_lifetimes(
        32, 6, seed=23207, task="visible_numerosity_equality")
    reversed_batch = generate_lifetimes(
        32, 6, seed=23207, task="visible_numerosity_equality",
        reverse_contexts=True)
    assert torch.equal(normal.frames, duplicate.frames)
    assert torch.equal(normal.correct_actions, duplicate.correct_actions)
    assert normal.context_ids is not None
    assert all(set(row.tolist()) == {0, 1} for row in normal.context_ids)
    assert torch.equal(normal.correct_actions, normal.context_ids)
    assert torch.equal(
        reversed_batch.correct_actions, 1 - normal.correct_actions)
    # The first count field and every nuisance draw stay fixed. Only the
    # second count field changes to make the counterfactual true.
    assert torch.equal(
        normal.frames[:, :, :, :, :16],
        reversed_batch.frames[:, :, :, :, :16])
    assert not torch.equal(normal.frames, reversed_batch.frames)


def test_numerosity_equality_has_a_public_nonanswer_operation_cue() -> None:
    equality = generate_lifetimes(
        32, 6, seed=23208, task="visible_numerosity_equality")
    cue = equality.frames[:, :, :, 5:29, 0:2]
    assert torch.equal(cue, torch.full_like(cue, 0.98))
    # The cue is constant across both verifier outcomes, so it announces the
    # operation but cannot announce the answer.
    assert torch.equal(
        cue[equality.correct_actions == 0].mean(),
        cue[equality.correct_actions == 1].mean())


def test_numerosity_equality_uses_direct_visual_gates() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    report = evaluate(
        model, count=32, trials=6, seed=23209,
        device=torch.device("cpu"), task="visible_numerosity_equality",
        feedback_trials=1)
    assert "overall_accuracy" in report
    assert "pixel_counterfactual_flip_at_least_80" in report["gate"]
    accuracy = _operation_cue_ablation_accuracy(
        model, count=32, seed=23210, device=torch.device("cpu"),
        support_trials=1, new_task="visible_numerosity_equality")
    assert 0.0 <= accuracy <= 1.0


def test_smaller_operation_is_cued_inverse_of_larger() -> None:
    larger = generate_lifetimes(
        32, 6, seed=23211, task="visible_pair_numerosity",
        numerosity_appearance_blend=0.248)
    smaller = generate_lifetimes(
        32, 6, seed=23211, task="visible_pair_numerosity_smaller",
        numerosity_appearance_blend=0.248)
    assert torch.equal(
        smaller.correct_actions, 1 - larger.correct_actions)
    assert torch.equal(smaller.context_ids, larger.context_ids)
    # The operation mark is the only pixel-level difference.
    unmarked_smaller = smaller.frames.clone()
    unmarked_larger = larger.frames.clone()
    unmarked_smaller[:, :, :, 5:29, 2:4] = 0
    unmarked_larger[:, :, :, 5:29, 2:4] = 0
    assert torch.equal(unmarked_smaller, unmarked_larger)


def test_smaller_operation_has_valid_counterfactual() -> None:
    normal = generate_lifetimes(
        32, 6, seed=23212, task="visible_pair_numerosity_smaller",
        numerosity_appearance_blend=0.248)
    reversed_batch = generate_lifetimes(
        32, 6, seed=23212, task="visible_pair_numerosity_smaller",
        numerosity_appearance_blend=0.248, reverse_contexts=True)
    assert torch.equal(
        reversed_batch.correct_actions, 1 - normal.correct_actions)
    assert not torch.equal(normal.frames, reversed_batch.frames)
