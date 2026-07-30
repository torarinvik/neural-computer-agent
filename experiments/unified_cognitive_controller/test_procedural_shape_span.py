import torch

from .environment import NULL_ACTION
from .model import UnifiedCognitiveController
from .train_procedural_shape_span import (
    ShapeNuisance, binary_outcome_complete_targets,
    generate_procedural_shape_batch, nuisance_from_level,
    nuisance_with_overrides, rollout_procedural_shape_span)


def test_shape_batch_is_deterministic_balanced_and_independently_rendered() -> None:
    nuisance = nuisance_from_level(0.0)
    first = generate_procedural_shape_batch(
        32, span=2, vocabulary=2, seed=27001, nuisance=nuisance)
    again = generate_procedural_shape_batch(
        32, span=2, vocabulary=2, seed=27001, nuisance=nuisance)
    assert torch.equal(first.presentation_frames, again.presentation_frames)
    assert torch.equal(first.query_frames, again.query_frames)
    assert torch.equal(first.query_ordinals, again.query_ordinals)
    assert first.correct_actions.flatten().bincount(
        minlength=2).tolist() == [32, 32]
    assert all(
        first.correct_actions[:, ordinal].bincount(
            minlength=2).tolist() == [16, 16]
        for ordinal in range(2))
    for ordinal in range(2):
        for candidate in range(2):
            rows = first.candidate_identities[:, ordinal] == candidate
            assert first.correct_actions[
                rows, ordinal].bincount(minlength=2).tolist() == [8, 8]
    sequence_ids = first.sequence_identities[:, 0] * 2
    sequence_ids += first.sequence_identities[:, 1]
    assert sequence_ids.bincount(minlength=4).tolist() == [8, 8, 8, 8]
    # A matching candidate is a fresh render, never the stored bitmap.
    matching = first.correct_actions.bool()
    repeated = first.presentation_frames[
        torch.arange(32).unsqueeze(1), first.query_ordinals]
    assert not torch.equal(repeated[matching], first.query_frames[matching])


def test_binary_outcome_completion_uses_only_action_and_success() -> None:
    attempts = torch.tensor([0, 0, 1, 1])
    outcomes = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert binary_outcome_complete_targets(
        attempts, outcomes).tolist() == [0, 1, 1, 0]


def test_nuisance_level_has_nonzero_floor_and_monotonic_axes() -> None:
    floor = nuisance_from_level(0.0)
    middle = nuisance_from_level(0.5)
    maximum = nuisance_from_level(1.0)
    assert floor.position_px > 0
    assert floor.size_fraction > 0
    assert floor.rotation_degrees > 0
    for field in (
            "position_px", "size_fraction", "rotation_degrees",
            "color_delta", "background_delta", "deformation"):
        assert getattr(floor, field) <= getattr(middle, field)
        assert getattr(middle, field) <= getattr(maximum, field)


def test_nuisance_overrides_change_only_selected_axes() -> None:
    floor = nuisance_from_level(0.0)
    rotation = nuisance_with_overrides(floor, rotation_degrees=19.0)
    assert rotation.rotation_degrees == 19.0
    assert rotation.position_px == floor.position_px
    assert rotation.size_fraction == floor.size_fraction
    assert rotation.color_delta == floor.color_delta
    assert rotation.background_delta == floor.background_delta
    assert rotation.deformation == floor.deformation


def test_blank_presentation_removes_evidence_only() -> None:
    nuisance = ShapeNuisance()
    normal = generate_procedural_shape_batch(
        32, span=2, vocabulary=2, seed=27002, nuisance=nuisance)
    blank = generate_procedural_shape_batch(
        32, span=2, vocabulary=2, seed=27002, nuisance=nuisance,
        blank_presentation=True)
    assert not torch.equal(
        normal.presentation_frames, blank.presentation_frames)
    assert torch.equal(normal.query_frames, blank.query_frames)
    assert torch.equal(normal.correct_actions, blank.correct_actions)
    assert torch.equal(
        normal.sequence_identities, blank.sequence_identities)


def test_visible_identity_precursor_uses_candidate_not_private_memory() -> None:
    batch = generate_procedural_shape_batch(
        32, span=1, vocabulary=2, seed=27005,
        nuisance=ShapeNuisance(), objective="visible_identity")
    assert torch.equal(
        batch.correct_actions, batch.candidate_identities)
    assert batch.correct_actions.flatten().bincount(
        minlength=2).tolist() == [16, 16]


def test_counterfactuals_are_pixel_rerenders_with_recomputed_answers() -> None:
    nuisance = ShapeNuisance()
    normal = generate_procedural_shape_batch(
        32, span=2, vocabulary=2, seed=27003, nuisance=nuisance)
    reversed_batch = generate_procedural_shape_batch(
        32, span=2, vocabulary=2, seed=27003, nuisance=nuisance,
        reverse_presentation=True)
    flipped = generate_procedural_shape_batch(
        32, span=2, vocabulary=2, seed=27003, nuisance=nuisance,
        flip_candidates=True)
    assert torch.equal(
        normal.sequence_identities.flip(1),
        reversed_batch.sequence_identities)
    assert not torch.equal(
        normal.presentation_frames, reversed_batch.presentation_frames)
    assert torch.equal(normal.query_frames, reversed_batch.query_frames)
    assert torch.equal(normal.query_ordinals, reversed_batch.query_ordinals)
    assert torch.equal(
        normal.candidate_identities, reversed_batch.candidate_identities)
    assert torch.equal(
        normal.correct_actions, 1 - flipped.correct_actions)
    assert not torch.equal(normal.query_frames, flipped.query_frames)


def test_rollout_uses_literal_model_device_fast_memory() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=2, intention_width=8)
    batch = generate_procedural_shape_batch(
        32, span=2, vocabulary=2, seed=27004,
        nuisance=nuisance_from_level(0.0))
    result = rollout_procedural_shape_span(
        model, batch, sample_actions=False)
    assert result["final_hidden"].device == model.actuator.weight.device
    assert result["final_workspace"].shape == (32, 2, 32)
    assert result["actions"].shape == (32, 2)


def test_addressed_workspace_breaks_content_addressing_symmetry() -> None:
    batch = generate_procedural_shape_batch(
        384, span=3, vocabulary=2, seed=27010,
        nuisance=nuisance_from_level(0.135), query_count=2,
        new_slot_difficulty=0.3)
    null = torch.full((batch.batch_size,), NULL_ACTION, dtype=torch.long)
    zeros = torch.zeros(batch.batch_size)
    collapsed = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    addressed = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        workspace_slot_addressing=True)
    collapsed_state = collapsed.initial_state(
        batch.batch_size, device="cpu")
    addressed_state = addressed.initial_state(
        batch.batch_size, device="cpu")

    for index in range(batch.span):
        _, collapsed_state = collapsed.step(
            batch.presentation_frames[:, index], collapsed_state,
            null, zeros, zeros)
        _, addressed_state = addressed.step(
            batch.presentation_frames[:, index], addressed_state,
            null, zeros, zeros)

    assert torch.equal(
        collapsed_state.workspace,
        collapsed_state.workspace[:, :1].expand_as(
            collapsed_state.workspace))
    assert not torch.equal(
        addressed_state.workspace,
        addressed_state.workspace[:, :1].expand_as(
            addressed_state.workspace))


def test_single_query_span_three_still_balances_every_ordinal() -> None:
    batch = generate_procedural_shape_batch(
        384, span=3, query_count=1, vocabulary=2, seed=27006,
        nuisance=nuisance_from_level(0.0))
    assert batch.presentation_frames.shape[:2] == (384, 3)
    assert batch.query_frames.shape[:2] == (384, 1)
    assert batch.correct_actions.flatten().bincount(
        minlength=2).tolist() == [192, 192]
    assert batch.query_ordinals.flatten().bincount(
        minlength=3).tolist() == [128, 128, 128]


def test_zero_difficulty_third_slot_is_redundant_but_fully_visible() -> None:
    batch = generate_procedural_shape_batch(
        384, span=3, query_count=1, vocabulary=2, seed=27007,
        nuisance=nuisance_from_level(0.0), new_slot_difficulty=0.0)
    assert torch.equal(
        batch.sequence_identities[:, 2],
        batch.sequence_identities[:, 0])
    third = batch.presentation_frames[:, 2]
    assert not torch.equal(third, third[:, :, :1, :1].expand_as(third))
    assert batch.correct_actions.flatten().bincount(
        minlength=2).tolist() == [192, 192]


def test_zero_difficulty_teaches_every_ordinal_with_redundant_content() -> None:
    floor = generate_procedural_shape_batch(
        384, span=3, query_count=2, vocabulary=2, seed=27008,
        nuisance=nuisance_from_level(0.0), new_slot_difficulty=0.0)
    assert floor.query_ordinals.flatten().bincount(
        minlength=3).tolist() == [256, 256, 256]
    assert floor.correct_actions.flatten().bincount(
        minlength=2).tolist() == [384, 384]
