import torch

from .model import UnifiedCognitiveController
from .train_procedural_shape_span import (
    ShapeNuisance, generate_procedural_shape_batch, nuisance_from_level,
    rollout_procedural_shape_span)


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
