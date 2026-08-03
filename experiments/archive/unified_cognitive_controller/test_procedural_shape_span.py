import torch

from .environment import NULL_ACTION
from .legacy_model import UnifiedCognitiveController
from .train_procedural_shape_span import (
    ShapeNuisance, binary_outcome_complete_targets,
    evaluate_procedural_shape_span, generate_procedural_shape_batch,
    nuisance_from_level, project_gradient_against_reference,
    project_parameter_update_against_reference,
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


def test_gradient_projection_removes_only_conflicting_component() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([-1.0, 1.0])
    reference = {"value": torch.tensor([1.0, 0.0])}
    applied, cosine, post_dot = project_gradient_against_reference(
        [("value", parameter)], reference, 1.0)
    assert applied
    assert abs(cosine + 2 ** -0.5) < 1e-6
    assert torch.allclose(parameter.grad, torch.tensor([0.0, 1.0]))
    assert abs(post_dot) < 1e-6

    parameter.grad = torch.tensor([1.0, 1.0])
    applied, _, post_dot = project_gradient_against_reference(
        [("value", parameter)], reference, 1.0)
    assert not applied
    assert torch.equal(parameter.grad, torch.tensor([1.0, 1.0]))
    assert post_dot == 1.0


def test_optimizer_update_projection_removes_harmful_component() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    before = {"value": torch.zeros(2)}
    reference = {"value": torch.tensor([1.0, 0.0])}
    applied, cosine, post_dot = project_parameter_update_against_reference(
        [("value", parameter)], before, reference, 1.0)
    assert applied
    assert abs(cosine - 2 ** -0.5) < 1e-6
    assert torch.allclose(parameter, torch.tensor([0.0, 1.0]))
    assert abs(post_dot) < 1e-6

    with torch.no_grad():
        parameter.copy_(torch.tensor([-1.0, 1.0]))
    applied, _, post_dot = project_parameter_update_against_reference(
        [("value", parameter)], before, reference, 1.0)
    assert not applied
    assert torch.equal(parameter, torch.tensor([-1.0, 1.0]))
    assert post_dot == -1.0


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


def test_evaluation_exposes_every_query_position_by_ordinal_cell() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    audit = evaluate_procedural_shape_span(
        model, count=384, span=3, vocabulary=2, seed=27011,
        nuisance=nuisance_from_level(0.135), device=torch.device("cpu"),
        query_count=2)
    cells = audit[
        "accuracy_by_query_position_and_presented_ordinal"]
    assert len(cells) == 2
    assert all(len(row) == 3 for row in cells)
    assert all(value is not None for row in cells for value in row)
    assert audit["crossed_history_frontier_queries"] == 128
    assert audit["repeated_history_frontier_queries"] == 0

    next_audit = evaluate_procedural_shape_span(
        model, count=384, span=3, vocabulary=2, seed=27012,
        nuisance=nuisance_from_level(0.135), device=torch.device("cpu"),
        query_count=1, next_query_stage=2, next_query_anchor_focus=1)
    assert (
        next_audit["next_conflict_queries"]
        + next_audit["next_nonconflict_queries"]
        == 192)
    next_cells = next_audit["next_accuracy_by_conflict_and_action"]
    assert len(next_cells) == 2
    assert all(len(row) == 2 for row in next_cells)
    assert all(value is not None for row in next_cells for value in row)


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


def test_query_frontier_starts_without_third_ordinal_second_queries() -> None:
    floor = generate_procedural_shape_batch(
        384, span=3, query_count=2, vocabulary=2, seed=27012,
        nuisance=nuisance_from_level(0.135),
        query_frontier_difficulty=0.0)
    full = generate_procedural_shape_batch(
        384, span=3, query_count=2, vocabulary=2, seed=27012,
        nuisance=nuisance_from_level(0.135),
        query_frontier_difficulty=1.0)
    assert not bool((floor.query_ordinals[:, 1] == 2).any())
    assert int((full.query_ordinals[:, 1] == 2).sum()) == 128


def test_query_history_bridge_repeats_third_lookup_without_starving_it() -> None:
    repeat = generate_procedural_shape_batch(
        384, span=3, query_count=2, vocabulary=2, seed=27013,
        nuisance=nuisance_from_level(0.135),
        query_history_difficulty=0.0)
    crossed = generate_procedural_shape_batch(
        384, span=3, query_count=2, vocabulary=2, seed=27013,
        nuisance=nuisance_from_level(0.135),
        query_history_difficulty=1.0)
    repeat_frontier = repeat.query_ordinals[:, 1] == 2
    crossed_frontier = crossed.query_ordinals[:, 1] == 2
    assert int(repeat_frontier.sum()) == 128
    assert torch.equal(
        repeat.query_ordinals[repeat_frontier, 0],
        repeat.query_ordinals[repeat_frontier, 1])
    assert bool(
        (crossed.query_ordinals[crossed_frontier, 0] != 2).all())


def test_third_query_history_stages_are_minimal_and_deterministic() -> None:
    batches = [
        generate_procedural_shape_batch(
            384, span=3, query_count=3, vocabulary=2, seed=27014,
            nuisance=nuisance_from_level(0.135),
            third_query_history_stage=stage)
        for stage in range(3)]
    immediate, delayed, novel = batches
    assert torch.equal(
        immediate.query_ordinals[:, 2], immediate.query_ordinals[:, 1])
    assert torch.equal(
        delayed.query_ordinals[:, 2], delayed.query_ordinals[:, 0])
    assert bool(
        (novel.query_ordinals[:, 2] != novel.query_ordinals[:, 0]).all())
    assert bool(
        (novel.query_ordinals[:, 2] != novel.query_ordinals[:, 1]).all())


def test_previous_query_curriculum_changes_one_operation_at_a_time() -> None:
    batches = [
        generate_procedural_shape_batch(
            384, span=3, query_count=3, vocabulary=2, seed=27015,
            nuisance=nuisance_from_level(0.135),
            previous_query_stage=stage)
        for stage in range(3)]
    direct, first_anchor, both_anchors = batches
    assert not bool(direct.query_operations.any())
    assert torch.equal(direct.query_cue_ordinals, direct.query_ordinals)

    assert torch.equal(
        first_anchor.query_operations.bool(),
        first_anchor.query_ordinals == 0)
    assert torch.equal(
        first_anchor.query_cue_ordinals,
        first_anchor.query_ordinals + first_anchor.query_operations)

    assert bool(
        both_anchors.query_operations[both_anchors.query_ordinals == 0]
        .bool().all())
    assert not bool(
        both_anchors.query_operations[both_anchors.query_ordinals == 2]
        .bool().any())
    middle_operations = both_anchors.query_operations[
        both_anchors.query_ordinals == 1]
    assert middle_operations.bincount(minlength=2).tolist() == [192, 192]
    assert torch.equal(
        both_anchors.query_cue_ordinals,
        both_anchors.query_ordinals + both_anchors.query_operations)


def test_previous_query_atom_crosses_operations_on_span_two() -> None:
    batch = generate_procedural_shape_batch(
        384, span=2, query_count=1, vocabulary=2, seed=27019,
        nuisance=nuisance_from_level(0.135), previous_query_stage=1)
    assert batch.query_cue_ordinals.unique().tolist() == [1]
    assert batch.query_operations.flatten().bincount(
        minlength=2).tolist() == [192, 192]
    for operation in range(2):
        selected = batch.query_operations == operation
        assert batch.correct_actions[selected].bincount(
            minlength=2).tolist() == [96, 96]


def test_previous_query_scope_focuses_without_starving_either_operation() -> None:
    focused = generate_procedural_shape_batch(
        384, span=3, query_count=1, vocabulary=2, seed=27020,
        nuisance=nuisance_from_level(0.135), previous_query_stage=1,
        previous_query_scope_difficulty=0.0)
    full = generate_procedural_shape_batch(
        384, span=3, query_count=1, vocabulary=2, seed=27020,
        nuisance=nuisance_from_level(0.135), previous_query_stage=1,
        previous_query_scope_difficulty=1.0)
    assert focused.query_ordinals.flatten().bincount(
        minlength=3).tolist() == [192, 192, 0]
    assert focused.query_operations.flatten().bincount(
        minlength=2).tolist() == [192, 192]
    assert full.query_ordinals.flatten().bincount(
        minlength=3).tolist() == [128, 128, 128]
    for operation in range(2):
        selected = focused.query_operations == operation
        assert focused.correct_actions[selected].bincount(
            minlength=2).tolist() == [96, 96]


def test_previous_query_position_changes_only_history_depth() -> None:
    first = generate_procedural_shape_batch(
        384, span=3, query_count=2, vocabulary=2, seed=27021,
        nuisance=nuisance_from_level(0.135), previous_query_stage=1,
        previous_query_position=0)
    second = generate_procedural_shape_batch(
        384, span=3, query_count=2, vocabulary=2, seed=27021,
        nuisance=nuisance_from_level(0.135), previous_query_stage=1,
        previous_query_position=1)
    assert bool((first.query_ordinals[:, 0] == 0).all())
    assert bool((second.query_ordinals[:, 1] == 0).all())
    assert bool(first.query_operations[:, 0].bool().all())
    assert bool(second.query_operations[:, 1].bool().all())
    assert torch.equal(
        first.sequence_identities, second.sequence_identities)
    assert first.correct_actions.flatten().bincount(
        minlength=2).tolist() == [384, 384]
    assert second.correct_actions.flatten().bincount(
        minlength=2).tolist() == [384, 384]


def test_second_anchor_focus_crosses_previous_and_direct_with_same_cue() -> None:
    batch = generate_procedural_shape_batch(
        384, span=3, query_count=1, vocabulary=2, seed=27022,
        nuisance=nuisance_from_level(0.135), previous_query_stage=2,
        previous_query_anchor_focus=1)
    assert batch.query_ordinals.flatten().bincount(
        minlength=3).tolist() == [0, 192, 192]
    assert batch.query_operations.flatten().bincount(
        minlength=2).tolist() == [192, 192]
    assert batch.query_cue_ordinals.unique().tolist() == [2]
    assert bool(
        (batch.query_operations[batch.query_ordinals[:, :1] == 1] == 1)
        .all())
    for operation in range(2):
        selected = batch.query_operations == operation
        assert batch.correct_actions[selected].bincount(
            minlength=2).tolist() == [96, 96]


def test_second_anchor_can_be_forced_to_a_later_query_position() -> None:
    batch = generate_procedural_shape_batch(
        384, span=3, query_count=2, vocabulary=2, seed=27023,
        nuisance=nuisance_from_level(0.135), previous_query_stage=2,
        previous_query_anchor_focus=1, previous_query_position=1)
    assert bool((batch.query_ordinals[:, 1] == 1).all())
    assert bool(batch.query_operations[:, 1].bool().all())
    assert bool((batch.query_cue_ordinals[:, 1] == 2).all())
    assert not bool(batch.query_operations[:, 0].bool().any())
    assert batch.correct_actions.flatten().bincount(
        minlength=2).tolist() == [384, 384]


def test_previous_query_cues_disambiguate_shared_anchors() -> None:
    batch = generate_procedural_shape_batch(
        384, span=3, query_count=3, vocabulary=2, seed=27016,
        nuisance=nuisance_from_level(0.135), previous_query_stage=2)
    for cue in (1, 2):
        selected = batch.query_cue_ordinals == cue
        assert sorted(
            batch.query_operations[selected].unique().tolist()) == [0, 1]
        assert batch.correct_actions[selected].bincount(
            minlength=2).tolist() == [288, 288]


def test_operation_glyph_bridge_preserves_all_other_pixels() -> None:
    legacy = generate_procedural_shape_batch(
        384, span=3, query_count=3, vocabulary=2, seed=27017,
        nuisance=nuisance_from_level(0.135), previous_query_stage=-1)
    bridge = generate_procedural_shape_batch(
        384, span=3, query_count=3, vocabulary=2, seed=27017,
        nuisance=nuisance_from_level(0.135), previous_query_stage=0)
    mask = torch.ones_like(legacy.query_frames, dtype=torch.bool)
    mask[:, :, :, 2:8, 2:8] = False
    mask[:, :, :, 2:8, 24:30] = False
    assert torch.equal(legacy.query_frames[mask], bridge.query_frames[mask])
    assert not torch.equal(legacy.query_frames, bridge.query_frames)
    assert torch.equal(legacy.correct_actions, bridge.correct_actions)


def test_operation_flip_is_a_valid_pixel_level_counterfactual() -> None:
    normal = generate_procedural_shape_batch(
        384, span=3, query_count=3, vocabulary=2, seed=27018,
        nuisance=nuisance_from_level(0.135), previous_query_stage=2)
    flipped = generate_procedural_shape_batch(
        384, span=3, query_count=3, vocabulary=2, seed=27018,
        nuisance=nuisance_from_level(0.135), previous_query_stage=2,
        flip_query_operations=True)
    assert torch.equal(
        normal.query_operations + flipped.query_operations,
        torch.ones_like(normal.query_operations))
    assert torch.equal(
        normal.query_cue_ordinals, flipped.query_cue_ordinals)
    assert torch.equal(
        normal.candidate_identities, flipped.candidate_identities)
    changed = normal.correct_actions != flipped.correct_actions
    assert 0.45 < float(changed.float().mean()) < 0.55


def test_next_query_atom_crosses_operations_on_span_two() -> None:
    batch = generate_procedural_shape_batch(
        384, span=2, query_count=1, vocabulary=2, seed=27024,
        nuisance=nuisance_from_level(0.135), next_query_stage=1)
    assert batch.query_cue_ordinals.unique().tolist() == [0]
    assert batch.query_operations.flatten().bincount(
        minlength=3).tolist() == [192, 0, 192]
    assert batch.query_ordinals.flatten().bincount(
        minlength=2).tolist() == [192, 192]
    for operation in (0, 2):
        selected = batch.query_operations == operation
        assert batch.correct_actions[selected].bincount(
            minlength=2).tolist() == [96, 96]


def test_second_next_anchor_crosses_direct_and_next_with_same_cue() -> None:
    batch = generate_procedural_shape_batch(
        384, span=3, query_count=1, vocabulary=2, seed=27025,
        nuisance=nuisance_from_level(0.135), next_query_stage=2,
        next_query_anchor_focus=1)
    assert batch.query_ordinals.flatten().bincount(
        minlength=3).tolist() == [0, 192, 192]
    assert batch.query_operations.flatten().bincount(
        minlength=3).tolist() == [192, 0, 192]
    assert batch.query_cue_ordinals.unique().tolist() == [1]
    assert bool(
        (batch.query_operations[batch.query_ordinals[:, :1] == 2] == 2)
        .all())
    for operation in (0, 2):
        selected = batch.query_operations == operation
        assert batch.correct_actions[selected].bincount(
            minlength=2).tolist() == [96, 96]


def test_second_next_anchor_can_align_direct_and_next_targets() -> None:
    batch = generate_procedural_shape_batch(
        384, span=3, query_count=1, vocabulary=2, seed=27030,
        nuisance=nuisance_from_level(0.135), next_query_stage=2,
        next_query_anchor_focus=1, next_query_target_aligned=True)
    assert batch.query_ordinals.unique().tolist() == [2]
    assert batch.query_operations.flatten().bincount(
        minlength=3).tolist() == [192, 0, 192]
    direct = batch.query_operations == 0
    following = batch.query_operations == 2
    assert bool((batch.query_cue_ordinals[direct] == 2).all())
    assert bool((batch.query_cue_ordinals[following] == 1).all())
    for selected in (direct, following):
        assert batch.correct_actions[selected].bincount(
            minlength=2).tolist() == [96, 96]


def test_third_next_anchor_can_align_a_fourth_item_target() -> None:
    batch = generate_procedural_shape_batch(
        256, span=4, query_count=1, vocabulary=2, seed=27031,
        nuisance=nuisance_from_level(0.135), next_query_stage=2,
        next_query_anchor_focus=2, next_query_target_aligned=True)
    assert batch.query_ordinals.unique().tolist() == [3]
    assert batch.query_operations.flatten().bincount(
        minlength=3).tolist() == [128, 0, 128]
    direct = batch.query_operations == 0
    following = batch.query_operations == 2
    assert bool((batch.query_cue_ordinals[direct] == 3).all())
    assert bool((batch.query_cue_ordinals[following] == 2).all())
    for selected in (direct, following):
        assert batch.correct_actions[selected].bincount(
            minlength=2).tolist() == [64, 64]


def test_next_anchor_can_be_forced_to_a_later_query_position() -> None:
    batch = generate_procedural_shape_batch(
        384, span=3, query_count=2, vocabulary=2, seed=27026,
        nuisance=nuisance_from_level(0.135), next_query_stage=2,
        next_query_anchor_focus=1, next_query_position=1)
    assert bool((batch.query_ordinals[:, 1] == 2).all())
    assert bool((batch.query_operations[:, 1] == 2).all())
    assert bool((batch.query_cue_ordinals[:, 1] == 1).all())
    assert bool((batch.query_operations[:, 0] == 0).all())
    assert batch.correct_actions.flatten().bincount(
        minlength=2).tolist() == [384, 384]


def test_next_query_stage_two_preserves_shared_direct_anchors() -> None:
    batch = generate_procedural_shape_batch(
        384, span=3, query_count=3, vocabulary=2, seed=27027,
        nuisance=nuisance_from_level(0.135), next_query_stage=2)
    for cue in (0, 1):
        selected = batch.query_cue_ordinals == cue
        assert sorted(
            batch.query_operations[selected].unique().tolist()) == [0, 2]
        outcomes = batch.correct_actions[selected].bincount(minlength=2)
        assert int(outcomes[0]) == int(outcomes[1])


def test_combined_relative_stage_balances_all_seven_valid_mappings() -> None:
    batch = generate_procedural_shape_batch(
        1152, span=3, query_count=3, vocabulary=2, seed=27028,
        nuisance=nuisance_from_level(0.135), next_query_stage=3)
    expected = {
        (0, 0), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
    }
    populated = {
        (cue, operation)
        for cue in range(3)
        for operation in range(3)
        if bool((
            (batch.query_cue_ordinals == cue)
            & (batch.query_operations == operation)).any())
    }
    assert populated == expected
    recomputed = torch.where(
        batch.query_operations == 1,
        batch.query_cue_ordinals - 1,
        torch.where(
            batch.query_operations == 2,
            batch.query_cue_ordinals + 1,
            batch.query_cue_ordinals))
    assert torch.equal(batch.query_ordinals, recomputed)
    for cue, operation in expected:
        selected = (
            (batch.query_cue_ordinals == cue)
            & (batch.query_operations == operation))
        outcomes = batch.correct_actions[selected].bincount(minlength=2)
        assert abs(int(outcomes[0]) - int(outcomes[1])) <= 1


def test_three_operation_flip_is_a_valid_pixel_counterfactual() -> None:
    normal = generate_procedural_shape_batch(
        1152, span=3, query_count=3, vocabulary=2, seed=27029,
        nuisance=nuisance_from_level(0.135), next_query_stage=3)
    flipped = generate_procedural_shape_batch(
        1152, span=3, query_count=3, vocabulary=2, seed=27029,
        nuisance=nuisance_from_level(0.135), next_query_stage=3,
        flip_query_operations=True)
    assert torch.equal(
        normal.query_cue_ordinals, flipped.query_cue_ordinals)
    assert torch.equal(
        normal.candidate_identities, flipped.candidate_identities)
    assert bool((normal.query_operations != flipped.query_operations).all())
    recomputed = torch.where(
        flipped.query_operations == 1,
        flipped.query_cue_ordinals - 1,
        torch.where(
            flipped.query_operations == 2,
            flipped.query_cue_ordinals + 1,
            flipped.query_cue_ordinals))
    assert torch.equal(flipped.query_ordinals, recomputed)
    changed = normal.correct_actions != flipped.correct_actions
    assert 0.45 < float(changed.float().mean()) < 0.55


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
