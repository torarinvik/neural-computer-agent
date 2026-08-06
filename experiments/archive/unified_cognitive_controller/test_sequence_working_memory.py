import torch

from .legacy_model import UnifiedCognitiveController
from .train_sequence_reward_buffer import (
    _action_conditioned_policy_loss,
    _balanced_provenance_loss,
    _base_mistake_weights,
    _collect_buffer,
    _outcome_only_position_weights,
    _outcome_only_query_weights,
    _protected_rehearsal_mask,
    _query_curriculum_indices,
    _query_window_indices,
    _remove_final_slot_from_logits,
    _replay_refinement_indices,
    _skill_slot_logits,
    _weighted_binary_complement_loss,
    _weighted_binary_margin_loss,
)
from .train_sequence_working_memory import (
    generate_sequence_memory_batch,
    rollout_sequence_memory,
)


def test_sequence_memory_generation_is_deterministic_and_balanced() -> None:
    first = generate_sequence_memory_batch(
        32, span=2, distractors=1, seed=26001, operation="mixed")
    duplicate = generate_sequence_memory_batch(
        32, span=2, distractors=1, seed=26001, operation="mixed")
    assert torch.equal(first.input_frames, duplicate.input_frames)
    assert torch.equal(first.query_frames, duplicate.query_frames)
    assert torch.equal(first.correct_actions, duplicate.correct_actions)
    assert first.operation_bits.bincount(minlength=2).tolist() == [16, 16]
    sequence_ids = first.sequence[:, 0] * 2 + first.sequence[:, 1]
    assert sequence_ids.bincount(minlength=4).tolist() == [8, 8, 8, 8]


def test_operation_counterfactual_changes_only_query_cue_and_answers() -> None:
    normal = generate_sequence_memory_batch(
        32, span=2, distractors=1, seed=26002, operation="mixed")
    reversed_operation = generate_sequence_memory_batch(
        32, span=2, distractors=1, seed=26002, operation="mixed",
        reverse_operations=True)
    assert torch.equal(normal.input_frames, reversed_operation.input_frames)
    assert torch.equal(
        normal.distractor_frames, reversed_operation.distractor_frames)
    assert torch.equal(normal.sequence, reversed_operation.sequence)
    assert torch.equal(
        normal.operation_bits, 1 - reversed_operation.operation_bits)
    assert torch.equal(
        normal.correct_actions, reversed_operation.correct_actions.flip(1))


def test_complement_operation_is_a_distinct_visible_adjacent_primitive() -> None:
    forward = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260021, operation="forward")
    complement = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260021, operation="complement")
    assert torch.equal(forward.input_frames, complement.input_frames)
    assert torch.equal(forward.distractor_frames, complement.distractor_frames)
    assert torch.equal(complement.operation_bits, torch.zeros(16, dtype=torch.long))
    assert torch.equal(complement.correct_actions, 1 - complement.sequence)
    assert not torch.equal(forward.query_frames, complement.query_frames)


def test_complement_reverse_is_a_distinct_visible_primitive() -> None:
    batch = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260022,
        operation="complement_reverse")
    expected = 1 - batch.sequence.flip(1)
    assert torch.equal(batch.operation_bits, torch.zeros(16, dtype=torch.long))
    assert torch.equal(batch.correct_actions, expected)
    complement = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260022, operation="complement")
    assert not torch.equal(batch.query_frames, complement.query_frames)


def test_undo_complement_exposes_producer_and_consumer_cues() -> None:
    batch = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260027,
        operation="undo_complement")
    complement = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260027, operation="complement")
    assert torch.equal(batch.correct_actions, batch.sequence)
    assert torch.equal(batch.operation_bits, torch.zeros(16, dtype=torch.long))
    assert not torch.equal(batch.query_frames, complement.query_frames)


def test_producer_global_parity_keeps_the_producer_cue_visible() -> None:
    batch = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260028,
        operation="producer_global_parity")
    assert torch.equal(
        batch.correct_actions,
        batch.sequence.sum(dim=1, keepdim=True).remainder(2).expand_as(
            batch.sequence),
    )
    complement = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260028, operation="complement")
    assert not torch.equal(batch.query_frames, complement.query_frames)


def test_complement_rotate_is_a_distinct_visible_primitive() -> None:
    batch = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260023,
        operation="complement_rotate")
    expected = 1 - batch.sequence.roll(-1, dims=1)
    assert torch.equal(batch.correct_actions, expected)
    rotate = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260023, operation="rotate")
    assert not torch.equal(batch.query_frames, rotate.query_frames)


def test_adjacent_xor_is_a_distinct_visible_primitive() -> None:
    batch = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260024, operation="adjacent_xor")
    expected = (batch.sequence != batch.sequence.roll(-1, dims=1)).long()
    assert torch.equal(batch.correct_actions, expected)
    rotate = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260024, operation="rotate")
    assert not torch.equal(batch.query_frames, rotate.query_frames)


def test_prefix_parity_is_a_distinct_visible_primitive() -> None:
    batch = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260025, operation="prefix_parity")
    expected = torch.cumsum(batch.sequence, dim=1).remainder(2)
    assert torch.equal(batch.correct_actions, expected)
    rotate = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260025, operation="rotate")
    assert not torch.equal(batch.query_frames, rotate.query_frames)


def test_global_parity_is_a_distinct_visible_primitive() -> None:
    batch = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260026, operation="global_parity")
    expected = batch.sequence.sum(dim=1, keepdim=True).remainder(2)
    assert torch.equal(batch.correct_actions, expected.expand_as(batch.sequence))
    rotate = generate_sequence_memory_batch(
        16, span=3, distractors=1, seed=260026, operation="rotate")
    assert not torch.equal(batch.query_frames, rotate.query_frames)


def test_generated_composition_is_deterministic_and_renders_two_primitive_cues() -> None:
    from .train_sequence_working_memory import _GENERATED_PRIMITIVE_COLUMNS

    batch = generate_sequence_memory_batch(
        48, span=4, distractors=1, seed=260029,
        operation="generated_composition")
    duplicate = generate_sequence_memory_batch(
        48, span=4, distractors=1, seed=260029,
        operation="generated_composition")
    assert torch.equal(batch.input_frames, duplicate.input_frames)
    assert torch.equal(batch.query_frames, duplicate.query_frames)
    assert torch.equal(batch.correct_actions, duplicate.correct_actions)

    cue_columns = tuple(_GENERATED_PRIMITIVE_COLUMNS.values())
    for row in range(batch.batch_size):
        active = [
            column
            for column in cue_columns
            if bool(
                (
                    batch.query_frames[row, :, :, 2:5, column:column + 3]
                    > 0.9
                ).any()
            )
        ]
        assert len(active) == 2


def test_sequence_counterfactual_is_a_valid_pixel_rerender() -> None:
    normal = generate_sequence_memory_batch(
        32, span=2, distractors=1, seed=26003, operation="mixed")
    reversed_sequence = generate_sequence_memory_batch(
        32, span=2, distractors=1, seed=26003, operation="mixed",
        reverse_sequence=True)
    assert torch.equal(
        normal.input_frames.flip(1), reversed_sequence.input_frames)
    assert torch.equal(
        normal.distractor_frames, reversed_sequence.distractor_frames)
    assert torch.equal(normal.query_frames, reversed_sequence.query_frames)
    assert torch.equal(normal.sequence.flip(1), reversed_sequence.sequence)


def test_blank_sequence_removes_only_evidence() -> None:
    normal = generate_sequence_memory_batch(
        16, span=2, distractors=1, seed=26004, operation="mixed")
    blank = generate_sequence_memory_batch(
        16, span=2, distractors=1, seed=26004, operation="mixed",
        blank_sequence=True)
    assert not torch.equal(normal.input_frames, blank.input_frames)
    assert torch.equal(normal.query_frames, blank.query_frames)
    assert torch.equal(normal.correct_actions, blank.correct_actions)


def test_position_blend_is_a_gradual_label_preserving_axis() -> None:
    base = generate_sequence_memory_batch(
        16, span=2, distractors=0, seed=26006, operation="mixed",
        position_blend=0.0)
    middle = generate_sequence_memory_batch(
        16, span=2, distractors=0, seed=26006, operation="mixed",
        position_blend=0.5)
    shifted = generate_sequence_memory_batch(
        16, span=2, distractors=0, seed=26006, operation="mixed",
        position_blend=1.0)
    assert torch.equal(base.sequence, middle.sequence)
    assert torch.equal(base.sequence, shifted.sequence)
    assert torch.equal(base.correct_actions, middle.correct_actions)
    assert torch.equal(base.correct_actions, shifted.correct_actions)
    assert torch.equal(base.query_frames, middle.query_frames)
    assert torch.equal(base.query_frames, shifted.query_frames)
    assert not torch.equal(base.input_frames, middle.input_frames)
    assert not torch.equal(middle.input_frames, shifted.input_frames)


def test_position_augmentation_changes_only_nuisance_pixels() -> None:
    fixed = generate_sequence_memory_batch(
        20, span=2, distractors=0, seed=26007, operation="mixed")
    augmented = generate_sequence_memory_batch(
        20, span=2, distractors=0, seed=26007, operation="mixed",
        position_augmentation=True)
    assert torch.equal(fixed.sequence, augmented.sequence)
    assert torch.equal(fixed.operation_bits, augmented.operation_bits)
    assert torch.equal(fixed.correct_actions, augmented.correct_actions)
    assert torch.equal(fixed.query_frames, augmented.query_frames)
    assert not torch.equal(fixed.input_frames, augmented.input_frames)


def test_rerender_override_preserves_logical_task_but_changes_pixels() -> None:
    base = generate_sequence_memory_batch(
        16, span=3, distractors=2, seed=26008, operation="mixed",
        position_augmentation=True)
    rerendered = generate_sequence_memory_batch(
        16, span=3, distractors=2, seed=26009, operation="mixed",
        position_augmentation=True, sequence_override=base.sequence,
        operation_bits_override=base.operation_bits)
    assert torch.equal(base.sequence, rerendered.sequence)
    assert torch.equal(base.operation_bits, rerendered.operation_bits)
    assert torch.equal(base.correct_actions, rerendered.correct_actions)
    assert not torch.equal(base.input_frames, rerendered.input_frames)
    assert not torch.equal(
        base.distractor_frames, rerendered.distractor_frames)


def test_rollout_keeps_all_fast_memory_on_the_model_device() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=2, intention_width=8)
    batch = generate_sequence_memory_batch(
        8, span=2, distractors=1, seed=26005, operation="mixed")
    result = rollout_sequence_memory(
        model, batch, sample_actions=False)
    assert result["final_hidden"].device == model.actuator.weight.device
    assert result["final_workspace"].device == model.actuator.weight.device
    assert result["final_workspace"].shape == (8, 2, 32)
    assert result["actions"].shape == (8, 2)


def test_rollout_slot_activity_is_opt_in_and_tracks_every_event() -> None:
    model = UnifiedCognitiveController(
        width=16, workspace_slots=2, intention_width=8,
        skill_adapter_widths=(8,), skill_adapter_gate_mode="relu")
    batch = generate_sequence_memory_batch(
        8, span=3, distractors=2, seed=26011, operation="mixed")
    ordinary = rollout_sequence_memory(model, batch, sample_actions=False)
    assert "skill_adapter_openings" not in ordinary
    result = rollout_sequence_memory(
        model, batch, sample_actions=False, return_slot_activity=True)
    assert result["skill_adapter_openings"].shape == (8, 8, 1)
    assert result["skill_adapter_residual_norms"].shape == (8, 8, 1)


def test_appended_slot_replay_helper_uses_the_final_slot_projection() -> None:
    model = UnifiedCognitiveController(
        width=16, workspace_slots=2, intention_width=8,
        skill_adapter_widths=(16, 16), skill_adapter_gate_mode="relu",
        skill_adapter_legacy_read_from=None,
        skill_adapter_reads_workspace_from=0,
        skill_adapter_reads_workspace_usage_from=0,
        skill_adapter_reads_event_age_from=0, event_age=True,
        skill_adapter_read_bottleneck=4)
    features = torch.randn(5, 16 * 2 + 2 * 16 + 2 + 1)
    logits, *_ = _skill_slot_logits(model, features)
    logits.sum().backward()
    assert model.skill_adapters[1][2].weight.grad is not None
    assert model.skill_adapter_read_projections[1].weight.grad is not None
    assert model.skill_adapters[0][2].weight.grad is None
    assert model.skill_adapter_read_projections[0].weight.grad is None


def test_existing_slot_continuation_removes_its_inherited_logits_once() -> None:
    model = UnifiedCognitiveController(
        width=16, workspace_slots=2, intention_width=8,
        skill_adapter_widths=(16,), skill_adapter_gate_mode="relu")
    with torch.no_grad():
        model.skill_adapters[0][2].bias.fill_(0.25)
        model.skill_adapter_gates[0].bias.fill_(1.0)
    features, base_logits, _, _ = _collect_buffer(
        model, count=8, span=2, distractors=1, seed=260111,
        device=torch.device("cpu"), position_augmentation=True)
    removed = _remove_final_slot_from_logits(
        model, features, base_logits)
    residual, *_ = _skill_slot_logits(model, features)
    assert torch.allclose(removed + residual, base_logits, atol=1e-6)


def test_replay_buffer_can_supply_a_generic_event_snapshot_read() -> None:
    model = UnifiedCognitiveController(
        width=16, workspace_slots=2, intention_width=8,
        skill_adapter_widths=(16,),
        skill_adapter_reads_event_snapshot_from=0)
    features, base_logits, actions, outcomes = _collect_buffer(
        model, count=8, span=2, distractors=1, seed=26012,
        device=torch.device("cpu"), position_augmentation=True,
        include_event_snapshot=True)
    assert features.shape == (16, 16 * 2 + 16)
    assert base_logits.shape == actions.shape + (2,)
    assert outcomes.shape == actions.shape
    logits, *_ = _skill_slot_logits(model, features)
    assert logits.shape == (16, 2)


def test_replay_buffer_can_supply_parent_action_context() -> None:
    model = UnifiedCognitiveController(
        width=16, workspace_slots=2, intention_width=8,
        skill_adapter_widths=(16,),
        skill_adapter_reads_parent_action_from=0)
    features = torch.randn(16, 16 * 2 + 2)
    logits, *_ = _skill_slot_logits(model, features)
    assert logits.shape == (16, 2)


def test_replay_buffer_can_supply_parent_entropy_context() -> None:
    model = UnifiedCognitiveController(
        width=16, workspace_slots=2, intention_width=8,
        skill_adapter_widths=(16,),
        skill_adapter_reads_parent_entropy_from=0)
    features = torch.randn(16, 16 * 2 + 1)
    logits, *_ = _skill_slot_logits(model, features)
    assert logits.shape == (16, 2)


def test_critic_policy_bridge_trains_the_slot_but_not_the_critic_target() -> None:
    logits = torch.zeros(4, 2, requires_grad=True)
    critic = torch.tensor(
        ((2.0, -1.0), (-1.0, 2.0), (0.5, 0.5), (2.0, -1.0)),
        requires_grad=True)
    fresh = torch.tensor([True, False, True, True])
    loss = _action_conditioned_policy_loss(
        logits, critic, fresh, temperature=1.0)
    loss.backward()
    assert logits.grad is not None
    assert critic.grad is None


def test_binary_complement_loss_uses_only_attempted_action_and_outcome() -> None:
    logits = torch.tensor(
        ((2.0, -2.0), (-2.0, 2.0), (-2.0, 2.0), (2.0, -2.0)),
        requires_grad=True)
    attempted = torch.tensor((0, 1, 0, 1))
    outcomes = torch.tensor((1.0, 1.0, 0.0, 0.0))
    replay = torch.zeros(4, dtype=torch.bool)
    loss = _weighted_binary_complement_loss(
        logits, attempted, outcomes, replay, 1.0)
    assert float(loss.detach()) < 0.03
    loss.backward()
    assert logits.grad is not None


def test_query_difficulty_weights_use_only_fresh_age_buckets() -> None:
    features = torch.zeros(7, 3)
    features[:, -1] = torch.tensor((1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0))
    outcomes = torch.tensor((1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0))
    replay = torch.tensor((False, False, False, False, False, False, True))
    weights = _outcome_only_query_weights(
        features, outcomes, replay, age_column=2, power=1.0, floor=0.25)
    assert float(weights[2]) > float(weights[0]) > float(weights[4])
    assert float(weights[6]) == 1.0
    assert torch.isclose(weights[~replay].mean(), torch.tensor(1.0))


def test_query_position_difficulty_weights_use_only_fresh_positions() -> None:
    outcomes = torch.tensor((1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0))
    replay = torch.tensor(
        (False, False, False, False, False, False, True, True))
    weights = _outcome_only_position_weights(
        outcomes, replay, target_lifetimes=2, span=4, power=1.0,
        floor=0.25)
    assert float(weights[2]) > float(weights[0]) > float(weights[4])
    assert float(weights[0]) == float(weights[1])
    assert float(weights[2]) == float(weights[3])
    assert float(weights[6]) == 1.0
    assert float(weights[7]) == 1.0
    assert torch.isclose(weights[~replay].mean(), torch.tensor(1.0))


def test_query_curriculum_keeps_replay_and_a_target_prefix() -> None:
    indices = _query_curriculum_indices(
        14, target_lifetimes=2, span=4, cutoff=2,
        device=torch.device("cpu"))
    assert indices.tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13]


def test_query_window_keeps_replay_and_a_target_suffix() -> None:
    indices = _query_window_indices(
        14, target_lifetimes=2, span=4, start=2, end=4,
        device=torch.device("cpu"))
    assert indices.tolist() == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def test_protected_rehearsal_mask_marks_only_non_target_fresh_rows() -> None:
    replay = torch.tensor(
        [False] * 8 + [False] * 6 + [True] * 3)
    protected = _protected_rehearsal_mask(
        replay, target_rows=8, protect_rehearsal=True)
    assert protected.tolist() == (
        [False] * 8 + [True] * 6 + [True] * 3)
    assert torch.equal(
        _protected_rehearsal_mask(
            replay, target_rows=8, protect_rehearsal=False), replay)


def test_base_mistake_weights_ignore_replay_rows() -> None:
    base_logits = torch.tensor(
        ((3.0, -3.0), (-3.0, 3.0), (3.0, -3.0), (-3.0, 3.0)))
    attempted = torch.tensor((0, 1, 0, 1))
    outcomes = torch.tensor((1.0, 0.0, 0.0, 1.0))
    replay = torch.tensor((False, False, True, False))
    weights = _base_mistake_weights(
        base_logits, attempted, outcomes, replay, weight=5.0)
    assert weights.tolist() == [1.0, 5.0, 1.0, 1.0]


def test_binary_margin_loss_pushes_against_a_large_wrong_parent_margin() -> None:
    logits = torch.tensor(((100.0, -100.0), (100.0, -100.0)))
    attempted = torch.tensor((1, 0))
    outcomes = torch.tensor((1.0, 0.0))
    replay = torch.zeros(2, dtype=torch.bool)
    loss = _weighted_binary_margin_loss(
        logits, attempted, outcomes, replay, 1.0, margin=1.0)
    assert float(loss) > 100.0


def test_address_conditioned_write_content_breaks_slot_symmetry() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=2, intention_width=8,
        workspace_slot_addressing=True)
    with torch.no_grad():
        model.workspace_read_address_scale.zero_()
        model.workspace_write_address_scale.zero_()
        model.workspace_write_content_address_scale.zero_()
    event = torch.randn(4, 32)
    action = torch.zeros(4, dtype=torch.long)
    reward = torch.zeros(4)
    state = model.initial_state(4, device=torch.device("cpu"))
    _, symmetric = model.step_event(event, state, action, reward, reward)
    assert torch.equal(symmetric.workspace[:, 0], symmetric.workspace[:, 1])
    with torch.no_grad():
        model.workspace_write_content_address_scale.fill_(0.25)
    state = model.initial_state(4, device=torch.device("cpu"))
    _, addressed = model.step_event(event, state, action, reward, reward)
    assert not torch.equal(addressed.workspace[:, 0], addressed.workspace[:, 1])


def test_zero_initialized_workspace_volatility_is_an_exact_noop() -> None:
    torch.manual_seed(26010)
    base = UnifiedCognitiveController(
        width=32, workspace_slots=2, intention_width=8,
        workspace_slot_addressing=True)
    volatile = UnifiedCognitiveController(
        width=32, workspace_slots=2, intention_width=8,
        workspace_slot_addressing=True, workspace_volatility=True)
    volatile.load_state_dict(base.state_dict(), strict=False)
    assert volatile.workspace_volatility_write_scale is not None
    with torch.no_grad():
        volatile.workspace_volatility_write_scale.zero_()
    action = torch.zeros(4, dtype=torch.long)
    state_base = base.initial_state(4, device=torch.device("cpu"))
    state_volatile = volatile.initial_state(4, device=torch.device("cpu"))
    for step in range(3):
        event = torch.randn(4, 32)
        reward = torch.full((4,), float(step > 0))
        has_feedback = torch.full((4,), float(step > 0))
        out_base, state_base = base.step_event(
            event, state_base, action, reward, has_feedback)
        out_volatile, state_volatile = volatile.step_event(
            event, state_volatile, action, reward, has_feedback)
        assert torch.equal(out_base.intent_event.payload,
                           out_volatile.intent_event.payload)
        assert torch.equal(state_base.workspace, state_volatile.workspace)
    assert state_volatile.workspace_volatility is not None


def test_event_age_is_generic_state_and_advances_once_per_event() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=2, intention_width=8,
        skill_adapter_widths=(8,),
        skill_adapter_reads_event_age_from=0,
        event_age=True, event_age_scale=8.0)
    state = model.initial_state(4, device=torch.device("cpu"))
    assert state.event_age is not None
    assert torch.equal(state.event_age, torch.zeros(4, 1))
    event = torch.randn(4, 32)
    action = torch.zeros(4, dtype=torch.long)
    reward = torch.zeros(4)
    _, next_state = model.step_event(
        event, state, action, reward, reward)
    assert next_state.event_age is not None
    assert torch.equal(next_state.event_age, torch.full((4, 1), 0.125))


def test_provenance_refinement_contains_both_sources_when_weighted() -> None:
    all_indices = torch.arange(10)
    replay_indices = all_indices[6:]
    assert torch.equal(
        _replay_refinement_indices(all_indices, replay_indices, 0.0),
        replay_indices)
    assert torch.equal(
        _replay_refinement_indices(all_indices, replay_indices, 1.0),
        all_indices)
    assert torch.equal(
        _replay_refinement_indices(all_indices, replay_indices, 0.0, 1.0),
        all_indices)


def test_provenance_loss_balances_unequal_source_counts() -> None:
    score = torch.tensor([0.0, 0.0, 0.0, 0.0])
    balanced = _balanced_provenance_loss(
        score, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    imbalanced = _balanced_provenance_loss(
        score, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.isfinite(balanced)
    assert torch.isfinite(imbalanced)
    assert balanced < imbalanced
