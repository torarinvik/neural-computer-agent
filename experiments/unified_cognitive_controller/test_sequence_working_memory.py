import torch

from .model import UnifiedCognitiveController
from .train_sequence_working_memory import (
    generate_sequence_memory_batch, rollout_sequence_memory)
from .train_sequence_reward_buffer import (
    _balanced_provenance_loss, _replay_refinement_indices, _skill_slot_logits)


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
