from __future__ import annotations

import pytest
import torch
from torch import nn

from neural_computer import (
    MEMORY_BACKEND_FORMAT,
    MEMORY_SNAPSHOT_FORMAT,
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    AmodalEventWindowBuffer,
    AppendOnlyContentAddressedMemory,
    ArtifactConsolidationReceipt,
    ConditionedOpaqueProtocolDecoder,
    ContentAddressedMemory,
    ControllerFeedback,
    EventWaitPolicy,
    ExecutableArtifactMemory,
    MemoryBackend,
    MemoryQuery,
    OpaqueProtocolDecoder,
    PersistentAppendOnlyContentAddressedMemory,
    PersistentContentAddressedMemory,
    freeze_core,
    load_growth_artifact,
    load_runtime_components,
    save_runtime,
)


def _feedback(batch: int, width: int) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch, width),
        reward=torch.zeros(batch),
        propensity=torch.ones(batch),
        has_feedback=torch.zeros(batch),
    )


def test_clean_controller_has_no_modality_or_protocol_ownership() -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )

    assert not hasattr(controller, "vision")
    assert not hasattr(controller, "actuator")
    assert not hasattr(controller, "action_embedding")
    assert controller.width == 16


def test_experiment_namespace_does_not_define_the_production_agent() -> None:
    import experiments.archive.unified_cognitive_controller as historical

    assert historical.__file__ is not None
    assert not hasattr(historical, "UnifiedCognitiveController")
    assert not hasattr(historical, "ActionIntentDecoder")


def test_event_metadata_survives_collection_without_early_reduction() -> None:
    batch, width = 2, 16
    events = AmodalEventCollection.from_events(
        [
            AmodalEvent(
                torch.ones(batch, width),
                source_key=torch.ones(batch, 4),
                timestamp=torch.full((batch,), 1.0),
                duration=torch.full((batch,), 0.5),
                confidence=torch.ones(batch),
            ),
            AmodalEvent(
                torch.zeros(batch, width),
                source_key=torch.zeros(batch, 4),
                timestamp=torch.full((batch,), 2.0),
                duration=torch.full((batch,), 0.25),
                confidence=torch.full((batch,), 0.5),
            ),
        ],
        width=width,
    )

    assert events.payload.shape == (batch, 2, width)
    assert events.source_key is not None
    assert events.source_key.shape == (batch, 2, 4)
    assert torch.equal(events.timestamp[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(events.duration[0], torch.tensor([0.5, 0.25]))


def test_quiet_tick_is_valid_and_does_not_fabricate_evidence() -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    state = controller.initial_state(3, device="cpu")
    quiet = AmodalEventCollection.empty(3, 16)
    output, next_state = controller.step(quiet, state, _feedback(3, 3))

    assert torch.equal(output.intention.confidence, torch.zeros(3, 1))
    assert torch.equal(next_state.latest_event, torch.zeros(3, 16))


def test_stochastic_memory_write_exposes_training_credit_without_protocol_fields() -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    memory = ContentAddressedMemory(width=16, capacity=1)
    events = AmodalEventCollection.from_events([AmodalEvent(torch.randn(2, 16))])
    state = controller.initial_state(2, device="cpu")

    output, _ = controller.step(
        events,
        state,
        _feedback(2, 3),
        memory,
        sample_memory_writes=True,
    )

    assert output.memory_write_log_probability is not None
    assert output.memory_write_sample is not None
    assert set(output.memory_write_sample.tolist()) <= {0.0, 1.0}
    output.memory_write_log_probability.sum().backward()
    policy_gradient = controller.memory_write_policy[0].weight.grad
    assert policy_gradient is not None
    assert torch.isfinite(policy_gradient).all()


def test_uniform_memory_write_intervention_controls_the_sample() -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    memory = ContentAddressedMemory(width=16, capacity=1)
    events = AmodalEventCollection.from_events([AmodalEvent(torch.randn(2, 16))])
    state = controller.initial_state(2, device="cpu")

    write_output, _ = controller.step(
        events,
        state,
        _feedback(2, 3),
        memory,
        memory_write_uniform=torch.zeros(2),
    )
    skip_output, _ = controller.step(
        events,
        state,
        _feedback(2, 3),
        memory,
        memory_write_uniform=torch.ones(2),
    )

    assert torch.equal(write_output.memory_write_sample, torch.ones(2))
    assert torch.equal(skip_output.memory_write_sample, torch.zeros(2))
    assert write_output.memory_write_log_probability is not None
    assert skip_output.memory_write_log_probability is not None


def test_forced_write_can_detach_gate_gradient_but_keep_value_gradient() -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    memory = ContentAddressedMemory(width=16, capacity=1)
    events = AmodalEventCollection.from_events([AmodalEvent(torch.randn(2, 16))])
    state = controller.initial_state(2, device="cpu")

    with memory.differentiable_transaction():
        controller.step(
            events,
            state,
            _feedback(2, 3),
            memory,
            memory_write_override=torch.ones(2),
            memory_write_gradient=False,
        )
        query_output, _ = controller.step(
            events,
            controller.initial_state(2, device="cpu"),
            _feedback(2, 3),
            memory,
        )
        assert query_output.memory_read is not None
        query_output.memory_read.value.sum().backward()

    assert controller.memory_write_policy[0].weight.grad is None
    assert controller.memory_value.weight.grad is not None
    assert torch.isfinite(controller.memory_value.weight.grad).all()


def test_transport_boundaries_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        AmodalEvent(torch.tensor([[float("nan")]])).validate()

    feedback = ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.tensor([float("inf")]),
        propensity=torch.ones(1),
        has_feedback=torch.ones(1),
    )
    with pytest.raises(ValueError, match="finite"):
        feedback.validate(batch=1, action_width=3)

    memory = ContentAddressedMemory(width=2, capacity=1)
    with pytest.raises(ValueError, match="finite"):
        memory.read(MemoryQuery(torch.tensor([[float("nan"), 0.0]])))


def test_runtime_supports_variable_encoders_and_decoders_without_resize() -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    runtime = AmodalControllerRuntime(controller)
    runtime.register_encoder("one", nn.Linear(7, 16))
    runtime.register_encoder("two", nn.Linear(9, 16))
    runtime.register_decoder("json", OpaqueProtocolDecoder(5, 4))
    runtime.register_decoder("speech", OpaqueProtocolDecoder(5, 6))
    state = runtime.initial_state(3, device="cpu")

    output, _ = runtime.step_streams(
        {"one": torch.randn(3, 7), "two": torch.randn(3, 9)},
        state,
        _feedback(3, 3),
    )

    assert set(output.decoded) == {"json", "speech"}
    assert output.decoded["json"].shape == (3, 4)
    assert output.decoded["speech"].shape == (3, 6)
    assert output.execution_logits.shape == (3, 3)


def test_conditioned_decoder_shares_weights_and_consumes_opaque_context() -> None:
    decoder = ConditionedOpaqueProtocolDecoder(5, 4, 3, hidden=8)
    intention = torch.randn(2, 5)
    context = torch.randn(2, 4)
    first = decoder(intention, context)
    second = decoder(intention, context.flip(0))

    assert first.shape == (2, 3)
    assert not torch.equal(first, second)
    assert decoder.configuration()["context"] == "opaque_learned_program_state_v1"


def test_deliberation_is_bounded_and_uses_quiet_internal_ticks() -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    with torch.no_grad():
        controller.execution_policy[-1].weight.zero_()
        controller.execution_policy[-1].bias.copy_(torch.tensor([0.0, 10.0, -10.0]))
        controller.execution_transport_policy[-1].weight.zero_()
        controller.execution_transport_policy[-1].bias.zero_()
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    event = AmodalEventCollection.from_events([AmodalEvent(torch.randn(1, 16))])

    result = runtime.deliberate(event, state, _feedback(1, 3), think_budget=1)

    assert result.decision == "commit"
    assert result.forced_commit
    assert result.think_ticks == 1
    assert len(result.trace) == 2
    assert result.trace[-1].controller.intention.confidence.shape == (1, 1)


def test_timeout_residual_is_zero_on_age_zero_states() -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    event = AmodalEventCollection.from_events([AmodalEvent(torch.randn(1, 16))])
    feedback = _feedback(1, 3)
    first, _ = controller.step(event, controller.initial_state(1, device="cpu"), feedback)
    with torch.no_grad():
        controller.execution_timeout_policy[-1].bias.copy_(torch.tensor([9.0, -7.0, 5.0]))
    second, _ = controller.step(event, controller.initial_state(1, device="cpu"), feedback)
    assert torch.equal(first.execution_logits, second.execution_logits)


def test_event_tokens_persist_and_evict_as_transport_not_semantic_slots() -> None:
    controller = AmodalCognitiveController(
        width=16,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        event_window_capacity=2,
    )
    state = controller.initial_state(1, device="cpu")
    feedback = _feedback(1, 3)
    first = AmodalEvent(
        torch.ones(1, 16),
        timestamp=torch.tensor([1.0]),
        duration=torch.tensor([0.25]),
        confidence=torch.tensor([1.0]),
    )
    second = AmodalEvent(
        torch.full((1, 16), 2.0),
        timestamp=torch.tensor([2.0]),
        duration=torch.tensor([0.5]),
        confidence=torch.tensor([0.8]),
    )
    third = AmodalEvent(
        torch.full((1, 16), 3.0),
        timestamp=torch.tensor([3.0]),
        duration=torch.tensor([0.75]),
        confidence=torch.tensor([0.6]),
    )

    _, state = controller.step([first], state, feedback)
    _, state = controller.step([second], state, feedback)
    assert state.event_window.present[0].tolist() == [True, True]
    assert torch.equal(state.event_window.timestamp[0], torch.tensor([1.0, 2.0]))
    _, state = controller.step([third], state, feedback)

    assert torch.equal(state.event_window.timestamp[0], torch.tensor([2.0, 3.0]))
    assert torch.equal(state.event_window.duration[0], torch.tensor([0.5, 0.75]))
    assert state.event_window.age[0, 0] > 0


def test_source_keys_are_generic_controller_input_metadata() -> None:
    controller = AmodalCognitiveController(
        width=16,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        source_key_width=4,
    )
    state = controller.initial_state(1, device="cpu")
    events = AmodalEventCollection.from_events(
        [
            AmodalEvent(
                torch.randn(1, 16),
                source_key=torch.ones(1, 4),
                confidence=torch.ones(1),
            ),
            AmodalEvent(
                torch.randn(1, 16),
                source_key=torch.zeros(1, 4),
                confidence=torch.ones(1),
            ),
        ]
    )

    output, next_state = controller.step(events, state, _feedback(1, 3))

    assert next_state.event_window.source_key is not None
    assert next_state.event_window.source_key.shape == (1, 32, 4)
    assert output.event_reliability.shape == (1, 32)


def test_source_credit_head_starts_without_a_source_preference() -> None:
    controller = AmodalCognitiveController(
        width=16,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        source_key_width=4,
    )

    assert controller.source_credit_policy is not None
    assert torch.count_nonzero(controller.source_credit_policy[-1].bias) == 0


def test_learned_wait_policy_can_release_partial_windows_without_payload_access() -> None:
    policy = EventWaitPolicy(hidden=4)
    with torch.no_grad():
        policy.network[-1].bias.fill_(-10.0)
    buffer = AmodalEventWindowBuffer(("left", "right"), wait_policy=policy)
    event = AmodalEvent(
        torch.randn(1, 16), timestamp=torch.tensor([0.0]), confidence=torch.ones(1)
    )

    released = buffer.push({"left": event})

    assert len(released) == 1
    assert not released[0].complete
    assert released[0].collection.present.tolist() == [[True, False]]
    assert released[0].collection.confidence[0, 1] == 0


def test_content_addressed_memory_round_trips_and_reports_receipts(tmp_path) -> None:
    memory = ContentAddressedMemory(width=4, capacity=2, write_threshold=0.5)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    receipt = memory.write(key, value, torch.ones(1), timestamp=torch.tensor([7.0]))
    read = memory.read(MemoryQuery(key))

    assert receipt.committed.tolist() == [True]
    assert read.hit.tolist() == [True]
    assert torch.allclose(read.value, value, atol=1e-5)
    path = tmp_path / "memory.pt"
    memory.snapshot(path)
    restored = ContentAddressedMemory(width=4, capacity=2, write_threshold=0.5)
    restored.load_snapshot(path)
    assert torch.equal(restored.state_dict()["keys"], memory.state_dict()["keys"])
    assert torch.equal(restored.state_dict()["values"], memory.state_dict()["values"])
    assert memory.configuration()["format"] == MEMORY_BACKEND_FORMAT
    payload = torch.load(path, weights_only=False)
    assert payload["format"] == MEMORY_SNAPSHOT_FORMAT

    legacy_payload = dict(payload)
    legacy_payload["format"] = "neural-computer.memory-snapshot.v1"
    legacy_payload.pop("state_checksum")
    legacy_payload["configuration"] = dict(legacy_payload["configuration"])
    legacy_payload["configuration"].pop("write_match_threshold")
    legacy_path = tmp_path / "legacy-memory-v1.pt"
    torch.save(legacy_payload, legacy_path)
    legacy_restored = ContentAddressedMemory(width=4, capacity=2, write_threshold=0.5)
    legacy_restored.load_snapshot(legacy_path)
    assert torch.equal(legacy_restored.state_dict()["values"], memory.state_dict()["values"])


def test_memory_snapshot_rejects_corrupt_format_and_state(tmp_path) -> None:
    memory = ContentAddressedMemory(width=4, capacity=2)
    path = tmp_path / "memory.pt"
    memory.snapshot(path)
    payload = torch.load(path, weights_only=False)

    payload["format"] = "neural-computer.memory-snapshot.invalid"
    invalid_format = tmp_path / "invalid-format.pt"
    torch.save(payload, invalid_format)
    with pytest.raises(ValueError, match="snapshot format"):
        memory.load_snapshot(invalid_format)

    payload["format"] = MEMORY_SNAPSHOT_FORMAT
    payload["state_dict"]["strengths"][0] = 2.0
    invalid_state = tmp_path / "invalid-state.pt"
    torch.save(payload, invalid_state)
    with pytest.raises(ValueError, match="strengths"):
        memory.load_snapshot(invalid_state)

    payload = torch.load(path, weights_only=False)
    payload["state_dict"]["keys"][0, 0] = 0.25
    invalid_checksum = tmp_path / "invalid-checksum.pt"
    torch.save(payload, invalid_checksum)
    with pytest.raises(ValueError, match="checksum"):
        memory.load_snapshot(invalid_checksum)


def test_memory_read_keeps_query_alignment_differentiable() -> None:
    memory = ContentAddressedMemory(width=4, capacity=2, write_threshold=0.5)
    memory.write(
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        torch.ones(2),
    )
    query_key = torch.tensor([[0.8, 0.6, 0.0, 0.0]], requires_grad=True)
    read = memory.read(MemoryQuery(query_key, top_k=2))

    read.value[:, 0].sum().backward()

    assert read.value.requires_grad
    assert query_key.grad is not None
    assert torch.isfinite(query_key.grad).all()
    assert not memory.keys.requires_grad


def test_differentiable_memory_transaction_connects_write_and_read() -> None:
    memory = ContentAddressedMemory(width=4, capacity=2, write_threshold=0.5)
    keys = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], requires_grad=True
    )
    values = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], requires_grad=True
    )
    query_key = torch.tensor([[0.8, 0.6, 0.0, 0.0]], requires_grad=True)

    with memory.differentiable_transaction():
        memory.write(keys, values, torch.ones(2))
        read = memory.read(MemoryQuery(query_key, top_k=2))
        (read.value[:, 0].sum() + read.scores[:, 0].sum()).backward()

    assert keys.grad is not None and torch.isfinite(keys.grad).all()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert query_key.grad is not None and torch.isfinite(query_key.grad).all()
    assert not memory.read(MemoryQuery(query_key.detach(), top_k=2)).value.requires_grad


def test_differentiable_memory_transaction_trains_write_gate() -> None:
    memory = ContentAddressedMemory(width=4, capacity=1, write_threshold=0.5)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    strength = torch.tensor([0.2], requires_grad=True)

    with memory.differentiable_transaction():
        memory.write(key, value, strength)
        read = memory.read(MemoryQuery(key))
        read.value.sum().backward()

    assert strength.grad is not None
    assert torch.isfinite(strength.grad).all()
    assert memory.occupied.sum().item() == 0


def test_write_threshold_equality_remains_differentiable_pending() -> None:
    memory = ContentAddressedMemory(width=4, capacity=1, write_threshold=0.5)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    strength = torch.tensor([0.5], requires_grad=True)

    with memory.differentiable_transaction():
        receipt = memory.write(key, value, strength)
        read = memory.read(MemoryQuery(key))
        read.value.sum().backward()

    assert receipt.committed.tolist() == [False]
    assert strength.grad is not None
    assert torch.isfinite(strength.grad).all()
    assert memory.occupied.sum().item() == 0


def test_differentiable_transaction_keeps_write_gate_gradient_after_commit() -> None:
    memory = ContentAddressedMemory(width=4, capacity=1, write_threshold=0.5)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    strength = torch.tensor([0.75], requires_grad=True)

    with memory.differentiable_transaction():
        receipt = memory.write(key, value, strength)
        read = memory.read(MemoryQuery(key))
        read.value.sum().backward()

    assert receipt.committed.tolist() == [True]
    assert strength.grad is not None
    assert torch.isfinite(strength.grad).all()
    assert strength.grad.abs().item() > 1e-6
    assert memory.occupied.sum().item() == 1


def test_content_addressed_memory_upserts_matching_keys() -> None:
    memory = ContentAddressedMemory(width=4, capacity=2, write_threshold=0.5)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    first_value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    second_value = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    first = memory.write(key, first_value, torch.ones(1))
    second = memory.write(key * 0.99, second_value, torch.ones(1))

    assert first.indices.tolist() == [0]
    assert second.indices.tolist() == [0]
    assert memory.occupied.sum().item() == 1
    assert torch.allclose(memory.read(MemoryQuery(key)).value, second_value, atol=1e-5)


def test_memory_side_policy_can_select_an_explicit_eviction_row() -> None:
    memory = ContentAddressedMemory(width=4, capacity=2, write_threshold=0.5)
    keys = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    values = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    memory.write(keys, values, torch.ones(2))
    candidates = memory.candidates(torch.tensor([0], dtype=torch.long))
    assert candidates.keys.shape == (1, 2, 4)
    replacement_key = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    replacement_value = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    receipt = memory.write(
        replacement_key,
        replacement_value,
        torch.ones(1),
        scope=torch.tensor([0], dtype=torch.long),
        target_index=torch.tensor([1], dtype=torch.long),
    )

    assert receipt.indices.tolist() == [1]
    assert torch.allclose(memory.values[1], replacement_value[0])
    assert torch.allclose(memory.values[0], values[0])


def test_scoped_memory_keeps_same_key_bindings_separate() -> None:
    memory = ContentAddressedMemory(width=4, capacity=1, scope_capacity=2)
    key = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    values = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    scope = torch.tensor([0, 1], dtype=torch.long)

    receipt = memory.write(key, values, torch.ones(2), scope=scope)
    read = memory.read(MemoryQuery(key, scope=scope))

    assert receipt.indices.tolist() == [0, 0]
    assert read.hit.tolist() == [True, True]
    assert torch.allclose(read.value, values, atol=1e-5)
    assert memory.occupied.shape == (2, 1)


def test_content_addressed_memory_binds_two_rows_within_one_scope() -> None:
    memory = ContentAddressedMemory(width=4, capacity=2)
    keys = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    values = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    memory.write(keys, values, torch.ones(2))
    read = memory.read(MemoryQuery(keys))

    assert read.hit.tolist() == [True, True]
    assert torch.allclose(read.value, values, atol=1e-5)
    assert read.indices.tolist() == [[0], [1]]


def test_scoped_memory_snapshot_round_trips_and_rejects_invalid_scope(tmp_path) -> None:
    memory = ContentAddressedMemory(width=4, capacity=1, scope_capacity=2)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    memory.write(key, value, torch.ones(1), scope=torch.tensor([1]))
    path = tmp_path / "scoped-memory.pt"
    memory.snapshot(path)

    restored = ContentAddressedMemory(width=4, capacity=1, scope_capacity=2)
    restored.load_snapshot(path)
    assert torch.allclose(
        restored.read(MemoryQuery(key, scope=torch.tensor([1]))).value,
        value,
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="scope"):
        restored.read(MemoryQuery(key, scope=torch.tensor([2])))


def test_scoped_memory_transaction_preserves_gradients_per_batch_row() -> None:
    memory = ContentAddressedMemory(width=4, capacity=1, scope_capacity=2)
    keys = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], requires_grad=True
    )
    values = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], requires_grad=True
    )
    scope = torch.tensor([0, 1], dtype=torch.long)

    with memory.differentiable_transaction():
        memory.write(keys, values, torch.ones(2), scope=scope)
        read = memory.read(MemoryQuery(keys, scope=scope))
        read.value.sum().backward()

    assert keys.grad is not None and torch.isfinite(keys.grad).all()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_controller_passes_opaque_memory_scopes() -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    memory = ContentAddressedMemory(
        width=16, capacity=1, scope_capacity=2, write_threshold=0.0
    )
    state = controller.initial_state(2, device="cpu")
    events = AmodalEventCollection.empty(2, 16)
    scope = torch.tensor([0, 1], dtype=torch.long)

    first, _ = controller.step(
        events, state, _feedback(2, 3), memory, memory_scope=scope
    )
    assert first.memory_write_receipt is not None
    second, _ = controller.step(
        events, state, _feedback(2, 3), memory, memory_scope=scope
    )

    assert second.memory_read is not None
    assert second.memory_read.hit.tolist() == [True, True]
    assert memory.occupied.sum().item() == 2


def test_memory_backend_contract_supports_persistent_replacement(tmp_path) -> None:
    path = tmp_path / "persistent-memory.pt"
    memory = PersistentContentAddressedMemory(width=4, capacity=2, path=path)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    memory.write(key, value, torch.ones(1), timestamp=torch.tensor([11.0]))

    replacement = PersistentContentAddressedMemory(width=4, capacity=2, path=path)
    read = replacement.read(MemoryQuery(key))

    assert isinstance(replacement, MemoryBackend)
    assert read.hit.tolist() == [True]
    assert torch.allclose(read.value, value, atol=1e-5)

    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    runtime = AmodalControllerRuntime(controller, memory=replacement)
    assert isinstance(runtime.memory, MemoryBackend)

    replacement.clear()
    cleared = PersistentContentAddressedMemory(width=4, capacity=2, path=path)
    assert not cleared.read(MemoryQuery(key)).hit.any()


def test_append_only_memory_grows_without_replacing_prior_records() -> None:
    memory = AppendOnlyContentAddressedMemory(
        width=8, write_threshold=0.0, write_match_threshold=0.999
    )
    keys = torch.eye(8)
    values = torch.roll(keys, shifts=1, dims=0)
    receipt = memory.write(keys, values, torch.ones(8))

    assert receipt.indices.tolist() == list(range(8))
    assert memory.record_count == 8
    extra_key = torch.zeros(1, 8)
    extra_key[0, :2] = torch.tensor([0.6, 0.8])
    extra_value = torch.zeros(1, 8)
    extra_value[0, 2:4] = torch.tensor([0.6, 0.8])
    memory.write(extra_key, extra_value, torch.ones(1))

    assert memory.record_count == 9
    for index in range(8):
        read = memory.read(MemoryQuery(keys[index : index + 1]))
        assert read.hit.tolist() == [True]
        assert torch.allclose(read.value, values[index : index + 1], atol=1e-5)
    assert memory.read(MemoryQuery(extra_key)).hit.tolist() == [True]
    assert memory.candidates().occupied.sum().item() == 9


def test_append_only_memory_isolated_scopes_and_empty_reads() -> None:
    memory = AppendOnlyContentAddressedMemory(
        width=4, write_threshold=0.0, scope_capacity=2
    )
    query_key = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    memory.write(
        query_key,
        torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        torch.ones(2),
        scope=torch.tensor([0, 1]),
    )

    read = memory.read(MemoryQuery(query_key, scope=torch.tensor([0, 1])))
    assert read.hit.tolist() == [True, True]
    assert torch.allclose(read.value[0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(read.value[1], torch.tensor([0.0, 0.0, 1.0, 0.0]))

    empty = AppendOnlyContentAddressedMemory(width=4)
    empty_read = empty.read(MemoryQuery(torch.ones(2, 4)))
    assert empty_read.hit.tolist() == [False, False]
    assert empty_read.value.shape == (2, 4)
    assert empty.candidates().keys.shape == (1, 0, 4)


def test_persistent_append_only_memory_reloads_growth_and_rejects_corruption(
    tmp_path,
) -> None:
    path = tmp_path / "append-only-memory.pt"
    memory = PersistentAppendOnlyContentAddressedMemory(
        width=8, path=path, write_threshold=0.0, write_match_threshold=0.999
    )
    keys = torch.eye(8)
    values = torch.roll(keys, shifts=1, dims=0)
    memory.write(keys, values, torch.ones(8))
    assert memory.record_count == 8

    restored = PersistentAppendOnlyContentAddressedMemory(
        width=8, path=path, write_threshold=0.0, write_match_threshold=0.999
    )
    assert restored.record_count == 8
    assert restored.read(MemoryQuery(keys[6:7])).hit.tolist() == [True]

    payload = torch.load(path, weights_only=False)
    payload["state_dict"]["values"][0, 0] += 0.25
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checksum"):
        PersistentAppendOnlyContentAddressedMemory(
            width=8, path=path, write_threshold=0.0, write_match_threshold=0.999
        )


def test_runtime_checkpoint_round_trips_variable_capacity_memory(tmp_path) -> None:
    key = torch.zeros(1, 16)
    key[0, 0] = 1.0
    value = torch.zeros(1, 16)
    value[0, 1] = 1.0
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        ),
        memory=AppendOnlyContentAddressedMemory(width=16, write_threshold=0.0),
    )
    assert source.memory is not None
    source.memory.write(key, value, torch.ones(1))
    checkpoint = tmp_path / "append-only-runtime.pt"
    save_runtime(source, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        ),
        memory=AppendOnlyContentAddressedMemory(width=16, write_threshold=0.0),
    )
    load_runtime_components(restored, checkpoint)
    assert restored.memory is not None
    assert restored.memory.record_count == 1
    assert restored.memory.read(MemoryQuery(key)).hit.tolist() == [True]


def test_memory_read_reports_and_returns_no_value_for_a_near_miss() -> None:
    memory = ContentAddressedMemory(width=4, capacity=1)
    memory.write(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        torch.ones(1),
    )
    read = memory.read(
        MemoryQuery(torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
    )
    assert not read.hit.any()
    assert torch.equal(read.value, torch.zeros_like(read.value))


def test_persistent_memory_rolls_back_when_snapshot_fails(tmp_path, monkeypatch) -> None:
    memory = PersistentContentAddressedMemory(
        width=4, capacity=2, path=tmp_path / "persistent-memory.pt"
    )
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    memory.write(key, value, torch.ones(1))
    prior = {name: tensor.detach().clone() for name, tensor in memory.state_dict().items()}

    def fail_snapshot(_path) -> None:
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(memory, "snapshot", fail_snapshot)
    with pytest.raises(OSError, match="persistence"):
        memory.write(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), key, torch.ones(1))

    for name, tensor in prior.items():
        assert torch.equal(memory.state_dict()[name], tensor)


def test_memory_rejects_out_of_range_write_strength() -> None:
    memory = ContentAddressedMemory(width=4, capacity=2)
    with pytest.raises(ValueError, match="strength"):
        memory.write(torch.zeros(1, 4), torch.zeros(1, 4), torch.tensor([1.1]))


def test_controller_owns_memory_query_and_commit_path() -> None:
    controller = AmodalCognitiveController(
        width=16,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        event_window_capacity=4,
    )
    memory = ContentAddressedMemory(width=16, capacity=4, write_threshold=0.0)
    state = controller.initial_state(1, device="cpu")
    event = AmodalEventCollection.from_events([AmodalEvent(torch.randn(1, 16))])

    first, state = controller.step(event, state, _feedback(1, 3), memory)
    second, _ = controller.step(event, state, _feedback(1, 3), memory)

    assert first.memory_write_receipt is not None
    assert first.memory_write_receipt.committed.tolist() == [True]
    assert second.memory_read is not None
    assert second.memory_read.hit.tolist() == [True]


def test_memory_address_is_shared_and_stable_across_feedback() -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    event = AmodalEventCollection.from_events([AmodalEvent(torch.randn(1, 16))])
    state = controller.initial_state(1, device="cpu")
    first, _ = controller.step(event, state, _feedback(1, 3))
    second, _ = controller.step(
        event,
        state,
        ControllerFeedback(
            action=torch.ones(1, 3),
            reward=torch.ones(1),
            propensity=torch.ones(1),
            has_feedback=torch.ones(1),
        ),
    )

    assert torch.allclose(first.memory_key, first.memory_query_key)
    assert torch.allclose(second.memory_key, second.memory_query_key)
    assert torch.allclose(first.memory_key, second.memory_key)


def test_memory_address_is_invariant_to_event_age() -> None:
    controller = AmodalCognitiveController(
        width=16,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        stable_memory_address=True,
    )
    event = AmodalEventCollection.from_events([AmodalEvent(torch.randn(1, 16))])
    quiet = AmodalEventCollection.empty(1, 16)
    state = controller.initial_state(1, device="cpu")
    first, state = controller.step(event, state, _feedback(1, 3))
    aged, _ = controller.step(quiet, state, _feedback(1, 3), elapsed=7.0)

    assert torch.allclose(first.memory_key, aged.memory_key)


def test_memory_address_is_invariant_to_irrelevant_prior_event() -> None:
    controller = AmodalCognitiveController(
        width=16,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        event_window_capacity=4,
        stable_memory_address=True,
    )
    prior = AmodalEvent(torch.randn(1, 16))
    latest = AmodalEvent(torch.randn(1, 16))
    single = AmodalEventCollection.from_events([latest])
    pair = AmodalEventCollection.from_events([prior, latest])
    state = controller.initial_state(1, device="cpu")
    feedback = _feedback(1, 3)

    single_output, _ = controller.step(single, state, feedback)
    pair_output, _ = controller.step(pair, state, feedback)

    assert torch.allclose(single_output.memory_key, pair_output.memory_key)


def test_runtime_checkpoint_loads_independent_components(tmp_path) -> None:
    def build() -> AmodalControllerRuntime:
        controller = AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            event_window_capacity=4,
        )
        runtime = AmodalControllerRuntime(
            controller,
            encoders={"sensor": nn.Linear(7, 16)},
            output_bus=None,
            memory=ContentAddressedMemory(width=16, capacity=4),
        )
        runtime.register_decoder("protocol", OpaqueProtocolDecoder(5, 3))
        return runtime

    source = build()
    checkpoint = tmp_path / "runtime.pt"
    save_runtime(source, checkpoint, provenance={"test": True})
    restored = build()
    payload = load_runtime_components(restored, checkpoint)

    assert payload["format"] == "neural-computer.amodal-runtime.v29"
    for left, right in zip(source.parameters(), restored.parameters(), strict=True):
        assert torch.equal(left, right)


def test_growth_register_boundary_is_prior_only_and_stateful() -> None:
    controller = AmodalCognitiveController(
        width=16,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        growth_register_widths=(8, 10),
        growth_prior_only_from=1,
        growth_recurrent_from=1,
    )
    assert controller.growth_slots[0]["input"].in_features == 48
    assert controller.growth_slots[1]["input"].in_features == 8
    assert "recurrent" not in controller.growth_slots[0]
    assert "recurrent" in controller.growth_slots[1]

    state = controller.initial_state(2, device="cpu")
    assert state.growth_registers is not None
    assert tuple(register.shape for register in state.growth_registers) == (
        (2, 8),
        (2, 10),
    )
    event = AmodalEventCollection.from_events([AmodalEvent(torch.randn(2, 16))])
    output, next_state = controller.step(event, state, _feedback(2, 3))
    assert next_state.growth_registers is not None
    assert not torch.equal(next_state.growth_registers[1], state.growth_registers[1])
    assert output.intention.payload.shape == (2, 5)


def test_growth_registers_are_zero_output_until_artifact_is_loaded() -> None:
    torch.manual_seed(801)
    base = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    torch.manual_seed(801)
    expanded = AmodalCognitiveController(
        width=16,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        growth_register_widths=(8,),
    )
    base_state = base.initial_state(1, device="cpu")
    expanded_state = expanded.initial_state(1, device="cpu")
    event = AmodalEventCollection.from_events([AmodalEvent(torch.randn(1, 16))])
    feedback = _feedback(1, 3)
    base_output, _ = base.step(event, base_state, feedback)
    expanded_output, _ = expanded.step(event, expanded_state, feedback)
    assert torch.allclose(base_output.intention.payload, expanded_output.intention.payload)


def test_intention_conditioned_growth_is_zero_output_until_trained() -> None:
    torch.manual_seed(802)
    base = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    torch.manual_seed(802)
    expanded = AmodalCognitiveController(
        width=16,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        growth_register_widths=(8,),
        growth_recurrent_from=0,
        growth_gated=True,
        growth_from_intention=True,
        growth_gate_from_context=True,
    )
    assert expanded.growth_slots[0]["input"].in_features == 53
    assert expanded.growth_slots[0]["gate"].in_features == 53
    assert expanded.configuration()["growth_from_intention"] is True
    assert expanded.configuration()["growth_gate_from_context"] is True
    state = base.initial_state(1, device="cpu")
    expanded_state = expanded.initial_state(1, device="cpu")
    event = AmodalEventCollection.from_events([AmodalEvent(torch.randn(1, 16))])
    feedback = _feedback(1, 3)
    base_output, _ = base.step(event, state, feedback)
    expanded_output, _ = expanded.step(event, expanded_state, feedback)
    assert torch.allclose(base_output.intention.payload, expanded_output.intention.payload)


def test_growth_artifact_load_preserves_canonical_core() -> None:
    controller = AmodalCognitiveController(
        width=16,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        growth_register_widths=(8,),
    )
    growth_prefix = ("growth_slots.0.",)
    freeze_core(controller, growth_prefix)
    assert all(
        parameter.requires_grad == name.startswith(growth_prefix)
        for name, parameter in controller.named_parameters()
    )
    artifact = {
        name: value.detach().clone() + 0.1
        for name, value in controller.state_dict().items()
        if name.startswith(growth_prefix)
    }
    receipt = load_growth_artifact(controller, artifact, growth_prefixes=growth_prefix)
    assert receipt.core_unchanged


def test_artifact_consolidation_verifies_candidate_without_mutating_source(tmp_path) -> None:
    source = ExecutableArtifactMemory(tmp_path / "source", width=4, capacity=3)
    artifacts = [
        {"growth_slots.0.weight": torch.full((2, 2), float(index))}
        for index in range(3)
    ]
    keys = torch.eye(4)[:3]
    for key, artifact in zip(keys, artifacts, strict=True):
        source.put(key, artifact)

    def verifier(candidate: ExecutableArtifactMemory) -> bool:
        assert len(candidate.occupied) == 2
        _, merged = candidate.promote_index(0)
        _, survivor = candidate.promote_index(1)
        return (
            merged["growth_slots.0.weight"].shape == (2, 2)
            and survivor["growth_slots.0.weight"][0, 0].item() in {1.0, 2.0}
        )

    candidate, receipt = source.consolidate_verified(
        (0, 1),
        torch.tensor([0.0, 0.0, 0.6, 0.8]),
        {
            "growth_slots.0.weight": torch.ones(2, 2),
            "growth_slots.1.weight": torch.full((2, 2), 2.0),
        },
        tmp_path / "accepted",
        verifier=verifier,
    )
    assert isinstance(receipt, ArtifactConsolidationReceipt)
    assert receipt.accepted
    assert receipt.rows_before == 3
    assert receipt.rows_after == 2
    assert receipt.rows_saved == 1
    assert candidate is not None and len(candidate.occupied) == 2
    assert len(source.occupied) == 3

    rejected, rejected_receipt = source.consolidate_verified(
        (0, 1),
        torch.tensor([0.0, 0.0, 0.8, 0.6]),
        {"growth_slots.0.weight": torch.ones(2, 2)},
        tmp_path / "rejected",
        verifier=lambda _: False,
    )
    assert rejected is None
    assert not rejected_receipt.accepted
    assert rejected_receipt.rows_saved == 0
    assert len(source.occupied) == 3


def test_v27_runtime_checkpoint_migrates_without_growth_boundary(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v27"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v27"
    payload["configuration"]["controller"]["schema"] = (
        "neural-computer.controller.v27"
    )
    for field in (
        "growth_register_widths",
        "growth_prior_only_from",
        "growth_recurrent_from",
        "growth_boundary",
    ):
        payload["configuration"]["controller"].pop(field)
    checkpoint = tmp_path / "legacy-runtime-v27.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)
    assert restored.controller.configuration()["schema"] == (
        "neural-computer.controller.v29"
    )


def test_v28_runtime_checkpoint_migrates_without_stable_value_head(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v28"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v28"
    payload["configuration"]["controller"]["schema"] = (
        "neural-computer.controller.v28"
    )
    payload["configuration"]["controller"].pop("memory_value_stable")
    payload["components"]["controller"] = {
        key: value
        for key, value in payload["components"]["controller"].items()
        if not key.startswith("memory_value_stable.")
    }
    checkpoint = tmp_path / "legacy-runtime-v28.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == (
        "neural-computer.controller.v29"
    )
    assert restored.controller.memory_value_stable is not None
    assert torch.count_nonzero(restored.controller.memory_value_stable.weight) == 0


def test_v23_runtime_checkpoint_migrates_transport_augmented_address(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v23"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v23"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v23"
    payload["configuration"]["controller"]["memory_address"] = (
        "latest_event_token_v1"
    )
    checkpoint = tmp_path / "legacy-runtime-v23.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"
    assert restored.controller.configuration()["memory_address"] == (
        "latest_event_token_v1"
    )
    assert restored.controller.stable_memory_address is False


def test_v24_runtime_checkpoint_migrates_without_feedback_residual(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v24"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v24"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v24"
    payload["configuration"]["controller"].pop("memory_value_feedback")
    payload["components"]["controller"] = {
        key: value
        for key, value in payload["components"]["controller"].items()
        if not key.startswith("memory_value_feedback.")
    }
    checkpoint = tmp_path / "legacy-runtime-v24.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"
    assert restored.controller.configuration()["memory_value_feedback"] == "none_v1"
    assert restored.controller.memory_value_feedback_enabled is False
    assert restored.controller.stable_memory_address is True


def test_v25_runtime_checkpoint_migrates_without_address_residual(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v25"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v25"
    controller_configuration = payload["configuration"]["controller"]
    controller_configuration["schema"] = "neural-computer.controller.v25"
    controller_configuration["memory_address"] = "latest_event_payload_v1"
    controller_configuration["memory_write_event_window"] = (
        "latest_pair_context_and_match_v2"
    )
    controller_configuration["memory_write_event_match"] = "latest_prior_cosine_v1"
    checkpoint = tmp_path / "legacy-runtime-v25.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"
    assert restored.controller.configuration()["memory_address"] == (
        "latest_event_payload_v1"
    )
    assert restored.controller.memory_address_residual is False


def test_runtime_checkpoint_round_trips_replacement_memory_backend(tmp_path) -> None:
    source_memory = PersistentContentAddressedMemory(
        width=16, capacity=2, path=tmp_path / "source-memory.pt"
    )
    key = torch.zeros(1, 16)
    key[0, 0] = 1.0
    value = torch.zeros(1, 16)
    value[0, 1] = 1.0
    source_memory.write(key, value, torch.ones(1))
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        ),
        memory=source_memory,
    )
    checkpoint = tmp_path / "runtime-with-memory.pt"
    save_runtime(source, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        ),
        memory=PersistentContentAddressedMemory(
            width=16, capacity=2, path=tmp_path / "replacement-memory.pt"
        ),
    )
    load_runtime_components(restored, checkpoint)

    assert restored.memory is not None
    read = restored.memory.read(MemoryQuery(key))
    assert read.hit.tolist() == [True]
    assert torch.allclose(read.value, value, atol=1e-5)


def test_runtime_checkpoint_accepts_legacy_memory_config_and_backend_replacement(
    tmp_path,
) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        ),
        memory=ContentAddressedMemory(width=16, capacity=2),
    )
    payload = source.checkpoint_payload()
    payload["configuration"]["memory"] = dict(payload["configuration"]["memory"])
    payload["configuration"]["memory"].pop("write_match_threshold")
    checkpoint = tmp_path / "legacy-memory-config.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        ),
        memory=PersistentContentAddressedMemory(
            width=16, capacity=2, path=tmp_path / "replacement-memory.pt"
        ),
    )
    load_runtime_components(restored, checkpoint)

    assert isinstance(restored.memory, PersistentContentAddressedMemory)


def test_v17_runtime_checkpoint_migrates_to_shared_memory_address(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v17"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v17"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v17"
    payload["configuration"]["controller"].pop("memory_address")
    payload["configuration"]["controller"].pop("event_address_relevance")
    controller_state = dict(payload["components"]["controller"])
    address_weight = controller_state.pop("memory_address.weight")
    address_bias = controller_state.pop("memory_address.bias")
    controller_state.pop("event_address_relevance.weight")
    controller_state.pop("event_address_relevance.bias")
    legacy_query_weight = torch.cat(
        [address_weight, torch.zeros_like(address_weight)], dim=1
    )
    controller_state["memory_query.weight"] = legacy_query_weight
    controller_state["memory_query.bias"] = address_bias
    controller_state["memory_key.weight"] = legacy_query_weight.clone()
    controller_state["memory_key.bias"] = address_bias.clone()
    payload["components"]["controller"] = controller_state
    checkpoint = tmp_path / "legacy-runtime-v17.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"
    assert torch.allclose(restored.controller.memory_address.weight, address_weight)


def test_v19_runtime_checkpoint_migrates_pair_context_write_policy(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v19"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v19"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v19"
    payload["configuration"]["controller"].pop("memory_write_event_window")
    checkpoint = tmp_path / "legacy-runtime-v19.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"


def test_v20_runtime_checkpoint_migrates_to_stochastic_write_capable_runtime(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v20"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v20"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v20"
    payload["configuration"]["controller"]["memory_address"] = "event_window_shared"
    payload["configuration"]["controller"][
        "memory_write_event_window"
    ] = "latest_pair_context_v1"
    payload["configuration"]["controller"].pop("memory_write_sampling")
    checkpoint = tmp_path / "legacy-runtime-v20.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    configuration = restored.controller.configuration()
    assert configuration["schema"] == "neural-computer.controller.v29"
    assert configuration["memory_write_sampling"] == "bernoulli_straight_through_v1"


def test_v21_runtime_checkpoint_migrates_latest_event_address_policy(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v21"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v21"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v21"
    payload["configuration"]["controller"]["memory_address"] = "event_window_shared"
    checkpoint = tmp_path / "legacy-runtime-v21.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    configuration = restored.controller.configuration()
    assert configuration["schema"] == "neural-computer.controller.v29"
    assert configuration["memory_address"] == "latest_event_token_v1"


def test_v22_runtime_checkpoint_migrates_latest_prior_match_policy(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v22"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v22"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v22"
    payload["configuration"]["controller"].pop("memory_write_event_match")
    checkpoint = tmp_path / "legacy-runtime-v22.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    configuration = restored.controller.configuration()
    assert configuration["schema"] == "neural-computer.controller.v29"
    assert (
        configuration["memory_write_event_match"]
        == "latest_prior_stable_content_cosine_and_max_v3"
    )


def test_runtime_checkpoint_rejects_corrupt_memory_state(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        ),
        memory=ContentAddressedMemory(width=16, capacity=2),
    )
    payload = source.checkpoint_payload()
    payload["components"]["memory"]["strengths"][0] = 2.0
    checkpoint = tmp_path / "corrupt-runtime-memory.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        ),
        memory=ContentAddressedMemory(width=16, capacity=2),
    )
    with pytest.raises(ValueError, match="strengths"):
        load_runtime_components(restored, checkpoint)


def test_v1_runtime_checkpoint_migrates_without_execution_weights(tmp_path) -> None:
    controller = AmodalCognitiveController(
        width=16, workspace_slots=2, intention_width=5, feedback_width=3
    )
    source = AmodalControllerRuntime(controller)
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v1"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v1"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v1"
    payload["configuration"]["controller"].pop("execution_hidden")
    payload["configuration"]["controller"].pop("execution_states")
    payload["configuration"]["controller"].pop("execution_transport_policy")
    payload["configuration"]["controller"].pop("execution_transport_features")
    payload["configuration"]["controller"].pop("execution_timeout_policy")
    payload["configuration"]["controller"].pop("event_pair_attention")
    payload["configuration"]["controller"].pop("event_pair_relevance")
    payload["configuration"]["controller"].pop("event_feedback_relevance")
    payload["configuration"]["controller"].pop("event_feedback_source_relevance")
    payload["configuration"]["controller"].pop("source_credit_state")
    payload["configuration"]["controller"].pop("source_credit_decay")
    payload["configuration"]["controller"].pop("source_credit_policy")
    payload["configuration"]["controller"].pop("source_credit_hidden")
    payload["configuration"]["controller"].pop("source_trust_binding")
    payload["configuration"]["controller"].pop("source_trust_binding_scale")
    payload["components"]["controller"] = {
        key: value
        for key, value in payload["components"]["controller"].items()
        if not key.startswith("execution_policy.")
        and not key.startswith("execution_transport_policy.")
        and not key.startswith("execution_timeout_policy.")
        and not key.startswith("event_pair_")
    }
    checkpoint = tmp_path / "legacy-runtime.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.execution_policy[-1].bias.tolist() == [0.0, 0.0, 1.0]


def test_v2_runtime_checkpoint_migrates_without_transport_policy(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v2"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v2"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v2"
    payload["configuration"]["controller"].pop("execution_transport_policy")
    payload["configuration"]["controller"].pop("execution_transport_features")
    payload["configuration"]["controller"].pop("execution_timeout_policy")
    payload["configuration"]["controller"].pop("event_pair_attention")
    payload["configuration"]["controller"].pop("event_pair_relevance")
    payload["configuration"]["controller"].pop("event_feedback_relevance")
    payload["configuration"]["controller"].pop("event_feedback_source_relevance")
    payload["configuration"]["controller"].pop("source_credit_state")
    payload["configuration"]["controller"].pop("source_credit_decay")
    payload["configuration"]["controller"].pop("source_credit_policy")
    payload["configuration"]["controller"].pop("source_credit_hidden")
    payload["configuration"]["controller"].pop("source_trust_binding")
    payload["configuration"]["controller"].pop("source_trust_binding_scale")
    payload["components"]["controller"] = {
        key: value
        for key, value in payload["components"]["controller"].items()
        if not key.startswith("execution_transport_policy.")
        and not key.startswith("execution_timeout_policy.")
        and not key.startswith("event_pair_")
    }
    checkpoint = tmp_path / "legacy-runtime-v2.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.execution_transport_policy[-1].bias.tolist() == [0.0, 0.0, 1.0]


def test_v3_runtime_checkpoint_migrates_scalar_transport_policy(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v3"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v3"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v3"
    payload["configuration"]["controller"].pop("execution_transport_features")
    payload["configuration"]["controller"].pop("execution_timeout_policy")
    payload["configuration"]["controller"].pop("event_pair_attention")
    payload["configuration"]["controller"].pop("event_pair_relevance")
    payload["configuration"]["controller"].pop("event_feedback_relevance")
    payload["configuration"]["controller"].pop("event_feedback_source_relevance")
    payload["configuration"]["controller"].pop("source_credit_state")
    payload["configuration"]["controller"].pop("source_credit_decay")
    payload["configuration"]["controller"].pop("source_credit_policy")
    payload["configuration"]["controller"].pop("source_credit_hidden")
    payload["configuration"]["controller"].pop("source_trust_binding")
    payload["configuration"]["controller"].pop("source_trust_binding_scale")
    old_transport = nn.Linear(1, 3)
    with torch.no_grad():
        old_transport.weight.zero_()
        old_transport.bias.copy_(torch.tensor([0.0, 0.0, 1.0]))
    controller_state = {
        key: value
        for key, value in payload["components"]["controller"].items()
        if not key.startswith("execution_transport_policy.")
        and not key.startswith("execution_timeout_policy.")
        and not key.startswith("event_pair_")
    }
    controller_state.update(
        {
            "execution_transport_policy.weight": old_transport.weight,
            "execution_transport_policy.bias": old_transport.bias,
        }
    )
    payload["components"]["controller"] = controller_state
    checkpoint = tmp_path / "legacy-runtime-v3.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.execution_transport_policy[-1].bias.tolist() == [
        0.0,
        0.0,
        1.0,
    ]


def test_v6_runtime_checkpoint_migrates_without_pairwise_event_weights(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v6"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v6"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v6"
    payload["configuration"]["controller"].pop("event_pair_attention")
    payload["configuration"]["controller"].pop("event_pair_relevance")
    payload["configuration"]["controller"].pop("event_feedback_relevance")
    payload["configuration"]["controller"].pop("event_feedback_source_relevance")
    payload["configuration"]["controller"].pop("source_credit_state")
    payload["configuration"]["controller"].pop("source_credit_decay")
    payload["configuration"]["controller"].pop("source_credit_policy")
    payload["configuration"]["controller"].pop("source_credit_hidden")
    payload["configuration"]["controller"].pop("source_trust_binding")
    payload["configuration"]["controller"].pop("source_trust_binding_scale")
    payload["components"]["controller"] = {
        key: value
        for key, value in payload["components"]["controller"].items()
        if not key.startswith("event_pair_")
    }
    checkpoint = tmp_path / "legacy-runtime-v6.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"


def test_v8_runtime_checkpoint_migrates_without_feedback_binding_weights(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v8"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v8"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v8"
    payload["configuration"]["controller"].pop("event_feedback_relevance")
    payload["configuration"]["controller"].pop("event_feedback_source_relevance")
    payload["configuration"]["controller"].pop("source_credit_state")
    payload["configuration"]["controller"].pop("source_credit_decay")
    payload["configuration"]["controller"].pop("source_credit_policy")
    payload["configuration"]["controller"].pop("source_credit_hidden")
    payload["configuration"]["controller"].pop("source_trust_binding")
    payload["configuration"]["controller"].pop("source_trust_binding_scale")
    payload["components"]["controller"] = {
        key: value
        for key, value in payload["components"]["controller"].items()
        if not key.startswith("event_feedback_relevance.")
    }
    checkpoint = tmp_path / "legacy-runtime-v8.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"


def test_v9_runtime_checkpoint_migrates_without_source_credit_weights(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v9"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v9"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v9"
    payload["configuration"]["controller"].pop("event_feedback_source_relevance")
    payload["configuration"]["controller"].pop("source_credit_state")
    payload["configuration"]["controller"].pop("source_credit_decay")
    payload["configuration"]["controller"].pop("source_credit_policy")
    payload["configuration"]["controller"].pop("source_credit_hidden")
    payload["configuration"]["controller"].pop("source_trust_binding")
    payload["configuration"]["controller"].pop("source_trust_binding_scale")
    payload["components"]["controller"] = {
        key: value
        for key, value in payload["components"]["controller"].items()
        if not key.startswith("event_feedback_source_relevance.")
    }
    checkpoint = tmp_path / "legacy-runtime-v9.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16, workspace_slots=2, intention_width=5, feedback_width=3
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"


def test_v10_runtime_checkpoint_migrates_without_source_credit_policy(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            source_key_width=4,
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v10"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v10"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v10"
    payload["configuration"]["controller"].pop("source_credit_state")
    payload["configuration"]["controller"].pop("source_credit_decay")
    payload["configuration"]["controller"].pop("source_credit_policy")
    payload["configuration"]["controller"].pop("source_credit_hidden")
    payload["configuration"]["controller"].pop("source_trust_binding")
    payload["configuration"]["controller"].pop("source_trust_binding_scale")
    checkpoint = tmp_path / "legacy-runtime-v10.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            source_key_width=4,
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"


def test_v11_runtime_checkpoint_migrates_without_source_credit_policy(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            source_key_width=4,
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v11"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v11"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v11"
    payload["configuration"]["controller"].pop("source_credit_policy")
    payload["configuration"]["controller"].pop("source_credit_hidden")
    payload["configuration"]["controller"].pop("source_trust_binding")
    payload["configuration"]["controller"].pop("source_trust_binding_scale")
    checkpoint = tmp_path / "legacy-runtime-v11.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            source_key_width=4,
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"


def test_v12_runtime_checkpoint_migrates_without_direct_trust_binding(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            source_key_width=4,
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v12"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v12"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v12"
    payload["configuration"]["controller"].pop("source_trust_binding")
    payload["configuration"]["controller"].pop("source_trust_binding_scale")
    checkpoint = tmp_path / "legacy-runtime-v12.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            source_key_width=4,
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"


def test_v13_runtime_checkpoint_preserves_previous_trust_binding_scale(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            source_key_width=4,
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v13"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v13"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v13"
    payload["configuration"]["controller"]["source_trust_binding_scale"] = 0.5
    checkpoint = tmp_path / "legacy-runtime-v13.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            source_key_width=4,
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"
    assert restored.controller.source_trust_binding_scale == 0.5


def test_v14_runtime_checkpoint_migrates_vector_credit_head(tmp_path) -> None:
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            source_key_width=4,
        )
    )
    payload = source.checkpoint_payload()
    payload["format"] = "neural-computer.amodal-runtime.v14"
    payload["configuration"]["format"] = "neural-computer.amodal-runtime.v14"
    payload["configuration"]["controller"]["schema"] = "neural-computer.controller.v14"
    payload["configuration"]["controller"].pop("source_credit_projection")
    payload["components"]["controller"]["source_credit_policy.2.weight"] = torch.randn(
        source.controller.source_key_width, source.controller.source_credit_hidden
    )
    payload["components"]["controller"]["source_credit_policy.2.bias"] = torch.randn(
        source.controller.source_key_width
    )
    checkpoint = tmp_path / "legacy-runtime-v14.pt"
    torch.save(payload, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=16,
            workspace_slots=2,
            intention_width=5,
            feedback_width=3,
            source_key_width=4,
        )
    )
    load_runtime_components(restored, checkpoint)

    assert restored.controller.configuration()["schema"] == "neural-computer.controller.v29"
    assert restored.controller.source_credit_policy[-1].out_features == 4
    assert restored.controller.source_trust_binding_scale == 0.25
