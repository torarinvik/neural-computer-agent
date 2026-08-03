from __future__ import annotations

import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    AmodalEventWindowBuffer,
    ContentAddressedMemory,
    ControllerFeedback,
    EventWaitPolicy,
    MemoryQuery,
    OpaqueProtocolDecoder,
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

    assert payload["format"] == "neural-computer.amodal-runtime.v1"
    for left, right in zip(source.parameters(), restored.parameters(), strict=True):
        assert torch.equal(left, right)
