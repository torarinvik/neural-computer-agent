from pathlib import Path

import pytest
import torch

from .amodal_interface import AmodalEvent, AmodalEventCollection, IntentEvent
from .amodal_runtime import (
    EXTRACTED_CHECKPOINT_FORMAT,
    ActionIntentDecoder,
    AmodalControllerRuntime,
    AmodalEventWindowBuffer,
    AmodalEventWindowStatus,
    AmodalEventTimeline,
    AmodalInputBus,
    AmodalOutputBus,
    ExtractedAmodalRuntime,
    FactorizedOpaqueProtocolDecoder,
    OpaqueProtocolDecoder,
    canonicalize_action_adapter_payload,
    convert_legacy_checkpoint,
    runtime_from_extracted_payload,
    runtime_from_legacy_payload,
)
from .environment import NULL_ACTION
from .model import ControllerOutput, ControllerState, UnifiedCognitiveController
from .train_amodal_latent_alignment import LatentBasisFrontend
from .train_amodal_audio_alignment import (
    PairRelationAudioEncoder,
    render_pair_relation_audio,
)


def _configuration(**updates: object) -> dict[str, object]:
    configuration: dict[str, object] = {
        "width": 32,
        "workspace_slots": 3,
        "intention_width": 8,
    }
    configuration.update(updates)
    return configuration


def _assert_optional_equal(
    left: torch.Tensor | None, right: torch.Tensor | None
) -> None:
    assert (left is None) == (right is None)
    if left is not None and right is not None:
        assert torch.equal(left, right)


def _assert_output_equal(left: ControllerOutput, right: ControllerOutput) -> None:
    for name in (
        "logits",
        "intention",
        "memory_key",
        "memory_value",
        "memory_write_strength",
        "workspace_read",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name)), name
    _assert_optional_equal(left.skill_adapter_openings, right.skill_adapter_openings)
    _assert_optional_equal(
        left.skill_adapter_residual_norms, right.skill_adapter_residual_norms
    )


def _assert_state_equal(left: ControllerState, right: ControllerState) -> None:
    assert torch.equal(left.hidden, right.hidden)
    assert torch.equal(left.workspace, right.workspace)
    assert torch.equal(left.latest_event, right.latest_event)


def test_latent_basis_frontend_exposes_only_a_trainable_alignment_adapter() -> None:
    frontend = torch.nn.Identity()
    adapter = LatentBasisFrontend(frontend, width=4)
    source = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(adapter(source), source.flip(-1))
    assert not any(parameter.requires_grad for parameter in frontend.parameters())
    assert all(parameter.requires_grad for parameter in adapter.adapter.parameters())
    assert torch.equal(
        adapter.basis_permutation,
        torch.tensor([3, 2, 1, 0]),
    )


def test_audio_frontend_has_device_independent_raw_waveform_contract() -> None:
    frames = torch.rand(3, 3, 32, 32)
    waveform = render_pair_relation_audio(frames, samples=2048)
    assert waveform.shape == (3, 1, 2048)
    frontend = PairRelationAudioEncoder(32, samples=2048)
    encoded = frontend(waveform)
    assert encoded.shape == (3, 32)
    assert torch.isfinite(encoded).all()


@pytest.mark.parametrize(
    "configuration",
    [
        _configuration(),
        _configuration(action_adapter_width=8, action_adapter_gated=True),
        _configuration(
            action_adapter_width=8,
            action_adapter_gated=True,
            action_adapter_into_intention=True,
        ),
        _configuration(
            relation_adapter_width=8,
            relation_adapter_gated=True,
            skill_adapter_widths=(8,),
            skill_adapter_gate_mode="relu",
        ),
    ],
)
def test_extracted_runtime_is_bit_identical_across_legacy_paths(
    configuration: dict[str, object],
) -> None:
    torch.manual_seed(1701)
    legacy = UnifiedCognitiveController(**configuration).eval()
    runtime = ExtractedAmodalRuntime.from_legacy(legacy).eval()
    legacy_state = legacy.initial_state(4, device="cpu")
    runtime_state = runtime.initial_state(4, device="cpu")
    previous_action = torch.full((4,), NULL_ACTION, dtype=torch.long)
    previous_reward = torch.zeros(4)
    for step in range(4):
        frame = torch.randn(4, 3, 32, 32)
        has_feedback = torch.full((4,), float(step > 0))
        legacy_output, legacy_state = legacy.step(
            frame, legacy_state, previous_action, previous_reward, has_feedback
        )
        runtime_output, runtime_state = runtime.step(
            frame, runtime_state, previous_action, previous_reward, has_feedback
        )
        _assert_output_equal(legacy_output, runtime_output)
        _assert_state_equal(legacy_state, runtime_state)
        previous_action = legacy_output.logits.argmax(dim=-1)
        previous_reward = (previous_action == (step % 2)).float()


def test_controller_owns_no_adapter_after_extraction() -> None:
    model = UnifiedCognitiveController(**_configuration())
    original_keys = set(model.state_dict())
    runtime = ExtractedAmodalRuntime.from_legacy(model)
    assert model.vision is not None
    assert model.actuator is not None
    assert runtime.controller.vision is None
    assert runtime.controller.actuator is None
    assert not any(
        name.startswith(("vision.", "actuator."))
        for name in runtime.controller.state_dict()
    )
    parameter_sets = [
        {id(parameter) for parameter in module.parameters()}
        for module in (runtime.encoder, runtime.controller, runtime.decoder)
    ]
    assert not (parameter_sets[0] & parameter_sets[1])
    assert not (parameter_sets[0] & parameter_sets[2])
    assert not (parameter_sets[1] & parameter_sets[2])
    assert set(runtime.legacy_state_dict()) == original_keys
    for name, value in model.state_dict().items():
        assert torch.equal(value, runtime.legacy_state_dict()[name]), name


def test_explicit_event_path_matches_extracted_frame_path() -> None:
    torch.manual_seed(1702)
    runtime = ExtractedAmodalRuntime.from_legacy(
        UnifiedCognitiveController(**_configuration())
    ).eval()
    frame = torch.randn(3, 3, 32, 32)
    state = runtime.initial_state(3, device="cpu")
    previous_action = torch.full((3,), NULL_ACTION, dtype=torch.long)
    previous_reward = torch.zeros(3)
    has_feedback = torch.zeros(3)
    event = runtime.encode(frame)
    assert isinstance(event, AmodalEvent)
    frame_output, frame_state = runtime.step(
        frame, state, previous_action, previous_reward, has_feedback
    )
    event_output, event_state = runtime.step_event(
        event, state, previous_action, previous_reward, has_feedback
    )
    _assert_output_equal(frame_output, event_output)
    _assert_state_equal(frame_state, event_state)


def test_extracted_components_round_trip_independently(tmp_path: Path) -> None:
    torch.manual_seed(1703)
    configuration = _configuration(
        action_adapter_width=8,
        action_adapter_gated=True,
        relation_adapter_width=8,
        relation_adapter_gated=True,
    )
    legacy = UnifiedCognitiveController(**configuration)
    legacy_payload = {
        "model_configuration": configuration,
        "state_dict": legacy.state_dict(),
    }
    runtime = runtime_from_legacy_payload(legacy_payload)
    extracted = runtime.extracted_payload(
        model_configuration=configuration, provenance={"test": True}
    )
    assert extracted["format"] == EXTRACTED_CHECKPOINT_FORMAT
    restored = runtime_from_extracted_payload(extracted)
    assert set(restored.legacy_state_dict()) == set(legacy.state_dict())
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, restored.legacy_state_dict()[name]), name

    source = tmp_path / "legacy.pt"
    destination = tmp_path / "extracted.pt"
    torch.save(legacy_payload, source)
    convert_legacy_checkpoint(source, destination)
    on_disk = torch.load(destination, weights_only=False)
    assert on_disk["format"] == EXTRACTED_CHECKPOINT_FORMAT
    disk_runtime = runtime_from_extracted_payload(on_disk)
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, disk_runtime.legacy_state_dict()[name]), name


def test_legacy_loader_allows_only_new_zero_initialized_critic_scale() -> None:
    configuration = _configuration(skill_adapter_widths=(8,))
    legacy = UnifiedCognitiveController(**configuration)
    state = dict(legacy.state_dict())
    key = "skill_adapter_critic_scales.0"
    assert key in state
    del state[key]
    runtime = runtime_from_legacy_payload(
        {"model_configuration": configuration, "state_dict": state}
    )
    assert torch.equal(
        runtime.controller.skill_adapter_critic_scales[0],
        torch.zeros_like(runtime.controller.skill_adapter_critic_scales[0]),
    )


def test_interface_rejects_wrong_width_and_schema() -> None:
    with pytest.raises(ValueError, match="event width"):
        AmodalEvent(torch.zeros(2, 3)).validate(width=4)
    with pytest.raises(ValueError, match="unsupported event schema"):
        AmodalEvent(torch.zeros(2, 3), schema="wrong").validate()
    with pytest.raises(ValueError, match="intention width"):
        IntentEvent(torch.zeros(2, 3)).validate(width=4)


def test_action_residual_can_be_folded_into_base_intention() -> None:
    torch.manual_seed(1704)
    configuration = _configuration(action_adapter_width=8, action_adapter_gated=True)
    source = UnifiedCognitiveController(**configuration).eval()
    with torch.no_grad():
        source.action_adapter[-1].weight.normal_(mean=0.0, std=0.1)
        source.action_adapter[-1].bias.normal_(mean=0.0, std=0.1)
        source.action_adapter_gate.bias.fill_(0.5)
    migrated_payload = canonicalize_action_adapter_payload(
        {
            "model_configuration": configuration,
            "state_dict": source.state_dict(),
        }
    )
    migrated = UnifiedCognitiveController(
        **migrated_payload["model_configuration"]
    ).eval()
    migrated.load_state_dict(migrated_payload["state_dict"])
    assert migrated.action_adapter_emits_intention

    source_state = source.initial_state(32, device="cpu")
    migrated_state = migrated.initial_state(32, device="cpu")
    previous_action = torch.full((32,), NULL_ACTION, dtype=torch.long)
    previous_reward = torch.zeros(32)
    maximum_difference = 0.0
    for step in range(5):
        frame = torch.randn(32, 3, 32, 32)
        feedback = torch.full((32,), float(step > 0))
        source_output, source_state = source.step(
            frame, source_state, previous_action, previous_reward, feedback
        )
        migrated_output, migrated_state = migrated.step(
            frame, migrated_state, previous_action, previous_reward, feedback
        )
        maximum_difference = max(
            maximum_difference,
            float((source_output.logits - migrated_output.logits).abs().max().detach()),
        )
        assert torch.equal(
            source_output.logits.argmax(dim=-1), migrated_output.logits.argmax(dim=-1)
        )
        assert torch.equal(source_state.hidden, migrated_state.hidden)
        assert torch.equal(source_state.workspace, migrated_state.workspace)
        previous_action = source_output.logits.argmax(dim=-1)
        previous_reward = (previous_action == (step % 2)).float()
    assert maximum_difference < 1e-6

    runtime = ExtractedAmodalRuntime.from_legacy(migrated).eval()
    assert not runtime.compatibility_suffix_active
    event = runtime.encode(torch.randn(4, 3, 32, 32))
    core_output, _ = runtime.controller.step_event(
        event,
        runtime.initial_state(4, device="cpu"),
        torch.full((4,), NULL_ACTION, dtype=torch.long),
        torch.zeros(4),
        torch.zeros(4),
    )
    assert torch.count_nonzero(core_output.intent_event.payload[:, -2:]) == 0


def test_canonicalization_rejects_missing_or_already_migrated_adapter() -> None:
    plain = UnifiedCognitiveController(**_configuration())
    with pytest.raises(ValueError, match="no action adapter"):
        canonicalize_action_adapter_payload(
            {
                "model_configuration": _configuration(),
                "state_dict": plain.state_dict(),
            }
        )
    configuration = _configuration(
        action_adapter_width=8, action_adapter_emits_intention=True
    )
    migrated = UnifiedCognitiveController(**configuration)
    with pytest.raises(ValueError, match="already emits"):
        canonicalize_action_adapter_payload(
            {
                "model_configuration": configuration,
                "state_dict": migrated.state_dict(),
            }
        )


def test_output_bus_fans_one_frozen_intention_to_multiple_decoders() -> None:
    torch.manual_seed(1705)
    runtime = ExtractedAmodalRuntime.from_legacy(
        UnifiedCognitiveController(**_configuration())
    ).eval()
    state = runtime.initial_state(5, device="cpu")
    frame = torch.randn(5, 3, 32, 32)
    previous_action = torch.full((5,), NULL_ACTION, dtype=torch.long)
    core_output, _ = runtime.step_intention(
        frame,
        state,
        previous_action,
        torch.zeros(5),
        torch.zeros(5),
    )
    primary = ActionIntentDecoder(
        runtime.decoder.projection,
        intention_width=runtime.controller.intention_width,
    )
    protocol = OpaqueProtocolDecoder(runtime.controller.intention_width)
    bus = AmodalOutputBus({"primary": primary})
    bus.register_decoder("protocol", protocol)
    outputs = bus(core_output.intent_event)
    assert set(outputs) == {"primary", "protocol"}
    assert torch.equal(outputs["primary"], runtime.decode(core_output.intent_event))
    assert outputs["protocol"].shape == (5, 2)


def test_output_bus_cardinality_is_runtime_variable() -> None:
    intention = IntentEvent(torch.randn(3, 10))
    empty = AmodalOutputBus()
    assert empty(intention) == {}
    bus = AmodalOutputBus({"one": OpaqueProtocolDecoder(8)})
    assert set(bus(intention)) == {"one"}
    bus.register_decoder("two", OpaqueProtocolDecoder(8, commands=3, hidden=4))
    outputs = bus(intention)
    assert outputs["one"].shape == (3, 2)
    assert outputs["two"].shape == (3, 3)


def _amodal_controller_runtime() -> AmodalControllerRuntime:
    extracted = ExtractedAmodalRuntime.from_legacy(
        UnifiedCognitiveController(**_configuration())
    )
    runtime = AmodalControllerRuntime(
        extracted.controller,
        encoders={"vision": torch.nn.Identity(), "audio": torch.nn.Identity()},
    )
    runtime.register_decoder("text", OpaqueProtocolDecoder(8, commands=3))
    runtime.register_decoder("device", OpaqueProtocolDecoder(8, commands=5))
    return runtime.eval()


def test_amodal_runtime_accepts_runtime_variable_encoder_and_decoder_counts() -> None:
    torch.manual_seed(1712)
    runtime = _amodal_controller_runtime()
    batch = 4
    state = runtime.initial_state(batch, device="cpu")
    previous_action = torch.full((batch,), NULL_ACTION, dtype=torch.long)
    streams = {
        "vision": torch.randn(batch, 32),
        "audio": torch.randn(batch, 32),
    }
    output, next_state = runtime.step_streams(
        streams,
        state,
        previous_action,
        torch.zeros(batch),
        torch.zeros(batch),
    )
    assert set(output.decoded) == {"text", "device"}
    assert output.decoded["text"].shape == (batch, 3)
    assert output.decoded["device"].shape == (batch, 5)
    assert next_state.hidden.shape == (batch, 32)
    # Adding another frontend event does not alter controller parameter shapes.
    assert runtime.controller.width == 32
    assert runtime.controller.intention_width == 8


def test_amodal_runtime_is_permutation_invariant_for_simultaneous_streams() -> None:
    torch.manual_seed(1713)
    runtime = _amodal_controller_runtime()
    batch = 3
    state = runtime.initial_state(batch, device="cpu")
    previous_action = torch.full((batch,), NULL_ACTION, dtype=torch.long)
    vision = torch.randn(batch, 32)
    audio = torch.randn(batch, 32)
    first, first_state = runtime.step_streams(
        {"vision": vision, "audio": audio},
        state,
        previous_action,
        torch.zeros(batch),
        torch.zeros(batch),
    )
    second, second_state = runtime.step_streams(
        {"audio": audio, "vision": vision},
        state,
        previous_action,
        torch.zeros(batch),
        torch.zeros(batch),
    )
    torch.testing.assert_close(first.intention.payload, second.intention.payload)
    torch.testing.assert_close(first_state.hidden, second_state.hidden)
    for name in first.decoded:
        torch.testing.assert_close(first.decoded[name], second.decoded[name])


def test_amodal_runtime_accepts_preencoded_events_without_source_semantics() -> None:
    runtime = _amodal_controller_runtime()
    event = AmodalEvent(torch.randn(2, 32))
    encoded = runtime.encode_streams({"arbitrary_sensor": event})
    assert len(encoded) == 1
    assert encoded[0].source_key is None
    assert torch.equal(encoded[0].payload, event.payload)


def test_amodal_runtime_rejects_missing_stream_encoder_and_empty_input() -> None:
    runtime = _amodal_controller_runtime()
    with pytest.raises(ValueError, match="at least one input stream"):
        runtime.encode_streams({})
    with pytest.raises(KeyError, match="no encoder registered"):
        runtime.encode_streams({"touch": torch.randn(2, 32)})


def test_factorized_decoder_emits_joint_opaque_bitmask_logits() -> None:
    torch.manual_seed(1711)
    decoder = FactorizedOpaqueProtocolDecoder(8, bits=2)
    logits = decoder(torch.randn(5, 8))
    assert logits.shape == (5, 4)
    assert torch.isfinite(logits).all()


def test_input_bus_preserves_one_event_and_identical_duplicates_exactly() -> None:
    torch.manual_seed(1706)
    payload = torch.randn(7, 12)
    event = AmodalEvent(payload)
    bus = AmodalInputBus(12)
    single = bus([event])
    duplicate = bus([event, event])
    assert torch.equal(single.payload, payload)
    assert torch.equal(duplicate.payload, payload)
    assert torch.equal(single.confidence, torch.ones(7))
    assert torch.equal(duplicate.confidence, torch.ones(7))


def test_input_bus_optional_event_normalization_balances_scales() -> None:
    large = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    small = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    bus = AmodalInputBus(4, normalize_events=True)
    result = bus([AmodalEvent(large), AmodalEvent(small)])
    torch.testing.assert_close(
        result.payload, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))


def test_input_bus_supports_per_example_cardinality_masks() -> None:
    payload = torch.tensor(
        [
            [[2.0, 4.0], [8.0, 12.0]],
            [[3.0, 5.0], [99.0, 99.0]],
        ]
    )
    collection = AmodalEventCollection(
        payload=payload,
        present=torch.tensor([[True, True], [True, False]]),
        confidence=torch.ones(2, 2),
    )
    combined = AmodalInputBus(2)(collection)
    assert torch.equal(combined.payload[0], torch.tensor([5.0, 8.0]))
    assert torch.equal(combined.payload[1], torch.tensor([3.0, 5.0]))


def test_input_bus_confidence_is_generic_attention_prior() -> None:
    collection = AmodalEventCollection(
        payload=torch.tensor([[[0.0], [10.0]]]),
        present=torch.ones(1, 2, dtype=torch.bool),
        confidence=torch.tensor([[3.0, 1.0]]),
    )
    combined = AmodalInputBus(1)(collection)
    assert torch.allclose(combined.payload, torch.tensor([[2.5]]))
    assert torch.allclose(combined.confidence, torch.tensor([2.5]))


def test_learned_input_residual_cannot_change_single_or_duplicate_events() -> None:
    torch.manual_seed(1707)
    event = AmodalEvent(torch.randn(4, 6))
    bus = AmodalInputBus(6, residual_hidden=5)
    assert bus.residual is not None
    with torch.no_grad():
        for parameter in bus.parameters():
            parameter.normal_()
    assert torch.equal(bus([event]).payload, event.payload)
    assert torch.equal(bus([event, event]).payload, event.payload)


def test_second_moment_input_residual_preserves_single_event_invariant() -> None:
    torch.manual_seed(1710)
    event = AmodalEvent(torch.randn(4, 6))
    bus = AmodalInputBus(6, residual_hidden=8, second_moment=True)
    assert bus.residual is not None
    with torch.no_grad():
        for parameter in bus.parameters():
            parameter.normal_()
    assert torch.equal(bus([event]).payload, event.payload)
    assert torch.equal(bus([event, event]).payload, event.payload)


def test_event_timeline_reorders_delivery_but_respects_timestamp_boundaries() -> None:
    torch.manual_seed(1708)
    first = torch.randn(3, 5)
    second = torch.randn(3, 5)
    event_a = AmodalEvent(first, timestamp=torch.zeros(3))
    event_b = AmodalEvent(second, timestamp=torch.zeros(3))
    windows = AmodalEventTimeline([event_b, event_a]).windows()
    assert len(windows) == 1
    assert torch.equal(windows[0].payload[:, 0], second)
    assert torch.equal(windows[0].payload[:, 1], first)

    delayed = AmodalEvent(second, timestamp=torch.ones(3))
    assert len(AmodalEventTimeline([event_a, delayed]).windows()) == 2
    assert len(AmodalEventTimeline([event_a, delayed], tolerance=1.0).windows()) == 1


def test_streaming_window_buffer_waits_for_all_handles_and_releases_in_order() -> None:
    first = AmodalEvent(torch.ones(2, 4), timestamp=torch.full((2,), 2.0))
    second = AmodalEvent(torch.full((2, 4), 2.0), timestamp=torch.full((2,), 2.0))
    earlier_first = AmodalEvent(
        torch.full((2, 4), 3.0), timestamp=torch.zeros(2)
    )
    earlier_second = AmodalEvent(
        torch.full((2, 4), 4.0), timestamp=torch.zeros(2)
    )
    buffer = AmodalEventWindowBuffer(("vision", "audio"))
    assert buffer.push({"audio": second}) == []
    assert buffer.pending_timestamps == (2.0,)
    assert buffer.push({"vision": earlier_first}) == []
    ready = buffer.push({"audio": earlier_second})
    assert [window.timestamp for window in ready] == [0.0]
    assert torch.equal(ready[0].collection.payload[:, 0], earlier_first.payload)
    ready = buffer.push({"vision": first})
    assert [window.timestamp for window in ready] == [2.0]
    assert buffer.pending_timestamps == ()


def test_streaming_window_buffer_timeout_marks_missing_payload_absent() -> None:
    buffer = AmodalEventWindowBuffer(("vision", "audio"), max_wait=1.0)
    first = AmodalEvent(torch.ones(2, 4), timestamp=torch.zeros(2))
    later = AmodalEvent(torch.full((2, 4), 2.0), timestamp=torch.ones(2))
    assert buffer.push({"vision": first}) == []
    ready = buffer.push({"vision": later})
    assert len(ready) == 1
    assert not ready[0].complete
    assert torch.equal(ready[0].collection.present[0], torch.tensor([True, False]))
    assert torch.equal(ready[0].collection.payload[:, 1], torch.zeros(2, 4))
    assert torch.equal(ready[0].collection.confidence[:, 1], torch.zeros(2))


def test_streaming_window_buffer_exposes_opaque_policy_status_and_manual_release() -> None:
    buffer = AmodalEventWindowBuffer(("vision", "audio"))
    event = AmodalEvent(torch.ones(2, 4), timestamp=torch.zeros(2))
    assert buffer.push({"vision": event}) == []
    assert buffer.pending_status(current_timestamp=2.0) == (
        AmodalEventWindowStatus(
            timestamp=0.0,
            age=2.0,
            present=(True, False),
            complete=False,
        ),
    )
    released = buffer.release_pending(0.0)
    assert not released.complete
    assert buffer.pending_status() == ()
