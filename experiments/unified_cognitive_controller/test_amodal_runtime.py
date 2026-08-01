from pathlib import Path

import pytest
import torch

from .amodal_interface import AmodalEvent, IntentEvent
from .amodal_runtime import (
    EXTRACTED_CHECKPOINT_FORMAT,
    ExtractedAmodalRuntime,
    convert_legacy_checkpoint,
    runtime_from_extracted_payload,
    runtime_from_legacy_payload,
)
from .environment import NULL_ACTION
from .model import ControllerOutput, ControllerState, UnifiedCognitiveController


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


def test_interface_rejects_wrong_width_and_schema() -> None:
    with pytest.raises(ValueError, match="event width"):
        AmodalEvent(torch.zeros(2, 3)).validate(width=4)
    with pytest.raises(ValueError, match="unsupported event schema"):
        AmodalEvent(torch.zeros(2, 3), schema="wrong").validate()
    with pytest.raises(ValueError, match="intention width"):
        IntentEvent(torch.zeros(2, 3)).validate(width=4)
