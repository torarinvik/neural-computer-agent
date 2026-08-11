from __future__ import annotations

import pytest
import torch

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalOutputBus,
    ControllerFeedback,
    IntentEvent,
    KeypressDecoder,
    KeypressEncoder,
    OpaqueProtocolDecoder,
)


def test_keypress_encoder_produces_opaque_feedback() -> None:
    encoder = KeypressEncoder(key_count=3, feedback_width=5)
    encoded = encoder(torch.tensor([0, 2], dtype=torch.long))
    assert encoded.shape == (2, 5)
    assert encoder.configuration()["schema"] == (
        "neural-computer.keypress-encoder.v1"
    )
    with pytest.raises(ValueError, match="outside"):
        encoder(torch.tensor([3], dtype=torch.long))


def test_keypress_decoder_returns_logits_and_exact_propensity() -> None:
    decoder = KeypressDecoder(intention_width=4, key_count=3, hidden=8)
    intention = torch.randn(5, 4)
    logits = decoder(intention)
    decision = decoder.decide(intention, sample=False)
    from_logits = decoder.decide_from_logits(logits, sample=False)
    assert logits.shape == (5, 3)
    assert decision.key_index.shape == (5,)
    assert torch.equal(decision.key_index, from_logits.key_index)
    assert torch.allclose(
        decision.propensity,
        torch.softmax(logits, dim=-1).amax(dim=-1),
    )
    assert decoder.configuration()["schema"] == (
        "neural-computer.keypress-decoder.v1"
    )


def test_keypress_decoder_is_a_replaceable_output_bus_backend() -> None:
    controller = AmodalCognitiveController(
        width=8, workspace_slots=1, intention_width=4, feedback_width=3
    )
    runtime = AmodalControllerRuntime(
        controller,
        output_bus=AmodalOutputBus(
            {"keypress": KeypressDecoder(4, 2), "other": OpaqueProtocolDecoder(4, 3)}
        ),
    )
    feedback = ControllerFeedback(
        action=torch.zeros(2, 3),
        reward=torch.zeros(2),
        propensity=torch.ones(2),
        has_feedback=torch.zeros(2),
    )
    output, _ = runtime.step_events(
        [AmodalEvent(torch.randn(2, 8))],
        runtime.initial_state(2, device="cpu"),
        feedback,
    )
    assert output.decoded["keypress"].shape == (2, 2)
    assert output.decoded["other"].shape == (2, 3)


def test_runtime_decodes_caller_owned_opaque_intention_without_controller_tick() -> None:
    controller = AmodalCognitiveController(
        width=8, workspace_slots=1, intention_width=4, feedback_width=3
    )
    runtime = AmodalControllerRuntime(
        controller,
        output_bus=AmodalOutputBus({"keypress": KeypressDecoder(4, 2)}),
    )
    before = {
        name: value.detach().clone() for name, value in controller.state_dict().items()
    }
    intention = torch.randn(4)

    decoded = runtime.decode_intention(intention)
    decoded_event = runtime.decode_intention(
        IntentEvent(payload=intention.unsqueeze(0))
    )

    assert decoded["keypress"].shape == (1, 2)
    assert torch.equal(decoded["keypress"], decoded_event["keypress"])
    assert all(torch.equal(value, controller.state_dict()[name]) for name, value in before.items())
    with pytest.raises(ValueError, match="shape"):
        runtime.decode_intention(torch.randn(2, 2, 4))
