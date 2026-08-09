import pytest
import torch

from experiments.external_register_composition_amodal.train import (
    _batch,
    _new_machine,
    _rollout,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import AmodalEventBridge, CapabilityConditionedEventBridge
from neural_computer import OpaqueProtocolDecoder


def test_event_bridge_starts_behavior_preserving_and_is_versioned() -> None:
    bridge = AmodalEventBridge(4, 6, 4, hidden=8)
    frontend = torch.randn(3, 4)
    state = torch.randn(3, 6)

    torch.testing.assert_close(bridge(frontend, state), frontend)
    assert bridge.configuration()["schema"] == "neural-computer.event-bridge.v1"


def test_event_bridge_validates_inputs() -> None:
    bridge = AmodalEventBridge(4, 6, 4, hidden=8)
    with pytest.raises(ValueError, match="frontend event"):
        bridge(torch.zeros(3, 5), torch.zeros(3, 6))
    with pytest.raises(ValueError, match="controller state"):
        bridge(torch.zeros(3, 4), torch.zeros(3, 5))


def test_conditioned_event_bridge_is_identity_and_context_versioned() -> None:
    bridge = CapabilityConditionedEventBridge(4, 6, 4, 3, hidden=8)
    frontend = torch.randn(3, 4)
    state = torch.randn(3, 6)
    bridge.set_context(torch.tensor([1.0, 2.0, 3.0]))

    torch.testing.assert_close(bridge(frontend, state), frontend)
    configuration = bridge.configuration()
    assert configuration["schema"] == "neural-computer.conditioned-event-bridge.v1"
    assert configuration["context_width"] == 3

    with pytest.raises(ValueError, match="capability context"):
        bridge.set_context(torch.zeros(2))


def test_external_rollout_trains_bridge_while_parent_stays_frozen() -> None:
    parent = _runtime(seed=19, growth=False)
    machine = _new_machine(1)
    decoder = OpaqueProtocolDecoder(32, 2, hidden=8)
    bridge = AmodalEventBridge(32, parent.controller.width, 32, hidden=8)
    batch = _batch("forward", count=4, span=2, seed=23)

    loss, _ = _rollout(
        parent,
        machine,
        decoder,
        batch,
        tuple(machine.instructions),
        train_decoder=True,
        credit_mode="attempted_bce",
        event_bridge=bridge,
    )
    loss.backward()

    assert any(parameter.grad is not None for parameter in bridge.parameters())
    assert all(parameter.grad is None for parameter in parent.parameters())
