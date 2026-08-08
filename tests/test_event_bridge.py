import pytest
import torch

from neural_computer import AmodalEventBridge


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
