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
from neural_computer import (
    AmodalEventBridge,
    CapabilityConditionedEventBridge,
    EpisodicIntentAdapter,
    ExternalSequenceOperatorMemory,
    OpaqueProtocolDecoder,
)


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


def test_external_rollout_binds_operator_route_once_and_keeps_credit_live() -> None:
    parent = _runtime(seed=43, growth=False)
    machine = _new_machine(
        2,
        operator_mode="factorized_protected_bounded_meta",
        operator_rank=2,
    )
    decoder = OpaqueProtocolDecoder(32, 2, hidden=8)
    memory = ExternalSequenceOperatorMemory(32, 16, operator_rank=2)
    memory.add_slot()
    memory.add_slot()
    query = memory.encode_program(torch.randn(1, 2, 16)).squeeze(0)
    calls = 0
    original_route_weights = memory.route_weights

    def counted_route_weights(route_query: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_route_weights(route_query)

    memory.route_weights = counted_route_weights  # type: ignore[method-assign]
    batch = _batch(
        "generated_composition",
        count=4,
        span=2,
        seed=47,
        generated_composition_ids=(0,),
        generated_compositions=(("reverse", "complement"),),
    )

    loss, rewards = _rollout(
        parent,
        machine,
        decoder,
        batch,
        tuple(machine.instructions),
        train_decoder=True,
        credit_mode="attempted_bce",
        sequence_operator_memory=memory,
        sequence_operator_route_query=query,
        bind_operator_route=True,
    )
    loss.backward()

    assert calls == 1
    assert rewards.shape == (4, 2)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in memory.parameters())
    assert all(parameter.grad is None for parameter in parent.parameters())


def test_external_rollout_keeps_route_probe_unbound_when_requested() -> None:
    parent = _runtime(seed=53, growth=False)
    machine = _new_machine(
        2,
        operator_mode="factorized_protected_bounded_meta",
        operator_rank=2,
    )
    decoder = OpaqueProtocolDecoder(32, 2, hidden=8)
    memory = ExternalSequenceOperatorMemory(32, 16, operator_rank=2)
    memory.add_slot()
    batch = _batch("generated_composition", count=2, span=1, seed=59)

    loss, rewards = _rollout(
        parent,
        machine,
        decoder,
        batch,
        tuple(machine.instructions),
        train_decoder=False,
        sequence_operator_memory=memory,
        sequence_operator_route_query=torch.randn(16),
        bind_operator_route=True,
        route_probe=True,
    )

    assert torch.isfinite(loss)
    assert rewards.shape == (2, 1)


def test_external_rollout_uses_bound_file_read_adapter_without_parent_gradients() -> None:
    parent = _runtime(seed=61, growth=False)
    machine = _new_machine(
        2,
        operator_mode="factorized_protected_bounded_meta",
        operator_rank=2,
    )
    decoder = OpaqueProtocolDecoder(32, 2, hidden=8)
    memory = ExternalSequenceOperatorMemory(32, 16, operator_rank=2)
    memory.add_slot()
    memory.add_slot()
    adapter = EpisodicIntentAdapter(16, 16, hidden=8)
    batch = _batch(
        "generated_composition",
        count=4,
        span=2,
        seed=67,
        generated_composition_ids=(0,),
        generated_compositions=(("reverse", "complement"),),
    )

    loss, _ = _rollout(
        parent,
        machine,
        decoder,
        batch,
        tuple(machine.instructions),
        train_decoder=True,
        credit_mode="attempted_bce",
        sequence_operator_memory=memory,
        sequence_operator_route_query=torch.randn(16),
        bind_operator_route=True,
        operator_read_adapter=adapter,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in adapter.parameters())
    assert all(parameter.grad is None for parameter in parent.parameters())


def test_bridge_corruption_control_keeps_scalar_credit_on_external_path() -> None:
    parent = _runtime(seed=29, growth=False)
    machine = _new_machine(1)
    decoder = OpaqueProtocolDecoder(32, 2, hidden=8)
    bridge = AmodalEventBridge(32, parent.controller.width, 32, hidden=8)
    batch = _batch("forward", count=4, span=2, seed=31)

    loss, _ = _rollout(
        parent,
        machine,
        decoder,
        batch,
        tuple(machine.instructions),
        train_decoder=True,
        credit_mode="reinforce_baseline",
        event_bridge=bridge,
        bridge_event_mode="norm_matched_noise",
        bridge_state_mode="zero",
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in bridge.parameters())
    assert all(parameter.grad is None for parameter in parent.parameters())

    bridge.zero_grad(set_to_none=True)
    cyclic_loss, _ = _rollout(
        parent,
        machine,
        decoder,
        batch,
        tuple(machine.instructions),
        train_decoder=True,
        credit_mode="reinforce_baseline",
        event_bridge=bridge,
        bridge_event_mode="cyclic_permutation",
        bridge_state_mode="zero",
    )
    cyclic_loss.backward()
    assert torch.isfinite(cyclic_loss)
    assert any(parameter.grad is not None for parameter in bridge.parameters())

    bridge.zero_grad(set_to_none=True)
    composed_loss, _ = _rollout(
        parent,
        machine,
        decoder,
        batch,
        tuple(machine.instructions),
        train_decoder=True,
        credit_mode="reinforce_baseline",
        event_bridge=bridge,
        bridge_event_mode="composed_orthogonal",
        bridge_state_mode="zero",
    )
    composed_loss.backward()
    assert torch.isfinite(composed_loss)
    assert any(parameter.grad is not None for parameter in bridge.parameters())


def test_bridge_input_override_requires_a_bridge() -> None:
    parent = _runtime(seed=37, growth=False)
    machine = _new_machine(1)
    decoder = OpaqueProtocolDecoder(32, 2, hidden=8)
    batch = _batch("forward", count=2, span=1, seed=41)

    with pytest.raises(ValueError, match="require an event bridge"):
        _rollout(
            parent,
            machine,
            decoder,
            batch,
            tuple(machine.instructions),
            train_decoder=False,
            event_bridge=None,
            bridge_event_mode="zero",
        )
