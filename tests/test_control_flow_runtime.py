from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControlFlowInstruction,
    ControlFlowIntentionAdapter,
    ControlFlowProgram,
    ControlFlowProgramAmodalRuntime,
    ControlFlowProgramMemory,
    ControllerFeedback,
    ExternalOutcomeProgramRouter,
    IntentEvent,
)


class _OpaqueCounterCodec(ControlFlowIntentionAdapter):
    """Test-only external codec; no production semantic mapping is implied."""

    def encode(
        self,
        intention: IntentEvent,
        previous_counters: torch.Tensor,
    ) -> torch.Tensor:
        encoded = previous_counters.clone()
        # Use one opaque sign bit only to make row permutation observable.
        encoded[:, 0] = (intention.payload[:, 0] > 0.0).to(torch.int64)
        return encoded

    def decode(
        self,
        counters: torch.Tensor,
        template: IntentEvent,
    ) -> IntentEvent:
        payload = counters.to(dtype=template.payload.dtype)
        return IntentEvent(
            payload=payload,
            timestamp=template.timestamp,
            confidence=template.confidence,
            target_key=template.target_key,
        )


class _EchoDecoder(nn.Module):
    def forward(self, intention: IntentEvent) -> torch.Tensor:
        return intention.payload


class _SignRouter(ExternalOutcomeProgramRouter):
    """Deterministic test router; production routing remains outcome-trained."""

    def behavior_probabilities(
        self,
        state,
        features: torch.Tensor,
        *,
        exploration: float = 0.0,
    ) -> torch.Tensor:
        self._validate_state(state)
        selected = (features[:, 0] <= 0.0).to(torch.long)
        probabilities = torch.zeros(
            features.shape[0],
            self.program_capacity,
            device=features.device,
            dtype=features.dtype,
        )
        probabilities.scatter_(1, selected.unsqueeze(-1), 1.0)
        return probabilities


def _program() -> ControlFlowProgram:
    return ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )


def _other_program() -> ControlFlowProgram:
    return ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("halt"),
        ),
    )


def _agent(
    *,
    memory: ControlFlowProgramMemory | None = None,
    router: ExternalOutcomeProgramRouter | None = None,
) -> ControlFlowProgramAmodalRuntime:
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    runtime.register_decoder("echo", _EchoDecoder())
    source = (
        {"program_memory": memory}
        if memory is not None
        else {"program": _program()}
    )
    return ControlFlowProgramAmodalRuntime(
        runtime,
        _OpaqueCounterCodec(intention_width=2, counter_count=2),
        **source,
        program_router=router,
        max_steps=8,
    )


def _feedback(batch: int) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch, 3),
        reward=torch.zeros(batch),
        propensity=torch.ones(batch),
        has_feedback=torch.zeros(batch, dtype=torch.bool),
    )


def test_control_flow_runtime_executes_external_file_through_opaque_bus() -> None:
    torch.manual_seed(1101)
    agent = _agent()
    before = {
        name: value.detach().clone()
        for name, value in agent.runtime.controller.state_dict().items()
    }
    state = agent.initial_state(2, device="cpu")
    events = [AmodalEvent(torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]))]

    output, state = agent.step_events(events, state, _feedback(2))
    assert output.schema == "neural-computer.control-flow-runtime.v1"
    assert output.program_slot is None
    assert output.program_digest == _program().digest()
    assert tuple(execution.status for execution in output.executions) == (
        "halted",
        "halted",
    )
    expected_counters = 1 + (
        output.controller.intention.payload[:, 0] > 0.0
    ).to(torch.int64)
    assert torch.equal(
        state.counters,
        torch.stack(
            [
                torch.tensor(execution.counters, dtype=torch.int64)
                for execution in output.executions
            ]
        ),
    )
    assert torch.equal(state.counters[:, 0], expected_counters)
    assert torch.equal(output.decoded["echo"], output.intention.payload)
    assert all(
        torch.equal(value, agent.runtime.controller.state_dict()[name])
        for name, value in before.items()
    )


def test_control_flow_runtime_preserves_permutation_and_state_reload() -> None:
    torch.manual_seed(1102)
    agent = _agent()
    payload = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]
    )
    first, state = agent.step_events(
        [AmodalEvent(payload)],
        agent.initial_state(2, device="cpu"),
        _feedback(2),
    )
    reversed_output, _ = agent.step_events(
        [AmodalEvent(payload.flip(0))],
        agent.initial_state(2, device="cpu"),
        _feedback(2),
    )
    assert torch.allclose(
        first.intention.payload,
        reversed_output.intention.payload.flip(0),
    )
    assert first.executions[0].counters == reversed_output.executions[1].counters
    assert first.executions[1].counters == reversed_output.executions[0].counters

    restored = agent.state_from_payload(state.payload())
    assert restored.digest() == state.digest()
    assert torch.equal(restored.counters, state.counters)

    corrupted = state.payload()
    corrupted["counters"] = corrupted["counters"].clone()
    corrupted["counters"][0, 0] += 1
    with pytest.raises(ValueError, match="checksum"):
        agent.state_from_payload(corrupted)


def test_control_flow_runtime_reads_a_checksummed_memory_backed_file() -> None:
    memory = ControlFlowProgramMemory(2)
    slot = memory.add_program(_program(), protect=True)
    agent = _agent(memory=memory)

    output, _ = agent.step_events(
        [AmodalEvent(torch.ones(1, 4))],
        agent.initial_state(1, device="cpu"),
        _feedback(1),
    )

    assert output.program_slot == slot
    assert output.program_digest == memory.program(slot).digest()
    assert memory.is_file_protected(slot)


def test_control_flow_runtime_routes_multiple_files_with_isolated_counter_state() -> None:
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    memory.add_program(_other_program(), protect=True)
    router = _SignRouter(
        feature_width=2,
        program_capacity=2,
        initial_programs=2,
    )
    agent = _agent(memory=memory, router=router)
    state = agent.initial_state(2, device="cpu")
    output, state = agent.step_events(
        [
            AmodalEvent(
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]
                )
            )
        ],
        state,
        _feedback(2),
    )

    expected_slots = (
        output.controller.intention.payload[:, 0] <= 0.0
    ).to(torch.long)
    assert torch.equal(output.selected_program_slots, expected_slots)
    assert output.program_route_probabilities is not None
    assert torch.equal(
        output.program_route_probabilities.argmax(dim=-1), expected_slots
    )
    assert set(state.program_counters) == {0, 1}
    assert output.program_digests == tuple(
        memory.program(int(slot)).digest() for slot in expected_slots.tolist()
    )

    restored = agent.state_from_payload(
        state.payload(program_router=router)
    )
    assert restored.digest(program_router=router) == state.digest(
        program_router=router
    )
    corrupted = state.payload(program_router=router)
    corrupted_router = corrupted["program_router"]
    assert isinstance(corrupted_router, dict)
    credit = corrupted_router["credit"]
    assert isinstance(credit, dict)
    credit["policy"] = credit["policy"].clone()
    credit["policy"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="checksum"):
        agent.state_from_payload(corrupted)


def test_control_flow_runtime_rejects_mismatched_adapter_or_program_width() -> None:
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
    )
    runtime = AmodalControllerRuntime(controller)
    with pytest.raises(ValueError, match="adapter width"):
        ControlFlowProgramAmodalRuntime(
            runtime,
            _OpaqueCounterCodec(intention_width=3, counter_count=2),
            program=_program(),
        )

    with pytest.raises(ValueError, match="program width"):
        ControlFlowProgramAmodalRuntime(
            runtime,
            _OpaqueCounterCodec(intention_width=2, counter_count=3),
            program=_program(),
        )


def test_control_flow_runtime_state_replacement_does_not_change_controller() -> None:
    agent = _agent()
    state = agent.initial_state(1, device="cpu")
    replaced = replace(state, counters=torch.tensor([[7, 0]], dtype=torch.int64))
    output, next_state = agent.step_events(
        [AmodalEvent(torch.zeros(1, 4))],
        replaced,
        _feedback(1),
    )
    assert output.executions[0].counters in ((1, 0), (2, 0))
    assert next_state.controller.hidden.shape == state.controller.hidden.shape
