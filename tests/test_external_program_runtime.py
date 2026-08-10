import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    ControllerFeedback,
    ExternalCapabilityRegisterMachine,
    ExternalProgramAmodalRuntime,
    ExternalProgramArtifact,
    ExternalSequenceProgramMemory,
)


class _EchoDecoder(nn.Module):
    def forward(self, intention):
        return intention.payload


def _feedback(batch_size: int) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch_size, 3),
        reward=torch.zeros(batch_size),
        propensity=torch.ones(batch_size),
        has_feedback=torch.zeros(batch_size, dtype=torch.bool),
    )


def _artifact() -> ExternalProgramArtifact:
    return ExternalProgramArtifact(
        codes=torch.randn(3, 5),
        interpreter_schema="neural-computer.external-register.v4",
        execution_schema="neural-computer.external-register-read-execute.v1",
    )


def _runtime(*, memory: ExternalSequenceProgramMemory | None = None):
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    runtime.register_decoder("echo", _EchoDecoder())
    machine = ExternalCapabilityRegisterMachine(
        event_width=12,
        action_width=3,
        intention_width=2,
        register_width=8,
        instruction_width=5,
        instructions=(),
    )
    return runtime, machine, ExternalProgramAmodalRuntime(
        runtime,
        machine,
        program=_artifact() if memory is None else None,
        program_memory=memory,
    )


def test_external_program_runtime_routes_file_result_to_decoders_without_core_mutation():
    torch.manual_seed(902)
    runtime, _machine, agent = _runtime()
    before = {
        name: value.detach().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    state = agent.initial_state(2, device="cpu")
    output, next_state = agent.step_events(
        [AmodalEvent(torch.randn(2, 4))],
        state,
        _feedback(2),
    )

    assert output.schema == "neural-computer.external-program-runtime.v1"
    assert output.execution.program_digest is not None
    assert len(output.execution.program_digest) == 64
    assert len(output.execution.trace) == 3
    assert output.intention.payload.shape == (2, 2)
    assert torch.equal(output.decoded["echo"], output.intention.payload)
    assert next_state.program.initialized.equal(torch.ones(2, dtype=torch.bool))
    assert all(torch.equal(value, runtime.controller.state_dict()[name]) for name, value in before.items())


def test_external_program_memory_selects_file_without_exposing_slot_to_controller():
    torch.manual_seed(903)
    memory = ExternalSequenceProgramMemory(5, content_addressing=True, hard_routing=True)
    first = _artifact()
    second = _artifact()
    memory.add_artifact(first)
    memory.add_artifact(second)
    _runtime_module, _machine, agent = _runtime(memory=memory)
    state = agent.initial_state(1, device="cpu")
    output, _ = agent.step_events(
        [AmodalEvent(torch.randn(1, 4))],
        state,
        _feedback(1),
    )

    assert output.selected_program_slot in (0, 1)
    assert output.execution.program_digest in {first.digest(), second.digest()}
    assert agent.configuration()["program_source"] == (
        "opaque_content_routed_external_program_memory_v1"
    )
    assert "selected_program_slot" not in agent.configuration()["machine"]


def test_external_program_runtime_quiet_tick_does_not_grow_program_state():
    torch.manual_seed(904)
    _runtime_components, _machine, agent = _runtime()
    state = agent.initial_state(1, device="cpu")
    output, next_state = agent.step_events(
        AmodalEventCollection.empty(1, 4),
        state,
        _feedback(1),
    )

    assert torch.equal(next_state.program.register, state.program.register)
    assert torch.equal(next_state.program.context, state.program.context)
    assert next_state.program.initialized.equal(state.program.initialized)
    assert output.execution.observed.initialized.equal(state.program.initialized)
