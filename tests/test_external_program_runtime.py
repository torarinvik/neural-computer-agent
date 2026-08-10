import pytest
import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    ControllerFeedback,
    ExternalCapabilityRegisterMachine,
    ExternalControllerStateAdapter,
    ExternalControllerTrajectoryQueryAdapter,
    ExternalOutcomeProgramRouter,
    ExternalProgramAmodalRuntime,
    ExternalProgramArtifact,
    ExternalProgramRuntimeState,
    ExternalSequenceProgramMemory,
)


class _EchoDecoder(nn.Module):
    def forward(self, intention):
        return intention.payload


class _AlternatingProgramMemory(ExternalSequenceProgramMemory):
    def __init__(self):
        super().__init__(5, content_addressing=True, hard_routing=True)
        self.route_index = 0

    def route_weights(self, query):
        selected = self.route_index % self.file_count
        self.route_index += 1
        weights = torch.zeros(
            query.shape[0],
            self.file_count,
            dtype=query.dtype,
            device=query.device,
        )
        weights[:, selected] = 1.0
        return weights


class _PartitionedProgramMemory(ExternalSequenceProgramMemory):
    def __init__(self):
        super().__init__(5, content_addressing=True, hard_routing=True)

    def route_weights(self, query):
        if self.file_count != 2:
            raise AssertionError("partitioned test memory requires two files")
        weights = torch.zeros(
            query.shape[0],
            self.file_count,
            dtype=query.dtype,
            device=query.device,
        )
        weights[:, 0] = 1.0
        weights[1::2, 0] = 0.0
        weights[1::2, 1] = 1.0
        return weights


class _RecordingTrajectoryQueryAdapter(ExternalControllerTrajectoryQueryAdapter):
    def __init__(self):
        super().__init__(4, 5)
        self.seen_present = None

    def forward(self, output, state):
        self.seen_present = state.event_window.present.detach().clone()
        return torch.zeros(
            output.state_representation.shape[0],
            self.query_width,
            device=output.state_representation.device,
            dtype=output.state_representation.dtype,
        )


class _ConstantTrajectoryQueryAdapter(ExternalControllerTrajectoryQueryAdapter):
    def __init__(self):
        super().__init__(4, 5)

    def forward(self, output, state):
        return torch.ones(
            output.state_representation.shape[0],
            self.query_width,
            device=output.state_representation.device,
            dtype=output.state_representation.dtype,
        )


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


def _runtime(
    *,
    memory: ExternalSequenceProgramMemory | None = None,
    program_query_adapter=None,
    program_route_exploration: float = 0.0,
    program_router: ExternalOutcomeProgramRouter | None = None,
):
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
        program_query_adapter=program_query_adapter,
        program_route_exploration=program_route_exploration,
        program_router=program_router,
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

    assert output.schema == "neural-computer.external-program-runtime.v6"
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
    assert output.program_route_query is not None
    assert output.program_route_query.shape == (1, 5)
    assert output.program_route_probabilities is not None
    assert output.program_route_probabilities.shape == (1, 2)
    torch.testing.assert_close(
        output.program_route_probabilities.sum(dim=-1),
        torch.ones(1),
    )
    assert output.program_route_propensities is not None
    torch.testing.assert_close(
        output.program_route_propensities,
        torch.ones(1),
    )
    assert output.execution.program_digest in {first.digest(), second.digest()}
    assert agent.configuration()["program_source"] == (
        "opaque_content_routed_external_program_memory_v1"
    )
    assert "selected_program_slot" not in agent.configuration()["machine"]


def test_external_program_runtime_default_route_query_keeps_current_event_window():
    torch.manual_seed(9030)
    memory = ExternalSequenceProgramMemory(5, content_addressing=True, hard_routing=True)
    memory.add_artifact(_artifact())
    adapter = _RecordingTrajectoryQueryAdapter()
    _runtime_module, _machine, agent = _runtime(
        memory=memory,
        program_query_adapter=adapter,
    )
    agent.step_events(
        [AmodalEvent(torch.randn(1, 4))],
        agent.initial_state(1, device="cpu"),
        _feedback(1),
    )

    assert adapter.seen_present is not None
    assert adapter.seen_present.any().item()
    assert agent.configuration()["program_query_adapter"]["schema"] == (
        "neural-computer.external-controller-trajectory-query-adapter.v1"
    )


def test_external_program_runtime_allows_explicit_final_state_route_adapter():
    memory = ExternalSequenceProgramMemory(5, content_addressing=True, hard_routing=True)
    memory.add_artifact(_artifact())
    adapter = ExternalControllerStateAdapter(12, 5)
    _runtime_module, _machine, agent = _runtime(
        memory=memory,
        program_query_adapter=adapter,
    )
    agent.step_events(
        [AmodalEvent(torch.randn(1, 4))],
        agent.initial_state(1, device="cpu"),
        _feedback(1),
    )

    assert agent.configuration()["program_query_adapter"]["schema"] == (
        "neural-computer.external-controller-state-adapter.v1"
    )


def test_external_program_runtime_exploration_reports_exact_route_propensity():
    torch.manual_seed(9033)
    memory = _PartitionedProgramMemory()
    memory.add_artifact(_artifact())
    memory.add_artifact(_artifact())
    _runtime_module, _machine, agent = _runtime(
        memory=memory,
        program_route_exploration=0.25,
    )
    output, _ = agent.step_events(
        [AmodalEvent(torch.randn(2, 4))],
        agent.initial_state(2, device="cpu"),
        _feedback(2),
    )

    assert output.program_route_probabilities is not None
    assert output.program_route_propensities is not None
    selected = output.selected_program_slots
    assert selected is not None
    expected = output.program_route_probabilities.gather(
        1,
        selected.unsqueeze(-1),
    ).squeeze(-1)
    torch.testing.assert_close(output.program_route_propensities, expected)
    assert bool(torch.all(output.program_route_propensities > 0.0))


def test_external_program_runtime_learns_route_from_scalar_feedback_only():
    torch.manual_seed(9034)
    memory = _PartitionedProgramMemory()
    memory.add_artifact(_artifact())
    memory.add_artifact(_artifact())
    router = ExternalOutcomeProgramRouter(
        feature_width=5,
        program_capacity=2,
        initial_programs=2,
        initial_learning_rate=0.2,
        initial_trace_decay=0.0,
        initial_baseline_rate=0.1,
    )
    _runtime_module, _machine, agent = _runtime(
        memory=memory,
        program_query_adapter=_ConstantTrajectoryQueryAdapter(),
        program_route_exploration=0.2,
        program_router=router,
    )
    state = agent.initial_state(1, device="cpu")
    initial_policy = state.program_router.credit.policy.detach().clone()
    feedback = _feedback(1)
    for _ in range(128):
        output, state = agent.step_events(
            [AmodalEvent(torch.ones(1, 4))],
            state,
            feedback,
        )
        selected = output.selected_program_slots
        assert selected is not None
        feedback = ControllerFeedback(
            action=torch.zeros(1, 3),
            reward=(selected == 1).to(torch.float32),
            propensity=torch.ones(1),
            has_feedback=torch.ones(1, dtype=torch.bool),
        )
    _, state = agent.step_events(
        [AmodalEvent(torch.ones(1, 4))],
        state,
        feedback,
    )

    assert state.program_router is not None
    policy = state.program_router.credit.policy
    assert not torch.equal(policy, initial_policy)
    assert float(policy[..., 1].mean().detach()) > float(policy[..., 0].mean().detach())
    assert state.program_router.credit.feedbacks.item() >= 128


def test_external_program_runtime_router_state_round_trips_and_activates_growth():
    torch.manual_seed(9035)
    memory = _PartitionedProgramMemory()
    memory.add_artifact(_artifact())
    memory.add_artifact(_artifact())
    router = ExternalOutcomeProgramRouter(
        feature_width=5,
        program_capacity=3,
        initial_programs=2,
    )
    _runtime_module, _machine, agent = _runtime(
        memory=memory,
        program_query_adapter=_ConstantTrajectoryQueryAdapter(),
        program_router=router,
    )
    state = agent.initial_state(1, device="cpu")
    _, state = agent.step_events(
        [AmodalEvent(torch.ones(1, 4))],
        state,
        _feedback(1),
    )
    restored = ExternalProgramRuntimeState.from_payload(
        state.payload(),
        program_router=router,
    )
    assert restored.program_router is not None
    torch.testing.assert_close(
        restored.program_router.credit.policy,
        state.program_router.credit.policy,
    )

    memory.add_artifact(_artifact())
    activated = agent.activate_program(restored)
    assert activated.program_router is not None
    assert activated.program_router.active_programs == 3
    assert set(activated.program_states) == {0, 1}


def test_external_program_runtime_supports_mixed_file_schedule_in_one_batch():
    torch.manual_seed(9031)
    memory = _PartitionedProgramMemory()
    memory.add_artifact(_artifact())
    memory.add_artifact(_artifact())
    _runtime_module, _machine, agent = _runtime(memory=memory)
    output, next_state = agent.step_events(
        [AmodalEvent(torch.randn(2, 4))],
        agent.initial_state(2, device="cpu"),
        _feedback(2),
    )

    assert output.selected_program_slot is None
    assert output.selected_program_logical_id is None
    assert output.selected_program_slots.tolist() == [0, 1]
    assert output.selected_program_logical_ids.tolist() == [0, 1]
    assert len(output.execution_snapshots) == 2
    assert output.execution.program_digest is None
    assert output.program_route_query is not None
    assert output.program_route_query.shape == (2, 5)
    assert output.program_route_probabilities is not None
    assert output.program_route_probabilities.shape == (2, 2)
    torch.testing.assert_close(
        output.program_route_probabilities.sum(dim=-1),
        torch.ones(2),
    )
    assert next_state.program_states.keys() == {0, 1}
    assert next_state.program_states[0].initialized.tolist() == [True, False]
    assert next_state.program_states[1].initialized.tolist() == [False, True]


def test_external_program_runtime_state_round_trip_resumes_exactly():
    torch.manual_seed(9032)
    memory = _PartitionedProgramMemory()
    memory.add_artifact(_artifact())
    memory.add_artifact(_artifact())
    _runtime_module, _machine, agent = _runtime(memory=memory)
    state = agent.initial_state(2, device="cpu")
    _, paused_state = agent.step_events(
        [AmodalEvent(torch.randn(2, 4))],
        state,
        _feedback(2),
    )
    checkpoint = paused_state.payload()
    restored_state = ExternalProgramRuntimeState.from_payload(checkpoint)

    next_events = [AmodalEvent(torch.randn(2, 4))]
    original_output, original_next = agent.step_events(
        next_events,
        paused_state,
        _feedback(2),
    )
    resumed_output, resumed_next = agent.step_events(
        next_events,
        restored_state,
        _feedback(2),
    )

    torch.testing.assert_close(
        original_output.execution.executed,
        resumed_output.execution.executed,
    )
    torch.testing.assert_close(
        original_output.intention.payload,
        resumed_output.intention.payload,
    )
    torch.testing.assert_close(
        original_output.decoded["echo"],
        resumed_output.decoded["echo"],
    )
    for logical_id in original_next.program_states:
        original_file = original_next.program_states[logical_id]
        resumed_file = resumed_next.program_states[logical_id]
        torch.testing.assert_close(original_file.register, resumed_file.register)
        torch.testing.assert_close(original_file.context, resumed_file.context)
        assert torch.equal(original_file.initialized, resumed_file.initialized)


def test_external_program_runtime_state_rejects_unknown_checkpoint_schema():
    _runtime_module, _machine, agent = _runtime()
    payload = agent.initial_state(1, device="cpu").payload()
    payload["schema"] = "neural-computer.external-program-runtime-state.unknown"
    with pytest.raises(ValueError, match="unsupported external program runtime state"):
        ExternalProgramRuntimeState.from_payload(payload)


def test_external_program_runtime_state_rejects_tensor_corruption():
    _runtime_module, _machine, agent = _runtime()
    payload = agent.initial_state(1, device="cpu").payload()
    controller_payload = dict(payload["controller"])
    hidden = controller_payload["hidden"].clone()
    hidden[0, 0] += 1.0
    controller_payload["hidden"] = hidden
    payload["controller"] = controller_payload
    with pytest.raises(ValueError, match="checksum mismatch"):
        ExternalProgramRuntimeState.from_payload(payload)


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


def test_external_program_runtime_isolates_recurrent_state_by_logical_file():
    torch.manual_seed(905)
    memory = _AlternatingProgramMemory()
    memory.add_artifact(_artifact())
    memory.add_artifact(_artifact())
    _runtime_components, _machine, agent = _runtime(memory=memory)
    state = agent.initial_state(1, device="cpu")
    initial = state.program_states

    _, first_state = agent.step_events(
        [AmodalEvent(torch.ones(1, 4))],
        state,
        _feedback(1),
    )
    assert first_state.program_states.keys() == {0, 1}
    assert first_state.program_states[0].initialized.equal(
        torch.ones(1, dtype=torch.bool)
    )
    assert first_state.program_states[1].initialized.equal(
        initial[1].initialized
    )

    second_output, second_state = agent.step_events(
        [AmodalEvent(torch.full((1, 4), 2.0))],
        first_state,
        _feedback(1),
    )
    assert second_output.selected_program_logical_id == 1
    assert torch.equal(
        second_state.program_states[0].context,
        first_state.program_states[0].context,
    )
    assert second_state.program_states[1].initialized.equal(
        torch.ones(1, dtype=torch.bool)
    )

    third_output, third_state = agent.step_events(
        [AmodalEvent(torch.full((1, 4), 3.0))],
        second_state,
        _feedback(1),
    )
    assert third_output.selected_program_logical_id == 0
    assert torch.equal(
        third_state.program_states[1].context,
        second_state.program_states[1].context,
    )
    assert not torch.equal(
        third_state.program_states[0].context,
        second_state.program_states[0].context,
    )


def test_external_program_runtime_prunes_retired_file_state_and_reloads_ids():
    torch.manual_seed(906)
    memory = ExternalSequenceProgramMemory(5, content_addressing=True, hard_routing=True)
    memory.add_artifact(_artifact())
    memory.add_artifact(_artifact())
    _runtime_components, _machine, agent = _runtime(memory=memory)
    state = agent.initial_state(1, device="cpu")
    _, state = agent.step_events(
        [AmodalEvent(torch.ones(1, 4))],
        state,
        _feedback(1),
    )
    assert state.program_states.keys() == {0, 1}

    receipt = memory.evict_verified(0, lambda _candidate: True)
    assert receipt.accepted
    assert memory.logical_slot_ids == (1,)

    output, next_state = agent.step_events(
        [AmodalEvent(torch.full((1, 4), 2.0))],
        state,
        _feedback(1),
    )
    assert output.selected_program_logical_id == 1
    assert next_state.program_states.keys() == {1}
    restored = ExternalSequenceProgramMemory.from_payload(memory.payload())
    assert restored.logical_slot_ids == (1,)
    restored_agent = _runtime(memory=restored)[2]
    assert restored_agent.initial_state(1, device="cpu").program_states.keys() == {1}
