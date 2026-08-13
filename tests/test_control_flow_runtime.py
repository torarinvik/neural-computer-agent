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
    ControlFlowProgramGrowthReceipt,
    ControlFlowProgramMemory,
    ControllerFeedback,
    ExternalControllerTrajectoryQueryAdapter,
    ExternalOutcomeProgramRouter,
    IntentEvent,
    PersistentOpaqueContextRouteEvidence,
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


class _ConstantRouteQuery(nn.Module):
    query_width = 2
    query_space_id = "opaque-route-query-v1"

    def forward(self, controller_output, controller_state) -> torch.Tensor:
        query = torch.zeros(
            controller_output.intention.payload.shape[0],
            self.query_width,
            device=controller_output.intention.payload.device,
        )
        query[:, 0] = 1.0
        return query


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
    query_adapter: nn.Module | None = None,
    route_evidence: PersistentOpaqueContextRouteEvidence | None = None,
    program_route_exploration: float = 0.0,
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
        program_route_evidence=route_evidence,
        program_route_query_adapter=query_adapter,
        program_route_exploration=program_route_exploration,
        max_steps=8,
    )


def _feedback(batch: int) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch, 3),
        reward=torch.zeros(batch),
        propensity=torch.ones(batch),
        has_feedback=torch.zeros(batch, dtype=torch.bool),
    )


def _route_feedback(reward: float) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.tensor([reward]),
        propensity=torch.ones(1),
        has_feedback=torch.ones(1, dtype=torch.bool),
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


def test_control_flow_runtime_state_v1_payload_migrates_to_v2() -> None:
    agent = _agent()
    state = agent.initial_state(1, device="cpu")
    payload = state.payload()
    payload["schema"] = "neural-computer.control-flow-runtime-state.v1"
    payload.pop("pending_route_query")
    payload.pop("pending_route_slots")
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}

    # Match the runtime's nested tensor checksum helper without importing an
    # implementation-private symbol into the public test contract.
    import hashlib

    def digest(value: object) -> str:
        result = hashlib.sha256()

        def visit(item: object) -> None:
            if isinstance(item, torch.Tensor):
                tensor = item.detach().cpu().contiguous()
                result.update(str(tensor.dtype).encode("utf-8"))
                result.update(repr(tuple(tensor.shape)).encode("utf-8"))
                result.update(tensor.numpy().tobytes())
            elif isinstance(item, dict):
                for key in sorted(item, key=str):
                    result.update(str(key).encode("utf-8"))
                    visit(item[key])
            elif isinstance(item, (tuple, list)):
                for child in item:
                    visit(child)
            else:
                result.update(repr(item).encode("utf-8"))

        visit(value)
        return result.hexdigest()

    payload["sha256"] = digest(unsigned)
    restored = agent.state_from_payload(payload)

    assert restored.schema == "neural-computer.control-flow-runtime-state.v2"
    assert restored.pending_route_query is None
    assert restored.pending_route_slots is None


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


def test_verified_program_growth_atomically_appends_memory_route_and_counters() -> None:
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    evidence = PersistentOpaqueContextRouteEvidence(width=20)
    evidence.append_slot()
    query_adapter = ExternalControllerTrajectoryQueryAdapter(
        controller_width=4,
        query_width=20,
    )
    agent = _agent(
        memory=memory,
        query_adapter=query_adapter,
        route_evidence=evidence,
    )
    state = agent.initial_state(1, device="cpu")
    controller_snapshot = {
        name: value.detach().clone()
        for name, value in agent.runtime.controller.state_dict().items()
    }

    receipt, grown = agent.admit_program_verified(
        state,
        _other_program(),
        (1.0, 1.0),
        protect=True,
    )

    assert isinstance(receipt, ControlFlowProgramGrowthReceipt)
    assert receipt.accepted
    assert receipt.slot == 1
    assert memory.file_count == 1
    assert agent.program_memory is not None
    assert agent.program_memory.file_count == 2
    assert agent.program_route_evidence is not None
    assert evidence.slot_count == 1
    assert agent.program_route_evidence.slot_count == 2
    assert set(grown.program_counters) == {0, 1}
    assert all(
        torch.equal(value, agent.runtime.controller.state_dict()[name])
        for name, value in controller_snapshot.items()
    )

    output, _ = agent.step_events(
        [AmodalEvent(torch.ones(1, 4))],
        grown,
        _feedback(1),
        program_route_override=torch.tensor([1], dtype=torch.int64),
    )
    assert output.program_slot == 1
    assert output.program_digest == agent.program_memory.program(1).digest()


def test_verified_program_growth_expands_router_capacity_without_changing_old_columns() -> None:
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    router = ExternalOutcomeProgramRouter(
        feature_width=2,
        program_capacity=1,
        initial_programs=1,
    )
    agent = _agent(memory=memory, router=router)
    state = agent.initial_state(1, device="cpu")
    old_policy = state.program_router.credit.policy.clone()

    receipt, grown = agent.admit_program_verified(
        state,
        _other_program(),
        (1.0, 1.0),
        protect=True,
    )

    assert receipt.accepted
    assert receipt.source_router_capacity == 1
    assert receipt.destination_router_capacity == 2
    assert agent.program_router is not None
    assert agent.program_router.program_capacity == 2
    assert grown.program_router is not None
    assert grown.program_router.active_programs == 2
    assert torch.equal(grown.program_router.credit.policy[..., :1], old_policy)
    assert set(grown.program_counters) == {0, 1}


def test_rejected_verified_program_growth_is_transactional() -> None:
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    evidence = PersistentOpaqueContextRouteEvidence(width=20)
    evidence.append_slot()
    query_adapter = ExternalControllerTrajectoryQueryAdapter(
        controller_width=4,
        query_width=20,
    )
    agent = _agent(
        memory=memory,
        query_adapter=query_adapter,
        route_evidence=evidence,
    )
    state = agent.initial_state(1, device="cpu")
    before_digest = state.digest()

    receipt, unchanged = agent.admit_program_verified(
        state,
        _other_program(),
        (0.0, 0.0),
        threshold=1.0,
        min_observations=2,
        min_stable_observations=2,
    )

    assert not receipt.accepted
    assert unchanged is state
    assert receipt.state_digest_before == before_digest
    assert receipt.state_digest_after == before_digest
    assert memory.file_count == 1
    assert agent.program_memory is not None
    assert agent.program_memory.file_count == 1
    assert evidence.slot_count == 1
    assert agent.program_route_evidence is not None
    assert agent.program_route_evidence.slot_count == 1


def test_control_flow_runtime_accepts_external_opaque_route_override() -> None:
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    memory.add_program(_other_program(), protect=True)
    query_adapter = ExternalControllerTrajectoryQueryAdapter(
        controller_width=4,
        query_width=20,
    )
    agent = _agent(memory=memory, query_adapter=query_adapter)

    output, state = agent.step_events(
        [AmodalEvent(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))],
        agent.initial_state(1, device="cpu"),
        _feedback(1),
        program_route_override=torch.tensor([1], dtype=torch.int64),
    )

    assert torch.equal(output.selected_program_slots, torch.tensor([1]))
    assert output.program_digest == memory.program(1).digest()
    assert output.program_route_query is not None
    assert output.program_route_query.shape == (1, 20)
    assert set(state.program_counters) == {0, 1}

    with pytest.raises(ValueError, match="out of range"):
        agent.step_events(
            [AmodalEvent(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))],
            agent.initial_state(1, device="cpu"),
            _feedback(1),
            program_route_override=torch.tensor([2], dtype=torch.int64),
        )


def test_control_flow_runtime_allows_external_exploration_with_route_evidence() -> None:
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    memory.add_program(_other_program(), protect=True)
    evidence = PersistentOpaqueContextRouteEvidence(width=20)
    evidence.append_slot()
    evidence.append_slot()
    query_adapter = ExternalControllerTrajectoryQueryAdapter(
        controller_width=4,
        query_width=20,
    )
    agent = _agent(
        memory=memory,
        query_adapter=query_adapter,
        route_evidence=evidence,
    )

    output, _ = agent.step_events(
        [AmodalEvent(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))],
        agent.initial_state(1, device="cpu"),
        _feedback(1),
        program_route_override=torch.tensor([1], dtype=torch.int64),
    )

    assert torch.equal(output.selected_program_slots, torch.tensor([1]))
    assert output.program_route_query is not None


def test_control_flow_runtime_consumes_context_route_evidence_in_cycle() -> None:
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    memory.add_program(_other_program(), protect=True)
    evidence = PersistentOpaqueContextRouteEvidence(
        width=20,
        min_mastery_observations=2,
    )
    evidence.append_slot()
    evidence.append_slot()
    query_adapter = ExternalControllerTrajectoryQueryAdapter(
        controller_width=4,
        query_width=20,
    )
    agent = _agent(
        memory=memory,
        query_adapter=query_adapter,
        route_evidence=evidence,
    )
    first, _ = agent.step_events(
        [AmodalEvent(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))],
        agent.initial_state(1, device="cpu"),
        _feedback(1),
    )
    assert first.program_slot == 0
    assert first.program_route_query is not None
    for _ in range(2):
        evidence.observe(first.program_route_query[0], 1, 1.0)

    routed, _ = agent.step_events(
        [AmodalEvent(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))],
        agent.initial_state(1, device="cpu"),
        _feedback(1),
    )

    assert routed.program_slot == 1
    assert routed.program_route_query is not None


def test_route_evidence_auto_credit_explores_and_reaches_newly_admitted_file() -> None:
    torch.manual_seed(1401)
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    evidence = PersistentOpaqueContextRouteEvidence(
        width=2,
        min_mastery_observations=2,
    )
    evidence.append_slot()
    agent = _agent(
        memory=memory,
        query_adapter=_ConstantRouteQuery(),
        route_evidence=evidence,
        program_route_exploration=0.4,
    )
    state = agent.initial_state(1, device="cpu")
    receipt, state = agent.admit_program_verified(
        state,
        _other_program(),
        (1.0, 1.0),
        protect=True,
    )
    assert receipt.accepted

    output, state = agent.step_events(
        [AmodalEvent(torch.ones(1, 4))],
        state,
        _feedback(1),
    )
    assert output.program_route_probabilities is not None
    assert output.program_route_propensities is not None
    assert state.pending_route_query is not None
    assert state.pending_route_slots is not None

    for _ in range(160):
        selected = int(output.selected_program_slots[0])
        outcome = 1.0 if selected == 1 else 0.0
        output, state = agent.step_events(
            [AmodalEvent(torch.ones(1, 4))],
            state,
            _feedback(1),
            route_feedback=_route_feedback(outcome),
        )
    agent.program_route_exploration = 0.0
    output, state = agent.step_events(
        [AmodalEvent(torch.ones(1, 4))],
        state,
        _feedback(1),
        route_feedback=_route_feedback(
            1.0 if int(output.selected_program_slots[0]) == 1 else 0.0
        ),
    )

    assert output.program_slot == 1
    assert output.program_digest == agent.program_memory.program(1).digest()
    assert agent.program_route_evidence is not None
    assert agent.program_route_evidence.preferred_slots(
        torch.tensor([[1.0, 0.0]])
    ).item() == 1

    restored = agent.state_from_payload(state.payload())
    assert restored.pending_route_query is not None
    assert torch.equal(restored.pending_route_slots, state.pending_route_slots)


def test_control_flow_runtime_rejects_incompatible_route_query_space() -> None:
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    memory.add_program(_other_program(), protect=True)
    evidence = PersistentOpaqueContextRouteEvidence(
        width=20,
        query_space_id="route-query-v2",
    )
    evidence.append_slot()
    evidence.append_slot()
    query_adapter = ExternalControllerTrajectoryQueryAdapter(
        controller_width=4,
        query_width=20,
        query_space_id="route-query-v1",
    )

    with pytest.raises(ValueError, match="query space"):
        _agent(
            memory=memory,
            query_adapter=query_adapter,
            route_evidence=evidence,
        )


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


def test_control_flow_runtime_exposes_replaceable_trajectory_route_query() -> None:
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    memory.add_program(_other_program(), protect=True)
    router = ExternalOutcomeProgramRouter(
        feature_width=20,
        program_capacity=2,
        initial_programs=2,
    )
    query_adapter = ExternalControllerTrajectoryQueryAdapter(
        controller_width=4,
        query_width=20,
    )
    agent = _agent(
        memory=memory,
        router=router,
        query_adapter=query_adapter,
    )

    output, _ = agent.step_events(
        [AmodalEvent(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))],
        agent.initial_state(1, device="cpu"),
        _feedback(1),
    )

    assert output.program_route_query is not None
    assert output.program_route_query.shape == (1, 20)
    assert bool(torch.isfinite(output.program_route_query).all())
    assert not output.program_route_query.requires_grad
    assert agent.configuration()["program_route_query_adapter"]["query_width"] == 20


def test_control_flow_runtime_can_credit_router_without_repeating_outcome_to_controller() -> None:
    memory = ControlFlowProgramMemory(2)
    memory.add_program(_program(), protect=True)
    memory.add_program(_other_program(), protect=True)
    router = ExternalOutcomeProgramRouter(
        feature_width=2,
        program_capacity=2,
        initial_programs=2,
        initial_learning_rate=0.5,
        initial_trace_decay=0.0,
        initial_baseline_rate=0.05,
    )
    agent = _agent(memory=memory, router=router)
    quiet = _feedback(1)
    state = agent.initial_state(1, device="cpu")
    first, state = agent.step_events(
        [AmodalEvent(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))],
        state,
        quiet,
    )
    policy_before = state.program_router.credit.policy.clone()
    route_feedback = replace(
        quiet,
        reward=torch.ones(1),
        has_feedback=torch.ones(1, dtype=torch.bool),
    )
    second, state = agent.step_events(
        [AmodalEvent(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))],
        state,
        quiet,
        route_feedback=route_feedback,
    )

    assert first.controller.intention.payload.shape == second.controller.intention.payload.shape
    assert state.program_router.credit.feedbacks.item() == 1
    assert not torch.equal(state.program_router.credit.policy, policy_before)


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
