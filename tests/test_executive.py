import pickle
from dataclasses import replace

import pytest
import torch

from neural_computer.executive import (
    ExecutiveInstruction,
    ExternalAmodalExecutive,
    ExternalExecutiveOperator,
    ExternalExecutiveOperatorRegistry,
    ExternalExecutiveOperatorState,
    ExternalExecutiveProgram,
    TrustedExternalExecutiveState,
    TypedWorkspaceValue,
)
from neural_computer.executive_memory import ExternalValueDelayOperator
from neural_computer.interface import AmodalEvent, AmodalEventCollection


class _EventEvidence(ExternalExecutiveOperator):
    def __init__(self) -> None:
        super().__init__(
            17,
            ("events",),
            "evidence",
            interface_version="test.event-evidence.v1",
        )

    def execute(
        self,
        arguments: tuple[TypedWorkspaceValue, ...],
    ) -> TypedWorkspaceValue:
        events = arguments[0].payload
        assert isinstance(events, AmodalEventCollection)
        batch = events.payload.shape[0]
        present = events.present.any(dim=1)
        score = torch.zeros(batch, 1, device=events.payload.device)
        if events.payload.shape[1]:
            score[:, 0] = events.payload[:, 0, 0]
        return TypedWorkspaceValue.from_tensor(
            "evidence",
            score,
            present=present,
            confidence=arguments[0].confidence,
        )


class _EvidenceIntention(ExternalExecutiveOperator):
    def __init__(self) -> None:
        super().__init__(
            23,
            ("evidence",),
            "intention",
            interface_version="test.evidence-intention.v1",
        )
        self.weights = torch.tensor([[1.0], [-1.0]])

    def execute(
        self,
        arguments: tuple[TypedWorkspaceValue, ...],
    ) -> TypedWorkspaceValue:
        evidence = arguments[0]
        assert isinstance(evidence.payload, torch.Tensor)
        payload = evidence.payload @ self.weights.T.to(evidence.payload)
        return TypedWorkspaceValue.from_tensor(
            "intention",
            payload,
            present=evidence.present,
            confidence=evidence.confidence,
        )


class _EventValue(ExternalExecutiveOperator):
    def __init__(self) -> None:
        super().__init__(31, ("events",), "value", interface_version="test.event-value.v1")

    def execute(
        self, arguments: tuple[TypedWorkspaceValue, ...]
    ) -> TypedWorkspaceValue:
        events = arguments[0].payload
        assert isinstance(events, AmodalEventCollection)
        batch = events.payload.shape[0]
        values = torch.zeros(batch, 2, dtype=events.payload.dtype)
        present = events.present.any(dim=1)
        if events.payload.shape[1]:
            values = events.payload[:, 0, :2]
        return TypedWorkspaceValue.from_tensor(
            "value", values, present=present, confidence=arguments[0].confidence
        )


class _ValueEquality(ExternalExecutiveOperator):
    def __init__(self) -> None:
        super().__init__(37, ("value", "value"), "evidence", interface_version="test.value-equality.v1")

    def execute(
        self, arguments: tuple[TypedWorkspaceValue, ...]
    ) -> TypedWorkspaceValue:
        left, right = arguments
        assert isinstance(left.payload, torch.Tensor)
        assert isinstance(right.payload, torch.Tensor)
        present = left.present & right.present
        equal = torch.isclose(left.payload, right.payload).all(dim=1, keepdim=True)
        score = torch.where(equal, torch.ones_like(equal, dtype=left.payload.dtype), -torch.ones_like(equal, dtype=left.payload.dtype))
        return TypedWorkspaceValue.from_tensor(
            "evidence",
            score,
            present=present,
            confidence=torch.minimum(left.confidence, right.confidence),
        )


def _collection(value: float | None, *, batch: int = 1) -> AmodalEventCollection:
    if value is None:
        return AmodalEventCollection.from_events(
            (), batch_size=batch, width=3, device="cpu"
        )
    return AmodalEventCollection.from_events(
        (
            AmodalEvent(
                payload=torch.tensor([[value, 0.25, -0.5]]).expand(batch, -1),
                source_key=torch.ones(batch, 2),
                confidence=torch.full((batch,), 0.75),
            ),
        )
    )


def _program() -> ExternalExecutiveProgram:
    return ExternalExecutiveProgram(
        slot_count=5,
        instructions=(
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction(
                "call", destination=1, operator_handle=17, arguments=(0,)
            ),
            ExecutiveInstruction(
                "branch",
                source=1,
                true_target=3,
                false_target=3,
                unknown_target=8,
            ),
            ExecutiveInstruction(
                "call", destination=2, operator_handle=23, arguments=(1,)
            ),
            ExecutiveInstruction("read", source=2),
            ExecutiveInstruction("write", destination=3),
            ExecutiveInstruction("copy", source=3, destination=4),
            ExecutiveInstruction("emit", source=4),
            ExecutiveInstruction("wait"),
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction(
                "call", destination=1, operator_handle=17, arguments=(0,)
            ),
            ExecutiveInstruction(
                "branch",
                source=1,
                true_target=3,
                false_target=3,
                unknown_target=8,
            ),
            ExecutiveInstruction("halt"),
        ),
    ).validate()


def test_typed_executive_waits_on_missing_input_then_emits_on_later_tick() -> None:
    intention_operator = _EvidenceIntention()
    executive = ExternalAmodalExecutive(
        _program(),
        ExternalExecutiveOperatorRegistry((_EventEvidence(), intention_operator)),
        intention_width=2,
    )
    state = executive.initial_state(1, device="cpu")
    frozen_weights = intention_operator.weights.clone()

    quiet, state = executive.tick(_collection(None), state)
    assert quiet.status == "waiting"
    assert quiet.intention is None
    assert state.instruction_pointer == 9
    assert state.workspace[0].kind == "events"
    assert not bool(state.workspace[0].present.item())
    assert state.workspace[1].kind == "evidence"
    assert not bool(state.workspace[1].present.item())

    emitted, state = executive.tick(_collection(0.8), state)
    assert emitted.status == "emitted"
    assert emitted.intention is not None
    assert torch.allclose(emitted.intention.payload, torch.tensor([[0.8, -0.8]]))
    received = state.workspace[0].payload
    assert isinstance(received, AmodalEventCollection)
    assert received.source_key is not None
    assert received.source_key.shape == (1, 1, 2)
    assert state.workspace[2].kind == "intention"
    assert state.workspace[3].kind == "intention"
    assert state.workspace[4].kind == "intention"
    assert isinstance(state.workspace[2].payload, torch.Tensor)
    assert isinstance(state.workspace[3].payload, torch.Tensor)
    assert isinstance(state.workspace[4].payload, torch.Tensor)
    assert (
        state.workspace[2].payload.data_ptr() != state.workspace[3].payload.data_ptr()
    )
    assert (
        state.workspace[3].payload.data_ptr() != state.workspace[4].payload.data_ptr()
    )
    assert state.ticks == 2
    assert torch.equal(intention_operator.weights, frozen_weights)

    paused, state = executive.tick(_collection(None), state)
    assert paused.status == "waiting"
    negative, state = executive.tick(_collection(-0.6), state)
    assert negative.status == "emitted"
    assert negative.intention is not None
    assert torch.allclose(negative.intention.payload, torch.tensor([[-0.6, 0.6]]))
    assert state.ticks == 4
    assert torch.equal(intention_operator.weights, frozen_weights)


def test_sealed_fast_path_matches_defensive_tick_and_rejects_foreign_leases() -> None:
    def build() -> ExternalAmodalExecutive:
        return ExternalAmodalExecutive(
            _program(),
            ExternalExecutiveOperatorRegistry((_EventEvidence(), _EvidenceIntention())),
            intention_width=2,
        )

    defensive = build()
    fast = build()
    defensive_state = defensive.initial_state(1, device="cpu")
    fast_state = fast.initial_sealed_state(1, device="cpu")
    inputs = (_collection(None), _collection(0.8), _collection(None), _collection(-0.6))

    for events in inputs:
        defensive_output, defensive_state = defensive.tick(events, defensive_state)
        fast_output, fast_state = fast.tick_fast(events, fast_state)
        assert fast_output.status == defensive_output.status
        assert fast_output.executed_instructions == defensive_output.executed_instructions
        assert fast_output.intention is None or defensive_output.intention is not None
        if fast_output.intention is not None:
            assert defensive_output.intention is not None
            assert torch.equal(
                fast_output.intention.payload, defensive_output.intention.payload
            )
        assert fast_state.state.instruction_pointer == defensive_state.instruction_pointer
        assert fast_state.state.ticks == defensive_state.ticks

    with pytest.raises(TypeError, match="sealed state lease"):
        fast.tick_fast(_collection(None), defensive_state)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="belongs to another executive"):
        build().tick_fast(_collection(None), fast_state)
    forged = TrustedExternalExecutiveState(fast_state.state, object())
    with pytest.raises(ValueError, match="belongs to another executive"):
        fast.tick_fast(_collection(None), forged)
    corrupted = TrustedExternalExecutiveState(
        replace(
            fast_state.state,
            instruction_pointer=len(fast.program.instructions),
        ),
        fast_state.owner_token,
    )
    with pytest.raises(ValueError, match="instruction pointer"):
        fast.tick_fast(_collection(None), corrupted)
    with pytest.raises(ValueError, match="handles are incompatible"):
        fast.seal_state(
            replace(fast.initial_state(1, device="cpu"), operator_states=())
        )
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(fast_state)

def test_executive_call_types_and_empty_reads_fail_closed() -> None:
    wrong_type = ExternalExecutiveProgram(
        2,
        (
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction(
                "call", destination=1, operator_handle=23, arguments=(0,)
            ),
            ExecutiveInstruction("halt"),
        ),
    )
    executive = ExternalAmodalExecutive(
        wrong_type,
        ExternalExecutiveOperatorRegistry((_EvidenceIntention(),)),
        intention_width=2,
    )
    with pytest.raises(TypeError, match="argument kind"):
        executive.tick(_collection(1.0), executive.initial_state(1, device="cpu"))

    empty_read = ExternalExecutiveProgram(
        1,
        (
            ExecutiveInstruction("read", source=0),
            ExecutiveInstruction("halt"),
        ),
    )
    executive = ExternalAmodalExecutive(
        empty_read,
        ExternalExecutiveOperatorRegistry(()),
        intention_width=2,
    )
    with pytest.raises(RuntimeError, match="empty slot"):
        executive.tick(_collection(None), executive.initial_state(1, device="cpu"))

    unknown_handle = ExternalExecutiveProgram(
        2,
        (
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction(
                "call", destination=1, operator_handle=999, arguments=(0,)
            ),
            ExecutiveInstruction("halt"),
        ),
    )
    executive = ExternalAmodalExecutive(
        unknown_handle,
        ExternalExecutiveOperatorRegistry(()),
        intention_width=2,
    )
    with pytest.raises(LookupError, match="unknown.*handle"):
        executive.tick(_collection(1.0), executive.initial_state(1, device="cpu"))


def test_branch_unknown_is_distinct_and_batch_divergence_fails_closed() -> None:
    divergent_program = ExternalExecutiveProgram(
        2,
        (
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction(
                "call", destination=1, operator_handle=17, arguments=(0,)
            ),
            ExecutiveInstruction(
                "branch",
                source=1,
                true_target=3,
                false_target=4,
                unknown_target=5,
            ),
            ExecutiveInstruction("wait"),
            ExecutiveInstruction("wait"),
            ExecutiveInstruction("wait"),
            ExecutiveInstruction("halt"),
        ),
    )
    divergent = ExternalAmodalExecutive(
        divergent_program,
        ExternalExecutiveOperatorRegistry((_EventEvidence(),)),
        intention_width=2,
    )
    state = divergent.initial_state(2, device="cpu")
    mixed_event = AmodalEvent(
        payload=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        confidence=torch.ones(2),
    )
    with pytest.raises(RuntimeError, match="control flow diverged"):
        divergent.tick(AmodalEventCollection.from_events((mixed_event,)), state)

    executive = ExternalAmodalExecutive(
        _program(),
        ExternalExecutiveOperatorRegistry((_EventEvidence(), _EvidenceIntention())),
        intention_width=2,
    )
    state = executive.initial_state(2, device="cpu")
    quiet, state = executive.tick(_collection(None, batch=2), state)
    assert quiet.status == "waiting"
    evidence = state.workspace[1]
    assert evidence.kind == "evidence"
    assert not bool(evidence.present.any())


def test_program_validation_and_instruction_budget_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown targets"):
        ExternalExecutiveProgram(
            1,
            (
                ExecutiveInstruction(
                    "branch",
                    source=0,
                    true_target=1,
                    false_target=1,
                    unknown_target=99,
                ),
                ExecutiveInstruction("halt"),
            ),
        ).validate()

    loop = ExternalExecutiveProgram(
        2,
        (
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction(
                "call", destination=1, operator_handle=17, arguments=(0,)
            ),
            ExecutiveInstruction(
                "branch",
                source=1,
                true_target=2,
                false_target=2,
                unknown_target=2,
            ),
            ExecutiveInstruction("halt"),
        ),
    )
    executive = ExternalAmodalExecutive(
        loop,
        ExternalExecutiveOperatorRegistry((_EventEvidence(),)),
        intention_width=2,
        max_instructions_per_tick=5,
    )
    output, state = executive.tick(
        _collection(1.0), executive.initial_state(1, device="cpu")
    )
    assert output.status == "step_budget_exhausted"
    assert state.status == "step_budget_exhausted"
    with pytest.raises(RuntimeError, match="cannot resume"):
        executive.tick(_collection(1.0), state)


def test_explicit_delay_state_composes_two_back_across_live_ticks() -> None:
    """The same frozen operators solve a temporal relation by external state."""
    program = ExternalExecutiveProgram(
        5,
        (
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction("call", destination=1, operator_handle=31, arguments=(0,)),
            ExecutiveInstruction("call", destination=2, operator_handle=41, arguments=(1,)),
            ExecutiveInstruction("call", destination=3, operator_handle=37, arguments=(1, 2)),
            ExecutiveInstruction("branch", source=3, true_target=5, false_target=5, unknown_target=7),
            ExecutiveInstruction("call", destination=4, operator_handle=23, arguments=(3,)),
            ExecutiveInstruction("emit", source=4, next_target=0),
            ExecutiveInstruction("wait", next_target=0),
            ExecutiveInstruction("halt"),
        ),
    ).validate()
    intention = _EvidenceIntention()
    delay = ExternalValueDelayOperator(41, width=2, delay=2)
    registry = ExternalExecutiveOperatorRegistry(
        (_EventValue(), _ValueEquality(), intention, delay)
    )
    executive = ExternalAmodalExecutive(program, registry, intention_width=2)
    state = executive.initial_state(1, device="cpu")
    frozen_weights = intention.weights.clone()

    first, state = executive.tick(_collection(1.0), state)
    second, state = executive.tick(_collection(2.0), state)
    before_decision = state
    matching, state = executive.tick(_collection(1.0), state)
    different, state = executive.tick(_collection(3.0), state)

    assert first.status == second.status == "waiting"
    assert first.intention is None and second.intention is None
    assert matching.status == different.status == "emitted"
    assert matching.intention is not None and different.intention is not None
    assert torch.equal(matching.intention.payload, torch.tensor([[1.0, -1.0]]))
    assert torch.equal(different.intention.payload, torch.tensor([[-1.0, 1.0]]))
    assert state.ticks == 4
    assert state.instruction_pointer == 0
    assert torch.equal(intention.weights, frozen_weights)

    # Causal control: erasing only temporal presence removes the decision.
    operator_states = dict(before_decision.operator_states)
    delay_state = operator_states[41]
    operator_states[41] = ExternalExecutiveOperatorState.from_mapping(
        delay_state.interface_version,
        {
            name: torch.zeros_like(value) if name == "present" else value
            for name, value in delay_state.tensors
        },
    )
    corrupted_state = replace(
        before_decision, operator_states=tuple(sorted(operator_states.items()))
    )
    corrupted, _ = executive.tick(_collection(1.0), corrupted_state)
    assert corrupted.status == "waiting" and corrupted.intention is None

    # A second executive sharing the exact same operator objects starts clean.
    independent = executive.initial_state(1, device="cpu")
    clean, independent = executive.tick(_collection(1.0), independent)
    assert clean.status == "waiting" and clean.intention is None
    independent_delay_state = dict(independent.operator_states)[41]
    assert independent_delay_state.tensor("present").tolist() == [[False, True]]


class _InvalidStateTransition(ExternalExecutiveOperator):
    def __init__(self) -> None:
        super().__init__(53, ("events",), "value", interface_version="test.invalid-state.v1")

    def initial_state(self, batch_size: int, *, device: torch.device | str, dtype: torch.dtype) -> ExternalExecutiveOperatorState:
        return ExternalExecutiveOperatorState.from_mapping(
            self.interface_version, {"count": torch.zeros(batch_size, device=device, dtype=dtype)}
        )

    def validate_state(self, state: ExternalExecutiveOperatorState, *, batch_size: int) -> ExternalExecutiveOperatorState:
        state.validate()
        if state.interface_version != self.interface_version or state.tensor("count").shape != (batch_size,):
            raise ValueError("invalid transition state")
        return state

    def execute(self, arguments: tuple[TypedWorkspaceValue, ...]) -> TypedWorkspaceValue:
        raise RuntimeError("state is required")

    def execute_with_state(self, arguments: tuple[TypedWorkspaceValue, ...], state: ExternalExecutiveOperatorState) -> tuple[TypedWorkspaceValue, ExternalExecutiveOperatorState]:
        changed = ExternalExecutiveOperatorState.from_mapping(
            self.interface_version, {"count": state.tensor("count") + 1}
        )
        # Wrong output kind forces validation failure after computing next state.
        return TypedWorkspaceValue.from_tensor("evidence", torch.ones(1, 1)), changed


def test_failed_stateful_call_does_not_commit_and_registry_mismatch_fails() -> None:
    program = ExternalExecutiveProgram(
        2,
        (
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction("call", destination=1, operator_handle=53, arguments=(0,)),
            ExecutiveInstruction("halt"),
        ),
    )
    executive = ExternalAmodalExecutive(
        program,
        ExternalExecutiveOperatorRegistry((_InvalidStateTransition(),)),
        intention_width=2,
    )
    state = executive.initial_state(1, device="cpu")
    before = dict(state.operator_states)[53].tensor("count").clone()
    with pytest.raises(TypeError, match="incompatible kind"):
        executive.tick(_collection(1.0), state)
    assert torch.equal(dict(state.operator_states)[53].tensor("count"), before)

    incompatible = ExternalAmodalExecutive(
        program,
        ExternalExecutiveOperatorRegistry((_EventValue(),)),
        intention_width=2,
    )
    with pytest.raises(ValueError, match="registry is incompatible"):
        incompatible.tick(_collection(1.0), state)
