import torch

from neural_computer import (
    ExternalCapabilityRegisterMachine,
    ExternalRegisterInstruction,
    IntentEvent,
)


def _machine() -> ExternalCapabilityRegisterMachine:
    return ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        instructions=(
            ExternalRegisterInstruction(5),
            ExternalRegisterInstruction(5),
        ),
    )


def test_external_register_machine_has_one_shared_interpreter_and_variable_program_data() -> None:
    machine = _machine()

    assert machine.configuration()["schema"] == (
        "neural-computer.external-register.v2"
    )
    assert machine.configuration()["instruction_count"] == 2
    assert machine.configuration()["execution"] == (
        "shared_interpreter_serial_instruction_chain_v1"
    )
    assert machine.configuration()["operator_mode"] == "factorized_low_rank"
    assert machine.configuration()["operator_rank"] == 8
    assert machine.configuration()["read_execute"] == (
        "neural-computer.external-register-read-execute.v1"
    )

    index = machine.add_instruction(ExternalRegisterInstruction(5))

    assert index == 2
    assert machine.configuration()["instruction_count"] == 3
    assert len(machine.instructions) == 3


def test_factorized_film_operator_is_shared_and_instruction_conditioned() -> None:
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_film",
        instructions=(ExternalRegisterInstruction(5),),
    )

    assert machine.configuration()["operator_mode"] == "factorized_film"
    register = torch.randn(3, 8)
    result = machine.execute(register, machine.instructions[0])

    assert result.shape == register.shape
    assert torch.isfinite(result).all()


def test_factorized_hybrid_operator_starts_with_the_composable_base_path() -> None:
    torch.manual_seed(909)
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_hybrid",
        instructions=(ExternalRegisterInstruction(5),),
    )

    assert machine.configuration()["operator_mode"] == "factorized_hybrid"
    register = torch.randn(3, 8)
    result = machine.execute(register, machine.instructions[0])

    assert result.shape == register.shape
    assert torch.isfinite(result).all()


def test_bounded_residual_operator_is_finite_and_configuration_is_explicit() -> None:
    torch.manual_seed(910)
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_bounded_residual",
        instructions=(ExternalRegisterInstruction(5),),
    )

    register = torch.randn(3, 8)
    result = machine.execute(register, machine.instructions[0])

    assert machine.configuration()["operator_mode"] == (
        "factorized_bounded_residual"
    )
    assert result.shape == register.shape
    assert torch.isfinite(result).all()


def test_external_register_state_is_external_and_quiet_ticks_do_not_mutate_it() -> None:
    torch.manual_seed(903)
    machine = _machine()
    state = machine.initial_state(2, device="cpu")
    intention = IntentEvent(torch.randn(2, 6))
    event = torch.randn(2, 4)
    action = torch.randn(2, 2)
    outcome = torch.zeros(2)

    _, active_state = machine.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=state,
    )
    quiet_event = torch.randn(2, 4)
    _, quiet_state = machine.step(
        event=quiet_event,
        action=torch.randn(2, 2),
        outcome=torch.ones(2),
        intention=IntentEvent(torch.randn(2, 6)),
        state=active_state,
        present=torch.zeros(2, dtype=torch.bool),
    )

    assert active_state.initialized.equal(torch.ones(2, dtype=torch.bool))
    assert torch.equal(quiet_state.register, active_state.register)
    assert torch.equal(quiet_state.initialized, active_state.initialized)

    empty_state = machine.initial_state(2, device="cpu")
    _, still_empty = machine.step(
        event=quiet_event,
        action=torch.randn(2, 2),
        outcome=torch.ones(2),
        intention=IntentEvent(torch.randn(2, 6)),
        state=empty_state,
        present=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.equal(still_empty.register, empty_state.register)
    assert torch.equal(still_empty.initialized, empty_state.initialized)


def test_downstream_instruction_executes_on_register_only() -> None:
    torch.manual_seed(904)
    machine = _machine()
    register = torch.randn(3, 8)
    first = machine.execute(register, machine.instructions[0])
    second_a = machine.execute(first, machine.instructions[1])
    second_b = machine.execute(first, machine.instructions[1])

    assert torch.equal(second_a, second_b)
    assert second_a.shape == (3, 8)
    assert machine.instructions[1].configuration()["storage"] == (
        "one_opaque_learned_vector_v1"
    )


def test_read_execute_uses_a_transient_snapshot_without_mutating_observed_state() -> None:
    torch.manual_seed(906)
    machine = _machine()
    state = machine.initial_state(2, device="cpu")
    kwargs = {
        "event": torch.randn(2, 4),
        "action": torch.zeros(2, 2),
        "outcome": torch.zeros(2),
        "intention": IntentEvent(torch.randn(2, 6)),
        "state": state,
    }

    observed, observed_state = machine.observe_register(**kwargs)
    expected = machine.execute_chain(observed, tuple(machine.instructions))
    snapshot, snapshot_state = machine.read_execute_register(**kwargs)

    assert torch.equal(snapshot, expected)
    assert torch.equal(snapshot_state.register, observed_state.register)
    assert torch.equal(snapshot_state.context, observed_state.context)
    assert not torch.equal(snapshot, snapshot_state.register)


def test_in_place_step_preserves_legacy_mutating_execution_contract() -> None:
    torch.manual_seed(907)
    machine = _machine()
    state = machine.initial_state(2, device="cpu")
    kwargs = {
        "event": torch.randn(2, 4),
        "action": torch.zeros(2, 2),
        "outcome": torch.zeros(2),
        "intention": IntentEvent(torch.randn(2, 6)),
        "state": state,
    }

    register, next_state = machine.step_register(**kwargs)

    assert torch.equal(register, next_state.register)


def test_external_decoder_can_consume_a_memory_selected_register_chain() -> None:
    torch.manual_seed(905)
    machine = _machine()
    state = machine.initial_state(2, device="cpu")
    kwargs = {
        "event": torch.randn(2, 4),
        "action": torch.zeros(2, 2),
        "outcome": torch.zeros(2),
        "intention": IntentEvent(torch.randn(2, 6)),
        "state": state,
    }

    first_register, next_state = machine.step_register(
        **kwargs,
        instructions=(machine.instructions[0],),
    )
    second_register, final_state = machine.step_register(
        **{**kwargs, "state": next_state},
        instructions=(machine.instructions[1],),
    )
    decoded = machine.to_intention(second_register)

    assert first_register.shape == (2, 8)
    assert second_register.shape == (2, 8)
    assert decoded.payload.shape == (2, 6)
    assert final_state.initialized.equal(torch.ones(2, dtype=torch.bool))
