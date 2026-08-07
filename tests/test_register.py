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

    index = machine.add_instruction(ExternalRegisterInstruction(5))

    assert index == 2
    assert machine.configuration()["instruction_count"] == 3
    assert len(machine.instructions) == 3


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
