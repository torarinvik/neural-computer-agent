import torch

from neural_computer import (
    ExternalCapabilityRegisterMachine,
    ExternalRegisterBasisCompatibilityPrior,
    ExternalRegisterComputeBasis,
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
        "neural-computer.external-register.v4"
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


def test_external_compute_basis_is_append_only_and_memory_addressable() -> None:
    machine = _machine()
    index = machine.add_basis_slot()

    assert index == 0
    assert machine.configuration()["basis_slot_count"] == 1
    assert machine.basis_slots[0].configuration()["storage"] == (
        "append_only_external_compute_slot_v1"
    )
    register = torch.randn(2, 8)
    default = machine.execute(register, machine.instructions[0])
    extended = machine.execute(
        register, machine.instructions[0], basis_slot=index
    )
    assert default.shape == extended.shape == register.shape
    assert torch.isfinite(extended).all()
    assert not torch.equal(default, extended)


def test_new_basis_slot_does_not_resize_controller_or_existing_instruction_data() -> None:
    machine = _machine()
    old_instruction = machine.instructions[0].code.detach().clone()
    old_parameter_count = sum(parameter.numel() for parameter in machine.parameters())
    machine.add_basis_slot(ExternalRegisterComputeBasis(8, 5, hidden=10))

    assert machine.instructions[0].code.shape == old_instruction.shape
    assert torch.equal(machine.instructions[0].code, old_instruction)
    assert sum(parameter.numel() for parameter in machine.parameters()) > old_parameter_count


def test_basis_selection_reuses_only_fresh_verified_slots_or_requests_growth() -> None:
    machine = _machine()
    machine.add_basis_slot()

    reuse = machine.select_basis_slot({0: (0.91, 0.88)}, threshold=0.8)
    grow = machine.select_basis_slot({0: (0.91, 0.79)}, threshold=0.8)

    assert reuse.action == "reuse"
    assert reuse.compute_slot_index == 0
    assert grow.action == "grow"
    assert grow.compute_slot_index is None


def test_basis_efficiency_selection_rejects_asymmetric_cross_operator_transfer() -> None:
    machine = _machine()
    machine.add_basis_slot()
    decision = machine.select_basis_slot_by_efficiency(
        {0: (0.98, 0.91)},
        {0: 16_384},
        fresh_stable_bits=8_192,
        threshold=0.8,
    )

    assert decision.action == "grow"
    assert decision.compute_slot_index is None


def test_opaque_basis_compatibility_prior_only_orders_and_never_admits() -> None:
    torch.manual_seed(912)
    machine = _machine()
    machine.add_basis_slot()
    machine.add_basis_slot()
    prior = ExternalRegisterBasisCompatibilityPrior(5, latent_width=6, hidden=8)
    keys = prior.basis_keys(machine.basis_slots)
    query = torch.randn(2, 5)
    cold_scores = prior(query, keys)
    assert torch.equal(cold_scores, torch.zeros_like(cold_scores))

    prior.enable()
    assert sorted(prior.order(query[0], keys)) == [0, 1]
    scheduled = machine.order_basis_candidates(prior, query[0], (1, 0))
    assert set(scheduled) == {0, 1}
    assert machine.configuration()["basis_slot_count"] == len(machine.basis_slots)
    outcomes = torch.tensor([[0.9, 0.4], [0.2, 0.8]])
    loss, pair_count = prior.outcome_ranking_loss(query, keys, outcomes)
    assert torch.isfinite(loss)
    assert pair_count == 2
    assert prior.configuration()["role"] == (
        "screening_only_fresh_admission_required"
    )


def test_basis_slots_can_be_frozen_and_unpromoted_growth_can_be_rolled_back() -> None:
    machine = _machine()
    first = machine.add_basis_slot()
    second = machine.add_basis_slot()

    machine.freeze_basis_slot(first)
    assert all(
        not parameter.requires_grad for parameter in machine.basis_slots[first].parameters()
    )
    machine.remove_basis_slot(second)
    assert len(machine.basis_slots) == 1


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


def test_protected_meta_operator_starts_with_an_inert_residual_family() -> None:
    torch.manual_seed(911)
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_protected_meta",
        instructions=(ExternalRegisterInstruction(5),),
    )

    register = torch.randn(3, 8)
    result = machine.execute(register, machine.instructions[0])

    assert machine.configuration()["operator_mode"] == (
        "factorized_protected_meta"
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


def test_event_window_is_persistent_and_available_to_new_basis_slots() -> None:
    torch.manual_seed(908)
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        event_window_size=3,
        instructions=(ExternalRegisterInstruction(5),),
    )
    basis_slot = machine.add_basis_slot()
    state = machine.initial_state(2, device="cpu")
    events = [torch.randn(2, 4) for _ in range(4)]
    for event in events:
        register, state = machine.read_execute_register(
            event=event,
            action=torch.zeros(2, 2),
            outcome=torch.zeros(2),
            intention=IntentEvent(torch.zeros(2, 6)),
            state=state,
            instructions=(machine.instructions[0],),
            basis_slots=(basis_slot,),
        )

    assert state.event_window is not None
    assert state.event_window_mask is not None
    assert state.event_window.shape == (2, 3, 4)
    assert state.event_window_mask.equal(torch.ones(2, 3, dtype=torch.bool))
    assert torch.equal(state.event_window[:, -1], events[-1])
    assert register.shape == (2, 8)


def test_quiet_ticks_preserve_the_entire_event_window() -> None:
    torch.manual_seed(909)
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        event_window_size=3,
        instructions=(ExternalRegisterInstruction(5),),
    )
    state = machine.initial_state(2, device="cpu")
    kwargs = {
        "action": torch.zeros(2, 2),
        "outcome": torch.zeros(2),
        "intention": IntentEvent(torch.zeros(2, 6)),
    }
    for event in (torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4)):
        _, state = machine.observe_register(event=event, state=state, **kwargs)
    before = state.event_window.clone()
    before_mask = state.event_window_mask.clone()
    _, after = machine.observe_register(
        event=torch.randn(2, 4),
        state=state,
        present=torch.zeros(2, dtype=torch.bool),
        **kwargs,
    )
    assert torch.equal(after.event_window, before)
    assert torch.equal(after.event_window_mask, before_mask)
