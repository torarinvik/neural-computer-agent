import torch

from neural_computer import (
    CanonicalRegisterReadout,
    ExternalCapabilityRegisterMachine,
    ExternalRegisterBasisCompatibilityPrior,
    ExternalRegisterComputeBasis,
    ExternalRegisterInstruction,
    ExternalSequenceMemory,
    ExternalSequenceOperatorMemory,
    IntentEvent,
    LearnedRegisterRoleBinding,
)


def test_canonical_register_readout_is_identity_initialized_and_versioned() -> None:
    torch.manual_seed(913)
    readout = CanonicalRegisterReadout(8, 8, hidden=12)
    register = torch.randn(3, 8)

    assert readout.configuration()["schema"] == (
        "neural-computer.external-register-canonical-readout.v1"
    )
    assert torch.equal(readout(register), register)

    with torch.no_grad():
        readout.base.bias[0] = 1.0
    assert not torch.equal(readout(register), register)


def test_canonical_register_readout_validates_register_shape_and_finiteness() -> None:
    readout = CanonicalRegisterReadout(8, 6)
    try:
        readout(torch.zeros(2, 7))
    except ValueError as error:
        assert "wrong shape" in str(error)
    else:
        raise AssertionError("expected shape validation")
    try:
        readout(torch.full((2, 8), float("nan")))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("expected finite-value validation")


def _machine(**kwargs: object) -> ExternalCapabilityRegisterMachine:
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
        **kwargs,
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


def test_shared_interpreter_mode_uses_one_instruction_family_for_addressed_slots() -> None:
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_shared_interpreter",
        instructions=(ExternalRegisterInstruction(5),),
        basis_slots=(ExternalRegisterComputeBasis(8, 5, hidden=10),),
    )
    register = torch.randn(2, 8)
    shared = machine.execute(register, machine.instructions[0])
    addressed = machine.execute(register, machine.instructions[0], basis_slot=0)

    assert machine.configuration()["operator_mode"] == (
        "factorized_shared_interpreter"
    )
    assert machine.configuration()["basis_binding"] == (
        "instruction_vector_selects_shared_interpreter_v1"
    )
    assert torch.equal(shared, addressed)


def test_shared_bounded_mode_limits_serial_state_drift() -> None:
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_shared_bounded",
        instructions=tuple(ExternalRegisterInstruction(5) for _ in range(3)),
    )
    register = torch.randn(2, 8)
    result = register
    for instruction in machine.instructions:
        result = machine.execute(result, instruction)

    assert machine.configuration()["operator_mode"] == "factorized_shared_bounded"
    assert machine.configuration()["basis_binding"] == (
        "instruction_vector_selects_shared_interpreter_v1"
    )
    assert torch.isfinite(result).all()
    assert float(result.abs().mean().detach()) < 10.0


def test_execution_trace_preserves_each_opaque_intermediate_state() -> None:
    machine = _machine()
    register = torch.randn(2, 8)
    trace_final, trace = machine.execute_chain_trace(
        register,
        machine.instructions,
    )
    final = machine.execute_chain(register, machine.instructions)

    assert machine.configuration()["execution_trace"] == (
        "neural-computer.external-register-execution-trace.v1"
    )
    assert len(trace) == len(machine.instructions)
    assert all(state.shape == register.shape for state in trace)
    assert torch.equal(trace_final, trace[-1])
    assert torch.equal(final, trace[-1])
    assert not torch.equal(trace[0], trace[1])


def test_shared_banked_mode_reads_only_prior_intermediate_states() -> None:
    torch.manual_seed(914)
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_shared_banked",
        instructions=tuple(ExternalRegisterInstruction(5) for _ in range(3)),
    )
    register = torch.randn(2, 8)
    final, trace = machine.execute_chain_trace(register, machine.instructions)
    first_only = machine.execute(register, machine.instructions[0])

    assert machine.configuration()["operator_mode"] == "factorized_shared_banked"
    assert machine.configuration()["basis_binding"] == (
        "instruction_vector_selects_shared_interpreter_v1"
    )
    assert len(trace) == 3
    assert torch.isfinite(final).all()
    assert not torch.equal(final, first_only)


def test_shared_relational_mode_integrates_role_relations_into_transition() -> None:
    torch.manual_seed(917)
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_shared_relational",
        role_count=2,
        instructions=tuple(ExternalRegisterInstruction(5) for _ in range(2)),
    )
    register = torch.randn(3, 8)
    result = machine.execute_chain(register, machine.instructions)
    loss = result.square().mean()
    loss.backward()

    assert machine.configuration()["operator_mode"] == (
        "factorized_shared_relational"
    )
    assert machine.relational_transition.configuration()["schema"] == (
        "neural-computer.external-register-relational-transition.v1"
    )
    assert torch.isfinite(result).all()
    assert machine.relational_transition.binding.role_seed.grad is not None


def test_stable_relational_mode_keeps_role_binding_instruction_independent() -> None:
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_shared_stable_relational",
        role_count=2,
        instructions=(ExternalRegisterInstruction(5),),
    )
    assert machine.configuration()["operator_mode"] == (
        "factorized_shared_stable_relational"
    )
    assert machine.relational_transition.configuration()[
        "instruction_conditioned_binding"
    ] is False

    register = torch.randn(2, 8)
    code_a = torch.randn(2, 5)
    code_b = torch.randn(2, 5)
    roles_a = machine.relational_transition.binding(register, code_a)
    roles_b = machine.relational_transition.binding(register, code_b)
    assert torch.equal(roles_a, roles_b)


def test_shared_canonical_mode_applies_one_internal_state_contract_per_step() -> None:
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_shared_canonical",
        instructions=tuple(ExternalRegisterInstruction(5) for _ in range(2)),
    )
    register = torch.randn(2, 8)
    final, trace = machine.execute_chain_trace(register, machine.instructions)

    assert machine.configuration()["operator_mode"] == (
        "factorized_shared_canonical"
    )
    assert len(trace) == 2
    assert torch.isfinite(final).all()
    assert torch.allclose(trace[0].mean(dim=-1), torch.zeros(2), atol=1e-5)


def test_learned_role_binding_is_shared_and_shape_preserving() -> None:
    torch.manual_seed(915)
    binding = LearnedRegisterRoleBinding(8, 5, role_count=2)
    register = torch.randn(3, 8)
    code = torch.randn(3, 5)
    roles = binding(register, code)

    assert binding.configuration()["schema"] == (
        "neural-computer.external-register-learned-role-binding.v1"
    )
    assert roles.shape == (3, 2, 4)
    assert torch.isfinite(roles).all()
    assert roles.reshape(3, -1).shape[1] == register.shape[1]
    roles.sum().backward()
    assert binding.role_seed.grad is not None


def test_shared_role_bound_mode_returns_learned_role_trace() -> None:
    torch.manual_seed(916)
    machine = ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        interpreter_hidden=12,
        operator_mode="factorized_shared_role_bound",
        role_count=2,
        instructions=tuple(ExternalRegisterInstruction(5) for _ in range(2)),
    )
    register = torch.randn(3, 8)
    final, roles = machine.execute_chain_role_trace(register, machine.instructions)

    assert machine.configuration()["operator_mode"] == (
        "factorized_shared_role_bound"
    )
    assert machine.configuration()["role_count"] == 2
    assert len(roles) == 2
    assert all(role.shape == (3, 2, 4) for role in roles)
    assert torch.isfinite(final).all()


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
    machine.add_basis_slot()
    addressed_result = machine.execute(
        register, machine.instructions[0], basis_slot=0
    )

    assert machine.configuration()["operator_mode"] == (
        "factorized_protected_meta"
    )
    assert result.shape == register.shape
    assert torch.isfinite(result).all()
    assert torch.equal(result, addressed_result)


def test_protected_bounded_meta_operator_is_bounded_and_basis_independent() -> None:
    machine = _machine(operator_mode="factorized_protected_bounded_meta")
    register = torch.randn(3, 8)
    instruction = machine.instructions[0]
    result = machine.execute(register, instruction)
    machine.add_basis_slot()
    addressed_result = machine.execute(register, instruction, basis_slot=0)

    assert machine.configuration()["operator_mode"] == (
        "factorized_protected_bounded_meta"
    )
    assert torch.equal(result, addressed_result)
    assert torch.isfinite(result).all()
    assert (result - register).abs().max() <= 1.0


def test_external_sequence_memory_grows_without_resizing_controller() -> None:
    memory = ExternalSequenceMemory(8)
    first = memory.add_slot()
    second = memory.add_slot(torch.ones(8))

    assert first == 0
    assert second == 1
    assert memory.configuration()["slot_count"] == 2
    assert torch.equal(memory.read(1, batch_size=3, device="cpu"), torch.ones(3, 8))


def test_protected_meta_execution_accepts_external_sequence_context() -> None:
    machine = _machine(operator_mode="factorized_protected_bounded_meta")
    register = torch.randn(3, 8)
    instruction = machine.instructions[0]
    plain = machine.execute(register, instruction)
    contextual = machine.execute(
        register,
        instruction,
        meta_context=torch.ones_like(register),
    )

    assert contextual.shape == plain.shape
    assert not torch.equal(plain, contextual)


def test_external_sequence_operator_memory_grows_and_executes_opaque_slots() -> None:
    machine = _machine(operator_mode="factorized_protected_bounded_meta")
    memory = ExternalSequenceOperatorMemory(8, 5, operator_rank=2)
    slot = memory.add_slot()
    register = torch.randn(3, 8)
    instruction = machine.instructions[0]

    result = machine.execute(
        register,
        instruction,
        sequence_operator_memory=memory,
        sequence_operator_slot=slot,
    )

    assert memory.configuration()["slot_count"] == 1
    assert result.shape == register.shape
    assert torch.isfinite(result).all()


def test_external_sequence_operator_memory_learns_opaque_route_weights() -> None:
    torch.manual_seed(904)
    memory = ExternalSequenceOperatorMemory(8, 5, operator_rank=2)
    memory.add_slot()
    memory.add_slot()
    query = torch.randn(3, 5)
    weights = memory.route_weights(query)

    assert weights.shape == (3, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3))
    assert torch.isfinite(weights).all()


def test_external_sequence_operator_memory_encodes_ordered_programs() -> None:
    torch.manual_seed(905)
    memory = ExternalSequenceOperatorMemory(8, 5, operator_rank=2)
    codes = torch.randn(1, 3, 5)
    forward = memory.encode_program(codes)
    reversed_codes = memory.encode_program(codes.flip(1))

    assert forward.shape == (1, 5)
    assert reversed_codes.shape == (1, 5)
    assert not torch.equal(forward, reversed_codes)


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
