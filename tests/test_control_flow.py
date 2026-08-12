from __future__ import annotations

import pytest
import torch

from neural_computer import (
    ControlFlowCompositionSearch,
    ControlFlowFrontierState,
    ControlFlowInstruction,
    ControlFlowOutcomeSearch,
    ControlFlowProgram,
    ControlFlowProgramFrontier,
    ControlFlowProgramMemory,
    compose_control_flow_programs,
    iter_control_flow_programs,
)


def _transfer_program() -> ControlFlowProgram:
    return ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=4),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )


def test_control_flow_executes_data_dependent_loop_with_opaque_counters() -> None:
    program = _transfer_program()

    for amount in (0, 1, 4, 11):
        result = program.execute((amount, 0), max_steps=100)

        assert result.status == "halted"
        assert result.counters == (0, amount)


def test_control_flow_fails_closed_on_resource_bounds() -> None:
    program = _transfer_program()

    exhausted = program.execute((4, 0), max_steps=3)
    assert exhausted.status == "step_budget_exhausted"
    assert exhausted.counters == (3, 1)

    with pytest.raises(ValueError, match="must end with halt"):
        ControlFlowProgram(
            2,
            (ControlFlowInstruction("jump", target=0),),
        ).validate()


def test_control_flow_memory_admission_is_scalar_gated_and_persistent() -> None:
    program = _transfer_program()
    memory = ControlFlowProgramMemory(2)
    before = memory.digest()

    rejected = memory.admit_verified(
        program,
        [1.0, 0.0],
        min_observations=2,
        min_stable_observations=2,
    )
    assert not rejected.accepted
    assert memory.file_count == 0
    assert memory.digest() == before

    accepted = memory.admit_verified(
        program,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=2,
        protect=True,
    )
    assert accepted.accepted and accepted.slot == 0
    assert memory.is_file_protected(0)
    restored = ControlFlowProgramMemory.from_payload(memory.payload())
    assert restored.digest() == memory.digest()
    assert restored.program(0).execute((7, 0), max_steps=100).counters == (0, 7)


def test_control_flow_composition_relocates_internal_and_terminal_jumps() -> None:
    first = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=3),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )
    second = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump_if_zero", counter=1, target=3),
            ControlFlowInstruction("dec", counter=1),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )

    composed = compose_control_flow_programs((first, second))

    assert composed.instructions[0].target == 3
    assert composed.instructions[3].op == "jump_if_zero"
    assert composed.instructions[3].target == 6
    assert composed.instructions[6].op == "halt"
    for initial in ((0, 0), (2, 0), (0, 3), (4, 2)):
        first_result = first.execute(initial, max_steps=100)
        second_result = second.execute(first_result.counters, max_steps=100)
        composed_result = composed.execute(initial, max_steps=200)
        assert composed_result.status == "halted"
        assert composed_result.counters == second_result.counters


def test_control_flow_composition_rejects_ambiguous_or_incompatible_files() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compose_control_flow_programs(())
    with pytest.raises(ValueError, match="internal halt"):
        compose_control_flow_programs(
            (
                ControlFlowProgram(
                    2,
                    (
                        ControlFlowInstruction("halt"),
                        ControlFlowInstruction("halt"),
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="common counter width"):
        compose_control_flow_programs(
            (
                ControlFlowProgram(2, (ControlFlowInstruction("halt"),)),
                ControlFlowProgram(3, (ControlFlowInstruction("halt"),)),
            )
        )


def test_control_flow_memory_composes_and_verifies_existing_files() -> None:
    memory = ControlFlowProgramMemory(2)
    first = memory.add_program(
        ControlFlowProgram(
            2,
            (
                ControlFlowInstruction("inc", counter=0),
                ControlFlowInstruction("halt"),
            ),
        ),
        protect=True,
    )
    second = memory.add_program(
        ControlFlowProgram(
            2,
            (
                ControlFlowInstruction("inc", counter=1),
                ControlFlowInstruction("halt"),
            ),
        ),
        protect=True,
    )
    candidate = memory.compose((first, second))
    assert candidate.execute((0, 0), max_steps=16).counters == (1, 1)
    receipt = memory.compose_verified(
        (first, second),
        (1.0, 1.0, 1.0),
        min_observations=3,
        min_stable_observations=2,
        protect=True,
    )
    assert receipt.accepted and receipt.slot == 2
    assert memory.is_file_protected(2)


def test_control_flow_composition_search_discovers_opaque_file_order() -> None:
    first = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )
    second = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=2),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("halt"),
        ),
    )
    decoy = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("halt"),
        ),
    )
    memory = ControlFlowProgramMemory(2)
    memory.add_program(first, protect=True)
    memory.add_program(second, protect=True)
    memory.add_program(decoy, protect=True)
    search = ControlFlowCompositionSearch(memory, min_program_length=2, max_program_length=2)
    state = search.initial_state()
    target = compose_control_flow_programs((first, second))
    initial_states = ((0, 0), (1, 0), (4, 0))

    accepted = None
    for _ in range(256):
        proposal = search.propose_exhaustive(state, scope="opaque-composition")
        outcomes = tuple(
            float(
                proposal.program.execute(initial, max_steps=32).status
                == target.execute(initial, max_steps=32).status
                and proposal.program.execute(initial, max_steps=32).counters
                == target.execute(initial, max_steps=32).counters
            )
            for initial in initial_states
        )
        feedback = search.record_outcomes(
            state,
            proposal,
            outcomes,
            min_observations=len(initial_states),
            min_stable_observations=len(initial_states),
        )
        state = feedback.state
        if feedback.receipt.accepted:
            accepted = feedback
            break

    assert accepted is not None
    assert accepted.proposal.slots == (0, 1)
    assert accepted.proposal.program.digest() == target.digest()
    assert accepted.receipt.slot is None
    assert "outcomes" not in state.payload()
    restored = type(state).from_payload(state.payload())
    assert restored == state

    memory.add_program(target)
    with pytest.raises(ValueError, match="memory changed"):
        search.propose_exhaustive(state, scope="opaque-composition")


def test_control_flow_outcome_search_can_learn_one_opaque_instruction_edit() -> None:
    target = _transfer_program()
    parent = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=4),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )
    search = ControlFlowOutcomeSearch()
    state = search.initial_state()
    generator = torch.Generator().manual_seed(23)

    accepted = None
    for _ in range(256):
        proposal = search.propose(
            state,
            parent,
            generator=generator,
            scope="opaque-context-a",
        )
        quality = [
            float(
                proposal.program.execute((amount, 0), max_steps=100).status == "halted"
                and proposal.program.execute((amount, 0), max_steps=100).counters
                == target.execute((amount, 0), max_steps=100).counters
            )
            for amount in (0, 1, 3, 7)
        ]
        feedback = search.record_outcomes(
            state,
            proposal,
            quality,
            min_observations=4,
            min_stable_observations=2,
        )
        state = feedback.state
        if feedback.accepted:
            accepted = feedback
            break

    assert accepted is not None
    assert accepted.proposal.program.digest() == target.digest()
    assert "outcomes" not in state.payload()


def test_control_flow_growth_promotes_loop_file_with_reversed_input_order() -> None:
    from experiments.recipe_expressibility.control_flow_program_growth import run

    report = run((17,))

    assert report["promoted"]
    assert all(
        bool(item["promoted"])
        for item in report["reports"]
    )
    assert all(
        bool(item["gates"]["missing_evidence_no_write"])
        and bool(item["gates"]["reward_shuffled_rejected"])
        for item in report["reports"]
    )


def test_control_flow_program_enumerator_has_a_certifiable_finite_bound() -> None:
    programs = tuple(
        iter_control_flow_programs(counter_count=2, min_length=1, max_length=1)
    )

    assert len(programs) == 1
    assert programs[0].instructions[-1].op == "halt"


def test_control_flow_frontier_acquires_a_two_edit_program_and_reloads() -> None:
    source = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=3),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )
    target = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=4),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )
    frontier = ControlFlowProgramFrontier(
        2,
        beam_width=32,
        max_depth=8,
        max_program_length=8,
        minimum_quality=0.25,
    )
    state = frontier.initial_state(source)
    generator = torch.Generator().manual_seed(10_017)
    accepted = None
    for _ in range(500):
        proposal = frontier.propose(state, generator=generator)
        outcomes = []
        for amount in range(8):
            actual = proposal.program.execute((amount, 0), max_steps=128)
            expected = target.execute((amount, 0), max_steps=128)
            outcomes.append(
                float(actual.status == "halted" and actual.counters == expected.counters)
            )
        feedback = frontier.record_outcomes(
            state,
            proposal,
            outcomes,
            min_observations=8,
            min_stable_observations=8,
        )
        state = feedback.state
        if feedback.accepted:
            accepted = feedback
            break

    assert accepted is not None
    assert accepted.proposal.program.digest() != source.digest()
    assert accepted.quality == 1.0
    restored = ControlFlowFrontierState.from_payload(state.payload())
    assert restored.digest() == state.digest()
    assert "outcomes" not in state.payload()


def test_control_flow_induction_promotes_from_scratch_loop_search() -> None:
    from experiments.recipe_expressibility.control_flow_program_induction import run

    report = run((17,))

    assert report["promoted"]
    assert all(
        item["search"]["status"] == "expressible"
        and item["gates"]["budget_exhaustion_is_not_inexpressible"]
        for item in report["reports"]
    )


def test_control_flow_frontier_growth_keeps_fresh_exhaustion_distinct() -> None:
    from experiments.recipe_expressibility.control_flow_frontier_growth import run

    report = run((17,))

    assert report["promoted"]
    assert not report["sample_efficiency_transfer_promoted"]
    assert all(
        item["fresh_search"]["status"] == "frontier_exhausted"
        and item["gates"]["termination_is_not_inexpressible"]
        for item in report["reports"]
    )
