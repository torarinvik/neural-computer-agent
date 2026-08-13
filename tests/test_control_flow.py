from __future__ import annotations

import pytest
import torch

from neural_computer import (
    ControlFlowCompositionSearch,
    ControlFlowFrontierProposalFactors,
    ControlFlowFrontierProposalMemory,
    ControlFlowFrontierState,
    ControlFlowInstruction,
    ControlFlowOutcomeSearch,
    ControlFlowProgram,
    ControlFlowProgramFrontier,
    ControlFlowProgramFrontierGrowth,
    ControlFlowProgramMemory,
    ControlFlowSpliceSearch,
    compose_control_flow_programs,
    delete_control_flow_instruction,
    insert_control_flow_instruction,
    iter_control_flow_programs,
    splice_control_flow_program,
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


def test_structural_control_flow_insertion_relocates_jump_targets() -> None:
    source = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=3),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )
    expanded = insert_control_flow_instruction(
        source,
        2,
        ControlFlowInstruction("inc", counter=1),
    )

    assert expanded.instructions[0].target == 4
    assert expanded.execute((3, 0), max_steps=100).counters == (0, 3)
    assert delete_control_flow_instruction(expanded, 2).digest() == source.digest()


def test_structural_control_flow_deletion_rejects_dangling_jump_targets() -> None:
    source = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump", target=1),
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )

    with pytest.raises(ValueError, match="jump target"):
        delete_control_flow_instruction(source, 1)


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


def test_control_flow_splice_rebases_parent_and_fragment_jumps() -> None:
    parent = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=3),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )
    fragment = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("jump_if_zero", counter=1, target=4),
            ControlFlowInstruction("dec", counter=1),
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )

    spliced = splice_control_flow_program(parent, 3, fragment)

    assert spliced.instructions[0].target == 3
    assert spliced.instructions[3].op == "jump_if_zero"
    assert spliced.instructions[3].target == 7
    assert spliced.instructions[6].target == 3
    assert spliced.instructions[2].op == "jump"
    assert spliced.instructions[2].target == 0
    for initial in ((0, 0), (3, 0), (0, 2), (4, 3)):
        parent_result = parent.execute(initial, max_steps=100)
        fragment_result = fragment.execute(parent_result.counters, max_steps=100)
        spliced_result = spliced.execute(initial, max_steps=300)
        assert spliced_result.status == "halted"
        assert spliced_result.counters == fragment_result.counters


def test_control_flow_splice_rejects_invalid_boundaries_and_abis() -> None:
    parent = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )
    fragment = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("halt"),
        ),
    )

    with pytest.raises(ValueError, match="position"):
        splice_control_flow_program(parent, 2, fragment)
    with pytest.raises(ValueError, match="position"):
        splice_control_flow_program(parent, -1, fragment)
    with pytest.raises(ValueError, match="common counter width"):
        splice_control_flow_program(parent, 0, ControlFlowProgram(3, fragment.instructions))
    with pytest.raises(ValueError, match="non-terminal body"):
        splice_control_flow_program(parent, 0, ControlFlowProgram(2, (ControlFlowInstruction("halt"),)))


def test_control_flow_memory_splices_verifies_and_reloads_existing_files() -> None:
    memory = ControlFlowProgramMemory(2)
    parent_slot = memory.add_program(
        ControlFlowProgram(
            2,
            (
                ControlFlowInstruction("inc", counter=0),
                ControlFlowInstruction("halt"),
            ),
        ),
        protect=True,
    )
    fragment_slot = memory.add_program(
        ControlFlowProgram(
            2,
            (
                ControlFlowInstruction("inc", counter=1),
                ControlFlowInstruction("halt"),
            ),
        ),
        protect=True,
    )

    receipt = memory.splice_verified(
        parent_slot,
        0,
        fragment_slot,
        (1.0, 1.0, 1.0),
        min_observations=3,
        min_stable_observations=2,
        protect=True,
    )

    assert receipt.accepted and receipt.slot == 2
    assert memory.is_file_protected(receipt.slot)
    assert memory.program(receipt.slot).execute((0, 0), max_steps=16).counters == (1, 1)
    restored = ControlFlowProgramMemory.from_payload(memory.payload())
    assert restored.digest() == memory.digest()


def test_control_flow_splice_search_discovers_opaque_fragment_insertion() -> None:
    parent = _transfer_program()
    fragment = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )
    decoy = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )
    memory = ControlFlowProgramMemory(2)
    parent_slot = memory.add_program(parent, protect=True)
    fragment_slot = memory.add_program(fragment, protect=True)
    memory.add_program(decoy, protect=True)
    target = memory.splice(parent_slot, len(parent.instructions) - 1, fragment_slot)
    search = ControlFlowSpliceSearch(memory, min_program_length=2, max_program_length=8)
    state = search.initial_state()
    initial_states = ((0, 0), (1, 0), (4, 0), (0, 2))

    accepted = None
    for _ in range(256):
        proposal = search.propose_exhaustive(state, scope="opaque-splice")
        outcomes = tuple(
            float(
                proposal.program.execute(initial, max_steps=300).status
                == target.execute(initial, max_steps=300).status
                and proposal.program.execute(initial, max_steps=300).counters
                == target.execute(initial, max_steps=300).counters
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
    assert accepted.proposal.parent_slot == parent_slot
    assert accepted.proposal.position == len(parent.instructions) - 1
    assert accepted.proposal.fragment_slot == fragment_slot
    assert accepted.proposal.program.digest() == target.digest()
    assert "outcomes" not in state.payload()
    assert type(state).from_payload(state.payload()) == state
    memory.add_program(target)
    with pytest.raises(ValueError, match="memory changed"):
        search.propose_exhaustive(state, scope="opaque-splice")


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


def test_control_flow_factorized_credit_uses_relative_position_without_candidate_rows() -> None:
    source = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )
    target = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("halt"),
        ),
    )
    factors = ControlFlowProgramFrontier.proposal_factors(source, 1, target)

    assert factors.primary_position == 0
    assert len(factors.instruction_digests) == 1

    memory = ControlFlowFrontierProposalMemory(
        exploration_floor=0.2,
        shared_prior_weight=0.25,
    )
    memory.record("context-a", 1.0, factors=factors)
    alternative = ControlFlowFrontierProposalFactors(
        operator_index=1,
        primary_position=0,
        instruction_digests=("1" * 64,),
    )
    memory.record("context-a", 0.0, factors=alternative)
    probabilities = memory.proposal_probabilities("context-a", (factors, alternative))

    assert float(probabilities[0]) > float(probabilities[1])
    assert bool(torch.all(probabilities >= 0.1))
    payload_text = str(memory.payload())
    assert target.digest() not in payload_text

    restored = ControlFlowFrontierProposalMemory.from_payload(memory.payload())
    assert restored.digest() == memory.digest()
    assert torch.equal(
        restored.proposal_probabilities("context-a", (factors, alternative)),
        probabilities,
    )


def test_control_flow_factorized_frontier_preserves_generic_runtime_boundary() -> None:
    source = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )
    policy = ControlFlowFrontierProposalMemory(exploration_floor=0.05)
    frontier = ControlFlowProgramFrontier(
        2,
        max_program_length=3,
        proposal_policy=policy,
    )
    state = frontier.initial_state(source, root_quality=1.0)
    proposal = frontier.propose(
        state,
        generator=torch.Generator().manual_seed(77),
        context="opaque-context",
    )

    assert proposal.context == "opaque-context"
    assert proposal.factors is not None
    feedback = frontier.record_outcomes(state, proposal, (0.0, 0.0))
    assert feedback.state.evaluations == 1
    assert feedback.state.accepted == 0
    assert policy.payload()["configuration"]["credit"] == (
        "scalar_factor_aggregate_without_candidate_rows_v1"
    )


def test_control_flow_fragment_splice_audit_promotes_behavior_only_assembly() -> None:
    from experiments.recipe_expressibility.control_flow_fragment_splice import run

    report = run((31,))

    assert report["status"] == "promoted_outcome_only_reusable_fragment_splicing"
    assert all(
        bool(item["promoted"])
        and all(
            float(stage["heldout_accuracy"]) == 1.0
            for stage in item["stage_reports"]
        )
        for item in report["warm_reports"]
    )
    assert all(item["not_promoted"] for item in report["shuffled_reports"])


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


def test_adaptive_frontier_growth_preserves_credit_and_rejects_stale_proposals() -> None:
    root = ControlFlowProgram(
        2,
        (
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )
    growth = ControlFlowProgramFrontierGrowth(
        2,
        initial_horizon=2,
        maximum_horizon=4,
        beam_width=16,
        max_depth=8,
    )
    state = growth.initial_state(root, root_quality=1.0)
    before_reject = state.digest()
    rejected_receipt, rejected_state = growth.expand_horizon_verified(
        state,
        lambda _: False,
    )
    assert not rejected_receipt.accepted
    assert rejected_state.digest() == before_reject

    expanded_receipt, state = growth.expand_horizon_verified(state, lambda _: True)
    assert expanded_receipt.accepted
    assert state.horizon == 3

    generator = torch.Generator().manual_seed(9017)
    qualified = None
    stale_proposal = None
    for _ in range(256):
        proposal = growth.propose(state, generator=generator)
        if len(proposal.proposal.program.instructions) == 3:
            feedback = growth.record_outcomes(
                state,
                proposal,
                (1.0, 1.0),
                min_observations=2,
                min_stable_observations=2,
            )
            state = feedback.state
            if feedback.accepted:
                qualified = proposal.proposal.program
                break
        else:
            state = growth.record_outcomes(
                state,
                proposal,
                (0.0, 0.0),
                min_observations=2,
                min_stable_observations=2,
            ).state
    assert qualified is not None
    old_evaluations = state.frontier.evaluations
    stale_proposal = growth.propose(state, generator=generator)
    promoted_receipt, state = growth.promote_root_verified(
        state,
        qualified,
        lambda candidate: candidate.frontier.root_digest == qualified.digest(),
    )
    assert promoted_receipt.accepted
    assert state.frontier.root_digest == qualified.digest()
    assert state.rung == 1
    assert state.frontier.evaluations == old_evaluations
    assert len(state.qualified_programs) == 2
    with pytest.raises(ValueError, match="rung"):
        growth.record_outcomes(state, stale_proposal, (1.0, 1.0))

    expanded_receipt, expanded_state = growth.expand_horizon_verified(state, lambda _: True)
    assert expanded_receipt.accepted
    assert expanded_state.horizon == 4
    with pytest.raises(ValueError, match="stale"):
        growth.record_outcomes(expanded_state, stale_proposal, (1.0, 1.0))

    restored = type(expanded_state).from_payload(expanded_state.payload())
    assert restored.digest() == expanded_state.digest()
    assert "outcomes" not in expanded_state.payload()


def test_adaptive_control_flow_growth_promotes_curriculum_and_retention() -> None:
    from experiments.recipe_expressibility.control_flow_adaptive_growth import run

    report = run((17,))

    assert report["status"] == "promoted_replay_free_adaptive_control_flow_growth"
    assert report["gates"] == {
        "positive_arms_promoted": True,
        "shuffled_feedback_not_promoted": True,
    }
    assert all(
        bool(item["gates"]["all_prior_programs_retained"])
        and bool(item["gates"]["horizon_grew_one_step_at_a_time"])
        for item in report["positive_reports"]
    )
    assert report["fresh_reports"][0]["stage_reports"][-1]["found"] is False


def test_adaptive_loop_growth_promotes_noncommuting_programs() -> None:
    from experiments.recipe_expressibility.control_flow_adaptive_loop_growth import run

    report = run((17,))

    assert report["status"] == "promoted_replay_free_adaptive_loop_growth"
    assert report["gates"] == {
        "positive_arms_promoted": True,
        "fresh_final_rung_not_found": True,
        "shuffled_feedback_not_promoted": True,
    }
    assert all(
        [item["horizon"] for item in arm["stage_reports"]] == [5, 6, 7]
        and all(item["retention_accuracy"] == 1.0 for item in arm["stage_reports"])
        for arm in report["positive_reports"]
    )
