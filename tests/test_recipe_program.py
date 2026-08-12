from __future__ import annotations

from itertools import product

import torch

from neural_computer import (
    ExternalRecipeProgramMemory,
    OpaqueContextRecipeProposalMemory,
    OutcomeOnlyRecipeSequenceSearch,
    RecipeBasis,
    RecipeInstruction,
    RecipeProgram,
    RecipeProgramSearchState,
)


def _basis() -> RecipeBasis:
    return RecipeBasis(slot_count=2, slot_values=(2, 2))


def _source() -> RecipeProgram:
    return RecipeProgram(
        (2, 2),
        (RecipeInstruction("inc", 0, modulus=2),),
    )


def _target() -> RecipeProgram:
    return RecipeProgram(
        (2, 2),
        (
            RecipeInstruction("inc", 0, modulus=2),
            RecipeInstruction("inc", 1, modulus=2),
        ),
    )


def _states() -> tuple[tuple[int, ...], ...]:
    return tuple(product(range(2), repeat=2))


def _outcomes(candidate: RecipeProgram, target: RecipeProgram) -> torch.Tensor:
    return torch.tensor(
        [float(candidate.execute(state) == target.execute(state)) for state in _states()]
    )


def test_recipe_program_round_trip_is_checksum_verified() -> None:
    program = _target()
    restored = RecipeProgram.from_payload(program.payload())

    assert restored.digest() == program.digest()
    assert restored.execute((1, 0)) == (0, 1)


def test_outcome_only_search_discovers_and_admits_toggle_sequence() -> None:
    basis = _basis()
    source = _source()
    target = _target()
    search = OutcomeOnlyRecipeSequenceSearch(
        basis,
        max_program_length=2,
    )
    state = search.initial_state()
    found = None
    for _ in range(256):
        proposal = search.propose_exhaustive(state, source)
        feedback = search.record_outcomes(
            state,
            proposal,
            _outcomes(proposal.program, target),
            threshold=1.0,
            min_observations=4,
            min_stable_observations=4,
        )
        state = feedback.state
        if feedback.receipt.accepted:
            found = proposal.program
            break

    assert found is not None
    assert found.digest() != source.digest()
    assert all(found.execute(state) == target.execute(state) for state in _states())
    restored_state = RecipeProgramSearchState.from_payload(state.payload())
    assert restored_state.proposals == state.proposals
    assert "outcomes" not in state.payload()

    memory = ExternalRecipeProgramMemory((2, 2))
    source_slot = memory.add_program(source)
    memory.protect_file(source_slot)
    receipt = memory.admit_verified_program(
        found,
        _outcomes(found, target),
        threshold=1.0,
        min_observations=4,
        min_stable_observations=4,
        protect=True,
    )

    assert receipt.accepted
    assert memory.file_count == 2
    assert memory.is_file_protected(source_slot)
    assert memory.is_file_protected(receipt.slot)
    assert all(memory.execute(receipt.slot, state) == target.execute(state) for state in _states())

    reloaded = ExternalRecipeProgramMemory.from_payload(memory.payload())
    assert reloaded.digest() == memory.digest()
    assert reloaded.program(receipt.slot).digest() == found.digest()


def test_failed_recipe_admission_is_a_noop() -> None:
    memory = ExternalRecipeProgramMemory((2, 2))
    source_slot = memory.add_program(_source())
    memory.protect_file(source_slot)
    before = memory.digest()
    wrong = RecipeProgram(
        (2, 2),
        (
            RecipeInstruction("dec", 0, modulus=2),
            RecipeInstruction("inc", 1, modulus=2),
        ),
    )
    receipt = memory.admit_verified_program(
        wrong,
        torch.zeros(4),
        threshold=1.0,
        min_observations=4,
        min_stable_observations=4,
        protect=True,
    )

    assert not receipt.accepted
    assert memory.file_count == 1
    assert memory.digest() == before


def test_candidate_history_is_scoped_while_scalar_priors_can_transfer() -> None:
    search = OutcomeOnlyRecipeSequenceSearch(_basis(), max_program_length=2)
    source = _source()
    state = search.initial_state()
    first = search.propose_exhaustive(state, source, scope="opaque-a")
    state = search.record_outcomes(
        state,
        first,
        torch.zeros(4),
    ).state

    second = search.propose_exhaustive(state, source, scope="opaque-b")

    assert second.program.digest() == first.program.digest()
    assert second.scope == "opaque-b"
    assert state.proposals == 1


def test_context_proposal_credit_is_opaque_persistent_and_has_an_exploration_floor() -> None:
    first = "0" * 64
    second = "1" * 64
    memory = OpaqueContextRecipeProposalMemory(
        exploration_floor=0.2,
        global_prior_weight=0.0,
    )
    memory.record("context-a", first, 1.0)
    memory.record("context-a", second, 0.0)
    memory.record("context-b", second, 1.0)

    context_a = memory.proposal_probabilities("context-a", (first, second))
    context_b = memory.proposal_probabilities("context-b", (first, second))
    unseen = memory.proposal_probabilities("context-new", (first, second))

    assert float(context_a[0]) > float(context_a[1])
    assert float(context_b[1]) > float(context_b[0])
    assert bool(torch.all(unseen >= 0.1))

    restored = OpaqueContextRecipeProposalMemory.from_payload(memory.payload())
    assert restored.digest() == memory.digest()
    assert torch.equal(
        restored.proposal_probabilities("context-a", (first, second)),
        context_a,
    )
    assert "outcomes" not in memory.payload()
