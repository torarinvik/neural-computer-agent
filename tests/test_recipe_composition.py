from __future__ import annotations

from itertools import product

import torch

from neural_computer import (
    ExternalRecipeCompositionMemory,
    OpaqueContextRecipeCompositionMemory,
    RecipeInstruction,
    RecipeProgram,
    RecipeProgramCompositionFactors,
)

SLOT_VALUES = (2, 8)
STATES = tuple(product(range(2), range(8)))


def _fragment_a() -> RecipeProgram:
    return RecipeProgram(SLOT_VALUES, (_instruction("inc", 0, 2),))


def _instruction(
    operation: str,
    first: int,
    modulus: int | None = None,
    second: int | None = None,
) -> RecipeInstruction:
    return RecipeInstruction(operation, first, second, modulus=modulus)


def _fragment_b() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (RecipeInstruction("cinc", 1, 0, modulus=8),),
    )


def _target() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (
            _instruction("inc", 0, 2),
            _instruction("cinc", 1, 8, second=0),
        ),
    )


def _outcomes(candidate: RecipeProgram, target: RecipeProgram) -> torch.Tensor:
    return torch.tensor(
        [float(candidate.execute(state) == target.execute(state)) for state in STATES]
    )


def test_composition_memory_is_verifier_gated_provenanced_and_persistent() -> None:
    memory = ExternalRecipeCompositionMemory(SLOT_VALUES)
    left = memory.add_program(_fragment_a())
    right = memory.add_program(_fragment_b())
    memory.protect_file(left)
    memory.protect_file(right)
    target = _target()
    candidate = next(
        item
        for item in memory.composition_candidates(max_program_length=2)
        if item.program.digest() == target.digest()
    )

    before = memory.digest()
    rejected = memory.admit_verified_composition(
        candidate,
        torch.zeros(len(STATES)),
        threshold=1.0,
        min_observations=len(STATES),
        min_stable_observations=len(STATES),
    )
    assert not rejected.accepted
    assert memory.file_count == 2
    assert memory.digest() == before

    accepted = memory.admit_verified_composition(
        candidate,
        _outcomes(candidate.program, target),
        threshold=1.0,
        min_observations=len(STATES),
        min_stable_observations=len(STATES),
        protect=True,
    )
    assert accepted.accepted
    assert accepted.slot == 2
    assert memory.provenance(2) == candidate.factors
    assert memory.is_file_protected(2)
    assert all(memory.execute(2, state) == target.execute(state) for state in STATES)

    restored = ExternalRecipeCompositionMemory.from_payload(memory.payload())
    assert restored.digest() == memory.digest()
    assert restored.provenance(2) == candidate.factors
    assert restored.program(2).digest() == target.digest()


def test_composition_policy_reuses_source_factors_without_candidate_rows() -> None:
    left = _fragment_a().digest()
    right = _fragment_b().digest()
    reverse = RecipeProgramCompositionFactors(right, left, "prepend")
    append = RecipeProgramCompositionFactors(left, right, "append")
    policy = OpaqueContextRecipeCompositionMemory(
        exploration_floor=0.2,
        shared_prior_weight=0.25,
    )
    policy.record("context-a", append, 1.0)
    policy.record("context-a", reverse, 0.0)

    probabilities = policy.proposal_probabilities("context-new", (append, reverse))
    assert float(probabilities[0]) > float(probabilities[1])
    assert bool(torch.all(probabilities >= 0.1))

    restored = OpaqueContextRecipeCompositionMemory.from_payload(policy.payload())
    assert restored.digest() == policy.digest()
    assert torch.equal(
        restored.proposal_probabilities("context-new", (append, reverse)),
        probabilities,
    )
    assert "outcomes" not in policy.payload()
