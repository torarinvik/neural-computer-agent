from __future__ import annotations

from itertools import product

import pytest

from neural_computer.recipe_basis import (
    RecipeBasis,
    RecipeInstruction,
    apply_sequence,
    paired_increment_target,
)


def _pair_states() -> tuple[tuple[int, ...], ...]:
    return tuple(
        (first, second, 0, 1, 0, 1)
        for first, second in product(range(8), repeat=2)
    )


def test_baseline_fail_closed_reports_atomic_gap() -> None:
    target = paired_increment_target(0, 1)
    result = RecipeBasis(allow_parallel=False).expressibility_probe(
        target,
        states=_pair_states(),
    )

    assert result.status == "inexpressible"
    assert result.instruction is None
    assert result.checked_candidates > 0
    assert "outside" in result.reason


def test_parallel_composition_closes_generic_pair_effect() -> None:
    target = paired_increment_target(0, 1, modulus=8)
    result = RecipeBasis(allow_parallel=True).expressibility_probe(
        target,
        states=_pair_states(),
    )

    assert result.status == "expressible"
    assert result.instruction is not None
    assert result.instruction.op == "parallel"
    assert result.instruction.apply((0, 1, 0, 1, 0, 1), values=8) == (
        1,
        2,
        0,
        1,
        0,
        1,
    )


def test_parallel_requires_disjoint_writes() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        RecipeInstruction(
            "parallel",
            children=(
                RecipeInstruction("inc", 0, modulus=8),
                RecipeInstruction("dec", 0, modulus=8),
            ),
        ).validate(slot_count=6)


def test_two_valued_toggle_is_an_existing_increment_sequence() -> None:
    values = (2, 2, 8, 8, 8, 8)
    state = (1, 0, 3, 4, 5, 6)

    result = apply_sequence(
        (
            RecipeInstruction("inc", 0, modulus=2),
            RecipeInstruction("inc", 1, modulus=2),
        ),
        state,
        values=values,
    )

    assert result == (0, 1, 3, 4, 5, 6)


def test_sequence_probe_finds_two_increment_toggle_without_pair_primitive() -> None:
    values = (2, 2, 8, 8, 8, 8)
    states = tuple(
        (first, second, 0, 1, 0, 1)
        for first, second in product(range(2), repeat=2)
    )
    result = RecipeBasis(slot_values=values).sequence_probe(
        lambda state: (
            (state[0] + 1) % 2,
            (state[1] + 1) % 2,
            *state[2:],
        ),
        max_length=2,
        states=states,
    )

    assert result.status == "expressible"
    assert result.instructions is not None
    assert tuple(instruction.op for instruction in result.instructions) == (
        "inc",
        "inc",
    )
    assert tuple(instruction.first for instruction in result.instructions) == (0, 1)


def test_sequence_probe_distinguishes_bounded_failure_from_search_budget() -> None:
    values = (2, 2, 8, 8, 8, 8)
    states = ((0, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0))
    target = lambda state: (
        (state[0] + 1) % 2,
        (state[1] + 1) % 2,
        *state[2:],
    )

    bounded = RecipeBasis(slot_values=values).sequence_probe(
        target, max_length=1, states=states
    )
    limited = RecipeBasis(slot_values=values).sequence_probe(
        target, max_length=2, states=states, max_expansions=1
    )

    assert bounded.status == "inexpressible"
    assert limited.status == "budget_exhausted"
    assert limited.checked_candidates == 1


def test_arithmetic_rejects_a_hidden_global_modulus() -> None:
    with pytest.raises(ValueError, match="must match"):
        RecipeInstruction("inc", 0, modulus=8).apply(
            (1, 0, 3, 4, 5, 6),
            values=(2, 2, 8, 8, 8, 8),
        )


def test_all_arithmetic_variants_honor_the_explicit_modulus() -> None:
    values = (2, 2, 8, 8, 8, 8)

    assert RecipeInstruction("dec", 0, modulus=2).apply(
        (0, 1, 3, 4, 5, 6), values=values
    )[0] == 1
    assert RecipeInstruction("cinc", 0, 1, modulus=2).apply(
        (0, 1, 3, 4, 5, 6), values=values
    )[0] == 1
    assert RecipeInstruction("cinc", 0, 1, modulus=2).apply(
        (0, 0, 3, 4, 5, 6), values=values
    )[0] == 0
    assert RecipeInstruction("cdec", 0, 1, modulus=2).apply(
        (0, 1, 3, 4, 5, 6), values=values
    )[0] == 1


def test_modulus_is_observable_in_basis_candidates() -> None:
    basis = RecipeBasis(slot_values=(2, 2, 8, 8, 8, 8))
    candidates = basis.atomic_candidates()

    assert RecipeInstruction("inc", 0, modulus=2) in candidates
    assert RecipeInstruction("inc", 2, modulus=8) in candidates


def test_basis_configuration_exposes_atomicity_and_no_task_names() -> None:
    configuration = RecipeBasis(allow_parallel=True).configuration()

    assert configuration["schema"] == "neural-computer.recipe-basis.v2"
    assert configuration["atomicity"] == "one_instruction_one_verifier_step_v1"
    assert "pair" not in repr(configuration).lower()
