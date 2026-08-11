from __future__ import annotations

from itertools import product

import pytest

from neural_computer.recipe_basis import (
    RecipeBasis,
    RecipeInstruction,
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
    target = paired_increment_target(0, 1)
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
                RecipeInstruction("inc", 0),
                RecipeInstruction("dec", 0),
            ),
        ).validate(slot_count=6)


def test_basis_configuration_exposes_atomicity_and_no_task_names() -> None:
    configuration = RecipeBasis(allow_parallel=True).configuration()

    assert configuration["schema"] == "neural-computer.recipe-basis.v1"
    assert configuration["atomicity"] == "one_instruction_one_verifier_step_v1"
    assert "pair" not in repr(configuration).lower()
