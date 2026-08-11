from __future__ import annotations

import torch

from experiments.recipe_expressibility.audit import (
    INSTRUCTION_FEATURE_WIDTH,
    LearnedRecipeInterpreter,
    instruction_features,
    modulus_boundary,
    sample_batch,
)
from neural_computer.recipe_basis import RecipeInstruction


def test_instruction_encoding_has_fixed_width_for_atomic_and_parallel() -> None:
    atomic = instruction_features(RecipeInstruction("inc", 0, modulus=2))
    parallel = instruction_features(
        RecipeInstruction(
            "parallel",
            children=(
                RecipeInstruction("inc", 0, modulus=2),
                RecipeInstruction("dec", 1, modulus=2),
            ),
        )
    )

    assert atomic.shape == (INSTRUCTION_FEATURE_WIDTH,)
    assert parallel.shape == (INSTRUCTION_FEATURE_WIDTH,)
    assert not torch.equal(atomic, parallel)


def test_modulus_boundary_exposes_the_legacy_global_mismatch() -> None:
    result = modulus_boundary()

    assert result["legacy_match_rates"] == [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    assert result["explicit_match_rates"] == [1.0] * 6


def test_random_batch_targets_match_the_opaque_instruction_execution() -> None:
    batch = sample_batch(
        generator=torch.Generator().manual_seed(4),
        batch_size=8,
        length=3,
        allow_parallel=True,
    )
    assert batch.initial.shape == (8, 6)
    assert batch.instructions.shape == (8, 3, INSTRUCTION_FEATURE_WIDTH)
    assert batch.targets.shape == (8, 3, 6)
    assert bool(batch.targets.ge(0).all())
    assert bool(batch.targets.lt(8).all())
    assert bool(batch.initial[:, 0:2].lt(2).all())


def test_interpreter_preserves_runtime_shapes() -> None:
    batch = sample_batch(
        generator=torch.Generator().manual_seed(5),
        batch_size=4,
        length=2,
        allow_parallel=True,
    )
    outputs = LearnedRecipeInterpreter(hidden=32)(
        batch.initial,
        batch.instructions,
    )

    assert len(outputs) == 2
    assert all(output.shape == (4, 6, 8) for output in outputs)
