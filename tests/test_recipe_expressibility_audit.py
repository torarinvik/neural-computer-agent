from __future__ import annotations

import torch

from experiments.recipe_expressibility.audit import (
    INSTRUCTION_FEATURE_WIDTH,
    LearnedRecipeInterpreter,
    instruction_features,
    sample_batch,
)
from neural_computer.recipe_basis import RecipeInstruction


def test_instruction_encoding_has_fixed_width_for_atomic_and_parallel() -> None:
    atomic = instruction_features(RecipeInstruction("inc", 0))
    parallel = instruction_features(
        RecipeInstruction(
            "parallel",
            children=(
                RecipeInstruction("inc", 0),
                RecipeInstruction("dec", 1),
            ),
        )
    )

    assert atomic.shape == (INSTRUCTION_FEATURE_WIDTH,)
    assert parallel.shape == (INSTRUCTION_FEATURE_WIDTH,)
    assert not torch.equal(atomic, parallel)


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
