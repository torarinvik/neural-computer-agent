from __future__ import annotations

import torch

from experiments.recipe_expressibility.audit import (
    INSTRUCTION_FEATURE_WIDTH,
    SLOT_VALUES,
    LearnedRecipeInterpreter,
    _sample_slot_values,
    evaluate_arithmetic_target,
    evaluate_single_modulus_target,
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


def test_randomized_domains_preserve_the_profile_without_fixed_positions() -> None:
    generator = torch.Generator().manual_seed(12)
    samples = tuple(
        _sample_slot_values(generator, SLOT_VALUES) for _ in range(8)
    )

    assert all(sorted(sample) == sorted(SLOT_VALUES) for sample in samples)
    assert any(sample != SLOT_VALUES for sample in samples)


def test_random_batch_targets_match_the_opaque_instruction_execution() -> None:
    batch = sample_batch(
        generator=torch.Generator().manual_seed(4),
        batch_size=8,
        length=3,
        allow_parallel=True,
        randomize_domains=False,
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
        randomize_domains=False,
    )
    outputs = LearnedRecipeInterpreter(hidden=32)(
        batch.initial,
        batch.instructions,
    )

    assert len(outputs) == 2
    assert all(output.shape == (4, 6, 8) for output in outputs)


def test_single_modulus_probe_uses_the_opaque_instruction_path() -> None:
    model = LearnedRecipeInterpreter(hidden=32)

    score_m2 = evaluate_single_modulus_target(
        model,
        seed=8,
        target_slot=0,
        batches=1,
        batch_size=4,
    )
    score_m8 = evaluate_single_modulus_target(
        model,
        seed=9,
        target_slot=2,
        batches=1,
        batch_size=4,
    )
    wrong_score = evaluate_single_modulus_target(
        model,
        seed=10,
        target_slot=0,
        instruction_modulus=8,
        batches=1,
        batch_size=4,
    )

    assert 0.0 <= score_m2 <= 1.0
    assert 0.0 <= score_m8 <= 1.0
    assert 0.0 <= wrong_score <= 1.0


def test_arithmetic_generalization_probes_preserve_runtime_contract() -> None:
    model = LearnedRecipeInterpreter(hidden=32)

    scores = tuple(
        evaluate_arithmetic_target(
            model,
            seed=20 + index,
            operation=operation,
            target_slot=0,
            condition_slot=1 if operation.startswith("c") else None,
            batches=1,
            batch_size=4,
        )
        for index, operation in enumerate(("dec", "cinc", "cdec"))
    )

    assert all(0.0 <= score <= 1.0 for score in scores)
