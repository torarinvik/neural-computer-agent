from __future__ import annotations

from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    generate_runtime_program_grammar,
)


def test_runtime_program_generation_is_deterministic_and_distinct() -> None:
    first = generate_runtime_program_grammar(seed=1739, count=5, depth=4)
    second = generate_runtime_program_grammar(seed=1739, count=5, depth=4)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5


def test_runtime_program_generation_rejects_invalid_budget() -> None:
    try:
        generate_runtime_program_grammar(seed=1739, count=1, depth=5)
    except ValueError as error:
        assert "depth" in str(error)
    else:
        raise AssertionError("depth five should be rejected")
