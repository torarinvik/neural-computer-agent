from __future__ import annotations

import torch

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
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
        generate_runtime_program_grammar(seed=1739, count=1, depth=9)
    except ValueError as error:
        assert "depth" in str(error)
    else:
        raise AssertionError("depth nine should be rejected")


def test_runtime_generated_program_renderer_supports_eight_steps() -> None:
    grammar = generate_runtime_program_grammar(seed=2718, count=1, depth=8)
    batch = generate_sequence_memory_batch(
        4,
        span=4,
        distractors=1,
        seed=99,
        operation="generated_composition",
        generated_composition_ids=(0,),
        generated_compositions=grammar,
    )

    assert len(grammar[0]) == 8
    assert batch.query_frames.shape == (4, 4, 3, 32, 32)
    assert torch.isfinite(batch.query_frames).all()
