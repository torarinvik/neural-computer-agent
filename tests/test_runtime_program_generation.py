from __future__ import annotations

import torch

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    _load_stack_artifact,
    _stack_artifact,
    generate_runtime_program_grammar,
)
from experiments.generated_composition_capability_amodal.train_distilled_consolidation import (
    _load_or_fresh,
)
from experiments.generated_composition_capability_amodal.train_multi_transfer import (
    _loaded_source_probe_outcomes,
)
from experiments.generated_composition_capability_amodal.train_pipeline import (
    _new_stack,
    expand_routed_stack,
)
from experiments.generated_composition_capability_amodal.train_sequential_distilled_consolidation import (
    _fresh_slot_mask,
    _train_expanded_new_only,
)
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    _capability_accuracy,
    _new_capability,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
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


def test_runtime_opaque_rule_generation_is_deterministic_and_renders() -> None:
    first = generate_runtime_program_grammar(
        seed=4242,
        count=3,
        depth=8,
        primitive_family="opaque_rule",
    )
    second = generate_runtime_program_grammar(
        seed=4242,
        count=3,
        depth=8,
        primitive_family="opaque_rule",
    )

    assert first == second
    assert all(
        primitive.startswith("rule:")
        for program in first
        for primitive in program
    )
    batch = generate_sequence_memory_batch(
        4,
        span=4,
        distractors=1,
        seed=101,
        operation="generated_composition",
        generated_composition_ids=(0,),
        generated_compositions=first,
    )
    assert batch.query_frames.shape == (4, 4, 3, 32, 32)
    assert torch.isfinite(batch.correct_actions.float()).all()


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


def test_batched_retention_probes_preserve_separate_outcomes() -> None:
    grammar = generate_runtime_program_grammar(
        seed=4242,
        count=3,
        depth=8,
        primitive_family="opaque_rule",
    )
    parent = _runtime(seed=69316, growth=False)
    stack = _new_stack(seed=69317, program_count=2, stack="routed")
    decoder = _new_capability(seed=69318)[1]
    loaded_stack, loaded_decoder = _load_stack_artifact(
        _stack_artifact(stack, decoder)
    )

    batched = _loaded_source_probe_outcomes(
        parent,
        loaded_stack,
        loaded_decoder,
        0,
        grammar,
        count=8,
        probes=3,
        seed=70_000,
    )
    separate = [
        _capability_accuracy(
            parent,
            loaded_stack,
            loaded_decoder,
            operation="generated_composition",
            span=4,
            count=8,
            seed=70_000 + probe * 101,
            generated_composition_ids=(0,),
            generated_compositions=grammar,
        )
        for probe in range(3)
    ]

    assert torch.allclose(torch.tensor(batched), torch.tensor(separate))


def test_routed_stack_expansion_preserves_existing_external_state() -> None:
    base = _new_stack(seed=17, program_count=2, stack="routed")
    expanded = expand_routed_stack(base, seed=19)

    assert len(expanded.programs) == 3
    for old, new in zip(base.programs, expanded.programs[:2], strict=True):
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                old.state_dict().values(), new.state_dict().values(), strict=True
            )
        )
    assert torch.equal(expanded.router[0].weight, base.router[0].weight)
    assert torch.equal(expanded.router[0].bias, base.router[0].bias)
    for step in range(base.composition_steps):
        old_slice = slice(step * 2, (step + 1) * 2)
        new_slice = slice(step * 3, step * 3 + 2)
        assert torch.equal(
            expanded.router[2].weight[new_slice], base.router[2].weight[old_slice]
        )
        assert torch.equal(
            expanded.router[2].bias[new_slice], base.router[2].bias[old_slice]
        )
        assert torch.all(expanded.router[2].bias[step * 3 + 2] == -8.0)


def test_artifact_reload_preserves_unspecified_expanded_slot_count() -> None:
    stack = _new_stack(seed=23, program_count=4, stack="routed")
    _, decoder = _new_capability(seed=29)
    artifact = _stack_artifact(stack, decoder)

    loaded, _ = _load_or_fresh(
        artifact,
        stack_seed=31,
        decoder_seed=37,
        program_count=None,
    )

    assert len(loaded.programs) == 4


def test_expanded_acquisition_helper_has_a_fresh_slot_isolation_contract() -> None:
    mask = _fresh_slot_mask(batch_size=3, slot_count=4, slot_index=2)
    assert mask.shape == (3, 4)
    assert torch.equal(mask.sum(dim=1), torch.ones(3, dtype=torch.long))
    assert torch.all(mask[:, 2])
    assert not torch.any(mask[:, :2])
    assert not torch.any(mask[:, 3:])
    assert _train_expanded_new_only.__doc__ is not None
