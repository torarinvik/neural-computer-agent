from __future__ import annotations

import torch

from experiments.brainworkshop_canonical.instruction_autonomy import (
    decide_header_program,
    existing_program_try_order,
)
from neural_computer import (
    RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
    TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    ExternalProgramArtifact,
    ExternalTemporalProgramBank,
    compose_recursive_temporal_program,
    one_hot_temporal_address_artifact,
    pad_recursive_temporal_program,
    recursive_temporal_primitive,
)


def _digest(seed: int = 0) -> str:
    return f"{seed:064x}"


def _legacy(values: tuple[float, ...]) -> ExternalProgramArtifact:
    return ExternalProgramArtifact(
        codes=torch.tensor([values], dtype=torch.float32),
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )


def test_same_slot_nuisance_matches_and_orthogonal_depth_fails_closed() -> None:
    bank = ExternalTemporalProgramBank(
        4,
        4,
        controller_digest=_digest(3),
        generalization_tolerance=0.0,
        min_mastery_observations=3,
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    )
    primitive = recursive_temporal_primitive(_legacy((8.0, -8.0, -8.0, -8.0)))
    composed = compose_recursive_temporal_program(primitive, 2)
    one_a = torch.nn.functional.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=0)
    one_b = torch.nn.functional.normalize(torch.tensor([0.8, 0.6, 0.0, 0.0]), dim=0)
    two = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0, 0.0]), dim=0)
    three = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 0.0, 1.0]), dim=0)
    bank.admit(primitive, one_a, [1.0, 1.0, 1.0], min_observations=3, min_stable_observations=3)
    bank.admit(primitive, one_b, [1.0, 1.0, 1.0], min_observations=3, min_stable_observations=3)
    bank.admit(composed, two, [1.0, 1.0, 1.0], min_observations=3, min_stable_observations=3)

    along = torch.nn.functional.normalize(0.4 * one_a + 0.6 * one_b, dim=0)
    matched = bank.router.invariant_preferred_slot(along, residual_tolerance=0.05)
    depth_miss = bank.router.invariant_preferred_slot(three, residual_tolerance=0.05)

    assert matched == 0
    assert depth_miss is None
    assert not bank.router.has_context(along)
    assert not bank.router.has_context(three)


def test_unknown_header_tries_nearer_shorter_file_then_composes() -> None:
    bank = ExternalTemporalProgramBank(
        4,
        4,
        controller_digest=_digest(5),
        generalization_tolerance=0.0,
        min_mastery_observations=3,
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    )
    primitive = recursive_temporal_primitive(_legacy((8.0, -8.0, -8.0, -8.0)))
    composed = compose_recursive_temporal_program(primitive, 2)
    one = torch.nn.functional.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=0)
    two = torch.nn.functional.normalize(torch.tensor([0.0, 1.0, 0.0, 0.0]), dim=0)
    query = torch.nn.functional.normalize(torch.tensor([0.1, 0.9, 0.0, 0.0]), dim=0)
    bank.admit(primitive, one, [1.0, 1.0, 1.0], min_observations=3, min_stable_observations=3)
    bank.admit(composed, two, [1.0, 1.0, 1.0], min_observations=3, min_stable_observations=3)

    order = existing_program_try_order(bank, query)
    first = decide_header_program(bank, query, primitive, max_history=4)
    after_nearest = decide_header_program(
        bank, query, primitive, failed_slots=frozenset({1}), max_history=4
    )
    after_compose_exists = decide_header_program(
        bank,
        query,
        primitive,
        failed_slots=frozenset({1}),
        failed_depths=frozenset({3}),
        max_history=4,
    )

    assert order[0] == 1
    assert first.kind == "try_existing" and first.slot == 1
    assert after_nearest.kind == "compose" and after_nearest.proposed_depth == 3
    assert after_nearest.artifact is not None
    assert after_nearest.artifact.program_length == 3
    assert after_compose_exists.kind == "compose"
    assert after_compose_exists.proposed_depth == 4


def test_after_nearest_failure_tries_existing_next_depth_before_compose() -> None:
    bank = ExternalTemporalProgramBank(
        4,
        4,
        controller_digest=_digest(7),
        generalization_tolerance=0.0,
        min_mastery_observations=3,
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    )
    primitive = recursive_temporal_primitive(_legacy((8.0, -8.0, -8.0, -8.0)))
    two = compose_recursive_temporal_program(primitive, 2)
    three = compose_recursive_temporal_program(primitive, 3)
    near = torch.nn.functional.normalize(torch.tensor([0.0, 1.0, 0.0, 0.0]), dim=0)
    far = torch.nn.functional.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=0)
    query = torch.nn.functional.normalize(torch.tensor([0.05, 0.95, 0.0, 0.0]), dim=0)
    bank.admit(primitive, far, [1.0, 1.0, 1.0], min_observations=3, min_stable_observations=3)
    bank.admit(two, near, [1.0, 1.0, 1.0], min_observations=3, min_stable_observations=3)
    bank.admit(three, far, [1.0, 1.0, 1.0], min_observations=3, min_stable_observations=3)

    after_two = decide_header_program(
        bank, query, primitive, failed_slots=frozenset({1}), max_history=4
    )

    assert after_two.kind == "try_existing"
    assert after_two.slot == 2
    assert after_two.proposed_depth == 3


def test_empty_bank_climbs_one_composed_depth_per_failure() -> None:
    bank = ExternalTemporalProgramBank(
        4,
        4,
        controller_digest=_digest(11),
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    )
    primitive = recursive_temporal_primitive(_legacy((8.0, -8.0, -8.0, -8.0)))
    query = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0, 0.0]), dim=0)
    first = decide_header_program(bank, query, primitive, max_history=4)
    second = decide_header_program(
        bank, query, primitive, failed_depths=frozenset({1}), max_history=4
    )
    third = decide_header_program(
        bank, query, primitive, failed_depths=frozenset({1, 2}), max_history=4
    )

    assert first.kind == "compose" and first.proposed_depth == 1
    assert second.kind == "compose" and second.proposed_depth == 2
    assert third.kind == "compose" and third.proposed_depth == 3


def test_one_hot_address_pads_and_composes_past_original_capacity() -> None:
    primitive = recursive_temporal_primitive(one_hot_temporal_address_artifact(0, 4))
    padded = pad_recursive_temporal_program(primitive, 8)
    five = compose_recursive_temporal_program(padded, 5)

    assert primitive.instruction_width == 4
    assert padded.instruction_width == 8
    assert five.program_length == 5
    assert torch.equal(five.codes[0, :4], primitive.codes[0])
    assert torch.equal(pad_recursive_temporal_program(padded, 8).codes, padded.codes)
