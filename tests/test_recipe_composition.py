from __future__ import annotations

import hashlib
import json
from itertools import product

import pytest
import torch

from neural_computer import (
    ExternalRecipeCompositionMemory,
    OpaqueContextRecipeCompositionMemory,
    RecipeInstruction,
    RecipeProgram,
    RecipeProgramCompositionFactors,
    RecipeProgramCompositionStructure,
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


def test_recursive_provenance_and_shape_survive_reload_and_reject_rewrites() -> None:
    memory = ExternalRecipeCompositionMemory(SLOT_VALUES)
    first = memory.add_program(_fragment_a())
    second = memory.add_program(_fragment_b())
    memory.protect_file(first)
    memory.protect_file(second)
    depth2 = next(
        candidate
        for candidate in memory.composition_candidates(max_program_length=2)
        if candidate.program.digest() == _target().digest()
    )
    receipt2 = memory.admit_verified_composition(
        depth2,
        _outcomes(depth2.program, _target()),
        threshold=1.0,
        min_observations=len(STATES),
        min_stable_observations=len(STATES),
        protect=True,
    )
    assert receipt2.accepted and receipt2.slot == 2
    depth3_target = RecipeProgram(
        SLOT_VALUES,
        _target().instructions + _fragment_b().instructions,
    )
    depth3 = next(
        candidate
        for candidate in memory.composition_candidates(max_program_length=3)
        if candidate.program.digest() == depth3_target.digest()
    )
    assert depth3.structure == RecipeProgramCompositionStructure(2, 1, True, False)
    receipt3 = memory.admit_verified_composition(
        depth3,
        _outcomes(depth3.program, depth3_target),
        threshold=1.0,
        min_observations=len(STATES),
        min_stable_observations=len(STATES),
        protect=True,
    )
    assert receipt3.accepted and receipt3.slot == 3
    assert memory.composition_depth(3) == 3

    restored = ExternalRecipeCompositionMemory.from_payload(memory.payload())
    assert restored.composition_depth(3) == 3
    assert restored.composition_structure(2, 1) == depth3.structure

    tampered = memory.payload()
    raw_factors = dict(tampered["provenance"][3])
    raw_factors["right_digest"] = _fragment_a().digest()
    tampered["provenance"] = list(tampered["provenance"])
    tampered["provenance"][3] = raw_factors
    content = {key: value for key, value in tampered.items() if key != "sha256"}
    tampered["sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="provenance"):
        ExternalRecipeCompositionMemory.from_payload(tampered)


def test_composition_policy_migrates_legacy_factor_only_payload() -> None:
    policy = OpaqueContextRecipeCompositionMemory()
    factors = RecipeProgramCompositionFactors("0" * 64, "1" * 64, "append")
    policy.record("context", factors, 1.0)
    current = policy.payload()
    legacy_configuration = {
        key: value
        for key, value in current["configuration"].items()
        if key
        in {
            "schema",
            "exploration_floor",
            "shared_prior_weight",
            "exploration_bonus",
            "left_weight",
            "right_weight",
            "mode_weight",
            "temperature",
            "context",
        }
    }
    legacy_configuration["credit"] = "scalar_composition_factor_aggregate_v1"
    legacy_factor_types = ("left", "right", "mode")

    def legacy_stats(stats: dict[str, dict[str, list[float]]]) -> dict[str, object]:
        return {factor_type: stats[factor_type] for factor_type in legacy_factor_types}

    legacy = {
        "schema": current["schema"],
        "configuration": legacy_configuration,
        "shared": legacy_stats(current["shared"]),
        "contexts": {
            context: legacy_stats(stats)
            for context, stats in current["contexts"].items()
        },
    }
    legacy["sha256"] = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    migrated = OpaqueContextRecipeCompositionMemory.from_payload(legacy)
    assert migrated.proposal_probabilities("context", (factors,))[0] > 0.0


def test_recursive_shape_credit_is_orientation_invariant() -> None:
    left_composite = RecipeProgramCompositionStructure(2, 1, True, False)
    right_composite = RecipeProgramCompositionStructure(1, 2, False, True)

    assert left_composite.canonical_shape_key() == "atomic+composite"
    assert right_composite.canonical_shape_key() == "atomic+composite"
    assert left_composite.canonical_depth_key() == "1:2"
    assert right_composite.canonical_depth_key() == "1:2"
    assert left_composite.depth_span_key() == right_composite.depth_span_key() == "1"


def test_composition_policy_migrates_v2_shape_payload() -> None:
    policy = OpaqueContextRecipeCompositionMemory()
    factors = RecipeProgramCompositionFactors("0" * 64, "1" * 64, "append")
    structure = RecipeProgramCompositionStructure(2, 1, True, False)
    policy.record("context", factors, 1.0, structure=structure)
    current = policy.payload()
    old_factor_types = (
        "left",
        "right",
        "mode",
        "left_depth",
        "right_depth",
        "left_shape",
        "right_shape",
    )
    v2_configuration = {
        key: value
        for key, value in current["configuration"].items()
        if key
        in {
            "schema",
            "exploration_floor",
            "shared_prior_weight",
            "exploration_bonus",
            "left_weight",
            "right_weight",
            "mode_weight",
            "left_depth_weight",
            "right_depth_weight",
            "shape_weight",
            "temperature",
            "context",
        }
    }
    v2_configuration["credit"] = "scalar_composition_factor_and_shape_aggregate_v2"

    def v2_stats(stats: dict[str, dict[str, list[float]]]) -> dict[str, object]:
        return {factor_type: stats[factor_type] for factor_type in old_factor_types}

    v2 = {
        "schema": current["schema"],
        "configuration": v2_configuration,
        "shared": v2_stats(current["shared"]),
        "contexts": {
            context: v2_stats(stats)
            for context, stats in current["contexts"].items()
        },
    }
    v2["sha256"] = hashlib.sha256(
        json.dumps(v2, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    migrated = OpaqueContextRecipeCompositionMemory.from_payload(v2)
    assert (
        migrated.configuration()["credit"]
        == "scalar_composition_factor_and_shape_profile_aggregate_v3"
    )
    assert migrated.proposal_probabilities("context", (factors,), (structure,))[0] > 0.0
