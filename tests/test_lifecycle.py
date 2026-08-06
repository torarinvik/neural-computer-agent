from __future__ import annotations

import pytest
import torch

from neural_computer import (
    CapabilityRetentionLedger,
    ExecutableArtifactMemory,
    ExternalCapabilityLifecycle,
    OpaqueCapacityPlanner,
    RetentionPolicyConfig,
)


def _artifact(value: float) -> dict[str, torch.Tensor]:
    return {
        "growth.weight": torch.tensor([[value, -value]], dtype=torch.float32),
        "growth.bias": torch.tensor([value], dtype=torch.float32),
    }


def _memory(tmp_path, *, capacity: int) -> ExecutableArtifactMemory:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(
            mastery_threshold=0.75,
            min_mastery_observations=2,
        ),
    )
    return ExecutableArtifactMemory(
        tmp_path / "memory",
        width=4,
        capacity=capacity,
        retention_ledger=ledger,
    )


def test_lifecycle_grows_when_every_existing_capability_is_protected(tmp_path) -> None:
    memory = _memory(tmp_path, capacity=2)
    lifecycle = ExternalCapabilityLifecycle(memory)
    keys = [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
    ]
    memory.put(keys[0], _artifact(1.0))
    memory.put(keys[1], _artifact(2.0))
    for key in keys[:2]:
        memory.observe_retention(key, 1.0)
        memory.observe_retention(key, 1.0)

    plan = lifecycle.plan_admission(keys[2], _artifact(3.0))
    assert plan.action == "grow"
    receipt = lifecycle.admit(
        keys[2],
        _artifact(3.0),
        plan=plan,
        grow_destination=tmp_path / "grown",
    )

    assert receipt.accepted
    assert receipt.action == "grow"
    assert lifecycle.memory.capacity == 3
    assert lifecycle.protection_mask().tolist() == [True, True, False]
    lifecycle.memory.observe_retention(keys[2], 1.0)
    lifecycle.memory.observe_retention(keys[2], 1.0)
    assert lifecycle.protection_mask().tolist() == [True, True, True]
    assert memory.capacity == 2
    assert memory.occupied == (0, 1)


def test_lifecycle_evicts_only_an_unprotected_row(tmp_path) -> None:
    memory = _memory(tmp_path, capacity=2)
    lifecycle = ExternalCapabilityLifecycle(memory)
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])
    third = torch.tensor([0.0, 0.0, 1.0, 0.0])
    memory.put(first, _artifact(1.0))
    memory.put(second, _artifact(2.0))
    memory.observe_retention(first, 1.0)
    memory.observe_retention(first, 1.0)

    plan = lifecycle.plan_admission(third, _artifact(3.0))
    assert plan.action == "evict"
    assert plan.eviction_index == 1
    receipt = lifecycle.admit(third, _artifact(3.0), plan=plan)

    assert receipt.accepted
    assert receipt.index == 1
    assert lifecycle.memory.promote(first)[0].index == 0
    assert lifecycle.memory.promote(third)[0].index == 1
    with pytest.raises(LookupError):
        lifecycle.memory.promote(second)


def test_lifecycle_planner_masks_protected_eviction_and_selects_growth(tmp_path) -> None:
    memory = _memory(tmp_path, capacity=2)
    lifecycle = ExternalCapabilityLifecycle(
        memory,
        planner=OpaqueCapacityPlanner(width=4, hidden=8),
    )
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])
    incoming = torch.tensor([0.0, 0.0, 1.0, 0.0])
    memory.put(first, _artifact(1.0))
    memory.put(second, _artifact(2.0))
    for key in (first, second):
        memory.observe_retention(key, 1.0)
        memory.observe_retention(key, 1.0)

    plan = lifecycle.plan_admission(incoming, _artifact(3.0))

    assert plan.action == "grow"
    assert plan.eviction_index is None
    assert lifecycle.protection_mask().tolist() == [True, True]


def test_lifecycle_rejects_unverified_consolidation_without_mutating_source(
    tmp_path,
) -> None:
    memory = _memory(tmp_path, capacity=2)
    lifecycle = ExternalCapabilityLifecycle(memory)
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])
    replacement = torch.tensor([0.0, 0.0, 1.0, 0.0])
    memory.put(first, _artifact(1.0))
    memory.put(second, _artifact(2.0))

    receipt = lifecycle.consolidate(
        (0, 1),
        replacement,
        _artifact(3.0),
        tmp_path / "rejected",
        verifier=lambda _: False,
        candidate_outcomes=[1.0, 1.0],
        retained_scores=[],
        min_candidate_observations=2,
    )

    assert not receipt.accepted
    assert lifecycle.memory is memory
    assert lifecycle.memory.capacity == 2
    assert lifecycle.memory.occupied == (0, 1)
