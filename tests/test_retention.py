from __future__ import annotations

import pytest
import torch

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    CapabilityRetentionLedger,
    ContentAddressedMemory,
    ExecutableArtifactMemory,
    PersistentContentAddressedMemory,
    RetentionPolicyConfig,
    evaluate_retention_gate,
    load_runtime_components,
    save_runtime,
    stable_prefix_minimum,
)


def _key(index: int) -> torch.Tensor:
    values = torch.zeros(4)
    values[index] = 1.0
    return values


def _artifact(value: float) -> dict[str, torch.Tensor]:
    return {"growth.bias": torch.tensor([value])}


def test_stable_prefix_minimum_rejects_late_recovery() -> None:
    assert stable_prefix_minimum([1.0, 1.0, 1.0, 1.0], min_observations=2) == 1.0
    assert stable_prefix_minimum([1.0, 1.0, 0.0, 1.0], min_observations=2) == pytest.approx(
        2.0 / 3.0
    )


def test_retention_gate_requires_candidate_and_all_retained_scores() -> None:
    accepted = evaluate_retention_gate(
        [1.0, 1.0, 1.0],
        [0.9, 0.85],
        candidate_threshold=0.8,
        retention_floor=0.8,
        min_candidate_observations=2,
    )
    assert accepted.accepted

    rejected = evaluate_retention_gate(
        [1.0, 1.0, 1.0],
        [0.9, 0.79],
        candidate_threshold=0.8,
        retention_floor=0.8,
        min_candidate_observations=2,
    )
    assert not rejected.accepted
    assert not rejected.retained


def test_retention_gate_uses_stable_prefix_floor_not_raw_probe_minimum() -> None:
    outcomes = [
        0.7109375,
        0.6953125,
        0.7265625,
        0.70703125,
        0.6875,
        0.71484375,
        0.703125,
        0.7109375,
    ]
    stable_floor = stable_prefix_minimum(outcomes, min_observations=8)
    assert min(outcomes) < 0.70 < stable_floor

    decision = evaluate_retention_gate(
        outcomes,
        [stable_floor],
        candidate_threshold=0.70,
        retention_floor=0.70,
        min_candidate_observations=8,
    )
    assert decision.accepted


def test_ledger_protects_mastery_and_allows_only_unprotected_eviction() -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(
            mastery_threshold=0.75,
            min_mastery_observations=3,
            reversal_patience=3,
        ),
    )
    mastered = _key(0)
    other = _key(1)
    for _ in range(3):
        ledger.observe(mastered, 1.0)
    assert ledger.is_protected(mastered)

    scores, protected = ledger.mask_eviction_scores(
        torch.stack((mastered, other)), torch.tensor([10.0, 1.0])
    )
    assert protected.tolist() == [True, False]
    assert int(scores.argmax()) == 1

    for _ in range(3):
        ledger.observe(other, 1.0)
    assert ledger.choose_eviction_index(
        torch.stack((mastered, other)), torch.tensor([10.0, 1.0])
    ) is None


def test_ledger_reversal_has_hysteresis_and_requires_new_mastery() -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(
            mastery_threshold=0.75,
            min_mastery_observations=3,
            reversal_patience=3,
        ),
    )
    key = _key(0)
    for _ in range(3):
        ledger.observe(key, 1.0)
    ledger.observe(key, 0.0)
    assert ledger.is_protected(key)
    for _ in range(2):
        ledger.observe(key, 0.0)
    status = ledger.status(key)
    assert not status.protected
    assert status.reversal_count == 1

    for _ in range(3):
        ledger.observe(key, 1.0)
    assert ledger.is_protected(key)


def test_ledger_persists_opaque_state(tmp_path) -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    key = _key(2)
    ledger.observe(key, 1.0)
    ledger.observe(key, 1.0)
    path = tmp_path / "retention-ledger.json"
    ledger.save(path)

    restored = CapabilityRetentionLedger.load(path)
    assert restored.is_protected(key)
    assert restored.status(key) == ledger.status(key)


def test_artifact_memory_refuses_to_evict_protected_rows(tmp_path) -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    memory = ExecutableArtifactMemory(
        tmp_path / "memory", width=4, capacity=2, retention_ledger=ledger
    )
    first = _key(0)
    second = _key(1)
    replacement = _key(2)
    final = _key(3)
    memory.put(first, _artifact(1.0))
    memory.put(second, _artifact(2.0))
    memory.observe_retention(first, 1.0)
    memory.observe_retention(first, 1.0)

    replaced_index = memory.put(
        replacement, _artifact(3.0), eviction_scores=torch.tensor([10.0, 1.0])
    )
    assert replaced_index == 1
    memory.validate()
    memory.promote(first)
    memory.promote(replacement)
    memory.observe_retention(replacement, 1.0)
    memory.observe_retention(replacement, 1.0)
    with pytest.raises(MemoryError, match="protected"):
        memory.put(final, _artifact(4.0), eviction_scores=torch.tensor([1.0, 1.0]))

    restored = ExecutableArtifactMemory.load(tmp_path / "memory")
    assert restored.retention.is_protected(first)
    restored.validate()


def test_content_addressed_memory_refuses_implicit_protected_eviction() -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    memory = ContentAddressedMemory(
        4, capacity=2, retention_ledger=ledger, write_match_threshold=0.99
    )
    first = _key(0)
    second = _key(1)
    memory.write(
        first.unsqueeze(0), first.unsqueeze(0), torch.ones(1)
    )
    memory.write(
        second.unsqueeze(0), second.unsqueeze(0), torch.ones(1)
    )
    memory.observe_retention(first, 1.0)
    memory.observe_retention(first, 1.0)
    memory.observe_retention(second, 1.0)
    memory.observe_retention(second, 1.0)
    with pytest.raises(MemoryError, match="protected"):
        memory.write(
            _key(2).unsqueeze(0), _key(2).unsqueeze(0), torch.ones(1)
        )


def test_persistent_content_memory_reloads_retention_ledger(tmp_path) -> None:
    path = tmp_path / "memory.pt"
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    memory = PersistentContentAddressedMemory(
        4, capacity=1, path=path, retention_ledger=ledger
    )
    key = _key(0)
    memory.write(key.unsqueeze(0), key.unsqueeze(0), torch.ones(1))
    memory.observe_retention(key, 1.0)
    memory.observe_retention(key, 1.0)

    restored = PersistentContentAddressedMemory(4, capacity=1, path=path)
    assert restored.retention.is_protected(key)
    with pytest.raises(MemoryError, match="protected"):
        restored.write(_key(1).unsqueeze(0), _key(1).unsqueeze(0), torch.ones(1))


def test_artifact_consolidation_gate_rejects_retention_regression(tmp_path) -> None:
    memory = ExecutableArtifactMemory(tmp_path / "source", width=4, capacity=2)
    memory.put(_key(0), _artifact(1.0))
    memory.put(_key(1), _artifact(2.0))

    candidate, receipt = memory.consolidate_verified(
        (0, 1),
        _key(2),
        _artifact(3.0),
        tmp_path / "rejected",
        verifier=lambda _candidate: True,
        candidate_outcomes=[1.0, 1.0],
        retained_scores=[0.79],
        candidate_threshold=0.8,
        retention_floor=0.8,
        min_candidate_observations=2,
    )
    assert candidate is None
    assert not receipt.accepted
    assert "retention floor" in receipt.reason


def test_runtime_checkpoint_round_trips_retention_state(tmp_path) -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    source_memory = ContentAddressedMemory(4, capacity=2, retention_ledger=ledger)
    key = _key(0)
    source_memory.write(key.unsqueeze(0), key.unsqueeze(0), torch.ones(1))
    source_memory.observe_retention(key, 1.0)
    source_memory.observe_retention(key, 1.0)
    source = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=4, workspace_slots=1, intention_width=4, feedback_width=2
        ),
        memory=source_memory,
    )
    checkpoint = tmp_path / "runtime.pt"
    save_runtime(source, checkpoint)

    restored = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=4, workspace_slots=1, intention_width=4, feedback_width=2
        ),
        memory=ContentAddressedMemory(4, capacity=2),
    )
    load_runtime_components(restored, checkpoint)
    assert restored.memory is not None
    assert restored.memory.retention.is_protected(key)
