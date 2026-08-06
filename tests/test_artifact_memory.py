from __future__ import annotations

import json

import pytest
import torch

from neural_computer import (
    CapabilityRetentionLedger,
    CapabilityRetentionProbe,
    ExecutableArtifactMemory,
    RetentionPolicyConfig,
)


def _artifact(value: float) -> dict[str, torch.Tensor]:
    return {
        "growth.weight": torch.tensor([[value, -value]], dtype=torch.float32),
        "growth.bias": torch.tensor([value], dtype=torch.float32),
    }


def test_executable_artifact_memory_reloads_and_verifies(tmp_path) -> None:
    directory = tmp_path / "artifacts"
    memory = ExecutableArtifactMemory(directory, width=4, capacity=2)
    key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    index = memory.put(key, _artifact(2.0))
    memory.validate()

    restored = ExecutableArtifactMemory.load(directory)
    handle, loaded = restored.promote(key)
    assert handle.index == index
    assert handle.confidence == pytest.approx(1.0)
    assert torch.equal(loaded["growth.weight"], _artifact(2.0)["growth.weight"])
    assert torch.equal(loaded["growth.bias"], _artifact(2.0)["growth.bias"])


def test_planner_candidates_are_fixed_width_opaque_artifact_summaries(tmp_path) -> None:
    memory = ExecutableArtifactMemory(tmp_path / "planner", width=4, capacity=2)
    key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    memory.put(key, _artifact(2.0))

    candidates = memory.planner_candidates()

    candidates.validate(width=4, capacity=2, batch=1)
    assert candidates.occupied.tolist() == [[True, False]]
    assert torch.isfinite(candidates.values).all()
    assert torch.count_nonzero(candidates.values[0, 0]) > 0


def test_corrupted_artifact_is_rejected_without_affecting_address_rows(tmp_path) -> None:
    directory = tmp_path / "artifacts"
    memory = ExecutableArtifactMemory(directory, width=4, capacity=1)
    key = torch.tensor([0.0, 1.0, 0.0, 0.0])
    memory.put(key, _artifact(3.0))
    filename = memory.paths[0]
    assert filename is not None
    path = directory / filename
    path.write_bytes(path.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="hash mismatch"):
        ExecutableArtifactMemory.load(directory)


def test_compaction_preserves_selected_artifacts_and_drops_others(tmp_path) -> None:
    source = ExecutableArtifactMemory(tmp_path / "source", width=4, capacity=3)
    keys = [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
    ]
    for value, key in enumerate(keys, start=1):
        source.put(key, _artifact(float(value)))

    compacted = source.compact((2, 0), tmp_path / "compact")
    assert compacted.capacity == 2
    assert len(compacted.occupied) == 2
    _, first = compacted.promote(keys[0])
    _, third = compacted.promote(keys[2])
    assert float(first["growth.bias"][0]) == 1.0
    assert float(third["growth.bias"][0]) == 3.0
    with pytest.raises(LookupError):
        compacted.promote(keys[1])


def test_compaction_refuses_to_drop_a_protected_artifact(tmp_path) -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    source = ExecutableArtifactMemory(
        tmp_path / "protected-source",
        width=4,
        capacity=2,
        retention_ledger=ledger,
    )
    protected = torch.tensor([1.0, 0.0, 0.0, 0.0])
    survivor = torch.tensor([0.0, 1.0, 0.0, 0.0])
    source.put(protected, _artifact(1.0))
    source.put(survivor, _artifact(2.0))
    source.observe_retention(protected, 1.0)
    source.observe_retention(protected, 1.0)

    with pytest.raises(MemoryError, match="protected"):
        source.compact((1,), tmp_path / "unsafe-compact")
    with pytest.raises(ValueError, match="candidate outcomes"):
        source.consolidate_verified(
            (0,),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
            _artifact(3.0),
            tmp_path / "unsafe-consolidation",
            verifier=lambda _: True,
        )


def test_retention_batch_preserves_order_and_persists_once(tmp_path) -> None:
    memory = ExecutableArtifactMemory(tmp_path / "batch", width=4, capacity=1)
    key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    memory.put(key, _artifact(1.0))
    original_save = memory.save
    save_calls = 0

    def save_once() -> None:
        nonlocal save_calls
        save_calls += 1
        original_save()

    memory.save = save_once

    memory.observe_retention_batch(((key, 1.0), (key, 0.0), (key, 1.0)))

    status = memory.retention.status(key)
    assert status.observations == 3
    assert save_calls == 1
    restored = ExecutableArtifactMemory.load(tmp_path / "batch")
    assert restored.retention.status(key).observations == 3


def test_legacy_alias_manifest_defaults_null_bindings(tmp_path) -> None:
    memory = ExecutableArtifactMemory(tmp_path / "legacy", width=4, capacity=1)
    primary = torch.tensor([1.0, 0.0, 0.0, 0.0])
    alias = torch.tensor([0.0, 1.0, 0.0, 0.0])
    memory.put(primary, _artifact(1.0))
    memory.alias_keys[0] = [alias]
    memory.alias_views[0] = ["legacy-alias"]
    memory.save()
    manifest_path = tmp_path / "legacy" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("alias_bindings")
    manifest_path.write_text(json.dumps(manifest))

    restored = ExecutableArtifactMemory.load(tmp_path / "legacy")

    handle, _ = restored.promote(alias)
    assert handle.binding is None



def test_growth_preserves_protected_rows_and_allows_new_write(tmp_path) -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    source = ExecutableArtifactMemory(
        tmp_path / "full-source",
        width=4,
        capacity=2,
        retention_ledger=ledger,
    )
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    alias = torch.tensor([0.0, 0.0, 1.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])
    third = torch.tensor([0.0, 0.0, 0.0, 1.0])
    source.put(first, _artifact(1.0))
    source.put(second, _artifact(2.0))
    source.alias_keys[0] = [alias]
    source.alias_views[0] = ["first-alias"]
    for key in (first, alias, second):
        source.observe_retention(key, 1.0)
        source.observe_retention(key, 1.0)

    with pytest.raises(MemoryError, match="protected"):
        source.put(third, _artifact(3.0))

    grown = source.grow(tmp_path / "grown", capacity=3)
    assert source.capacity == 2
    assert len(source.occupied) == 2
    assert grown.capacity == 3
    assert grown.retention.is_protected(first)
    assert grown.retention.is_protected(alias)
    assert grown.alias_views[0] == ["first-alias"]
    grown.put(third, _artifact(3.0))
    assert len(grown.occupied) == 3
    restored = ExecutableArtifactMemory.load(tmp_path / "grown")
    assert restored.retention.is_protected(first)
    _, loaded = restored.promote(third)
    assert float(loaded["growth.bias"][0]) == 3.0


def test_protected_opaque_alias_protects_the_complete_artifact_row(tmp_path) -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    source = ExecutableArtifactMemory(
        tmp_path / "alias-protected-source",
        width=4,
        capacity=2,
        retention_ledger=ledger,
    )
    primary = torch.tensor([1.0, 0.0, 0.0, 0.0])
    alias = torch.tensor([0.0, 0.0, 1.0, 0.0])
    other = torch.tensor([0.0, 1.0, 0.0, 0.0])
    source.put(primary, _artifact(1.0))
    source.put(other, _artifact(2.0))
    source.alias_keys[0] = [alias]
    source.alias_views[0] = ["0"]
    source.observe_retention(alias, 1.0)
    source.observe_retention(alias, 1.0)

    written = source.put(
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        _artifact(3.0),
        eviction_scores=torch.tensor([1.0, 0.0]),
    )
    assert written == 1
    with pytest.raises(MemoryError, match="protected"):
        source.compact((1,), tmp_path / "alias-unsafe-compact")

    compacted = source.compact((0,), tmp_path / "alias-safe-compact")
    assert compacted.retention.is_protected(alias)


def test_verified_consolidation_registers_replacement_mastery(tmp_path) -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    source = ExecutableArtifactMemory(
        tmp_path / "mastery-source",
        width=4,
        capacity=2,
        retention_ledger=ledger,
    )
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])
    replacement = torch.tensor([0.0, 0.0, 1.0, 0.0])
    source.put(first, _artifact(1.0))
    source.put(second, _artifact(2.0))

    candidate, receipt = source.consolidate_verified(
        (0, 1),
        replacement,
        _artifact(3.0),
        tmp_path / "mastery-consolidated",
        verifier=lambda _: True,
        candidate_outcomes=[1.0, 1.0],
        retained_scores=[],
        min_candidate_observations=2,
    )

    assert receipt.accepted
    assert candidate is not None
    assert candidate.retention.is_protected(replacement)
    restored = ExecutableArtifactMemory.load(tmp_path / "mastery-consolidated")
    assert restored.retention.is_protected(replacement)


def test_protected_consolidation_can_probe_candidate_before_retention_gate(tmp_path) -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    source = ExecutableArtifactMemory(
        tmp_path / "protected-source",
        width=4,
        capacity=2,
        retention_ledger=ledger,
    )
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])
    replacement = torch.tensor([0.0, 0.0, 1.0, 0.0])
    source.put(first, _artifact(1.0))
    source.put(second, _artifact(2.0))
    for key in (first, second):
        source.observe_retention(key, 1.0)
        source.observe_retention(key, 1.0)

    probed: list[int] = []

    def probe(candidate: ExecutableArtifactMemory) -> list[float]:
        probed.append(len(candidate.occupied))
        return [1.0, 1.0]

    candidate, receipt = source.consolidate_verified(
        (0, 1),
        replacement,
        _artifact(3.0),
        tmp_path / "protected-consolidated",
        verifier=lambda candidate: candidate.retention.is_protected(replacement),
        candidate_outcome_probe=probe,
        retained_scores=[1.0, 1.0],
        min_candidate_observations=2,
    )

    assert receipt.accepted
    assert candidate is not None
    assert probed == [1]
    assert candidate.retention.is_protected(replacement)
    assert len(source.occupied) == 2


def test_protected_consolidation_checks_each_opaque_alias_independently(tmp_path) -> None:
    ledger = CapabilityRetentionLedger(
        4,
        config=RetentionPolicyConfig(min_mastery_observations=2),
    )
    source = ExecutableArtifactMemory(
        tmp_path / "per-capability-source",
        width=4,
        capacity=2,
        retention_ledger=ledger,
    )
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    alias = torch.tensor([0.0, 1.0, 0.0, 0.0])
    replacement = torch.tensor([0.0, 0.0, 1.0, 0.0])
    source.put(first, _artifact(1.0))
    source.put(torch.tensor([0.0, 0.0, 0.0, 1.0]), _artifact(2.0))
    source.alias_keys[0] = [alias]
    source.alias_views[0] = ["alias"]
    for key in (first, alias):
        source.observe_retention(key, 1.0)
        source.observe_retention(key, 1.0)

    candidate, receipt = source.consolidate_verified(
        (0,),
        replacement,
        _artifact(3.0),
        tmp_path / "per-capability-rejected",
        replacement_aliases=(first, alias),
        replacement_alias_views=("first", "alias"),
        verifier=lambda _: True,
        candidate_outcome_probe=lambda _: (
            CapabilityRetentionProbe(first, [1.0, 1.0]),
            CapabilityRetentionProbe(alias, [0.0, 0.0]),
        ),
        retained_scores=[1.0, 1.0],
        candidate_threshold=0.8,
        retention_floor=0.8,
        min_candidate_observations=2,
    )

    assert candidate is None
    assert not receipt.accepted
    assert "capability 1" in receipt.reason
    assert len(source.occupied) == 2


def test_similar_but_distinct_procedure_addresses_do_not_collapse(tmp_path) -> None:
    memory = ExecutableArtifactMemory(tmp_path / "similar", width=4, capacity=2)
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.96677, 0.255, 0.0, 0.0])
    second = torch.nn.functional.normalize(second, dim=0)
    memory.put(first, _artifact(1.0))
    memory.put(second, _artifact(2.0))

    assert len(memory.occupied) == 2
    _, first_loaded = memory.promote(first)
    _, second_loaded = memory.promote(second)
    assert float(first_loaded["growth.bias"][0]) == 1.0
    assert float(second_loaded["growth.bias"][0]) == 2.0


def test_router_selected_rows_can_be_promoted_and_verified(tmp_path) -> None:
    memory = ExecutableArtifactMemory(tmp_path / "selected", width=4, capacity=2)
    first = memory.put(
        torch.tensor([1.0, 0.0, 0.0, 0.0]), _artifact(1.0)
    )
    second = memory.put(
        torch.tensor([0.0, 1.0, 0.0, 0.0]), _artifact(2.0)
    )

    assert [index for index, _ in memory.address_rows()] == [first, second]
    handle, loaded = memory.promote_index(second, confidence=0.8, margin=0.2)
    assert handle.index == second
    assert handle.confidence == pytest.approx(0.8)
    assert float(loaded["growth.bias"][0]) == 2.0


def test_top_k_promotion_returns_verified_compositional_candidates(tmp_path) -> None:
    memory = ExecutableArtifactMemory(tmp_path / "candidates", width=4, capacity=2)
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.nn.functional.normalize(
        torch.tensor([0.9, 0.435, 0.0, 0.0]), dim=0
    )
    memory.put(first, _artifact(1.0))
    memory.put(second, _artifact(2.0))

    query = torch.nn.functional.normalize(
        torch.tensor([0.95, 0.312, 0.0, 0.0]), dim=0
    )
    handles, artifacts = memory.promote_candidates(query, top_k=2)

    assert [handle.index for handle in handles] == [1, 0]
    assert [float(artifact["growth.bias"][0]) for artifact in artifacts] == [
        2.0,
        1.0,
    ]
    assert handles[0].margin > 0.0


def test_verified_consolidation_preserves_multiple_opaque_addresses(tmp_path) -> None:
    source = ExecutableArtifactMemory(tmp_path / "source", width=4, capacity=2)
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])
    source.put(first, _artifact(1.0))
    source.put(second, _artifact(2.0))

    def verifier(candidate: ExecutableArtifactMemory) -> bool:
        first_handle, first_artifact = candidate.promote(first)
        second_handle, second_artifact = candidate.promote(second)
        return (
            first_handle.index == second_handle.index == 0
            and float(first_artifact["growth.bias"][0]) == 3.0
            and float(second_artifact["growth.bias"][0]) == 3.0
        )

    candidate, receipt = source.consolidate_verified(
        (0, 1),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        _artifact(3.0),
        tmp_path / "consolidated",
        replacement_aliases=(first, second),
        replacement_alias_views=("left", "right"),
        replacement_alias_bindings=(
            {"schema": "opaque-slot-binding-v1", "slot_indices": [0, 1]},
            {"schema": "opaque-slot-binding-v1", "slot_indices": [0, 1, 2]},
        ),
        verifier=verifier,
    )
    assert receipt.accepted
    assert receipt.rows_saved == 1
    assert candidate is not None
    restored = ExecutableArtifactMemory.load(tmp_path / "consolidated")
    first_handle, _ = restored.promote(first)
    second_handle, _ = restored.promote(second)
    assert first_handle.index == second_handle.index == 0
    assert first_handle.view == "left"
    assert second_handle.view == "right"
    assert first_handle.binding == {
        "schema": "opaque-slot-binding-v1",
        "slot_indices": [0, 1],
    }
    assert second_handle.binding == {
        "schema": "opaque-slot-binding-v1",
        "slot_indices": [0, 1, 2],
    }
    assert [view for _, _, view in restored.view_candidates()] == ["left", "right"]
    promoted_view, _ = restored.promote_view(0, "right")
    assert promoted_view.view == "right"
    assert promoted_view.binding == {
        "schema": "opaque-slot-binding-v1",
        "slot_indices": [0, 1, 2],
    }
    assert len(source.occupied) == 2
