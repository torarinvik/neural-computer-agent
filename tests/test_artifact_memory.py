from __future__ import annotations

import pytest
import torch

from neural_computer import ExecutableArtifactMemory


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
