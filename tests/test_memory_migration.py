import pytest
import torch

from neural_computer import (
    ContentAddressedMemory,
    MemoryMigrationExample,
    MemoryQuery,
)


def _memory(key_space_id: str, value_space_id: str) -> ContentAddressedMemory:
    return ContentAddressedMemory(
        4,
        capacity=3,
        write_match_threshold=0.99,
        key_space_id=key_space_id,
        value_space_id=value_space_id,
    )


def test_memory_migration_preserves_opaque_addresses_and_retention_without_replay() -> None:
    source = _memory("memory-key-v1", "memory-value-v1")
    candidate = _memory("memory-key-v2", "memory-value-v2")
    source_keys = torch.tensor(
        [[1.0, 0.2, -0.1, 0.3], [-0.4, 0.8, 0.2, 0.1]]
    )
    target_keys = source_keys[:, [2, 0, 3, 1]]
    values = torch.tensor(
        [[0.0, 1.0, 0.5, -0.2], [0.7, -0.1, 0.2, 0.9]]
    )
    strengths = torch.ones(2)
    source.write(source_keys, values, strengths)
    candidate.write(target_keys, values, strengths)
    for _ in range(8):
        source.observe_retention(source_keys[0], 1.0)
    candidate.retention.adopt_transformed(
        source.retention,
        source_keys[0],
        target_keys[0],
    )
    pairs = [
        (source_keys[0], target_keys[0]),
        (source_keys[1], target_keys[1]),
    ]
    queries = [
        MemoryMigrationExample(
            MemoryQuery(source_keys[index].unsqueeze(0)),
            MemoryQuery(target_keys[index].unsqueeze(0)),
        )
        for index in range(2)
    ]
    source_digest = source._migration_digest()

    receipt = source.migrate_representation_verified(
        candidate,
        pairs,
        queries,
        retention_probe=lambda memory: memory.retention.is_protected(target_keys[0]),
    )

    assert receipt.accepted
    assert receipt.max_value_difference == 0.0
    assert receipt.protected_count == 1
    assert source._migration_digest() == source_digest

    candidate.values[0, 0].add_(0.5)
    rejected = source.migrate_representation_verified(candidate, pairs, queries)

    assert not rejected.accepted
    assert rejected.reason == "held-out memory values changed"


def test_memory_migration_rejects_untransferred_protected_evidence() -> None:
    source = _memory("memory-key-v1", "memory-value-v1")
    candidate = _memory("memory-key-v2", "memory-value-v2")
    source_key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    target_key = torch.tensor([0.0, 1.0, 0.0, 0.0])
    value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    source.write(source_key.unsqueeze(0), value, torch.ones(1))
    candidate.write(target_key.unsqueeze(0), value, torch.ones(1))
    for _ in range(8):
        source.observe_retention(source_key, 1.0)
    query_pairs = [
        MemoryMigrationExample(
            MemoryQuery(source_key.unsqueeze(0)),
            MemoryQuery(target_key.unsqueeze(0)),
        )
    ]

    with pytest.raises(ValueError, match="protected memory evidence"):
        source.migrate_representation_verified(
            candidate,
            [(source_key, target_key)],
            query_pairs,
        )
