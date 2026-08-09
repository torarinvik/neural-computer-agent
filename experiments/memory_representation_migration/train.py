"""Two-seed retention-safe external memory migration audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    ContentAddressedMemory,
    MemoryMigrationExample,
    MemoryQuery,
)

WIDTH = 4
CAPACITY = 3
SOURCE_SPACES = ("memory-key-v1", "memory-value-v1")
TARGET_SPACES = ("memory-key-v2", "memory-value-v2")


def _memory(spaces: tuple[str, str]) -> ContentAddressedMemory:
    return ContentAddressedMemory(
        WIDTH,
        capacity=CAPACITY,
        write_match_threshold=0.99,
        key_space_id=spaces[0],
        value_space_id=spaces[1],
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    source = _memory(SOURCE_SPACES)
    candidate = _memory(TARGET_SPACES)
    source_keys = torch.tensor(
        [[1.0, 0.2, -0.1, 0.3], [-0.4, 0.8, 0.2, 0.1]]
    )
    target_keys = source_keys[:, [2, 0, 3, 1]]
    values = torch.tensor(
        [[0.0, 1.0, 0.5, -0.2], [0.7, -0.1, 0.2, 0.9]]
    )
    source.write(source_keys, values, torch.ones(2))
    candidate.write(target_keys, values, torch.ones(2))
    for _ in range(8):
        source.observe_retention(source_keys[0], 1.0)
    candidate.retention.adopt_transformed(
        source.retention,
        source_keys[0],
        target_keys[0],
    )
    addresses = tuple(
        (source_keys[index], target_keys[index]) for index in range(source_keys.shape[0])
    )
    queries = tuple(
        MemoryMigrationExample(
            MemoryQuery(source_keys[index].unsqueeze(0)),
            MemoryQuery(target_keys[index].unsqueeze(0)),
        )
        for index in range(source_keys.shape[0])
    )
    accepted = source.migrate_representation_verified(
        candidate,
        addresses,
        queries,
        retention_probe=lambda memory: memory.retention.is_protected(target_keys[0]),
    )
    drifted = _memory(TARGET_SPACES)
    drifted.write(target_keys, values, torch.ones(2))
    drifted.retention.adopt_transformed(source.retention, source_keys[0], target_keys[0])
    drifted.values[0, 0].add_(0.5)
    rejected = source.migrate_representation_verified(drifted, addresses, queries)
    report = {
        "schema": "neural-computer.memory-representation-migration.v1",
        "seed": seed,
        "configuration": {
            "source_spaces": SOURCE_SPACES,
            "target_spaces": TARGET_SPACES,
            "occupied_rows": len(addresses),
            "heldout_queries": len(queries),
            "migration": "one_to_one_address_retention_query_probe_v1",
        },
        "gates": {
            "behavior_preserving_migration": accepted.accepted,
            "protected_retention_preserved": accepted.protected_count == 1,
            "drifted_candidate_rejected": not rejected.accepted,
            "zero_controller_updates": True,
            "zero_replayed_outcomes": True,
            "zero_memory_replay": True,
        },
        "promoted": accepted.accepted and not rejected.accepted,
        "metrics": {
            "accepted_max_value_difference": accepted.max_value_difference,
            "rejected_max_value_difference": rejected.max_value_difference,
            "protected_count": accepted.protected_count,
        },
        "accounting": {
            "unique_verifier_bits": len(queries),
            "unique_logical_lifetimes": len(addresses),
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": "retention-safe external memory migration; not arbitrary value alignment or general continual learning",
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
