"""Pressure-test protected artifact refusal followed by explicit capacity growth."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import (
    CapabilityRetentionLedger,
    ExecutableArtifactMemory,
    RetentionPolicyConfig,
)


def _artifact(value: float) -> dict[str, torch.Tensor]:
    return {
        "growth.weight": torch.tensor([[value, -value]], dtype=torch.float32),
        "growth.bias": torch.tensor([value], dtype=torch.float32),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.retention_probes < 1 or args.capacity < 2:
        raise ValueError("capacity must be at least two and probes must be positive")
    root = args.report_out.parent
    for directory in (root / "source", root / "grown"):
        if directory.exists():
            shutil.rmtree(directory)

    keys = [
        torch.eye(args.width, dtype=torch.float32)[index]
        for index in range(args.capacity + 1)
    ]
    ledger = CapabilityRetentionLedger(
        args.width,
        config=RetentionPolicyConfig(
            mastery_threshold=args.retention_threshold,
            min_mastery_observations=args.retention_probes,
        ),
    )
    source = ExecutableArtifactMemory(
        root / "source",
        width=args.width,
        capacity=args.capacity,
        retention_ledger=ledger,
    )
    for index in range(args.capacity):
        source.put(keys[index], _artifact(float(index + 1)))
        for _ in range(args.retention_probes):
            source.observe_retention(keys[index], 1.0)

    source_before = {
        "capacity": source.capacity,
        "occupied": list(source.occupied),
        "version": source.version,
        "protected": [source._row_is_protected(index) for index in source.occupied],
    }
    refused = False
    try:
        source.put(keys[-1], _artifact(float(args.capacity + 1)))
    except MemoryError as error:
        refused = "protected" in str(error)
    if not refused:
        raise RuntimeError("full protected artifact bank did not refuse eviction")

    grown = source.grow(root / "grown", capacity=args.capacity + 1)
    grown.put(keys[-1], _artifact(float(args.capacity + 1)))
    reloaded = ExecutableArtifactMemory.load(root / "grown")
    reloaded.validate()
    promoted = [reloaded.promote(key)[0].index for key in keys]
    source_after = {
        "capacity": source.capacity,
        "occupied": list(source.occupied),
        "version": source.version,
        "protected": [source._row_is_protected(index) for index in source.occupied],
    }
    report = {
        "schema": "neural-computer.artifact-capacity-growth-report.v1",
        "claim_boundary": (
            "A full protected executable-artifact bank refuses eviction, then "
            "grows transactionally into a separate verified capacity before "
            "admitting a new opaque artifact; this is a memory safety boundary, "
            "not a general continual-learning claim."
        ),
        "seed": args.seed,
        "width": args.width,
        "initial_capacity": args.capacity,
        "grown_capacity": args.capacity + 1,
        "retention_threshold": args.retention_threshold,
        "retention_probes": args.retention_probes,
        "source_before": source_before,
        "source_after": source_after,
        "reloaded_promoted_indices": promoted,
        "accounting": {
            "unique_verifier_bits": args.capacity * args.retention_probes,
            "unique_logical_lifetimes": args.capacity * args.retention_probes,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "retention_observations": args.capacity * args.retention_probes,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "full_protected_write_refused": refused,
            "source_immutable_after_refusal_and_growth": source_before == source_after,
            "all_source_rows_protected": all(source_before["protected"]),
            "retention_transferred": all(
                reloaded.retention.is_protected(keys[index])
                for index in range(args.capacity)
            ),
            "new_artifact_admitted_after_growth": len(grown.occupied)
            == args.capacity + 1,
            "reloaded_all_artifacts": promoted == list(range(args.capacity + 1)),
            "no_replayed_examples": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--retention-probes", type=int, default=8)
    parser.add_argument("--retention-threshold", type=float, default=0.70)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"promoted": report["promoted"], "gates": report["gates"]}))


if __name__ == "__main__":
    main()
