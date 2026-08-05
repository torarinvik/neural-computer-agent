"""Audit append, cold reload, eviction, and compaction of artifact memory."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import ExecutableArtifactMemory


def _artifact_digest(artifact: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(artifact):
        digest.update(name.encode("utf-8"))
        digest.update(artifact[name].contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_single(
    directory: Path, *, device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    memory = ExecutableArtifactMemory.load(directory, device=device)
    rows = memory.address_rows()
    if len(rows) != 1:
        raise ValueError(f"source memory must contain one row: {directory}")
    index, key = rows[0]
    _, artifact = memory.promote_index(index)
    return key, artifact


def _same_artifacts(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _random_keys(rows: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return F.normalize(torch.randn(rows, width, generator=generator), dim=-1)


def _corruption_control(
    memory: ExecutableArtifactMemory,
    destination: Path,
    *,
    device: torch.device,
) -> bool:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(memory.directory, destination)
    filename = memory.paths[0]
    if filename is None:
        raise RuntimeError("first row has no artifact path")
    path = destination / filename
    path.write_bytes(path.read_bytes() + b"corruption")
    try:
        ExecutableArtifactMemory.load(destination, device=device)
    except ValueError as error:
        return "hash mismatch" in str(error)
    return False


def run(args: argparse.Namespace) -> dict[str, object]:
    if len(args.sources) != 2:
        raise ValueError("exactly two single-row source memories are required")
    device = torch.device(args.device)
    first_key, first_artifact = _load_single(args.sources[0], device=device)
    second_key, second_artifact = _load_single(args.sources[1], device=device)
    width = int(first_key.numel())
    if int(second_key.numel()) != width:
        raise ValueError("source address widths do not match")
    canonical_keys = _random_keys(2, width, args.seed)

    memory = ExecutableArtifactMemory(
        args.bank, width=width, capacity=3, device=device
    )
    first_index = memory.put(canonical_keys[0], first_artifact)
    first_hash = memory.artifact_sha256[first_index]
    memory.evict()
    cold_after_first = ExecutableArtifactMemory.load(args.bank, device=device)
    _, first_cold_artifact = cold_after_first.promote_index(first_index)
    second_index = memory.put(canonical_keys[1], second_artifact)
    memory.validate()
    first_hash_after_append = memory.artifact_sha256[first_index]
    appended = ExecutableArtifactMemory.load(args.bank, device=device)
    _, first_after_append = appended.promote_index(first_index)
    _, second_after_append = appended.promote_index(second_index)

    memory.compact((first_index, second_index), args.compact)
    compacted_reloaded = ExecutableArtifactMemory.load(
        args.compact, device=device
    )
    compact_rows = compacted_reloaded.address_rows()
    compact_artifacts = [
        compacted_reloaded.promote_index(index)[1]
        for index, _ in compact_rows
    ]
    corruption_rejected = _corruption_control(
        compacted_reloaded,
        args.report.parent / "compact_corrupted",
        device=device,
    )
    report = {
        "schema": "online-executable-artifact-memory-audit-v1",
        "claim_boundary": (
            "The canonical artifact store can append a new opaque executable "
            "growth artifact after deployment, cold-reload it, evict hot "
            "cache entries, and compact selected rows without changing the "
            "stored tensors or learned address keys."
        ),
        "sources": [str(path) for path in args.sources],
        "address_policy": "canonical random opaque keys",
        "bank": str(args.bank),
        "compact": str(args.compact),
        "rows": {
            "first": first_index,
            "second": second_index,
            "compact": [index for index, _ in compact_rows],
        },
        "versions": {
            "after_first_write": cold_after_first.version,
            "after_second_write": appended.version,
            "after_compaction": compacted_reloaded.version,
        },
        "hashes": {
            "first_before_append": first_hash,
            "first_after_append": first_hash_after_append,
        },
        "artifact_digests": {
            "first_source": _artifact_digest(first_artifact),
            "first_cold_reload": _artifact_digest(first_cold_artifact),
            "first_after_append": _artifact_digest(first_after_append),
            "second_source": _artifact_digest(second_artifact),
            "second_after_append": _artifact_digest(second_after_append),
            "compact_first": _artifact_digest(compact_artifacts[0]),
            "compact_second": _artifact_digest(compact_artifacts[1]),
        },
        "gates": {
            "append_used_new_row": second_index == 1,
            "first_hash_preserved": (
                first_hash is not None
                and first_hash == first_hash_after_append
            ),
            "cold_reload_preserved_first": _same_artifacts(
                first_artifact, first_cold_artifact
            ),
            "append_reload_preserved_first": _same_artifacts(
                first_artifact, first_after_append
            ),
            "append_reload_preserved_second": _same_artifacts(
                second_artifact, second_after_append
            ),
            "compaction_preserved_first": _same_artifacts(
                first_artifact, compact_artifacts[0]
            ),
            "compaction_preserved_second": _same_artifacts(
                second_artifact, compact_artifacts[1]
            ),
            "corruption_rejected": corruption_rejected,
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, nargs=2, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=67001)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["gates"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
