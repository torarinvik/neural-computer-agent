"""Shared diagnostics for sequential external-capability pressure tests."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from neural_computer import ExecutableArtifactMemory, RetentionPolicyConfig

MASTERY_THRESHOLD = 0.75


def digest_artifact(artifact: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(artifact.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def digest_artifact_bank(bank: ExecutableArtifactMemory) -> str:
    """Digest opaque row metadata and verified artifact bytes for reload checks."""

    digest = hashlib.sha256()
    for index in bank.occupied:
        digest.update(str(index).encode("utf-8"))
        digest.update(
            bank.rows.keys[index].detach().cpu().contiguous().numpy().tobytes()
        )
        if bank.paths[index] is not None:
            digest.update(digest_artifact(bank._load_verified(index)).encode("utf-8"))
        for alias in bank.alias_keys[index]:
            digest.update(alias.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_bank(
    path: Path,
    *,
    reversal_patience: int,
    observations: int,
) -> ExecutableArtifactMemory:
    bank = ExecutableArtifactMemory(
        path,
        width=48,
        capacity=1,
        write_match_threshold=0.99999,
    )
    bank.retention.config = RetentionPolicyConfig(
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=observations,
        reversal_threshold=0.5,
        reversal_patience=reversal_patience,
    )
    return bank


def reversal_recovery(
    bank: ExecutableArtifactMemory,
    key: torch.Tensor,
    row: int,
    *,
    reversal_patience: int,
    recovery_observations: int,
) -> dict[str, object]:
    before = bank.retention.status(key)
    bank.observe_retention_batch(
        tuple((key, 0.0) for _ in range(reversal_patience))
    )
    after_reversal = bank.retention.status(key)
    row_protected_after_reversal = bool(bank.protection_mask()[row])
    bank.observe_retention_batch(
        tuple((key, 1.0) for _ in range(recovery_observations))
    )
    recovered = bank.retention.status(key)
    return {
        "before": asdict(before),
        "after_reversal": asdict(after_reversal),
        "after_reversal_row_protected": row_protected_after_reversal,
        "after_recovery": asdict(recovered),
        "reversal_detected": after_reversal.reversal_count == before.reversal_count + 1,
        "alias_released": not after_reversal.protected,
        "alias_recovered": recovered.protected,
    }


def corruption_control(
    bank: ExecutableArtifactMemory,
    destination: Path,
) -> dict[str, object]:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(bank.directory, destination)
    row = bank.occupied[0]
    artifact_name = bank.paths[row]
    if artifact_name is None:
        return {"rejected": False, "reason": "selected row has no artifact path"}
    artifact_path = destination / artifact_name
    with artifact_path.open("ab") as stream:
        stream.write(b"corrupted-after-checksum")
    try:
        ExecutableArtifactMemory.load(destination)
    except (OSError, RuntimeError, ValueError, EOFError, KeyError) as error:
        return {"rejected": True, "error_type": type(error).__name__}
    return {"rejected": False, "reason": "corrupted artifact loaded"}
