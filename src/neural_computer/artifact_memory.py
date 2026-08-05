"""Persistent storage for opaque, controller-native learned artifacts.

The controller never owns this store's parameters. A caller may load an
artifact into a zero-initialized growth state, execute it through the generic
controller boundary, and later evict or replace it without changing the
frozen controller. The store knows only learned address keys, opaque artifact
payloads, integrity hashes, and generic strength statistics; it does not
interpret task names, semantic labels, or device protocols.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .memory import (
    MEMORY_READ_MATCH_THRESHOLD,
    ContentAddressedMemory,
    MemoryQuery,
)

ARTIFACT_MEMORY_SCHEMA = "neural-computer.executable-artifact-memory.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(text)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary_path.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ArtifactHandle:
    """Verified location and address evidence for one opaque artifact."""

    index: int
    confidence: float
    margin: float
    version: int


class ExecutableArtifactMemory:
    """Bounded hot/cold memory for controller-native learned artifacts.

    The artifact is intentionally an opaque mapping of tensor names to
    tensors. The memory can route and verify it, but it cannot execute it;
    execution remains the responsibility of the caller's generic frozen-core
    controller and growth-state loader. This keeps storage independently
    replaceable and prevents a memory backend from becoming a hidden
    modality- or task-specific reasoning branch.
    """

    def __init__(
        self,
        directory: Path,
        *,
        width: int,
        capacity: int = 8,
        device: torch.device | str = "cpu",
        write_threshold: float = 0.5,
        write_match_threshold: float = 0.999,
    ) -> None:
        if width < 1 or capacity < 1:
            raise ValueError("artifact memory width and capacity must be positive")
        if not 0.0 <= write_threshold <= 1.0:
            raise ValueError("write_threshold must lie in [0, 1]")
        if not 0.0 <= write_match_threshold <= 1.0:
            raise ValueError("write_match_threshold must lie in [0, 1]")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.capacity = int(capacity)
        self.write_threshold = float(write_threshold)
        self.write_match_threshold = float(write_match_threshold)
        self.rows = ContentAddressedMemory(
            width,
            capacity,
            write_threshold=write_threshold,
            write_match_threshold=write_match_threshold,
        ).to(device)
        self.paths: list[str | None] = [None] * capacity
        self.artifact_sha256: list[str | None] = [None] * capacity
        self.hot: dict[int, dict[str, torch.Tensor]] = {}
        self._manifest_version = 0
        self._load_manifest_if_present()

    @property
    def version(self) -> int:
        return int(self.rows.store_version.item())

    @property
    def occupied(self) -> tuple[int, ...]:
        return tuple(
            int(index)
            for index in torch.nonzero(self.rows.occupied, as_tuple=False)
            .reshape(-1)
            .tolist()
        )

    def address_rows(self) -> tuple[tuple[int, torch.Tensor], ...]:
        """Return occupied physical rows and detached opaque address keys."""
        return tuple(
            (index, self.rows.keys[index].detach().cpu().clone())
            for index in self.occupied
        )

    @staticmethod
    def _validate_key(key: torch.Tensor, width: int, name: str = "key") -> None:
        if key.shape != (width,):
            raise ValueError(f"{name} must have shape [{width}]")
        if not bool(torch.isfinite(key).all()):
            raise ValueError(f"{name} must contain only finite values")

    @staticmethod
    def _validate_artifact(
        artifact: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not isinstance(artifact, Mapping) or not artifact:
            raise ValueError("artifact must be a nonempty tensor mapping")
        normalized: dict[str, torch.Tensor] = {}
        for name, value in artifact.items():
            if not isinstance(name, str) or not name:
                raise ValueError("artifact tensor names must be nonempty strings")
            if not isinstance(value, torch.Tensor):
                raise TypeError("artifact values must be tensors")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("artifact tensors must contain only finite values")
            normalized[name] = value.detach().cpu().clone()
        return normalized

    def _manifest_payload(self) -> dict[str, Any]:
        return {
            "schema": ARTIFACT_MEMORY_SCHEMA,
            "width": self.width,
            "capacity": self.capacity,
            "write_threshold": self.write_threshold,
            "write_match_threshold": self.write_match_threshold,
            "paths": self.paths,
            "artifact_sha256": self.artifact_sha256,
            "manifest_version": self._manifest_version,
        }

    def _load_manifest_if_present(self) -> None:
        path = self.directory / "manifest.json"
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        if payload.get("schema") != ARTIFACT_MEMORY_SCHEMA:
            raise ValueError("unsupported artifact-memory schema")
        if int(payload.get("width", -1)) != self.width:
            raise ValueError("artifact-memory width metadata mismatch")
        if int(payload.get("capacity", -1)) != self.capacity:
            raise ValueError("artifact-memory capacity metadata mismatch")
        if float(payload.get("write_threshold", -1.0)) != self.write_threshold:
            raise ValueError("artifact-memory threshold metadata mismatch")
        if float(payload.get("write_match_threshold", -1.0)) != self.write_match_threshold:
            raise ValueError("artifact-memory match threshold metadata mismatch")
        paths = payload.get("paths")
        hashes = payload.get("artifact_sha256")
        if not isinstance(paths, list) or len(paths) != self.capacity:
            raise ValueError("artifact-memory paths metadata mismatch")
        if not isinstance(hashes, list) or len(hashes) != self.capacity:
            raise ValueError("artifact-memory hash metadata mismatch")
        if any(value is not None and not isinstance(value, str) for value in paths):
            raise ValueError("artifact-memory paths must contain strings or null")
        if any(value is not None and not isinstance(value, str) for value in hashes):
            raise ValueError("artifact-memory hashes must contain strings or null")
        self.paths = list(paths)
        self.artifact_sha256 = list(hashes)
        self._manifest_version = int(payload.get("manifest_version", 0))
        if self._manifest_version < 0:
            raise ValueError("artifact-memory manifest version cannot be negative")

    def _artifact_path(self) -> Path:
        return self.directory / (
            f"artifact-v{self._manifest_version + 1:08d}.pt"
        )

    def _free_or_coldest(self) -> int:
        free = torch.nonzero(~self.rows.occupied, as_tuple=False).reshape(-1)
        if free.numel():
            return int(free[0])
        return int(self.rows.strengths.argmin())

    @torch.no_grad()
    def put(
        self,
        key: torch.Tensor,
        artifact: Mapping[str, torch.Tensor],
        *,
        strength: float = 1.0,
    ) -> int:
        """Atomically add one learned artifact and return its physical row."""
        self._validate_key(key, self.width)
        normalized_artifact = self._validate_artifact(artifact)
        if not self.write_threshold < strength <= 1.0:
            raise ValueError("artifact strength must exceed write_threshold and be <= 1")
        target_path = self._artifact_path()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target_path.parent, prefix=f".{target_path.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(normalized_artifact, temporary_path)
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary_path.replace(target_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        receipt = self.rows.write(
            key.detach().to(self.rows.keys).unsqueeze(0),
            key.detach().to(self.rows.values).unsqueeze(0),
            torch.tensor([strength], device=self.rows.keys.device),
        )
        if not bool(receipt.committed[0]) or int(receipt.indices[0]) < 0:
            raise RuntimeError("artifact write did not commit")
        index = int(receipt.indices[0])
        self.hot.pop(index, None)
        self.paths[index] = target_path.name
        self.artifact_sha256[index] = _sha256_file(target_path)
        self._manifest_version += 1
        self.save()
        return index

    @torch.no_grad()
    def _resolve(self, query: torch.Tensor) -> ArtifactHandle:
        self._validate_key(query, self.width, "query")
        read = self.rows.read(MemoryQuery(query.unsqueeze(0), top_k=2))
        if not bool(read.hit[0]):
            raise LookupError("artifact query did not meet the read threshold")
        scores = read.scores[0]
        index = int(read.indices[0, 0])
        confidence = float(scores[0])
        margin = float(scores[0] - scores[1]) if scores.numel() > 1 else 1.0
        return ArtifactHandle(index, confidence, margin, self.version)

    def _load_verified(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.capacity or self.paths[index] is None:
            raise LookupError("artifact row is empty")
        path = self.directory / self.paths[index]
        expected = self.artifact_sha256[index]
        if expected is None or not path.is_file():
            raise ValueError("artifact row is missing its integrity record")
        if _sha256_file(path) != expected:
            raise ValueError("artifact hash mismatch; refusing corrupted artifact")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return self._validate_artifact(payload)

    def promote(
        self, query: torch.Tensor
    ) -> tuple[ArtifactHandle, dict[str, torch.Tensor]]:
        """Resolve, verify, and load an artifact into the hot process cache."""
        handle = self._resolve(query)
        artifact = self._load_verified(handle.index)
        self.hot[handle.index] = artifact
        return handle, artifact

    @torch.no_grad()
    def promote_candidates(
        self, query: torch.Tensor, *, top_k: int = 2
    ) -> tuple[tuple[ArtifactHandle, ...], tuple[dict[str, torch.Tensor], ...]]:
        """Verify and load every strong opaque candidate for one query.

        The method exposes generic top-k retrieval for compositional callers.
        The memory backend still knows only learned keys, tensor payloads, and
        integrity metadata; deciding whether to execute one candidate or
        compose several remains outside the storage boundary.
        """
        self._validate_key(query, self.width, "query")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        read = self.rows.read(
            MemoryQuery(query.unsqueeze(0), top_k=min(top_k, self.capacity))
        )
        handles: list[ArtifactHandle] = []
        artifacts: list[dict[str, torch.Tensor]] = []
        scores = read.scores[0]
        indices = read.indices[0]
        for position, (score, index) in enumerate(zip(scores, indices)):
            if (
                not bool(torch.isfinite(score))
                or float(score) < MEMORY_READ_MATCH_THRESHOLD
            ):
                continue
            row = int(index)
            artifact = self._load_verified(row)
            next_score = (
                float(scores[position + 1])
                if position + 1 < scores.numel()
                and bool(torch.isfinite(scores[position + 1]))
                else 0.0
            )
            handles.append(
                ArtifactHandle(
                    row,
                    max(0.0, min(1.0, float(score))),
                    max(0.0, float(score) - next_score),
                    self.version,
                )
            )
            artifacts.append(artifact)
            self.hot[row] = artifact
        if not handles:
            raise LookupError("artifact query had no verified candidates")
        return tuple(handles), tuple(artifacts)

    def promote_index(
        self,
        index: int,
        *,
        confidence: float = 1.0,
        margin: float = 0.0,
    ) -> tuple[ArtifactHandle, dict[str, torch.Tensor]]:
        """Verify and load a row selected by an external memory-side router."""
        if index not in self.occupied:
            raise LookupError("artifact row is empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if margin < 0.0:
            raise ValueError("margin must be nonnegative")
        artifact = self._load_verified(index)
        self.hot[index] = artifact
        return ArtifactHandle(index, confidence, margin, self.version), artifact

    def evict(self, index: int | None = None) -> None:
        """Evict hot artifacts without deleting their persistent files."""
        if index is None:
            self.hot.clear()
        else:
            self.hot.pop(index, None)

    def save(self) -> None:
        """Persist the address rows and manifest through atomic snapshots."""
        self.rows.snapshot(self.directory / "rows.pt")
        _atomic_text_write(
            self.directory / "manifest.json",
            json.dumps(self._manifest_payload(), indent=2, sort_keys=True) + "\n",
        )

    def validate(self) -> None:
        """Verify metadata, rows, and every live artifact before serving them."""
        self.rows.validate_state()
        if len(self.paths) != self.capacity or len(self.artifact_sha256) != self.capacity:
            raise ValueError("artifact-memory metadata has the wrong capacity")
        for index in self.occupied:
            self._load_verified(index)

    def compact(
        self, indices: Sequence[int], destination: Path
    ) -> ExecutableArtifactMemory:
        """Create a smaller verified store containing selected live artifacts."""
        selected = tuple(int(index) for index in indices)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("compact requires distinct nonempty row indices")
        if any(index not in self.occupied for index in selected):
            raise ValueError("compact indices must refer to occupied rows")
        compacted = ExecutableArtifactMemory(
            destination,
            width=self.width,
            capacity=len(selected),
            device=self.rows.keys.device,
            write_threshold=self.write_threshold,
            write_match_threshold=self.write_match_threshold,
        )
        for index in selected:
            artifact = self._load_verified(index)
            compacted.put(
                self.rows.keys[index].detach().cpu(),
                artifact,
                strength=float(self.rows.strengths[index]),
            )
        compacted.validate()
        return compacted

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        device: torch.device | str = "cpu",
    ) -> ExecutableArtifactMemory:
        directory = Path(directory)
        payload = json.loads((directory / "manifest.json").read_text())
        instance = cls(
            directory,
            width=int(payload["width"]),
            capacity=int(payload["capacity"]),
            device=device,
            write_threshold=float(payload["write_threshold"]),
            write_match_threshold=float(payload["write_match_threshold"]),
        )
        rows_path = directory / "rows.pt"
        if not rows_path.exists():
            raise ValueError("artifact-memory rows snapshot is missing")
        instance.rows.load_snapshot(rows_path, map_location=device)
        instance.validate()
        return instance
