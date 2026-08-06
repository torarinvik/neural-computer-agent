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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .memory import (
    MEMORY_READ_MATCH_THRESHOLD,
    ContentAddressedMemory,
    MemoryCandidates,
)
from .retention import (
    CapabilityRetentionLedger,
    CapabilityRetentionProbe,
    evaluate_retention_gate,
)

LEGACY_ARTIFACT_MEMORY_SCHEMA = "neural-computer.executable-artifact-memory.v1"
ARTIFACT_MEMORY_SCHEMA = "neural-computer.executable-artifact-memory.v2"

CandidateRetentionOutcomes = (
    Sequence[float]
    | torch.Tensor
    | Sequence[CapabilityRetentionProbe]
)

ArtifactBinding = Mapping[str, Any]


def _normalize_alias_binding(binding: ArtifactBinding | None) -> dict[str, Any] | None:
    """Copy one JSON-safe opaque execution binding for durable storage."""

    if binding is None:
        return None
    if not isinstance(binding, Mapping):
        raise TypeError("artifact alias bindings must be mappings or null")
    try:
        normalized = json.loads(json.dumps(dict(binding), sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise TypeError("artifact alias bindings must be JSON serializable") from exc
    if not isinstance(normalized, dict):
        raise TypeError("artifact alias binding must normalize to an object")
    return normalized


def _structured_retention_outcomes(
    outcomes: CandidateRetentionOutcomes,
) -> tuple[CapabilityRetentionProbe, ...] | None:
    """Recognize per-capability candidate outcomes without ambiguity."""

    if isinstance(outcomes, torch.Tensor) or not isinstance(outcomes, Sequence):
        return None
    if not outcomes:
        return None
    if all(isinstance(item, CapabilityRetentionProbe) for item in outcomes):
        return tuple(outcomes)
    if any(isinstance(item, CapabilityRetentionProbe) for item in outcomes):
        raise TypeError(
            "candidate retention probes must contain only "
            "CapabilityRetentionProbe values"
        )
    return None


def _structured_retention_gate(
    outcomes: Sequence[CapabilityRetentionProbe],
    retained_scores: Sequence[float] | torch.Tensor,
    *,
    candidate_threshold: float,
    retention_floor: float,
    min_candidate_observations: int,
) -> tuple[bool, str]:
    """Require stable retention independently for every opaque capability."""

    retained = torch.as_tensor(retained_scores, dtype=torch.float64).reshape(-1)
    if retained.numel() != len(outcomes):
        raise ValueError(
            "per-capability candidate probes must align with retained scores"
        )
    for index, (probe, floor) in enumerate(zip(outcomes, retained.tolist())):
        decision = evaluate_retention_gate(
            probe.outcomes,
            [floor],
            candidate_threshold=candidate_threshold,
            retention_floor=retention_floor,
            min_candidate_observations=min_candidate_observations,
        )
        if not decision.accepted:
            return False, f"capability {index}: {decision.reason}"
    return True, "per-capability candidate mastery and retention passed"


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
    view: str | None = None
    binding: dict[str, Any] | None = None


@dataclass(frozen=True)
class ArtifactConsolidationReceipt:
    """Auditable result for a caller-verified smaller artifact bank."""

    accepted: bool
    source_indices: tuple[int, ...]
    rows_before: int
    rows_after: int
    rows_saved: int
    reason: str = ""


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
        retention_ledger: CapabilityRetentionLedger | None = None,
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
        if retention_ledger is not None and retention_ledger.width != self.width:
            raise ValueError("retention ledger width must match artifact memory")
        self.retention = retention_ledger or CapabilityRetentionLedger(self.width)
        self.rows = ContentAddressedMemory(
            width,
            capacity,
            write_threshold=write_threshold,
            write_match_threshold=write_match_threshold,
        ).to(device)
        self.paths: list[str | None] = [None] * capacity
        self.artifact_sha256: list[str | None] = [None] * capacity
        self.alias_keys: list[list[torch.Tensor]] = [[] for _ in range(capacity)]
        self.alias_views: list[list[str | None]] = [[] for _ in range(capacity)]
        self.alias_bindings: list[list[dict[str, Any] | None]] = [
            [] for _ in range(capacity)
        ]
        self.hot: dict[int, dict[str, torch.Tensor]] = {}
        self._manifest_version = 0
        self._load_manifest_if_present()
        retention_path = self._retention_path()
        if retention_ledger is None and retention_path.exists():
            self.retention = CapabilityRetentionLedger.load(retention_path)

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

    def protection_mask(self) -> torch.Tensor:
        """Return the current opaque protection state for every physical row."""

        return torch.tensor(
            [self._row_is_protected(index) for index in range(self.capacity)],
            dtype=torch.bool,
        )

    def address_rows(self) -> tuple[tuple[int, torch.Tensor], ...]:
        """Return occupied physical rows and detached opaque address keys."""
        return tuple(
            (index, self.rows.keys[index].detach().cpu().clone())
            for index in self.occupied
        )

    def planner_candidates(self) -> MemoryCandidates:
        """Expose opaque row summaries to a replaceable capacity planner.

        Tensor names and artifact contents are never interpreted.  A stable
        sorted flattening creates a fixed-width learned value summary solely
        so a memory-side policy can compare rows; execution and verification
        remain outside this storage helper.
        """

        keys = self.rows.keys.detach().cpu().clone().unsqueeze(0)
        values = torch.zeros(1, self.capacity, self.width, dtype=torch.float32)
        for index in self.occupied:
            artifact = self._load_verified(index)
            tensors = [
                value.reshape(-1).to(dtype=torch.float32)
                for _name, value in sorted(artifact.items())
            ]
            flat = torch.cat(tensors)
            positions = torch.linspace(0, flat.numel() - 1, self.width).round().long()
            values[0, index] = flat[positions]
        return MemoryCandidates(
            keys=keys,
            values=values,
            strengths=self.rows.strengths.detach().cpu().clone().unsqueeze(0),
            timestamps=self.rows.timestamps.detach().cpu().clone().unsqueeze(0),
            occupied=self.rows.occupied.detach().cpu().clone().unsqueeze(0),
        )

    def view_candidates(
        self,
    ) -> tuple[tuple[int, torch.Tensor, str], ...]:
        """Return opaque alias addresses that identify executable views.

        The backend exposes the address evidence and opaque view token without
        assigning either semantic meaning. A replaceable memory-side router
        may score these candidates, then call :meth:`promote_view` to obtain a
        verified artifact for the selected view.
        """
        candidates: list[tuple[int, torch.Tensor, str]] = []
        for index in self.occupied:
            for key, view in zip(
                self.alias_keys[index], self.alias_views[index], strict=True
            ):
                if view is not None:
                    candidates.append((index, key.detach().cpu().clone(), view))
        return tuple(candidates)

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
            "alias_keys": [
                [alias.tolist() for alias in aliases] for aliases in self.alias_keys
            ],
            "alias_views": self.alias_views,
            "alias_bindings": [
                list(self._row_alias_bindings(index)) for index in range(self.capacity)
            ],
            "manifest_version": self._manifest_version,
        }

    def _load_manifest_if_present(self) -> None:
        path = self.directory / "manifest.json"
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        if payload.get("schema") not in {
            LEGACY_ARTIFACT_MEMORY_SCHEMA,
            ARTIFACT_MEMORY_SCHEMA,
        }:
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
        aliases = payload.get("alias_keys", [[] for _ in range(self.capacity)])
        if not isinstance(aliases, list) or len(aliases) != self.capacity:
            raise ValueError("artifact-memory alias metadata mismatch")
        self.alias_keys = []
        alias_views = payload.get("alias_views", [[] for _ in range(self.capacity)])
        if not isinstance(alias_views, list) or len(alias_views) != self.capacity:
            raise ValueError("artifact-memory alias view metadata mismatch")
        self.alias_views = []
        alias_bindings = payload.get(
            "alias_bindings",
            [[None] * len(row_aliases) for row_aliases in aliases],
        )
        if not isinstance(alias_bindings, list) or len(alias_bindings) != self.capacity:
            raise ValueError("artifact-memory alias binding metadata mismatch")
        self.alias_bindings = []
        for row_aliases in aliases:
            if not isinstance(row_aliases, list):
                raise TypeError("artifact-memory aliases must be lists")
            normalized_aliases: list[torch.Tensor] = []
            for alias in row_aliases:
                tensor = torch.tensor(alias, dtype=torch.float32)
                self._validate_key(tensor, self.width, "alias key")
                normalized_aliases.append(tensor)
            self.alias_keys.append(normalized_aliases)
        for row_views in alias_views:
            if not isinstance(row_views, list):
                raise TypeError("artifact-memory alias views must be lists")
            if len(row_views) != len(self.alias_keys[len(self.alias_views)]):
                raise ValueError("artifact-memory alias views must align with keys")
            if any(view is not None and not isinstance(view, str) for view in row_views):
                raise TypeError("artifact-memory alias views must contain strings or null")
            self.alias_views.append(list(row_views))
        for row_bindings, row_aliases in zip(
            alias_bindings,
            self.alias_keys,
            strict=True,
        ):
            if not isinstance(row_bindings, list):
                raise TypeError("artifact-memory alias bindings must be lists")
            if len(row_bindings) != len(row_aliases):
                raise ValueError("artifact-memory alias bindings must align with keys")
            self.alias_bindings.append(
                [_normalize_alias_binding(binding) for binding in row_bindings]
            )
        self._manifest_version = int(payload.get("manifest_version", 0))
        if self._manifest_version < 0:
            raise ValueError("artifact-memory manifest version cannot be negative")

    def _artifact_path(self) -> Path:
        return self.directory / (
            f"artifact-v{self._manifest_version + 1:08d}.pt"
        )

    def _retention_path(self) -> Path:
        return self.directory / "retention-ledger.json"

    def _row_retention_keys(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return the primary and every opaque alias key for one row."""
        return (
            self.rows.keys[index].detach().cpu(),
            *(alias.detach().cpu() for alias in self.alias_keys[index]),
        )

    def _row_alias_bindings(
        self, index: int
    ) -> tuple[dict[str, Any] | None, ...]:
        """Return alias bindings aligned with keys, including legacy nulls."""

        return tuple(
            self.alias_bindings[index][position]
            if position < len(self.alias_bindings[index])
            else None
            for position in range(len(self.alias_keys[index]))
        )

    def _row_is_protected(self, index: int) -> bool:
        """Treat a mastered alias as protecting its complete physical row."""
        return any(
            self.retention.is_protected(key)
            for key in self._row_retention_keys(index)
        )

    def _choose_eviction_position(
        self, occupied_indices: Sequence[int], scores: torch.Tensor
    ) -> int | None:
        """Choose an unprotected row while masking primary keys and aliases."""
        normalized_scores = scores.reshape(-1)
        if normalized_scores.shape[0] != len(occupied_indices):
            raise ValueError("eviction scores must align with occupied artifact rows")
        if not bool(torch.isfinite(normalized_scores).all()):
            raise ValueError("eviction scores must be finite")
        protected = torch.tensor(
            [self._row_is_protected(index) for index in occupied_indices],
            dtype=torch.bool,
            device=normalized_scores.device,
        )
        masked = normalized_scores.masked_fill(protected, -torch.inf)
        if bool(protected.all()):
            return None
        return int(masked.argmax().item())

    def _free_or_coldest(self) -> int:
        free = torch.nonzero(~self.rows.occupied, as_tuple=False).reshape(-1)
        if free.numel():
            return int(free[0])
        return int(self.rows.strengths.argmin())

    def _matching_row(self, key: torch.Tensor) -> int | None:
        occupied = self.occupied
        if not occupied:
            return None
        normalized_key = torch.nn.functional.normalize(
            key.to(device=self.rows.keys.device, dtype=self.rows.keys.dtype), dim=0
        )
        row_keys = torch.stack([self.rows.keys[index] for index in occupied])
        scores = torch.nn.functional.normalize(row_keys, dim=-1) @ normalized_key
        best_score, best_position = scores.max(dim=0)
        if float(best_score) < self.write_match_threshold:
            return None
        return occupied[int(best_position)]

    def observe_retention(
        self, key: torch.Tensor, outcome: float | torch.Tensor
    ) -> None:
        """Update memory-side mastery state and persist it without replay."""

        self._validate_key(key, self.width)
        self.retention.observe(key, outcome)
        self.save()

    def observe_retention_batch(
        self,
        observations: Sequence[tuple[torch.Tensor, float | torch.Tensor]],
    ) -> None:
        """Record ordered verifier outcomes and persist the bank once."""

        entries = tuple(observations)
        if not entries:
            return
        for key, _outcome in entries:
            self._validate_key(key, self.width)
        for key, outcome in entries:
            self.retention.observe(key, outcome)
        self.save()

    @torch.no_grad()
    def put(
        self,
        key: torch.Tensor,
        artifact: Mapping[str, torch.Tensor],
        *,
        strength: float = 1.0,
        eviction_scores: torch.Tensor | None = None,
    ) -> int:
        """Atomically add an artifact without evicting protected capabilities.

        ``eviction_scores`` use the convention that larger values are more
        disposable.  If every occupied row is protected, the write fails
        explicitly so a caller can grow or transactionally consolidate the
        bank instead of silently forgetting a mastered capability.
        """
        self._validate_key(key, self.width)
        normalized_artifact = self._validate_artifact(artifact)
        if not self.write_threshold < strength <= 1.0:
            raise ValueError("artifact strength must exceed write_threshold and be <= 1")
        matching_row = self._matching_row(key)
        target_index: torch.Tensor | None = None
        if matching_row is None and len(self.occupied) == self.capacity:
            if eviction_scores is None:
                candidate_scores = torch.zeros(
                    len(self.occupied), device=self.rows.keys.device
                )
            else:
                candidate_scores = eviction_scores.reshape(-1).to(
                    device=self.rows.keys.device, dtype=torch.float32
                )
                if candidate_scores.shape[0] == self.capacity:
                    candidate_scores = candidate_scores[
                        torch.tensor(
                            self.occupied,
                            dtype=torch.long,
                            device=self.rows.keys.device,
                        )
                    ]
                if candidate_scores.shape[0] != len(self.occupied):
                    raise ValueError(
                        "eviction scores must align with occupied artifact rows"
                    )
            candidate_position = self._choose_eviction_position(
                self.occupied, candidate_scores
            )
            if candidate_position is None:
                raise MemoryError(
                    "all occupied artifact capabilities are protected; "
                    "grow or consolidate the memory bank"
                )
            target_index = torch.tensor(
                [self.occupied[candidate_position]],
                dtype=torch.long,
                device=self.rows.keys.device,
            )
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
            target_index=target_index,
        )
        if not bool(receipt.committed[0]) or int(receipt.indices[0]) < 0:
            raise RuntimeError("artifact write did not commit")
        index = int(receipt.indices[0])
        self.hot.pop(index, None)
        self.paths[index] = target_path.name
        self.artifact_sha256[index] = _sha256_file(target_path)
        self.alias_keys[index] = []
        self.alias_views[index] = []
        self.alias_bindings[index] = []
        self._manifest_version += 1
        self.save()
        return index

    @torch.no_grad()
    def _resolve(self, query: torch.Tensor) -> ArtifactHandle:
        self._validate_key(query, self.width, "query")
        scores, views, bindings = self._address_matches(query)
        top_scores, top_indices = torch.topk(
            scores, k=min(2, self.capacity)
        )
        if not bool(torch.isfinite(top_scores[0])) or float(
            top_scores[0]
        ) < MEMORY_READ_MATCH_THRESHOLD:
            raise LookupError("artifact query did not meet the read threshold")
        index = int(top_indices[0])
        confidence = float(top_scores[0])
        margin = float(top_scores[0] - top_scores[1]) if top_scores.numel() > 1 else 1.0
        return ArtifactHandle(
            index,
            confidence,
            margin,
            self.version,
            views[index],
            bindings[index],
        )

    @torch.no_grad()
    def _address_scores(self, query: torch.Tensor) -> torch.Tensor:
        """Score each artifact row by its primary or aliased opaque address."""
        return self._address_matches(query)[0]

    @torch.no_grad()
    def _address_matches(
        self, query: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        list[str | None],
        list[dict[str, Any] | None],
    ]:
        """Return row scores and the opaque view selected for each row."""
        normalized_query = torch.nn.functional.normalize(
            query.to(device=self.rows.keys.device, dtype=self.rows.keys.dtype), dim=0
        )
        scores = torch.full(
            (self.capacity,), -torch.inf, device=self.rows.keys.device
        )
        views: list[str | None] = [None] * self.capacity
        bindings: list[dict[str, Any] | None] = [None] * self.capacity
        for index in self.occupied:
            candidates = [self.rows.keys[index]] + [
                alias.to(device=self.rows.keys.device, dtype=self.rows.keys.dtype)
                for alias in self.alias_keys[index]
            ]
            candidate_keys = torch.stack(candidates)
            score = torch.nn.functional.normalize(candidate_keys, dim=-1) @ normalized_query
            best = int(score.argmax())
            scores[index] = score[best]
            views[index] = (
                self.alias_views[index][best - 1] if best > 0 else None
            )
            alias_bindings = self._row_alias_bindings(index)
            bindings[index] = alias_bindings[best - 1] if best > 0 else None
        return scores, views, bindings

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
        handles: list[ArtifactHandle] = []
        artifacts: list[dict[str, torch.Tensor]] = []
        address_scores, views, bindings = self._address_matches(query)
        scores, indices = torch.topk(address_scores, k=min(top_k, self.capacity))
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
                    views[row],
                    bindings[row],
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
        self.retention.save(self._retention_path())
        _atomic_text_write(
            self.directory / "manifest.json",
            json.dumps(self._manifest_payload(), indent=2, sort_keys=True) + "\n",
        )

    def validate(self) -> None:
        """Verify metadata, rows, and every live artifact before serving them."""
        self.rows.validate_state()
        self.retention.validate()
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
        selected_set = set(selected)
        dropped_protected = tuple(
            index
            for index in self.occupied
            if index not in selected_set
            and self._row_is_protected(index)
        )
        if dropped_protected:
            raise MemoryError(
                "cannot compact away protected artifact capabilities: "
                f"{dropped_protected}"
            )
        compacted = ExecutableArtifactMemory(
            destination,
            width=self.width,
            capacity=len(selected),
            device=self.rows.keys.device,
            write_threshold=self.write_threshold,
            write_match_threshold=self.write_match_threshold,
        )
        compacted.retention = self.retention.subset(
            [
                key
                for index in selected
                for key in self._row_retention_keys(index)
            ]
        )
        for index in selected:
            artifact = self._load_verified(index)
            compacted_index = compacted.put(
                self.rows.keys[index].detach().cpu(),
                artifact,
                strength=float(self.rows.strengths[index]),
            )
            compacted.alias_keys[compacted_index] = [
                alias.detach().cpu().clone() for alias in self.alias_keys[index]
            ]
            compacted.alias_views[compacted_index] = list(self.alias_views[index])
            compacted.alias_bindings[compacted_index] = list(
                self._row_alias_bindings(index)
            )
        compacted.save()
        compacted.validate()
        return compacted

    def grow(self, destination: Path, capacity: int) -> ExecutableArtifactMemory:
        """Create a larger verified store without mutating this source.

        Growth is the explicit escape hatch after a protected write refuses to
        evict a mastered row.  It copies every live artifact, opaque alias,
        strength, and retention record into a new capacity while preserving
        integrity verification.  Callers still decide when the returned store
        becomes canonical; the source remains available for rollback or audit.
        """
        if not isinstance(capacity, int) or capacity <= self.capacity:
            raise ValueError("grown artifact capacity must exceed current capacity")
        destination = Path(destination)
        if destination.resolve() == self.directory.resolve():
            raise ValueError("grown artifact memory must use a new destination")
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("grown artifact destination must be empty")
        grown = ExecutableArtifactMemory(
            destination,
            width=self.width,
            capacity=capacity,
            device=self.rows.keys.device,
            write_threshold=self.write_threshold,
            write_match_threshold=self.write_match_threshold,
        )
        grown.retention = self.retention.subset(
            [
                key
                for index in self.occupied
                for key in self._row_retention_keys(index)
            ]
        )
        for index in self.occupied:
            artifact = self._load_verified(index)
            grown_index = grown.put(
                self.rows.keys[index].detach().cpu(),
                artifact,
                strength=float(self.rows.strengths[index]),
            )
            grown.alias_keys[grown_index] = [
                alias.detach().cpu().clone() for alias in self.alias_keys[index]
            ]
            grown.alias_views[grown_index] = list(self.alias_views[index])
            grown.alias_bindings[grown_index] = list(self._row_alias_bindings(index))
        grown.save()
        grown.validate()
        return grown

    def consolidate_verified(
        self,
        source_indices: Sequence[int],
        replacement_key: torch.Tensor,
        replacement_artifact: Mapping[str, torch.Tensor],
        destination: Path,
        *,
        verifier: Callable[[ExecutableArtifactMemory], bool],
        strength: float = 1.0,
        replacement_aliases: Sequence[torch.Tensor] = (),
        replacement_alias_views: Sequence[str | None] = (),
        replacement_alias_bindings: Sequence[ArtifactBinding | None] = (),
        candidate_outcomes: CandidateRetentionOutcomes | None = None,
        candidate_outcome_probe: (
            Callable[[ExecutableArtifactMemory], CandidateRetentionOutcomes]
            | None
        ) = None,
        retained_scores: Sequence[float] | torch.Tensor | None = None,
        candidate_threshold: float = 0.8,
        retention_floor: float = 0.8,
        min_candidate_observations: int = 8,
    ) -> tuple[ExecutableArtifactMemory | None, ArtifactConsolidationReceipt]:
        """Build and externally verify a smaller bank before adoption.

        The replacement is opaque to the store. The caller may create it by
        composing or compressing learned growth state, then supply a
        behavior-only verifier over the candidate bank. The source bank is
        never mutated; callers adopt the returned candidate only when the
        verifier passes. This makes consolidation transactional at the
        memory boundary without introducing task, modality, or protocol
        semantics into storage.
        """
        selected = tuple(int(index) for index in source_indices)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("consolidation requires distinct nonempty source rows")
        if any(index not in self.occupied for index in selected):
            raise ValueError("consolidation sources must refer to occupied rows")
        if not callable(verifier):
            raise TypeError("consolidation verifier must be callable")
        if candidate_outcomes is not None and candidate_outcome_probe is not None:
            raise ValueError(
                "supply static candidate outcomes or a candidate outcome probe, "
                "not both"
            )
        if candidate_outcome_probe is not None and retained_scores is None:
            raise ValueError(
                "candidate outcome probes require retained capability scores"
            )
        selected_protected = tuple(
            index
            for index in selected
            if self._row_is_protected(index)
        )
        if (
            selected_protected
            and candidate_outcomes is None
            and candidate_outcome_probe is None
        ):
            raise ValueError(
                "consolidating protected artifact capabilities requires "
                "candidate outcomes or a candidate outcome probe"
            )
        if (
            candidate_outcome_probe is None
            and (candidate_outcomes is None) != (retained_scores is None)
        ):
            raise ValueError(
                "candidate outcomes and retained scores must be supplied together"
            )
        if candidate_outcomes is not None and retained_scores is not None:
            structured = _structured_retention_outcomes(candidate_outcomes)
            if structured is None:
                retention_decision = evaluate_retention_gate(
                    candidate_outcomes,
                    retained_scores,
                    candidate_threshold=candidate_threshold,
                    retention_floor=retention_floor,
                    min_candidate_observations=min_candidate_observations,
                )
                retention_accepted = retention_decision.accepted
                retention_reason = retention_decision.reason
            else:
                retention_accepted, retention_reason = _structured_retention_gate(
                    structured,
                    retained_scores,
                    candidate_threshold=candidate_threshold,
                    retention_floor=retention_floor,
                    min_candidate_observations=min_candidate_observations,
                )
            if not retention_accepted:
                rows_before = len(self.occupied)
                return None, ArtifactConsolidationReceipt(
                    accepted=False,
                    source_indices=selected,
                    rows_before=rows_before,
                    rows_after=rows_before,
                    rows_saved=0,
                    reason=retention_reason,
                )
        survivors = tuple(index for index in self.occupied if index not in selected)
        rows_before = len(self.occupied)
        compacted = ExecutableArtifactMemory(
            destination,
            width=self.width,
            capacity=len(survivors) + 1,
            device=self.rows.keys.device,
            write_threshold=self.write_threshold,
            write_match_threshold=self.write_match_threshold,
        )
        compacted.retention = self.retention.subset(
            [
                key
                for index in self.occupied
                if index not in selected
                for key in self._row_retention_keys(index)
            ]
        )
        replacement_index = compacted.put(
            replacement_key, replacement_artifact, strength=strength
        )
        normalized_aliases: list[torch.Tensor] = []
        for alias in replacement_aliases:
            self._validate_key(alias, self.width, "replacement alias key")
            normalized_aliases.append(alias.detach().cpu().clone())
        if replacement_alias_views and len(replacement_alias_views) != len(
            normalized_aliases
        ):
            raise ValueError("replacement alias views must align with aliases")
        normalized_alias_views = list(replacement_alias_views or (None,) * len(normalized_aliases))
        if any(view is not None and not isinstance(view, str) for view in normalized_alias_views):
            raise TypeError("replacement alias views must contain strings or null")
        named_views = [view for view in normalized_alias_views if view is not None]
        if len(set(named_views)) != len(named_views):
            raise ValueError("replacement alias views must be unique")
        if replacement_alias_bindings and len(replacement_alias_bindings) != len(
            normalized_aliases
        ):
            raise ValueError("replacement alias bindings must align with aliases")
        normalized_alias_bindings = [
            _normalize_alias_binding(binding)
            for binding in (
                replacement_alias_bindings
                or (None,) * len(normalized_aliases)
            )
        ]
        compacted.alias_keys[replacement_index] = normalized_aliases
        compacted.alias_views[replacement_index] = normalized_alias_views
        compacted.alias_bindings[replacement_index] = normalized_alias_bindings
        if candidate_outcomes is not None:
            structured = _structured_retention_outcomes(candidate_outcomes)
            if structured is None:
                for outcome in torch.as_tensor(candidate_outcomes).reshape(-1):
                    compacted.retention.observe(replacement_key, outcome)
            else:
                candidate_keys = compacted._row_retention_keys(replacement_index)
                for probe in structured:
                    compacted._validate_key(
                        probe.key, compacted.width, "candidate retention probe key"
                    )
                    probe_key = probe.key.detach().to(
                        device="cpu", dtype=torch.float32
                    ).contiguous()
                    if not any(torch.equal(probe_key, key) for key in candidate_keys):
                        raise ValueError(
                            "candidate retention probe key must address the "
                            "replacement row"
                        )
                    for outcome in torch.as_tensor(probe.outcomes).reshape(-1):
                        compacted.retention.observe(probe_key, outcome)
        for index in survivors:
            compacted_index = compacted.put(
                self.rows.keys[index].detach().cpu(),
                self._load_verified(index),
                strength=float(self.rows.strengths[index]),
            )
            compacted.alias_keys[compacted_index] = [
                alias.detach().cpu().clone() for alias in self.alias_keys[index]
            ]
            compacted.alias_views[compacted_index] = list(self.alias_views[index])
            compacted.alias_bindings[compacted_index] = list(
                self._row_alias_bindings(index)
            )
        compacted.save()
        compacted.validate()
        if candidate_outcome_probe is not None:
            resolved_candidate_outcomes = candidate_outcome_probe(compacted)
            structured = _structured_retention_outcomes(resolved_candidate_outcomes)
            if structured is None:
                retention_decision = evaluate_retention_gate(
                    resolved_candidate_outcomes,
                    retained_scores,
                    candidate_threshold=candidate_threshold,
                    retention_floor=retention_floor,
                    min_candidate_observations=min_candidate_observations,
                )
                retention_accepted = retention_decision.accepted
                retention_reason = retention_decision.reason
            else:
                retention_accepted, retention_reason = _structured_retention_gate(
                    structured,
                    retained_scores,
                    candidate_threshold=candidate_threshold,
                    retention_floor=retention_floor,
                    min_candidate_observations=min_candidate_observations,
                )
            if not retention_accepted:
                return None, ArtifactConsolidationReceipt(
                    accepted=False,
                    source_indices=selected,
                    rows_before=rows_before,
                    rows_after=rows_before,
                    rows_saved=0,
                    reason=retention_reason,
                )
            if structured is None:
                for outcome in torch.as_tensor(resolved_candidate_outcomes).reshape(-1):
                    compacted.retention.observe(replacement_key, outcome)
            else:
                candidate_keys = compacted._row_retention_keys(replacement_index)
                for probe in structured:
                    compacted._validate_key(
                        probe.key, compacted.width, "candidate retention probe key"
                    )
                    probe_key = probe.key.detach().to(
                        device="cpu", dtype=torch.float32
                    ).contiguous()
                    if not any(torch.equal(probe_key, key) for key in candidate_keys):
                        raise ValueError(
                            "candidate retention probe key must address the "
                            "replacement row"
                        )
                    for outcome in torch.as_tensor(probe.outcomes).reshape(-1):
                        compacted.retention.observe(probe_key, outcome)
            compacted.save()
            compacted.validate()
        rows_after = len(compacted.occupied)
        if rows_after != len(survivors) + 1:
            raise ValueError("consolidation replacement key collides with a survivor")
        accepted = bool(verifier(compacted))
        receipt = ArtifactConsolidationReceipt(
            accepted=accepted,
            source_indices=selected,
            rows_before=rows_before,
            rows_after=rows_after if accepted else rows_before,
            rows_saved=(rows_before - rows_after) if accepted else 0,
            reason=(
                "behavior verifier passed"
                if accepted
                else "behavior verifier rejected candidate bank"
            ),
        )
        return (compacted if accepted else None), receipt

    @torch.no_grad()
    def promote_view(
        self,
        index: int,
        view: str,
        *,
        confidence: float = 1.0,
        margin: float = 0.0,
    ) -> tuple[ArtifactHandle, dict[str, torch.Tensor]]:
        """Verify and load one opaque executable view from a physical row."""
        if index not in self.occupied:
            raise LookupError("artifact row is empty")
        if not isinstance(view, str) or not view:
            raise ValueError("view must be a nonempty string")
        if view not in self.alias_views[index]:
            raise LookupError("artifact row does not contain the requested view")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if margin < 0.0:
            raise ValueError("margin must be nonnegative")
        artifact = self._load_verified(index)
        self.hot[index] = artifact
        view_index = self.alias_views[index].index(view)
        bindings = self._row_alias_bindings(index)
        return ArtifactHandle(
            index,
            confidence,
            margin,
            self.version,
            view,
            bindings[view_index],
        ), artifact

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
