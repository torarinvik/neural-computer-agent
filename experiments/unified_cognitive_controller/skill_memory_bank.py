"""Bounded hot/cold storage for opaque learned skill artifacts.

The controller addresses rows with a generic latent key.  Cold rows keep the
key/value statistics in :class:`DiskLatentMemory` and the learned artifact on
disk; promotion loads one artifact into the hot process-local cache.  The
bank never interprets a task name, operation, or answer label.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .memory import DiskLatentMemory


class SkillArtifactBank:
    """A bounded content-addressed cold bank with an explicit hot cache."""

    SCHEMA = "skill-artifact-bank-v1"

    def __init__(
            self, directory: Path, *, width: int, capacity: int = 8,
            device: torch.device | str = "cpu") -> None:
        if width < 1 or capacity < 1:
            raise ValueError("width and capacity must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memory = DiskLatentMemory(width, capacity, device=device)
        self.paths: list[str | None] = [None] * capacity
        # Hashes are optional for backwards compatibility with banks created
        # before artifact integrity was recorded.  New writes always receive
        # a hash, and promotion verifies it before loading the artifact.
        self.artifact_sha256: list[str | None] = [None] * capacity
        self.hot: dict[int, dict[str, Any]] = {}

    @property
    def width(self) -> int:
        return self.memory.store.width

    @property
    def capacity(self) -> int:
        return self.memory.store.capacity

    @property
    def hot_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.hot))

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _free_or_coldest(self) -> int:
        valid = self.memory.store.valid
        free = (~valid).nonzero(as_tuple=False)
        if free.numel():
            return int(free[0, 0])
        usage = self.memory.store.usage.masked_fill(~valid, float("inf"))
        return int(usage.argmin())

    @torch.no_grad()
    def put(
            self, key: torch.Tensor, artifact: dict[str, Any], *,
            strength: float = 1.0, name: str | None = None,
            ) -> int:
        """Write one opaque artifact, replacing the least-used cold row."""
        if key.shape != (self.width,):
            raise ValueError("key must have shape [width]")
        if not 0.0 <= strength:
            raise ValueError("strength must be nonnegative")
        index = self._free_or_coldest()
        if self.memory.store.valid[index]:
            self.hot.pop(index, None)
        filename = name or f"skill-{index:04d}.pt"
        path = self.directory / filename
        torch.save(artifact, path)
        self.artifact_sha256[index] = self._file_sha256(path)
        self.memory.store.clock += 1
        self.memory.store.keys[index].copy_(key.detach())
        self.memory.store.values[index].copy_(key.detach())
        self.memory.store.usage[index] = float(strength)
        self.memory.store.age[index] = self.memory.store.clock
        self.memory.store.access_count[index] = 0
        self.memory.store.success_count[index] = 0
        self.memory.store.failure_count[index] = 0
        self.memory.store.volatility[index] = 1.0
        self.memory.store.valid[index] = True
        self.paths[index] = filename
        return index

    @torch.no_grad()
    def resolve(
            self, query: torch.Tensor, *, record_access: bool = True,
            ) -> tuple[int, float, float]:
        """Resolve a query to a row, cosine confidence, and score margin."""
        if query.shape != (self.width,):
            raise ValueError("query must have shape [width]")
        query = query.to(
            device=self.memory.store.keys.device,
            dtype=self.memory.store.keys.dtype)
        valid = self.memory.store.valid.nonzero(as_tuple=False).squeeze(1)
        if not valid.numel():
            raise KeyError("query resolved against an empty skill bank")
        keys = torch.nn.functional.normalize(
            self.memory.store.keys[valid], dim=-1)
        normalized_query = torch.nn.functional.normalize(query, dim=-1)
        cosine = normalized_query @ keys.T
        # Keep the same generic usage prior as DiskLatentMemory's normal read.
        scores = cosine + self.memory.store.usage[valid].clamp_min(
            1e-6).log()
        order = scores.argsort(descending=True)
        selected = valid[order[0]]
        margin = (
            scores[order[0]] - scores[order[1]]
            if order.numel() > 1 else scores.new_tensor(1.0))
        if record_access:
            self.memory.store.access_count[selected] += 1
        return int(selected), float(cosine[order[0]]), float(margin)

    @torch.no_grad()
    def resolve_with_selector(
            self, query: torch.Tensor, selector: torch.nn.Module, *,
            record_access: bool = True,
            ) -> tuple[int, float, float]:
        """Resolve through an explicitly supplied learned row selector.

        This is opt-in: the bank's production default remains cosine routing.
        The selector receives the query and all valid opaque row keys and must
        return one scalar score per row.  Confidence is the top softmax score
        and margin is the top-minus-runner-up raw-score difference, giving the
        same abstention vocabulary as :meth:`resolve`.
        """
        if query.shape != (self.width,):
            raise ValueError("query must have shape [width]")
        valid = self.memory.store.valid.nonzero(as_tuple=False).squeeze(1)
        if not valid.numel():
            raise KeyError("query resolved against an empty skill bank")
        bank_keys = self.memory.store.keys[valid]
        query = query.to(device=bank_keys.device, dtype=bank_keys.dtype)
        try:
            parameter = next(selector.parameters())
        except StopIteration:
            parameter = None
        if parameter is None:
            selector_query = query
            selector_keys = bank_keys
        else:
            selector_query = query.to(
                device=parameter.device, dtype=parameter.dtype)
            selector_keys = bank_keys.to(
                device=parameter.device, dtype=parameter.dtype)
        scores = selector(
            selector_query.unsqueeze(0), selector_keys.unsqueeze(0))
        if scores.shape != (1, valid.numel()):
            raise ValueError(
                "selector must return scores with shape [batch, rows]")
        scores = scores.squeeze(0).to(device=bank_keys.device)
        order = scores.argsort(descending=True)
        selected = valid[order[0]]
        probabilities = torch.softmax(scores, dim=0)
        margin = (
            scores[order[0]] - scores[order[1]]
            if order.numel() > 1 else scores.new_tensor(1.0))
        if record_access:
            self.memory.store.access_count[selected] += 1
        return (
            int(selected), float(probabilities[order[0]]), float(margin))

    def _load_checked_artifact(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.paths) or self.paths[index] is None:
            raise KeyError("query resolved to an empty skill row")
        path = self.directory / self.paths[index]
        expected_hash = self.artifact_sha256[index]
        if expected_hash is not None:
            actual_hash = self._file_sha256(path)
            if actual_hash != expected_hash:
                raise ValueError(
                    "skill artifact hash mismatch; refusing corrupted file")
        return torch.load(path, map_location="cpu", weights_only=False)

    @torch.no_grad()
    def retrieve_index(
            self, query: torch.Tensor, *, record_access: bool = True,
            ) -> tuple[int, float]:
        """Resolve a query to a physical row and cosine confidence."""
        index, confidence, _ = self.resolve(
            query, record_access=record_access)
        return index, confidence

    def promote(
            self, query: torch.Tensor, *, min_confidence: float | None = None,
            min_margin: float | None = None,
            ) -> tuple[int, float, dict[str, Any]]:
        """Load a cold artifact, optionally abstaining on weak addresses.

        The default preserves nearest-row behavior for existing experiments.
        A caller that cannot safely use an arbitrary fallback can set either
        threshold; the bank then raises ``LookupError`` instead of silently
        activating an unrelated skill.
        """
        if min_confidence is not None and not -1.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between -1 and 1")
        if min_margin is not None and min_margin < 0.0:
            raise ValueError("min_margin must be nonnegative")
        # Do not count a rejected or corrupted promotion as a successful
        # access; otherwise ambiguous requests could distort cold eviction.
        index, confidence, margin = self.resolve(query, record_access=False)
        if (
                min_confidence is not None and confidence < min_confidence
        ) or (
                min_margin is not None and margin < min_margin
        ):
            raise LookupError(
                "skill address below confidence/margin threshold; abstaining")
        artifact = self._load_checked_artifact(index)
        self.memory.store.access_count[index] += 1
        self.hot[index] = artifact
        return index, confidence, artifact

    def promote_with_selector(
            self, query: torch.Tensor, selector: torch.nn.Module, *,
            min_confidence: float | None = None,
            min_margin: float | None = None,
            ) -> tuple[int, float, dict[str, Any]]:
        """Promote an artifact using an explicitly supplied learned selector.

        The selector is never persisted or silently installed.  Callers must
        pass it on every use, and the same hash verification and abstention
        checks as cosine promotion apply.
        """
        if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
            raise ValueError("selector min_confidence must be between 0 and 1")
        if min_margin is not None and min_margin < 0.0:
            raise ValueError("min_margin must be nonnegative")
        index, confidence, margin = self.resolve_with_selector(
            query, selector, record_access=False)
        if (
                min_confidence is not None and confidence < min_confidence
        ) or (
                min_margin is not None and margin < min_margin
        ):
            raise LookupError(
                "skill selector address below threshold; abstaining")
        artifact = self._load_checked_artifact(index)
        self.memory.store.access_count[index] += 1
        self.hot[index] = artifact
        return index, confidence, artifact

    def evict_hot(self, index: int | None = None) -> None:
        """Drop one or all process-local hot artifacts; cold rows remain."""
        if index is None:
            self.hot.clear()
        else:
            self.hot.pop(index, None)

    def save(self) -> None:
        """Persist cold tensors and the path manifest atomically enough for a probe."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memory.save(self.directory / "rows.pt")
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps({
            "schema": self.SCHEMA,
            "width": self.width,
            "capacity": self.capacity,
            "paths": self.paths,
            "artifact_sha256": self.artifact_sha256,
        }, indent=2) + "\n")
        temporary.replace(self.directory / "manifest.json")

    @classmethod
    def load(
            cls, directory: Path, *,
            device: torch.device | str = "cpu",
            ) -> "SkillArtifactBank":
        directory = Path(directory)
        payload = json.loads((directory / "manifest.json").read_text())
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported skill-artifact-bank schema")
        instance = cls.__new__(cls)
        instance.directory = directory
        instance.memory = DiskLatentMemory.load(
            directory / "rows.pt", device=device)
        if int(payload["width"]) != instance.memory.store.width:
            raise ValueError("skill-bank width metadata mismatch")
        if int(payload["capacity"]) != instance.memory.store.capacity:
            raise ValueError("skill-bank capacity metadata mismatch")
        instance.paths = list(payload["paths"])
        if len(instance.paths) != instance.capacity:
            raise ValueError("skill-bank path metadata mismatch")
        hashes = payload.get("artifact_sha256")
        if hashes is None:
            # Legacy manifests did not include hashes.  Keep them loadable and
            # establish hashes for files that are present so the next save
            # upgrades the manifest without a separate migration step.
            hashes = [None] * instance.capacity
            for index, filename in enumerate(instance.paths):
                if filename is not None:
                    path = directory / filename
                    if path.is_file():
                        hashes[index] = cls._file_sha256(path)
        if len(hashes) != instance.capacity:
            raise ValueError("skill-bank artifact hash metadata mismatch")
        instance.artifact_sha256 = list(hashes)
        instance.hot = {}
        return instance
