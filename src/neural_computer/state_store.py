"""Durable, checksummed state for replaceable memory-side policies.

The controller does not own this state.  A route policy, utility policy, or
other external learner can persist its tensor state independently of the
controller and executable artifacts, then be replaced or reloaded through an
explicit configuration contract.  The store does not interpret tensor names
or assign semantic meaning to coordinates.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

OPAQUE_STATE_STORE_SCHEMA = "neural-computer.persistent-opaque-state.v1"


def _state_checksum(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _normalize_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("opaque state must be a nonempty tensor mapping")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("opaque state names must be nonempty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError("opaque state values must be tensors")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("opaque state tensors must be finite")
        normalized[name] = value.detach().cpu().clone()
    return normalized


def _normalize_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(configuration, Mapping):
        raise TypeError("opaque state configuration must be a mapping")
    normalized = dict(configuration)
    # Validate that the manifest is portable and deterministic before writing,
    # then round-trip it so callers cannot mutate nested metadata after the
    # store has been constructed.
    try:
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise TypeError("opaque state configuration must be JSON-serializable") from error
    return json.loads(encoded)


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


class PersistentOpaqueStateStore:
    """Persist one replaceable tensor state with version and integrity checks."""

    schema = OPAQUE_STATE_STORE_SCHEMA

    def __init__(
        self,
        path: Path,
        *,
        configuration: Mapping[str, Any],
    ) -> None:
        self.path = Path(path)
        self._configuration = _normalize_configuration(configuration)

    @property
    def configuration(self) -> dict[str, Any]:
        return dict(self._configuration)

    def save(self, state: Mapping[str, torch.Tensor]) -> str:
        """Atomically save state and return its SHA-256 digest."""

        normalized = _normalize_state(state)
        checksum = _state_checksum(normalized)
        _atomic_torch_save(
            {
                "format": self.schema,
                "configuration": self.configuration,
                "state_dict": normalized,
                "state_checksum": checksum,
            },
            self.path,
        )
        return checksum

    def load(
        self,
        *,
        map_location: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Load, validate, and return detached CPU tensors."""

        payload = torch.load(self.path, map_location=map_location, weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != self.schema:
            raise ValueError("unsupported opaque state store format")
        if payload.get("configuration") != self.configuration:
            raise ValueError("opaque state store configuration mismatch")
        state = payload.get("state_dict")
        if not isinstance(state, Mapping):
            raise TypeError("opaque state store is missing its state mapping")
        normalized = _normalize_state(state)
        if payload.get("state_checksum") != _state_checksum(normalized):
            raise ValueError("opaque state store checksum mismatch")
        return normalized

    def save_module(self, module: nn.Module) -> str:
        """Persist a module state without coupling the module to the store."""

        if not isinstance(module, nn.Module):
            raise TypeError("opaque state store module must be torch.nn.Module")
        return self.save(module.state_dict())

    def load_module(
        self,
        module: nn.Module,
        *,
        map_location: torch.device | str = "cpu",
    ) -> str:
        """Load state into a compatible module and return its verified digest."""

        if not isinstance(module, nn.Module):
            raise TypeError("opaque state store module must be torch.nn.Module")
        state = self.load(map_location=map_location)
        expected = module.state_dict()
        if set(state) != set(expected):
            raise ValueError("opaque state store module keys do not match")
        for name, value in state.items():
            if value.shape != expected[name].shape:
                raise ValueError(
                    f"opaque state store shape mismatch for parameter {name!r}"
                )
            if value.dtype != expected[name].dtype:
                raise ValueError(
                    f"opaque state store dtype mismatch for parameter {name!r}"
                )
        module.load_state_dict(state, strict=True)
        return _state_checksum(state)


__all__ = ["OPAQUE_STATE_STORE_SCHEMA", "PersistentOpaqueStateStore"]
