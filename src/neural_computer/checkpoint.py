"""Independent, versioned serialization for the production runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .runtime import AmodalControllerRuntime, RUNTIME_FORMAT
from .interface import EVENT_SCHEMA, INTENTION_SCHEMA


def save_runtime(
    runtime: AmodalControllerRuntime,
    path: Path,
    *,
    provenance: dict[str, object] | None = None,
) -> None:
    """Save independently loadable controller, adapter, and memory components."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(runtime.checkpoint_payload(provenance=provenance), path)


def load_runtime_components(
    runtime: AmodalControllerRuntime,
    path: Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a canonical payload into an already-constructed runtime.

    The caller constructs the encoder/decoder set explicitly.  This makes
    adapter replacement visible and prevents checkpoint metadata from creating
    hidden modality branches.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != RUNTIME_FORMAT:
        raise ValueError("unsupported amodal runtime checkpoint format")
    if payload.get("event_schema") != EVENT_SCHEMA or payload.get("intention_schema") != INTENTION_SCHEMA:
        raise ValueError("unsupported neural-IR schema in runtime checkpoint")
    if payload.get("configuration") != runtime.configuration():
        raise ValueError("checkpoint configuration does not match runtime")
    runtime.load_component_state_dicts(payload["components"])
    return payload
