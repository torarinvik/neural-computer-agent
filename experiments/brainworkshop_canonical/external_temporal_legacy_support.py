"""Compatibility helpers for historical temporal compaction harnesses.

These helpers are intentionally isolated from the canonical temporal address
path. The old compaction reports stored a random basis vector as an opaque
value and decoded it back into a relative offset; that representation is not
used by current content retrieval or address-index experiments. Keeping the
helper here preserves importability of those historical verifier/compaction
probes without making the legacy decoder part of the canonical API.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from neural_computer import AppendOnlyContentAddressedMemory, MemoryQuery

from .external_temporal_offset_growth import EVENT_WIDTH, MAX_OFFSET


def address_basis(seed: int) -> torch.Tensor:
    """Return the historical random value basis for legacy probes only."""

    generator = torch.Generator().manual_seed(seed + 8_101)
    return F.normalize(
        torch.randn(MAX_OFFSET, EVENT_WIDTH, generator=generator), dim=-1
    )


def legacy_probe(
    system,
    file,
    evidence,
    memory: AppendOnlyContentAddressedMemory,
    basis: torch.Tensor,
    *,
    label: str,
    key: torch.Tensor,
    query_symbol: int,
    depth: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    lifetimes: int,
) -> dict[str, object]:
    """Probe the pre-index content backend for historical compaction tests."""

    read = memory.read(MemoryQuery(key.unsqueeze(0), top_k=1))
    hit = bool(read.hit[0])
    if not hit:
        route = {
            "hit": False,
            "score": float(read.scores[0, 0])
            if read.scores.shape[1]
            else float("-inf"),
            "resolved_offset": None,
        }
        return {"label": label, "route": route, "accuracy": 0.5, "lifetimes": []}
    value = F.normalize(read.value[0], dim=0)
    route = {
        "hit": True,
        "score": float(read.scores[0, 0]),
        "resolved_offset": int((basis @ value).argmax()) + 1,
    }
    from .external_temporal_query_address_growth import _evaluate

    rows = _evaluate(
        system,
        file,
        evidence,
        query_symbol=query_symbol,
        depth=depth,
        batch_size=batch_size,
        data_steps=data_steps,
        seed=seed,
        lifetimes=lifetimes,
        forced_offset=int(route["resolved_offset"]),
    )
    return {
        "label": label,
        "route": route,
        "accuracy": sum(float(row["accuracy"]) for row in rows) / len(rows),
        "lifetimes": rows,
    }
