"""Safe loading of externally stored growth state into a frozen processor."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn


def _digest_state(
    module: nn.Module,
    *,
    excluded_prefixes: Sequence[str] = (),
) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class GrowthLoadReceipt:
    """Evidence that one external artifact changed only growth state."""

    loaded_keys: tuple[str, ...]
    core_digest_before: str
    core_digest_after: str

    @property
    def core_unchanged(self) -> bool:
        return self.core_digest_before == self.core_digest_after


def compose_growth_artifacts(
    artifacts: Sequence[Mapping[str, torch.Tensor]],
    *,
    prefix_maps: Sequence[Mapping[str, str] | None] | None = None,
) -> dict[str, torch.Tensor]:
    """Compose verified growth payloads into disjoint module namespaces.

    Storage and retrieval deliberately do not decide how multiple artifacts
    execute.  This caller-owned operation provides the generic, auditable
    merge needed by a controller with several independently replaceable
    growth slots.  Prefix maps remap source state names (longest matching
    prefix wins); every resulting name must be unique.  Collisions are
    rejected even when tensor values happen to match, because silently
    choosing one artifact would make composition order-dependent.

    Returned tensors are detached CPU clones, matching the artifact-memory
    boundary.  The function does not interpret task, modality, or protocol
    metadata and does not mutate any input mapping.
    """
    if not artifacts:
        raise ValueError("at least one growth artifact is required")
    if prefix_maps is None:
        maps: tuple[Mapping[str, str] | None, ...] = (None,) * len(artifacts)
    else:
        if len(prefix_maps) != len(artifacts):
            raise ValueError("prefix_maps must align with artifacts")
        maps = tuple(prefix_maps)

    composed: dict[str, torch.Tensor] = {}
    for artifact, prefix_map in zip(artifacts, maps, strict=True):
        if not isinstance(artifact, Mapping) or not artifact:
            raise ValueError("each growth artifact must be nonempty")
        remappings = tuple(
            sorted(
                ((prefix, target) for prefix, target in (prefix_map or {}).items()),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        )
        for name, value in artifact.items():
            if not isinstance(name, str) or not name:
                raise ValueError("growth artifact names must be nonempty strings")
            if not isinstance(value, torch.Tensor):
                raise TypeError("growth artifact values must be tensors")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"growth artifact entry {name!r} is non-finite")
            target_name = name
            for source_prefix, target_prefix in remappings:
                if name.startswith(source_prefix):
                    target_name = target_prefix + name[len(source_prefix):]
                    break
            if not target_name:
                raise ValueError("remapped growth artifact names must be nonempty")
            if target_name in composed:
                raise ValueError(
                    f"growth artifact namespace collision at {target_name!r}"
                )
            composed[target_name] = value.detach().cpu().clone()
    return composed


def freeze_core(
    module: nn.Module,
    growth_prefixes: Sequence[str],
) -> None:
    """Freeze all module parameters except explicitly declared growth state."""
    prefixes = tuple(growth_prefixes)
    if not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError("growth_prefixes must contain nonempty prefixes")
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(any(name.startswith(prefix) for prefix in prefixes))


def load_growth_artifact(
    module: nn.Module,
    artifact: Mapping[str, torch.Tensor],
    *,
    growth_prefixes: Sequence[str],
) -> GrowthLoadReceipt:
    """Load an opaque artifact without permitting shared-core mutation.

    The artifact may contain only existing module state entries whose names
    begin with a declared growth prefix. No task or modality metadata is
    interpreted here. The function copies tensors directly so missing core
    entries cannot be silently reinitialized by a partial checkpoint load.
    """
    prefixes = tuple(growth_prefixes)
    if not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError("growth_prefixes must contain nonempty prefixes")
    if not isinstance(artifact, Mapping) or not artifact:
        raise ValueError("growth artifact must be a nonempty tensor mapping")
    current = module.state_dict()
    for name, value in artifact.items():
        if not isinstance(name, str) or not any(name.startswith(prefix) for prefix in prefixes):
            raise ValueError(f"artifact entry {name!r} is outside the growth boundary")
        if name not in current:
            raise ValueError(f"artifact entry {name!r} is not a module state entry")
        if not isinstance(value, torch.Tensor):
            raise TypeError("growth artifact values must be tensors")
        if value.shape != current[name].shape:
            raise ValueError(f"artifact entry {name!r} has the wrong shape")
        if value.dtype != current[name].dtype:
            raise ValueError(f"artifact entry {name!r} has the wrong dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"artifact entry {name!r} is non-finite")

    core_before = _digest_state(module, excluded_prefixes=prefixes)
    with torch.no_grad():
        for name, value in artifact.items():
            current[name].copy_(value.to(device=current[name].device))
    core_after = _digest_state(module, excluded_prefixes=prefixes)
    receipt = GrowthLoadReceipt(
        loaded_keys=tuple(sorted(artifact)),
        core_digest_before=core_before,
        core_digest_after=core_after,
    )
    if not receipt.core_unchanged:
        raise RuntimeError("growth artifact changed frozen core state")
    return receipt
