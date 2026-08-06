"""Safe loading of externally stored growth state into a frozen processor."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod

import torch
from torch import nn

_QUANTIZATION_SCALE_SUFFIX = ".__scale__"
_QUANTIZATION_SHAPE_SUFFIX = ".__shape__"
_INT4_CODEC = "int4"


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


def select_growth_artifact_view(
    artifact: Mapping[str, torch.Tensor],
    *,
    source_prefix: str,
    target_prefix: str = "growth_slots.0.",
) -> dict[str, torch.Tensor]:
    """Project one opaque namespace from a composed growth artifact.

    Memory-side routing may return a view identifier with a verified artifact
    handle. The caller supplies the corresponding namespace prefixes here;
    this helper does not interpret the view as a task, modality, or protocol.
    """
    if not isinstance(source_prefix, str) or not source_prefix:
        raise ValueError("source_prefix must be a nonempty string")
    if not isinstance(target_prefix, str) or not target_prefix:
        raise ValueError("target_prefix must be a nonempty string")
    if not isinstance(artifact, Mapping) or not artifact:
        raise ValueError("growth artifact must be a nonempty tensor mapping")
    selected: dict[str, torch.Tensor] = {}
    for name, value in artifact.items():
        if not name.startswith(source_prefix):
            continue
        target_name = target_prefix + name[len(source_prefix) :]
        if target_name in selected:
            raise ValueError(f"growth artifact view collision at {target_name!r}")
        selected[target_name] = value.detach().cpu().clone()
    if not selected:
        raise ValueError(f"growth artifact has no entries under {source_prefix!r}")
    return selected


def compress_growth_artifact(
    artifact: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype | str = torch.float16,
    preserve_names: Sequence[str] = (),
    dtype_overrides: Mapping[str, torch.dtype | str] | None = None,
) -> dict[str, torch.Tensor]:
    """Create a smaller caller-owned tensor representation of an artifact.

    This is a replaceable storage codec, not a learned reasoning branch.
    Float16/bfloat16 entries are cast directly; int8 entries use a symmetric
    per-tensor scale; and ``"int4"`` entries use packed nibbles with
    per-output-row scales. Non-floating entries are copied unchanged. Names
    in ``preserve_names`` remain lossless float tensors, while
    ``dtype_overrides`` lets a caller choose a codec for selected tensors or
    opaque namespaces. This supports adaptive mixed-precision candidates
    without changing the controller boundary. The returned mapping must be
    behavior-verified after decompression before promotion.
    """
    if dtype not in (torch.float16, torch.bfloat16, torch.int8, _INT4_CODEC):
        raise ValueError(
            "growth compression dtype must be float16, bfloat16, int8, or int4"
        )
    preserve = tuple(preserve_names)
    if any(not isinstance(name, str) or not name for name in preserve):
        raise ValueError("preserve_names must contain nonempty tensor names")
    if len(set(preserve)) != len(preserve):
        raise ValueError("preserve_names must be unique")
    if not isinstance(artifact, Mapping) or not artifact:
        raise ValueError("growth artifact must be a nonempty tensor mapping")
    overrides = dict(dtype_overrides or {})
    valid_dtypes = (torch.float16, torch.bfloat16, torch.int8, _INT4_CODEC)
    if any(name not in artifact for name in overrides):
        raise ValueError("dtype_overrides must refer to artifact entries")
    if any(name in preserve for name in overrides):
        raise ValueError("preserved entries cannot also have dtype overrides")
    if any(dtype not in valid_dtypes for dtype in overrides.values()):
        raise ValueError(
            "growth dtype overrides must be float16, bfloat16, int8, or int4"
        )
    compressed: dict[str, torch.Tensor] = {}
    for name, value in artifact.items():
        if not isinstance(name, str) or not name:
            raise ValueError("growth artifact names must be nonempty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError("growth artifact values must be tensors")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"growth artifact entry {name!r} is non-finite")
        if name in preserve:
            compressed[name] = value.detach().cpu().clone()
            continue
        entry_dtype = overrides.get(name, dtype)
        if entry_dtype in (torch.int8, _INT4_CODEC) and value.is_floating_point():
            if name.endswith(
                (_QUANTIZATION_SCALE_SUFFIX, _QUANTIZATION_SHAPE_SUFFIX)
            ):
                raise ValueError("growth artifact name is reserved for quantization")
            work = value.detach().cpu().to(torch.float32)
            if entry_dtype == _INT4_CODEC:
                limit = 7.0
                if work.ndim > 1:
                    reduce_dims = tuple(range(1, work.ndim))
                    scale = work.abs().amax(dim=reduce_dims, keepdim=True) / limit
                else:
                    scale = work.abs() / limit
                scale = torch.where(
                    (scale == 0) | ~torch.isfinite(scale),
                    torch.ones_like(scale),
                    scale,
                )
                codes = torch.clamp(torch.round(work / scale), -7, 7).to(torch.int16)
                codes = codes.reshape(-1) + 8
                if codes.numel() % 2:
                    codes = torch.cat((codes, codes.new_zeros(1)))
                packed = codes[0::2] | (codes[1::2] << 4)
                compressed[name] = packed.to(torch.uint8)
                compressed[name + _QUANTIZATION_SCALE_SUFFIX] = scale
                compressed[name + _QUANTIZATION_SHAPE_SUFFIX] = torch.tensor(
                    work.shape, dtype=torch.int64
                )
            else:
                maximum = work.abs().max()
                scale = maximum / 127.0
                if not bool(torch.isfinite(scale)) or float(scale) == 0.0:
                    scale = torch.ones((), dtype=torch.float32)
                compressed[name] = torch.clamp(
                    torch.round(work / scale), -127, 127
                ).to(torch.int8)
                compressed[name + _QUANTIZATION_SCALE_SUFFIX] = scale.reshape(1)
        else:
            compressed[name] = (
                value.to(dtype=entry_dtype)
                if value.is_floating_point()
                else value.detach().cpu().clone()
            )
    return compressed


def decompress_growth_artifact(
    artifact: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reconstruct float tensors from an int8 or packed-int4 mapping."""
    if not isinstance(artifact, Mapping) or not artifact:
        raise ValueError("growth artifact must be a nonempty tensor mapping")
    scales = {
        name[: -len(_QUANTIZATION_SCALE_SUFFIX)]: value
        for name, value in artifact.items()
        if name.endswith(_QUANTIZATION_SCALE_SUFFIX)
    }
    shapes = {
        name[: -len(_QUANTIZATION_SHAPE_SUFFIX)]: value
        for name, value in artifact.items()
        if name.endswith(_QUANTIZATION_SHAPE_SUFFIX)
    }
    decompressed: dict[str, torch.Tensor] = {}
    for name, value in artifact.items():
        if name.endswith((_QUANTIZATION_SCALE_SUFFIX, _QUANTIZATION_SHAPE_SUFFIX)):
            continue
        if name in shapes:
            scale = scales.get(name)
            shape = shapes[name]
            if value.dtype != torch.uint8:
                raise ValueError(f"packed entry {name!r} must be uint8")
            if (
                shape.dtype != torch.int64
                or shape.ndim != 1
                or bool((shape < 0).any())
            ):
                raise ValueError(f"quantized shape for {name!r} is invalid")
            shape_tuple = tuple(int(dimension) for dimension in shape.tolist())
            expected_scale_shape = shape_tuple[:1] + (1,) * max(
                len(shape_tuple) - 1, 0
            )
            if (
                scale is None
                or scale.shape != expected_scale_shape
                or not scale.is_floating_point()
            ):
                raise ValueError(f"quantization scale for {name!r} is invalid")
            if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
                raise ValueError(f"quantization scale for {name!r} is invalid")
            element_count = prod(shape_tuple) if shape_tuple else 1
            if value.numel() * 2 < element_count:
                raise ValueError(f"packed entry {name!r} is too short")
            packed = value.to(torch.int16)
            codes = torch.stack((packed & 15, packed >> 4), dim=1).reshape(-1)
            quantized = (codes[:element_count] - 8).to(torch.float32)
            decompressed[name] = quantized.reshape(shape_tuple) * scale
        elif name in scales:
            scale = scales[name]
            if value.dtype != torch.int8:
                raise ValueError(f"quantized entry {name!r} must be int8")
            if scale.shape != (1,) or not scale.is_floating_point():
                raise ValueError(f"quantization scale for {name!r} is invalid")
            if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
                raise ValueError(f"quantization scale for {name!r} is invalid")
            decompressed[name] = value.to(torch.float32) * scale
        else:
            if value.dtype == torch.int8:
                raise ValueError(f"int8 entry {name!r} is missing its scale")
            decompressed[name] = value.detach().cpu().clone()
    if not decompressed:
        raise ValueError("growth artifact has no decompressed entries")
    return decompressed


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
    allow_dtype_cast: bool = False,
) -> GrowthLoadReceipt:
    """Load an opaque artifact without permitting shared-core mutation.

    The artifact may contain only existing module state entries whose names
    begin with a declared growth prefix. No task or modality metadata is
    interpreted here. The function copies tensors directly so missing core
    entries cannot be silently reinitialized by a partial checkpoint load.
    Dtype casting is opt-in for caller-owned compressed representations and
    never changes the frozen core.
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
            can_cast = (
                allow_dtype_cast
                and value.is_floating_point()
                and current[name].is_floating_point()
            )
            if not can_cast:
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
