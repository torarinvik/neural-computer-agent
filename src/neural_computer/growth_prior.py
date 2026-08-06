"""Reusable initialization state for independently replaceable growth modules.

The prior is external mutable state, not part of the controller.  It stores
only a validated tensor state snapshot and can initialize a fresh growth
module by copy-on-write.  Existing capability modules are never modified when
the prior is updated, which keeps reuse separate from retention.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import torch
from torch import nn

GROWTH_PRIOR_SCHEMA = "neural-computer.external-growth-prior.v1"


def _module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    if not isinstance(module, nn.Module):
        raise TypeError("growth prior sources must be torch modules")
    state = module.state_dict()
    if not state:
        raise ValueError("growth prior source must have nonempty state")
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _validate_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("growth prior state must be a nonempty tensor mapping")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("growth prior state names must be nonempty strings")
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            raise ValueError("growth prior state values must be nonempty tensors")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"growth prior state {name!r} must be finite")
        normalized[name] = value.detach().cpu().clone()
    return normalized


class ExternalGrowthPrior:
    """A copy-on-write learned initialization for external growth modules.

    ``from_modules`` computes a parameter-wise average across prior acquired
    adapters.  It does not merge capability identities or execute any module.
    ``load_into`` copies the snapshot into a newly constructed compatible
    module and rejects shape, dtype, or namespace drift.  The shared
    controller is outside this boundary by construction.
    """

    schema = GROWTH_PRIOR_SCHEMA

    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        source_count: int,
    ) -> None:
        if source_count < 1:
            raise ValueError("growth prior source_count must be positive")
        self._state = _validate_state(state)
        self.source_count = int(source_count)

    @classmethod
    def from_module(cls, module: nn.Module) -> ExternalGrowthPrior:
        return cls(_module_state(module), source_count=1)

    @classmethod
    def from_modules(
        cls,
        modules: Iterable[nn.Module],
    ) -> ExternalGrowthPrior:
        sources = tuple(modules)
        if not sources:
            raise ValueError("at least one growth module is required")
        states = tuple(_module_state(module) for module in sources)
        names = tuple(states[0])
        if any(tuple(state) != names for state in states[1:]):
            raise ValueError("growth prior modules must share state names")
        averaged: dict[str, torch.Tensor] = {}
        for name in names:
            first = states[0][name]
            if any(
                state[name].shape != first.shape
                or state[name].dtype != first.dtype
                for state in states[1:]
            ):
                raise ValueError("growth prior modules must share shapes and dtypes")
            if first.is_floating_point():
                averaged[name] = torch.stack(
                    [state[name].to(torch.float32) for state in states]
                ).mean(dim=0).to(dtype=first.dtype)
            else:
                averaged[name] = first.clone()
        return cls(averaged, source_count=len(sources))

    def state_payload(self) -> dict[str, torch.Tensor]:
        """Return detached state that callers may persist or inspect."""

        return {name: value.clone() for name, value in self._state.items()}

    def digest(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self._state.items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(repr(tuple(value.shape)).encode("utf-8"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def load_into(
        self,
        module: nn.Module,
        *,
        reset_prefixes: tuple[str, ...] = (),
        mix: float = 1.0,
    ) -> None:
        """Copy the prior into a compatible fresh growth module.

        ``reset_prefixes`` keeps selected target-owned state at its fresh
        initialization.  This is useful when a shared representation can be
        reused but a capability-specific output head must remain neutral until
        new verifier outcomes activate it.
        """

        current = module.state_dict()
        if any(not prefix for prefix in reset_prefixes):
            raise ValueError("reset_prefixes must contain nonempty prefixes")
        if not 0.0 <= mix <= 1.0:
            raise ValueError("growth prior mix must lie in [0, 1]")
        if tuple(current) != tuple(self._state):
            raise ValueError("growth module state names do not match the prior")
        for name, value in self._state.items():
            target = current[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"growth module state {name!r} is incompatible")
        payload = self.state_payload()
        for state_name, value in payload.items():
            if any(state_name.startswith(prefix) for prefix in reset_prefixes):
                payload[state_name] = current[state_name].detach().clone()
            elif mix < 1.0 and value.is_floating_point():
                payload[state_name] = (
                    mix * value.to(torch.float32)
                    + (1.0 - mix) * current[state_name].detach().cpu().to(torch.float32)
                ).to(dtype=value.dtype)
            elif mix < 1.0:
                payload[state_name] = current[state_name].detach().clone()
        module.load_state_dict(payload, strict=True)

    def update_from(self, module: nn.Module) -> ExternalGrowthPrior:
        """Return a new prior containing the old prior plus one module."""

        incoming = _module_state(module)
        if tuple(incoming) != tuple(self._state):
            raise ValueError("growth module state names do not match the prior")
        count = self.source_count
        updated: dict[str, torch.Tensor] = {}
        for name, old in self._state.items():
            value = incoming[name]
            if value.shape != old.shape or value.dtype != old.dtype:
                raise ValueError(f"growth module state {name!r} is incompatible")
            if old.is_floating_point():
                updated[name] = (
                    old.to(torch.float32) * count + value.to(torch.float32)
                ).div(count + 1).to(dtype=old.dtype)
            else:
                updated[name] = old.clone()
        return ExternalGrowthPrior(updated, source_count=count + 1)

    def update_many(self, modules: Iterable[nn.Module]) -> ExternalGrowthPrior:
        """Return a prior updated with modules without replaying their examples."""

        result = self
        for module in modules:
            result = result.update_from(module)
        return result


__all__ = ["GROWTH_PRIOR_SCHEMA", "ExternalGrowthPrior"]
