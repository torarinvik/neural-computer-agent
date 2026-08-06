from __future__ import annotations

import pytest
import torch
from torch import nn

from neural_computer import ExternalGrowthPrior, OpaqueViewRouteExtension


def _module(value: float) -> nn.Module:
    module = nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.fill_(value)
        module.bias.fill_(value + 1.0)
    return module


def test_growth_prior_averages_and_loads_copy_on_write() -> None:
    first = _module(1.0)
    second = _module(3.0)
    prior = ExternalGrowthPrior.from_modules((first, second))
    target = _module(0.0)
    prior.load_into(target)

    assert prior.source_count == 2
    assert torch.allclose(target.weight, torch.full_like(target.weight, 2.0))
    assert torch.allclose(target.bias, torch.full_like(target.bias, 3.0))
    with torch.no_grad():
        target.weight.zero_()
    assert torch.allclose(prior.state_payload()["weight"], torch.full((2, 2), 2.0))
    assert torch.allclose(first.weight, torch.ones_like(first.weight))


def test_growth_prior_updates_without_replaying_or_mutating_sources() -> None:
    first = _module(1.0)
    prior = ExternalGrowthPrior.from_module(first)
    second = _module(5.0)
    updated = prior.update_from(second)

    assert prior.source_count == 1
    assert updated.source_count == 2
    assert torch.allclose(updated.state_payload()["weight"], torch.full((2, 2), 3.0))
    assert torch.allclose(first.weight, torch.ones_like(first.weight))
    assert torch.allclose(second.weight, torch.full_like(second.weight, 5.0))
    assert updated.digest() != prior.digest()


def test_growth_prior_rejects_namespace_or_shape_drift() -> None:
    prior = ExternalGrowthPrior.from_module(_module(1.0))
    with pytest.raises(ValueError, match="state names"):
        prior.load_into(nn.Sequential(nn.Linear(2, 2)))
    with pytest.raises(ValueError, match="incompatible"):
        prior.update_from(nn.Linear(3, 3))


def test_growth_prior_can_reset_a_capability_specific_head() -> None:
    first = OpaqueViewRouteExtension(width=4, hidden=8)
    with torch.no_grad():
        for parameter in first.encoder.parameters():
            parameter.fill_(2.0)
        first.score.weight.fill_(3.0)
        first.score.bias.fill_(4.0)
    prior = ExternalGrowthPrior.from_module(first)
    target = OpaqueViewRouteExtension(width=4, hidden=8)
    prior.load_into(target, reset_prefixes=("score.",))

    assert torch.allclose(target.encoder[0].weight, torch.full_like(target.encoder[0].weight, 2.0))
    assert torch.count_nonzero(target.score.weight) == 0
    assert torch.count_nonzero(target.score.bias) == 0


def test_growth_prior_can_blend_with_fresh_initialization() -> None:
    first = _module(4.0)
    prior = ExternalGrowthPrior.from_module(first)
    target = _module(0.0)
    prior.load_into(target, mix=0.25)

    assert torch.allclose(target.weight, torch.full_like(target.weight, 1.0))
    assert torch.allclose(target.bias, torch.full_like(target.bias, 2.0))
