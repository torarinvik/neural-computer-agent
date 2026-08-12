from __future__ import annotations

import torch
from torch import nn

from experiments.episodic_context_credit_amodal.shared_growth_router import (
    _route_scores,
    _staged_admission_audit,
)
from experiments.episodic_context_credit_amodal.train import _pattern_bank


class _FixedRoute(nn.Module):
    def __init__(self, values: list[float], *, use_keys: bool = False) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values))
        self.use_keys = use_keys

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        del query
        if self.use_keys:
            return keys[..., 0].expand(keys.shape[0], -1)
        return self.values.expand(keys.shape[0], -1)


def test_generated_pattern_bank_scales_without_duplicate_same_statistics_rows() -> None:
    patterns = _pattern_bank(9, episode_length=6)

    assert patterns.shape == (20, 6)
    assert torch.equal(patterns.sum(dim=1), torch.full((20,), 3, dtype=torch.long))
    assert torch.unique(patterns, dim=0).shape[0] == 20


def test_large_generated_pattern_bank_materializes_only_addressed_prefix() -> None:
    patterns = _pattern_bank(100, episode_length=24)

    assert patterns.shape == (101, 24)
    assert torch.equal(
        patterns.sum(dim=1),
        torch.full((101,), 12, dtype=torch.long),
    )
    assert torch.unique(patterns, dim=0).shape[0] == 101


def test_permuted_physical_target_does_not_activate_later_growth_router() -> None:
    query = torch.zeros(4, 1)
    old_keys = torch.zeros(2, 1)
    expansions = (
        (
            _FixedRoute([], use_keys=True),
            torch.tensor([[1.0], [5.0]]),
            2,
        ),
        (
            _FixedRoute([], use_keys=True),
            torch.tensor([[100.0], [99.0]]),
            4,
        ),
    )
    base_router = _FixedRoute([1.0, 0.0])

    without_remap, _, _ = _route_scores(
        base_router,
        query,
        old_keys,
        expansions,
        family=2,
    )
    with_remap, _, _ = _route_scores(
        base_router,
        query,
        old_keys,
        expansions,
        family=2,
        expected_row=3,
    )

    assert int(without_remap.argmax(dim=-1)[0]) == 4
    assert int(with_remap.argmax(dim=-1)[0]) == 3


def test_staged_admission_keeps_unstable_rows_pending() -> None:
    keys = torch.eye(4)
    result = _staged_admission_audit(
        keys,
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )

    assert result["admitted_slots"] == [0, 2]
    assert result["pending_slots"] == [1, 3]
    assert result["occupied_count"] == 2
    assert result["all_occupied_rows_protected"]
    assert result["pending_does_not_consume_capacity"]
