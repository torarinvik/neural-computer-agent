from __future__ import annotations

import torch

from neural_computer import MemoryCandidates, OpaqueCapacityPlanner


def _bank(batch: int = 2, capacity: int = 4, width: int = 6) -> MemoryCandidates:
    generator = torch.Generator().manual_seed(1234)
    return MemoryCandidates(
        keys=torch.nn.functional.normalize(
            torch.randn(batch, capacity, width, generator=generator), dim=-1
        ),
        values=torch.randn(batch, capacity, width, generator=generator),
        strengths=torch.rand(batch, capacity, generator=generator),
        timestamps=torch.arange(capacity, dtype=torch.float32).expand(batch, -1),
        occupied=torch.ones(batch, capacity, dtype=torch.bool),
    )


def test_capacity_planner_is_equivariant_to_candidate_permutation() -> None:
    bank = _bank()
    incoming_key = torch.nn.functional.normalize(torch.randn(2, 6), dim=-1)
    incoming_value = torch.randn(2, 6)
    protected = torch.tensor(
        [[True, False, False, True], [False, True, False, False]],
        dtype=torch.bool,
    )
    planner = OpaqueCapacityPlanner(width=6, hidden=16).eval()
    original = planner(
        bank,
        incoming_key,
        incoming_value,
        protected,
        consolidation_available=torch.tensor([True, False]),
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_bank = MemoryCandidates(
        keys=bank.keys[:, permutation],
        values=bank.values[:, permutation],
        strengths=bank.strengths[:, permutation],
        timestamps=bank.timestamps[:, permutation],
        occupied=bank.occupied[:, permutation],
    )
    permuted = planner(
        permuted_bank,
        incoming_key,
        incoming_value,
        protected[:, permutation],
        consolidation_available=torch.tensor([True, False]),
    )
    assert torch.allclose(original.action_logits, permuted.action_logits)
    assert torch.allclose(
        original.eviction_scores[:, permutation], permuted.eviction_scores
    )
    assert torch.allclose(
        original.pair_scores[:, permutation][:, :, permutation],
        permuted.pair_scores,
    )
    assert torch.equal(
        original.available_actions, permuted.available_actions
    )


def test_capacity_planner_masks_protected_eviction_and_can_force_growth() -> None:
    bank = _bank(batch=1, capacity=3)
    planner = OpaqueCapacityPlanner(width=6, hidden=16).eval()
    protected = torch.ones(1, 3, dtype=torch.bool)
    output = planner(
        bank,
        torch.nn.functional.normalize(torch.randn(1, 6), dim=-1),
        torch.randn(1, 6),
        protected,
        consolidation_available=torch.zeros(1, dtype=torch.bool),
    )
    assert not bool(output.available_actions[0, 1])
    assert not bool(output.available_actions[0, 2])
    plan = planner.propose(
        bank,
        torch.nn.functional.normalize(torch.randn(1, 6), dim=-1),
        torch.randn(1, 6),
        protected,
        consolidation_available=torch.zeros(1, dtype=torch.bool),
    )
    assert plan.action == "grow"
    assert plan.eviction_index is None
    assert plan.pair is None


def test_capacity_planner_returns_one_plan_per_bank() -> None:
    bank = _bank(batch=2)
    planner = OpaqueCapacityPlanner(width=6, hidden=16).eval()
    plans = planner.propose(
        bank,
        torch.nn.functional.normalize(torch.randn(2, 6), dim=-1),
        torch.randn(2, 6),
        torch.zeros(2, 4, dtype=torch.bool),
    )
    assert isinstance(plans, tuple)
    assert len(plans) == 2
    assert all(plan.action in {"admit", "evict", "consolidate", "grow"} for plan in plans)
