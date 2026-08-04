from __future__ import annotations

import pytest
import torch

from experiments.cross_adapter_memory_amodal.environment import (
    CrossAdapterRecallVerifier,
)
from experiments.cross_adapter_memory_amodal.train import (
    ReaderEventAdapter,
    _slot_order,
    build_runtime,
)


def test_cross_adapter_verifier_exposes_only_scalar_outcomes() -> None:
    verifier = CrossAdapterRecallVerifier(batch_size=2, seed=17)
    verifier.reset()
    rewards = verifier.score_probe(
        torch.tensor([0, 1]), torch.tensor([0, 1], dtype=torch.long)
    )
    assert rewards.shape == (2,)
    assert verifier.score_recall(torch.zeros(2, dtype=torch.long)).shape == (2,)
    assert not hasattr(verifier, "bits")


def test_cross_adapter_verifier_supports_bounded_slot_populations() -> None:
    verifier = CrossAdapterRecallVerifier(batch_size=2, seed=17, slot_count=3)
    verifier.reset()
    assert verifier.slot_count == 3
    rewards = verifier.score_probe(
        torch.tensor([0, 2]), torch.tensor([0, 1], dtype=torch.long)
    )
    assert rewards.shape == (2,)
    assert verifier.score_recall(torch.zeros(2, dtype=torch.long)).shape == (2,)


def test_cross_adapter_verifier_supports_paired_hidden_worlds() -> None:
    verifier = CrossAdapterRecallVerifier(batch_size=2, seed=17, slot_count=3)
    verifier.reset()
    verifier.score_probe(
        torch.tensor([0, 2]), torch.tensor([0, 1], dtype=torch.long)
    )
    duplicate = verifier.duplicate_rows(2)
    assert duplicate.batch_size == 4
    assert duplicate.query_slot.tolist() == [
        verifier.query_slot[0].item(),
        verifier.query_slot[0].item(),
        verifier.query_slot[1].item(),
        verifier.query_slot[1].item(),
    ]


def test_reader_adapter_is_shape_checked_and_independently_parameterized() -> None:
    adapter = ReaderEventAdapter(16)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 272
    assert adapter(torch.zeros(3, 16)).shape == (3, 16)
    with pytest.raises(ValueError, match="reader raw event"):
        adapter(torch.zeros(3, 15))


def test_cross_adapter_runtime_has_two_addressable_memory_rows() -> None:
    runtime = build_runtime(seed=5, batch_size=2)
    assert runtime.controller.memory_top_k == 1
    assert runtime.memory.capacity == 2


def test_cross_adapter_runtime_expands_event_window_for_three_slots() -> None:
    runtime = build_runtime(seed=5, batch_size=2, slot_count=3, memory_capacity=2)
    assert runtime.controller.event_window_capacity == 4
    assert runtime.memory.capacity == 2


def test_target_cue_places_the_target_row_last() -> None:
    verifier = CrossAdapterRecallVerifier(batch_size=6, seed=17, slot_count=3)
    verifier.reset()
    order = _slot_order(verifier, target_cue=True)
    assert torch.equal(order[:, -1], verifier.query_slot)
    assert torch.equal(
        torch.sort(order, dim=1).values,
        torch.arange(3).expand(6, -1),
    )


def test_target_cue_can_randomize_position_without_losing_permutation() -> None:
    torch.manual_seed(17)
    verifier = CrossAdapterRecallVerifier(batch_size=6, seed=17, slot_count=3)
    verifier.reset()
    order = _slot_order(
        verifier,
        target_cue=True,
        randomize_slot_order=True,
    )
    assert torch.equal(
        torch.sort(order, dim=1).values,
        torch.arange(3).expand(6, -1),
    )
