import pytest
import torch

from neural_computer import (
    ExternalGoalRepresentationAlignmentBank,
    ExternalGoalRepresentationAlignmentStatistics,
)


def _adapter(
    offset: float,
) -> tuple[ExternalGoalRepresentationAlignmentStatistics, torch.Tensor, torch.Tensor]:
    source = torch.tensor(
        [[-2.0, 0.5], [-1.0, 0.0], [0.0, 0.5], [1.0, 1.0], [2.0, 1.5]],
        dtype=torch.float32,
    )
    target = (source[:, :1] * 0.25 + offset).clone()
    model = ExternalGoalRepresentationAlignmentStatistics(2, 1, ridge=1e-5)
    model.observe(source[:3], target[:3])
    return model, source[3:], target[3:]


def test_alignment_bank_quarantines_full_capacity_and_promotes_after_eviction() -> None:
    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=1,
        quarantine_capacity=2,
    )
    first, heldout_source, heldout_target = _adapter(0.0)
    receipt = bank.admit_verified(
        "frontend-a",
        first,
        heldout_source,
        heldout_target,
        prediction_tolerance=1e-3,
    )
    assert receipt.accepted and receipt.slot_id == 0
    active_digest = bank.active_digest()

    second, second_source, second_target = _adapter(1.0)
    blocked = bank.admit_verified(
        "frontend-b",
        second,
        second_source,
        second_target,
        prediction_tolerance=1e-3,
    )
    assert not blocked.accepted and blocked.quarantined
    assert bank.frontend_space_ids == ("frontend-a",)
    assert bank.quarantined_space_ids == ("frontend-b",)
    assert bank.active_digest() == active_digest

    failed = bank.evict_verified(0, lambda _candidate: False)
    assert not failed.accepted
    assert bank.frontend_space_ids == ("frontend-a",)

    evicted = bank.evict_verified(0, lambda candidate: candidate.active_count == 0)
    assert evicted.accepted and evicted.evicted_slot_id == 0
    promoted = bank.promote_quarantined_verified(
        "frontend-b",
        second_source,
        second_target,
        prediction_tolerance=1e-3,
    )
    assert promoted.accepted and promoted.slot_id == 1
    assert bank.frontend_space_ids == ("frontend-b",)
    assert bank.slot_ids == (1,)

    restored = ExternalGoalRepresentationAlignmentBank.from_payload(bank.state_payload())
    assert restored.digest() == bank.digest()
    assert torch.equal(
        restored.route("frontend-b", second_source),
        bank.route("frontend-b", second_source),
    )


def test_alignment_bank_growth_is_retention_gated_and_keeps_content_stable() -> None:
    bank = ExternalGoalRepresentationAlignmentBank(1, capacity=1)
    adapter, source, target = _adapter(0.0)
    assert bank.admit_verified(
        "frontend-a", adapter, source, target, prediction_tolerance=1e-3
    ).accepted
    before = bank.active_digest()
    rejected = bank.grow_verified(2, lambda _candidate: False)
    assert not rejected.accepted
    assert bank.capacity == 1 and bank.active_digest() == before
    accepted = bank.grow_verified(2, lambda candidate: candidate.active_count == 1)
    assert accepted.accepted
    assert bank.capacity == 2 and bank.active_digest() == before


def test_alignment_bank_rejects_unsafe_candidate_without_active_mutation() -> None:
    bank = ExternalGoalRepresentationAlignmentBank(1, capacity=1, quarantine_capacity=0)
    adapter, source, target = _adapter(0.0)
    digest_before = bank.active_digest()
    unsafe_target = target + 5.0
    rejected = bank.admit_verified(
        "unsafe",
        adapter,
        source,
        unsafe_target,
        prediction_tolerance=1e-3,
    )
    assert not rejected.accepted and not rejected.quarantined
    assert bank.active_count == 0 and bank.active_digest() == digest_before


def test_alignment_bank_rejects_duplicate_or_unknown_operations() -> None:
    bank = ExternalGoalRepresentationAlignmentBank(1, capacity=1)
    adapter, source, target = _adapter(0.0)
    bank.admit_verified("frontend-a", adapter, source, target, prediction_tolerance=1e-3)
    with pytest.raises(ValueError, match="already active"):
        bank.admit_verified("frontend-a", adapter, source, target, prediction_tolerance=1e-3)
    with pytest.raises(KeyError, match="unknown"):
        bank.route("unknown", source)
