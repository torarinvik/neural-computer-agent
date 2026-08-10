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


def test_alignment_bank_routes_by_opaque_signature_and_refuses_ambiguity() -> None:
    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=2,
        identity_width=3,
        identity_min_score=0.6,
        identity_min_margin=0.2,
    )
    first, source, target = _adapter(0.0)
    second, second_source, second_target = _adapter(1.0)
    assert bank.admit_verified(
        "opaque-a",
        first,
        source,
        target,
        prediction_tolerance=1e-3,
        identity_signature=torch.tensor([1.0, 0.0, 0.0]),
    ).accepted
    assert bank.admit_verified(
        "opaque-b",
        second,
        second_source,
        second_target,
        prediction_tolerance=1e-3,
        identity_signature=torch.tensor([0.0, 1.0, 0.0]),
    ).accepted

    routed = bank.route_by_signature(
        torch.tensor([1.0, 0.02, 0.0]),
        source,
    )
    assert routed.selected_slot_id == 0
    assert routed.aligned is not None
    assert torch.equal(routed.aligned, bank.route_slot(routed.selected_slot_id, source))

    ambiguous = bank.route_by_signature(
        torch.tensor([1.0, 1.0, 0.0]),
        source,
    )
    assert ambiguous.selected_slot_id is None
    assert ambiguous.aligned is None

    missing = bank.route_by_signature(torch.tensor([0.0, 0.0, 1.0]), source)
    assert missing.selected_slot_id is None
    assert missing.aligned is None

    restored = ExternalGoalRepresentationAlignmentBank.from_payload(bank.state_payload())
    restored_route = restored.route_by_signature(
        torch.tensor([0.0, 1.0, 0.02]),
        second_source,
    )
    assert restored.digest() == bank.digest()
    assert restored_route.selected_slot_id == 1
    assert torch.equal(
        restored_route.aligned,
        bank.route_slot(restored_route.selected_slot_id, second_source),
    )


def test_alignment_bank_identity_updates_are_explicitly_verifier_gated() -> None:
    bank = ExternalGoalRepresentationAlignmentBank(1, capacity=1, identity_width=2)
    adapter, source, target = _adapter(0.0)
    receipt = bank.admit_verified(
        "opaque-a",
        adapter,
        source,
        target,
        prediction_tolerance=1e-3,
        identity_signature=torch.tensor([1.0, 0.0]),
    )
    before = bank.digest()
    assert bank.observe_identity_verified(receipt.slot_id, torch.tensor([1.0, 0.01]))
    assert bank.digest() != before
    with pytest.raises(KeyError, match="unknown"):
        bank.observe_identity_verified(9, torch.tensor([1.0, 0.0]))


def test_alignment_bank_defers_overlapping_identity_and_resolves_from_later_anchor() -> None:
    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=2,
        identity_width=2,
        identity_quarantine_capacity=2,
    )
    first, source, target = _adapter(0.0)
    second, second_source, second_target = _adapter(1.0)
    first_receipt = bank.admit_verified(
        "opaque-a",
        first,
        source,
        target,
        prediction_tolerance=1e-3,
        identity_signature=torch.tensor([1.0, 0.0]),
    )
    second_receipt = bank.admit_verified(
        "opaque-b",
        second,
        second_source,
        second_target,
        prediction_tolerance=1e-3,
        identity_signature=torch.tensor([0.0, 1.0]),
    )
    deferred = bank.defer_identity_signature(
        torch.tensor([1.0, 1.0]),
        candidate_slot_ids=(first_receipt.slot_id, second_receipt.slot_id),
    )
    assert deferred.accepted and bank.identity_quarantined_count == 1

    blocked_eviction = bank.evict_verified(first_receipt.slot_id, lambda _candidate: True)
    assert not blocked_eviction.accepted
    rejected = bank.resolve_identity_quarantine(
        first_receipt.slot_id,
        verifier_accepted=False,
    )
    assert not rejected.accepted and bank.identity_quarantined_count == 1
    persisted = ExternalGoalRepresentationAlignmentBank.from_payload(bank.state_payload())
    assert persisted.identity_quarantined_count == 1
    assert persisted.digest() == bank.digest()

    resolved = bank.resolve_identity_quarantine(
        first_receipt.slot_id,
        verifier_accepted=True,
    )
    assert resolved.accepted and resolved.resolved_count == 1
    assert bank.identity_quarantined_count == 0
    evicted = bank.evict_verified(second_receipt.slot_id, lambda candidate: candidate.active_count == 1)
    assert evicted.accepted


def test_alignment_bank_routes_partial_evidence_and_selects_anchor_without_slot_id() -> None:
    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=2,
        identity_width=3,
        identity_min_score=0.6,
        identity_min_margin=0.2,
        identity_quarantine_capacity=1,
    )
    first, source, target = _adapter(0.0)
    second, second_source, second_target = _adapter(1.0)
    assert bank.admit_verified(
        "opaque-a",
        first,
        source,
        target,
        prediction_tolerance=1e-3,
        identity_signature=torch.tensor([1.0, 0.0, 0.0]),
    ).accepted
    assert bank.admit_verified(
        "opaque-b",
        second,
        second_source,
        second_target,
        prediction_tolerance=1e-3,
        identity_signature=torch.tensor([0.0, 1.0, 0.0]),
    ).accepted

    before = bank.digest()
    partial_a = bank.route_by_signature(
        torch.tensor([0.9, 0.0, 123.0]),
        source,
        signature_mask=torch.tensor([True, False, False]),
    )
    partial_b = bank.route_by_signature(
        torch.tensor([0.0, 0.8, -99.0]),
        second_source,
        signature_mask=torch.tensor([False, True, False]),
    )
    assert partial_a.selected_slot_id == 0
    assert partial_b.selected_slot_id == 1
    assert bank.digest() == before

    deferred = bank.defer_identity_signature(torch.tensor([1.0, 1.0, 0.0]))
    assert deferred.accepted
    rejected = bank.accept_identity_anchor(
        torch.tensor([0.98, 0.02, 42.0]),
        signature_mask=torch.tensor([True, True, False]),
        verifier_accepted=False,
    )
    assert not rejected.accepted and rejected.selected_slot_id == 0
    assert bank.identity_quarantined_count == 1

    partial_anchor = bank.accept_identity_anchor(
        torch.tensor([0.99, 0.01, 42.0]),
        signature_mask=torch.tensor([True, True, False]),
        verifier_accepted=True,
    )
    assert not partial_anchor.accepted
    assert not partial_anchor.anchor_update_stored
    assert bank.identity_quarantined_count == 1

    resolved = bank.accept_identity_anchor(
        torch.tensor([1.0, 0.0, 0.0]),
        verifier_accepted=True,
    )
    assert resolved.accepted
    assert resolved.selected_slot_id == 0
    assert resolved.anchor_update_stored
    assert resolved.resolved_count == 1
    assert bank.identity_quarantined_count == 0
