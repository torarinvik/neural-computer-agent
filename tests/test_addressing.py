from __future__ import annotations

import pytest
import torch

from neural_computer import (
    FactorizedOpaqueAddressRouter,
    OpaqueAddressRouter,
    OpaqueAppendOnlyRouteChain,
    OpaqueCandidateGrowthRouter,
    OpaqueViewRouteExtension,
    PersistentOpaqueContextRouteEvidence,
    PersistentOpaqueRouteEvidence,
    attempted_outcome_loss,
    failure_gated_candidate_scores,
    failure_gated_view_scores,
)


def test_opaque_address_router_is_row_permutation_equivariant() -> None:
    router = OpaqueAddressRouter(width=4, hidden=8)
    query = torch.randn(3, 4)
    keys = torch.randn(2, 4)
    permutation = torch.tensor([1, 0])

    scores = router(query, keys)
    permuted_scores = router(query, keys[permutation])

    assert torch.allclose(permuted_scores, scores[:, permutation])


def test_attempted_outcome_loss_rejects_invalid_rows_and_nonbinary_outcomes() -> None:
    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="out of range"):
        attempted_outcome_loss(logits, torch.tensor([0, 3]), torch.ones(2))
    with pytest.raises(ValueError, match="binary"):
        attempted_outcome_loss(logits, torch.tensor([0, 1]), torch.tensor([0.0, 0.5]))


def test_factorized_opaque_address_router_is_row_permutation_equivariant() -> None:
    router = FactorizedOpaqueAddressRouter(width=4, hidden=8)
    query = torch.randn(3, 4)
    keys = torch.randn(2, 4)
    permutation = torch.tensor([1, 0])

    scores = router(query, keys)
    permuted_scores = router(query, keys[permutation])

    assert torch.allclose(permuted_scores, scores[:, permutation])


def test_candidate_growth_router_is_neutral_and_permutation_equivariant() -> None:
    router = OpaqueCandidateGrowthRouter(width=4, hidden=8)
    query = torch.randn(3, 4)
    keys = torch.randn(5, 4)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    scores = router(query, keys)
    permuted_scores = router(query, keys[permutation])

    assert torch.equal(scores, torch.zeros_like(scores))
    assert torch.allclose(permuted_scores, scores[:, permutation])


def test_failure_gated_candidate_scores_preserve_old_bank_until_failure() -> None:
    old_scores = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    candidates = torch.tensor([[2.0, 1.0], [2.0, 1.0]])

    scores = failure_gated_candidate_scores(
        old_scores,
        candidates,
        torch.tensor([False, True]),
    )

    assert scores.argmax(dim=-1).tolist() == [1, 2]


def test_opaque_view_route_extension_is_neutral_before_learning() -> None:
    extension = OpaqueViewRouteExtension(width=4, hidden=8)
    query = torch.randn(5, 4)

    scores = extension(query)

    assert scores.shape == (5,)
    assert torch.equal(scores, torch.zeros_like(scores))


def test_opaque_view_route_extension_rejects_wrong_query_width() -> None:
    extension = OpaqueViewRouteExtension(width=4, hidden=8)
    with pytest.raises(ValueError, match="query"):
        extension(torch.randn(2, 5))


def test_failure_gated_view_scores_preserve_old_routes_until_failure() -> None:
    old_scores = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    extension_scores = torch.tensor([1.0, 1.0])

    scores = failure_gated_view_scores(
        old_scores,
        extension_scores,
        torch.tensor([False, True]),
    )

    assert scores.argmax(dim=-1).tolist() == [1, 2]


def test_append_only_route_chain_keeps_all_new_rows_cold_until_failure() -> None:
    base = OpaqueAddressRouter(width=4, hidden=8)
    chain = OpaqueAppendOnlyRouteChain(
        base,
        width=4,
        extensions=(
            OpaqueViewRouteExtension(width=4, hidden=8),
            OpaqueViewRouteExtension(width=4, hidden=8),
        ),
    )
    query = torch.randn(3, 4)
    keys = torch.randn(2, 4)

    cold = chain(query, keys)
    assert cold.shape == (3, 4)
    assert torch.equal(cold.argmax(dim=-1), base(query, keys).argmax(dim=-1))


def test_append_only_route_chain_activates_only_the_failed_stage() -> None:
    base = OpaqueAddressRouter(width=4, hidden=8)
    first = OpaqueViewRouteExtension(width=4, hidden=8)
    second = OpaqueViewRouteExtension(width=4, hidden=8)
    with torch.no_grad():
        first.score.bias.fill_(2.0)
        second.score.bias.fill_(2.0)
    chain = OpaqueAppendOnlyRouteChain(
        base,
        width=4,
        extensions=(first, second),
    )
    query = torch.randn(2, 4)
    keys = torch.randn(2, 4)

    first_failed = chain(
        query,
        keys,
        torch.tensor([[True, False], [True, False]]),
    )
    second_failed = chain(
        query,
        keys,
        torch.tensor([[True, True], [True, True]]),
    )
    assert first_failed.argmax(dim=-1).tolist() == [2, 2]
    assert second_failed.argmax(dim=-1).tolist() == [3, 3]


def test_append_only_route_chain_rejects_misaligned_failure_state() -> None:
    chain = OpaqueAppendOnlyRouteChain(
        OpaqueAddressRouter(width=4, hidden=8),
        width=4,
        extensions=(OpaqueViewRouteExtension(width=4, hidden=8),),
    )
    with pytest.raises(ValueError, match="failed_stages"):
        chain(torch.randn(2, 4), torch.randn(2, 4), torch.zeros(2, 2))


def test_persistent_route_evidence_prefers_last_stable_opaque_slot() -> None:
    evidence = PersistentOpaqueRouteEvidence()
    assert evidence.append_slot() == 0
    assert evidence.append_slot() == 1
    assert evidence.preferred_order() == (0, 1)

    for _ in range(evidence.min_mastery_observations):
        evidence.observe(1, 1.0)

    assert evidence.status().preferred_slot == 1
    assert evidence.preferred_order() == (1, 0)
    assert evidence.preferred_order(slot_count=2) == (1, 0)
    assert evidence.status().posterior[1] > 0.5
    assert evidence.status().protected == (False, True)


def test_persistent_route_evidence_can_reset_a_reused_slot() -> None:
    evidence = PersistentOpaqueRouteEvidence()
    evidence.append_slot()
    evidence.append_slot()
    for _ in range(evidence.min_mastery_observations):
        evidence.observe(1, 1.0)

    evidence.reset_slot(1)

    assert evidence.status().attempts == (0, 0)
    assert evidence.status().protected == (False, False)
    assert evidence.preferred_order() == (0, 1)


def test_persistent_route_evidence_round_trips_without_semantic_fields() -> None:
    evidence = PersistentOpaqueRouteEvidence(prior_strength=2.0)
    evidence.append_slot()
    evidence.observe(0, 0.75)
    restored = PersistentOpaqueRouteEvidence.from_payload(evidence.payload())

    assert restored.payload() == evidence.payload()
    assert "task" not in restored.payload()
    assert "label" not in restored.payload()


def test_context_route_evidence_conditions_preference_on_opaque_learned_key() -> None:
    table = PersistentOpaqueContextRouteEvidence(width=4)
    assert table.append_slot() == 0
    assert table.append_slot() == 1
    cue_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cue_b = torch.tensor([0.0, 1.0, 0.0, 0.0])

    for _ in range(8):
        table.observe(cue_a, 1, 1.0)

    assert table.preferred_order(cue_a) == (1, 0)
    assert table.preferred_order(cue_b) == (0, 1)
    assert table.preferred_slots(torch.stack((cue_a, cue_b))).tolist() == [1, 0]


def test_context_route_evidence_reset_clears_protection_for_reused_slot() -> None:
    table = PersistentOpaqueContextRouteEvidence(width=4)
    table.append_slot()
    table.append_slot()
    cue = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for _ in range(8):
        table.observe(cue, 1, 1.0)
    assert table.protected_slots() == (False, True)

    table.reset_slot(1)

    assert table.protected_slots() == (False, False)
    assert table.preferred_order(cue) == (0, 1)


def test_context_route_evidence_round_trips_opaque_rows() -> None:
    table = PersistentOpaqueContextRouteEvidence(width=3)
    table.append_slot()
    table.append_slot()
    context = torch.tensor([0.0, 2.0, 0.0])
    for _ in range(8):
        table.observe(context, 1, 1.0)

    restored = PersistentOpaqueContextRouteEvidence.from_payload(table.payload())

    assert restored.payload() == table.payload()
    assert restored.preferred_order(context) == (1, 0)
    assert "task" not in restored.payload()
    assert "label" not in restored.payload()


def test_route_evidence_retires_a_stale_mapping_after_patient_failures() -> None:
    evidence = PersistentOpaqueRouteEvidence(
        min_mastery_observations=2,
        reversal_patience=2,
    )
    evidence.append_slot()
    for _ in range(2):
        evidence.observe(0, 1.0)
    assert evidence.status().protected == (True,)

    evidence.observe(0, 0.0)
    evidence.observe(0, 0.0)

    status = evidence.status()
    assert status.preferred_slot is None
    assert status.protected == (False,)
    assert status.reversal_count == (1,)
    assert status.attempts == (0,)


def test_context_route_evidence_can_commit_a_new_slot_after_reversal() -> None:
    table = PersistentOpaqueContextRouteEvidence(
        width=3,
        min_mastery_observations=2,
        reversal_patience=2,
    )
    table.append_slot()
    table.append_slot()
    context = torch.tensor([1.0, 0.0, 0.0])
    for _ in range(2):
        table.observe(context, 0, 1.0)
    assert table.preferred_order(context) == (0, 1)

    table.observe(context, 0, 0.0)
    table.observe(context, 0, 0.0)
    for _ in range(2):
        table.observe(context, 1, 1.0)

    assert table.preferred_order(context) == (1, 0)


def test_context_route_evidence_groups_batch_outcomes_before_advancing_ledger() -> None:
    table = PersistentOpaqueContextRouteEvidence(
        width=3,
        min_mastery_observations=2,
    )
    table.append_slot()
    context = torch.tensor([1.0, 0.0, 0.0])

    table.observe_batch(
        contexts=torch.stack((context, context, context)),
        slots=torch.tensor([0, 0, 0]),
        outcomes=torch.tensor([1.0, 0.0, 1.0]),
    )

    first = table._records[0].evidence.status()
    assert first.attempts == (1,)
    assert first.successes == (2.0 / 3.0,)
    assert first.protected == (False,)

    table.observe_batch(
        contexts=torch.stack((context, context)),
        slots=torch.tensor([0, 0]),
        outcomes=torch.tensor([1.0, 1.0]),
    )

    second = table._records[0].evidence.status()
    assert second.attempts == (2,)
    assert second.protected == (True,)
