import math
from collections.abc import Callable

import pytest
import torch

from neural_computer import (
    EXTERNAL_FACTORED_TRANSITION_EXACT_RESIDUAL_MODE,
    EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    ExternalAffineTransitionStatistics,
    ExternalBoundTransitionModel,
    ExternalContextAddressResolver,
    ExternalContextualEvidenceCalibrator,
    ExternalContextualTransitionEvidenceStatistics,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalGoalEvaluator,
    ExternalModelBasedPlanner,
    ExternalOnlineContextAddressResolver,
    ExternalOnlineTransitionContextRouter,
    ExternalRoutedIntentionCostLedger,
    ExternalSignedEntryValueModel,
    ExternalSparseTransitionEvidenceIndex,
    ExternalTransitionContextAddressAdapter,
    ExternalTransitionContextEncoder,
    ExternalTransitionEvidenceEvaluator,
    ExternalTransitionEvidenceStatistics,
    ExternalTransitionMemory,
    ExternalTransitionModel,
    ExternalTransitionModelBank,
    ExternalTransitionModelLifetimePolicy,
    ExternalTransitionModelPriorSelectionReceipt,
    ExternalTransitionObservation,
    ExternalTransitionProbeContextualUtilityMemory,
    ExternalTransitionProbeUtilityMemory,
    ExternalTransitionRollout,
    ExternalTransitionRouteMemory,
    ExternalTransitionRouteQuery,
    ExternalTransitionSupportStatistics,
    OpaqueCandidateGrowthRouter,
    OpaqueCapacityPlanner,
)


def test_transition_route_query_is_opaque_proposal_and_persistent() -> None:
    query = ExternalTransitionRouteQuery(4)
    contexts = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    proposal = query.propose(
        torch.tensor([0.0, 0.9, 0.1, 0.0]),
        contexts,
        (11, 22, 33),
    )

    assert proposal.selected_slot_id == 22
    assert proposal.eligible_slot_ids == (11, 22, 33)
    assert proposal.margin is not None and proposal.margin > 0.0
    assert "verification" in proposal.reason

    restored = ExternalTransitionRouteQuery.from_payload(query.state_payload())
    assert restored.configuration() == query.configuration()
    assert restored.digest() == query.digest()

    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=6, context_width=4)
    adapter = ExternalTransitionContextAddressAdapter(encoder)
    observation = ExternalTransitionObservation(
        state=torch.randn(3, 2),
        intention=torch.randn(3, 1),
        next_state=torch.randn(3, 2),
        confidence=torch.ones(3),
    )
    query.register_slot(22, adapter, route_key=adapter.trajectory_stats(observation))
    restored_with_slot = ExternalTransitionRouteQuery.from_payload(
        query.state_payload()
    )
    proposal = restored_with_slot.propose_observation(
        observation,
        contexts,
        (11, 22, 33),
        fallback_query=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    assert proposal.selected_slot_id == 22
    assert restored_with_slot.digest() == query.digest()


def test_transition_route_query_empty_bank_is_explicit() -> None:
    proposal = ExternalTransitionRouteQuery(3).propose(
        torch.tensor([1.0, 0.0, 0.0]),
        torch.empty(0, 3),
        (),
    )

    assert proposal.selected_slot_id is None
    assert proposal.scores.numel() == 0
    assert proposal.margin is None


def test_sparse_transition_evidence_routes_partial_overlap_and_rejects_conflict() -> (
    None
):
    memory = ExternalSparseTransitionEvidenceIndex(
        2,
        1,
        input_match_tolerance=0.01,
        output_match_tolerance=0.01,
        minimum_matches=2,
    )
    source = ExternalTransitionObservation(
        state=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        intention=torch.tensor([[1.0], [2.0], [3.0]]),
        next_state=torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
    )
    memory.record(7, source)
    partial = ExternalTransitionObservation(
        state=source.state[[0, 2]],
        intention=source.intention[[0, 2]],
        next_state=source.next_state[[0, 2]],
    )
    proposal = memory.propose(partial, (7,))
    assert proposal.selected_slot_id == 7
    assert proposal.matched_observations == (2,)
    assert proposal.contradictory_observations == (0,)
    conflict = ExternalTransitionObservation(
        state=source.state[[0, 2]],
        intention=source.intention[[0, 2]],
        next_state=source.next_state[[0, 2]].roll(1, 0),
    )
    rejected = memory.propose(conflict, (7,))
    assert rejected.selected_slot_id is None
    assert rejected.contradictory_observations == (2,)


def test_sparse_transition_evidence_compacts_and_persists_unique_facts() -> None:
    memory = ExternalSparseTransitionEvidenceIndex(2, 1)
    observation = ExternalTransitionObservation(
        state=torch.tensor([[1.0, 0.0]]),
        intention=torch.tensor([[1.0]]),
        next_state=torch.tensor([[0.0, 1.0]]),
    )
    memory.record(3, observation)
    memory.record(
        3,
        ExternalTransitionObservation(
            state=observation.state + 0.001,
            intention=observation.intention,
            next_state=observation.next_state + 0.001,
        ),
    )
    assert memory.slot_record_count(3) == 1
    compact = memory.observation_for_slot(3)
    assert compact.state.shape == (1, 2)
    restored = ExternalSparseTransitionEvidenceIndex.from_payload(
        memory.state_payload()
    )
    assert restored.digest() == memory.digest()
    assert restored.record_count == memory.record_count


def test_sparse_transition_evidence_preserves_conflicting_drift_versions() -> None:
    memory = ExternalSparseTransitionEvidenceIndex(
        1,
        1,
        input_match_tolerance=1e-6,
        output_match_tolerance=0.01,
        minimum_matches=1,
        minimum_match_fraction=1.0,
    )
    state = torch.tensor([[0.5]])
    intention = torch.tensor([[1.0]])
    source = ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=torch.tensor([[1.0]]),
    )
    drift = ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=torch.tensor([[1.5]]),
    )
    memory.record(4, source)
    memory.record(4, drift)
    assert memory.slot_record_count(4) == 2
    assert memory.propose(source, (4,)).selected_slot_id == 4
    assert memory.propose(drift, (4,)).selected_slot_id == 4
    restored = ExternalSparseTransitionEvidenceIndex.from_payload(
        memory.state_payload()
    )
    assert restored.slot_record_count(4) == 2
    assert restored.propose(drift, (4,)).selected_slot_id == 4


def test_factored_sparse_proposal_cannot_bypass_factual_verification() -> None:
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_EXACT_RESIDUAL_MODE,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    sparse = ExternalSparseTransitionEvidenceIndex(
        1,
        1,
        input_match_tolerance=1e-6,
        output_match_tolerance=1e-6,
        minimum_matches=1,
        minimum_match_fraction=1.0,
    )
    router = ExternalFactoredTransitionRouter(
        model,
        ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2),
        max_contexts=1,
        match_tolerance=0.01,
        sparse_evidence=sparse,
    )
    router._contexts = [torch.zeros(2)]
    router._slot_ids = [0]
    observation = ExternalTransitionObservation(
        state=torch.zeros(1, 1),
        intention=torch.zeros(1, 1),
        next_state=torch.ones(1, 1),
    )
    sparse.record(0, observation)
    assert sparse.propose(observation, (0,)).selected_slot_id == 0
    assert router.route_partial_bundle((observation,)).status == "ambiguous"


def test_transition_route_memory_is_slot_local_bounded_and_persistent() -> None:
    memory = ExternalTransitionRouteMemory(
        4,
        max_prototypes_per_slot=2,
        merge_cosine=0.99,
    )
    memory.register_slot(10, prototype=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    memory.register_slot(20, prototype=torch.tensor([0.0, 1.0, 0.0, 0.0]))
    proposal = memory.propose(
        torch.tensor([0.0, 0.9, 0.1, 0.0]),
        (10, 20),
        minimum_score=0.5,
    )
    assert proposal.selected_slot_id == 20
    assert memory.observe(10, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert not memory.observe(10, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert memory.prototype_count(10) == 2

    restored = ExternalTransitionRouteMemory.from_payload(memory.state_payload())
    restored_proposal = restored.propose(
        torch.tensor([0.0, 0.0, 0.9, 0.1]),
        (10, 20),
        minimum_score=0.5,
    )
    assert restored.digest() == memory.digest()
    assert restored_proposal.selected_slot_id == 10
    assert "next_state" not in restored.state_payload()


def test_transition_route_memory_scores_partial_queries_without_mutating_state() -> None:
    memory = ExternalTransitionRouteMemory(4, merge_cosine=0.99)
    memory.register_slot(10, prototype=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    memory.register_slot(20, prototype=torch.tensor([0.0, 1.0, 0.0, 0.0]))
    before = memory.digest()
    proposal = memory.propose(
        torch.tensor([0.95, 0.0, 17.0, -23.0]),
        (10, 20),
        minimum_score=0.8,
        query_mask=torch.tensor([True, False, False, False]),
    )
    assert proposal.selected_slot_id == 10
    assert memory.digest() == before
    with pytest.raises(ValueError, match="retain evidence"):
        memory.propose(
            torch.ones(4),
            (10, 20),
            minimum_score=0.8,
            query_mask=torch.zeros(4, dtype=torch.bool),
        )


def test_transition_route_memory_learns_masked_prototypes_and_restores_them() -> None:
    memory = ExternalTransitionRouteMemory(4, merge_cosine=0.999999)
    memory.register_slot(10, prototype=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    before = memory.digest()
    assert memory.observe(
        10,
        torch.tensor([0.98, 0.02, 9.0, -7.0]),
        query_mask=torch.tensor([True, True, False, False]),
    )
    assert memory.digest() != before
    assert memory.masked_prototype_count == 1
    payload = memory.state_payload()
    row = payload["slots"]["10"][1]
    assert row["mask"] == [True, True, False, False]
    restored = ExternalTransitionRouteMemory.from_payload(payload)
    assert restored.digest() == memory.digest()
    assert restored.masked_prototype_count == 1
    query = restored.propose(
        torch.tensor([0.99, 0.01, 100.0, 100.0]),
        (10,),
        minimum_score=0.95,
        query_mask=torch.tensor([True, True, False, False]),
    )
    assert query.selected_slot_id == 10


def test_transition_route_memory_replacement_is_atomic_and_retention_gated() -> None:
    memory = ExternalTransitionRouteMemory(
        4,
        max_prototypes_per_slot=2,
        merge_cosine=0.99,
    )
    memory.register_slot(10, prototype=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert memory.observe(10, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert memory.observe(10, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert memory.prototype_count(10) == 2
    source_digest = memory.digest()
    new_query = torch.tensor([0.0, 0.0, 0.0, 1.0])
    rejected = memory.replace_verified(
        10,
        new_query,
        retention_probe=lambda _candidate: False,
    )
    assert not rejected.accepted
    assert memory.digest() == source_digest
    assert memory.prototype_count(10) == 2

    def retention_probe(candidate: ExternalTransitionRouteMemory) -> bool:
        return (
            candidate.propose(
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                (10,),
                minimum_score=0.9,
            ).selected_slot_id
            == 10
            and candidate.propose(
                new_query,
                (10,),
                minimum_score=0.9,
            ).selected_slot_id
            == 10
        )

    accepted = memory.replace_verified(
        10,
        new_query,
        retention_probe=retention_probe,
    )
    assert accepted.accepted
    assert accepted.replaced_index == 1
    assert memory.prototype_count(10) == 2
    restored = ExternalTransitionRouteMemory.from_payload(memory.state_payload())
    assert restored.digest() == memory.digest()
    assert restored.propose(new_query, (10,), minimum_score=0.9).selected_slot_id == 10


def test_transition_route_memory_honors_verified_planner_replacement_index() -> None:
    memory = ExternalTransitionRouteMemory(
        4,
        max_prototypes_per_slot=2,
        merge_cosine=0.99,
    )
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 0.0, 1.0, 0.0])
    replacement = torch.tensor([0.0, 0.0, 0.0, 1.0])
    memory.register_slot(10, prototype=first)
    assert memory.observe(10, second)
    receipt = memory.replace_verified(
        10,
        replacement,
        replacement_index=0,
        retention_probe=lambda candidate: (
            candidate.propose(second, (10,), minimum_score=0.9).selected_slot_id
            == 10
            and candidate.propose(replacement, (10,), minimum_score=0.9).selected_slot_id
            == 10
        ),
    )
    assert receipt.accepted
    assert receipt.replaced_index == 0
    assert memory.propose(first, (10,), minimum_score=0.99).selected_slot_id is None
    assert memory.propose(replacement, (10,), minimum_score=0.99).selected_slot_id == 10


def test_transition_route_memory_capacity_growth_is_atomic_and_replay_free() -> None:
    memory = ExternalTransitionRouteMemory(
        4,
        max_prototypes_per_slot=1,
        merge_cosine=0.99,
    )
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])
    third = torch.tensor([0.0, 0.0, 1.0, 0.0])
    fourth = torch.tensor([0.0, 0.0, 0.0, 1.0])
    memory.register_slot(10, prototype=first)
    source_digest = memory.digest()

    rejected = memory.grow_verified(3, lambda _candidate: False)
    assert not rejected.accepted
    assert memory.max_prototypes_per_slot == 1
    assert memory.digest() == source_digest

    def retention_probe(candidate: ExternalTransitionRouteMemory) -> bool:
        return (
            candidate.max_prototypes_per_slot == 3
            and candidate.propose(first, (10,), minimum_score=0.99).selected_slot_id
            == 10
        )

    accepted = memory.grow_verified(3, retention_probe)
    assert accepted.accepted
    assert accepted.source_capacity == 1
    assert accepted.destination_capacity == 3
    assert memory.max_prototypes_per_slot == 3
    assert memory.prototype_count(10) == 1
    assert memory.observe(10, second)
    assert memory.observe(10, third)
    assert not memory.observe(10, fourth)
    assert memory.prototype_count(10) == 3
    for query in (first, second, third):
        assert memory.propose(query, (10,), minimum_score=0.99).selected_slot_id == 10

    restored = ExternalTransitionRouteMemory.from_payload(memory.state_payload())
    assert restored.max_prototypes_per_slot == 3
    assert restored.total_prototype_count == 3
    assert restored.digest() == memory.digest()


def test_transition_route_memory_requires_mask_overlap_for_merges() -> None:
    boundary = ExternalTransitionRouteMemory(
        4,
        max_prototypes_per_slot=2,
        merge_cosine=0.9,
    )
    boundary.register_slot(10, prototype=torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert boundary.observe(
        10,
        torch.tensor([1.0, 1.0, 0.0, 9.0]),
        query_mask=torch.tensor([True, True, True, False]),
    )
    assert boundary.prototype_count(10) == 2

    memory = ExternalTransitionRouteMemory(
        4,
        max_prototypes_per_slot=3,
        merge_cosine=0.9,
    )
    full = torch.tensor([1.0, 0.0, 0.0, 0.0])
    first_partial = torch.tensor([0.0, 1.0, 0.0, 0.0])
    second_partial = torch.tensor([0.0, 0.0, 0.0, 1.0])
    first_mask = torch.tensor([True, True, True, False])
    second_mask = torch.tensor([True, False, False, True])
    memory.register_slot(10, prototype=full)
    assert memory.observe(10, first_partial, query_mask=first_mask)
    assert memory.prototype_count(10) == 2

    # Repeated evidence with the same mask still merges into its prototype.
    assert memory.observe(
        10,
        first_partial + torch.tensor([0.0, 0.01, 0.0, 0.0]),
        query_mask=first_mask,
    )
    assert memory.prototype_count(10) == 2

    # The two masks overlap on only one of four dimensions.  Their shared
    # cosine is not sufficient to erase the fact that this is new coverage.
    assert memory.observe(10, second_partial, query_mask=second_mask)
    assert memory.prototype_count(10) == 3
    restored = ExternalTransitionRouteMemory.from_payload(memory.state_payload())
    assert restored.merge_mask_overlap == 0.75
    assert restored.digest() == memory.digest()


def test_transition_route_memory_consolidation_is_retention_gated_and_persistent() -> None:
    memory = ExternalTransitionRouteMemory(
        4,
        max_prototypes_per_slot=4,
        merge_cosine=0.99,
    )
    full = torch.tensor([1.0, 0.0, 0.0, 0.0])
    first = torch.tensor([0.0, 1.0, 0.0, 0.0])
    second = torch.tensor([0.0, 0.0, 1.0, 0.0])
    third = torch.tensor([0.0, 0.0, 0.0, 1.0])
    first_mask = torch.tensor([True, True, False, False])
    second_mask = torch.tensor([False, False, True, True])
    third_mask = torch.tensor([False, False, False, True])
    memory.register_slot(10, prototype=full)
    assert memory.observe(10, first, query_mask=first_mask)
    assert memory.observe(10, second, query_mask=second_mask)
    assert memory.observe(10, third, query_mask=third_mask)
    assert memory.prototype_count(10) == 4
    source_digest = memory.digest()

    rejected = memory.consolidate_verified(
        10,
        (2, 1),
        retention_probe=lambda _candidate: False,
    )
    assert not rejected.accepted
    assert rejected.merged_indices == (1, 2)
    assert memory.prototype_count(10) == 4
    assert memory.digest() == source_digest

    def retention_probe(candidate: ExternalTransitionRouteMemory) -> bool:
        return all(
            candidate.propose(
                query,
                (10,),
                minimum_score=0.95,
                query_mask=query_mask,
            ).selected_slot_id
            == 10
            for query, query_mask in (
                (full, None),
                (first, first_mask),
                (second, second_mask),
                (third, third_mask),
            )
        )

    accepted = memory.consolidate_verified(10, (1, 2), retention_probe)
    assert accepted.accepted
    assert accepted.merged_indices == (1, 2)
    assert accepted.source_prototype_count == 4
    assert accepted.destination_prototype_count == 3
    assert memory.prototype_count(10) == 3
    restored = ExternalTransitionRouteMemory.from_payload(memory.state_payload())
    assert restored.prototype_count(10) == 3
    assert restored.digest() == memory.digest()
    for query, query_mask in (
        (full, None),
        (first, first_mask),
        (second, second_mask),
        (third, third_mask),
    ):
        assert (
            restored.propose(
                query,
                (10,),
                minimum_score=0.95,
                query_mask=query_mask,
            ).selected_slot_id
            == 10
        )


def test_transition_route_memory_adapts_to_opaque_capacity_planner() -> None:
    memory = ExternalTransitionRouteMemory(
        4,
        max_prototypes_per_slot=3,
        merge_cosine=0.99,
    )
    memory.register_slot(10, prototype=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert memory.observe(
        10,
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        query_mask=torch.tensor([False, True, True, False]),
    )
    candidates = memory.policy_candidates(10)
    assert candidates.keys.shape == (1, 3, 4)
    assert candidates.values[0, 1].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert candidates.occupied[0].tolist() == [True, True, False]
    source_digest = memory.digest()
    planner = OpaqueCapacityPlanner(width=4, hidden=8).eval()
    plan = memory.maintenance_plan(
        10,
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        planner=planner,
        query_mask=torch.tensor([True, False, False, True]),
        protected_indices=(0,),
    )
    assert plan.action in {"admit", "evict", "consolidate", "grow"}
    if plan.eviction_index is not None:
        assert plan.eviction_index != 0
    if plan.pair is not None:
        assert plan.pair[0] != plan.pair[1]
        assert all(index in {0, 1} for index in plan.pair)
    assert memory.digest() == source_digest


def test_transition_route_query_can_use_slot_local_prototype_memory() -> None:
    memory = ExternalTransitionRouteMemory(4)
    query = ExternalTransitionRouteQuery(
        4,
        minimum_score=0.8,
        route_memory=memory,
    )
    query.register_slot(10, route_key=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    query.register_slot(20, route_key=torch.tensor([0.0, 1.0, 0.0, 0.0]))
    observation = ExternalTransitionObservation(
        state=torch.randn(2, 2),
        intention=torch.randn(2, 1),
        next_state=torch.randn(2, 2),
    )
    proposal = query.propose_observation(
        observation,
        torch.randn(2, 4),
        (10, 20),
        fallback_query=torch.tensor([0.98, 0.02, 0.0, 0.0]),
    )
    restored = ExternalTransitionRouteQuery.from_payload(query.state_payload())
    restored_proposal = restored.propose_observation(
        observation,
        torch.randn(2, 4),
        (10, 20),
        fallback_query=torch.tensor([0.98, 0.02, 0.0, 0.0]),
    )
    assert proposal.selected_slot_id == 10
    assert restored_proposal.selected_slot_id == proposal.selected_slot_id
    assert restored.digest() == query.digest()


def test_transition_route_query_can_use_trained_context_key_feature() -> None:
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=6, context_width=4)
    adapter = ExternalTransitionContextAddressAdapter(encoder)
    memory = ExternalTransitionRouteMemory(4)
    query = ExternalTransitionRouteQuery(
        4,
        minimum_score=0.8,
        route_width=4,
        route_memory=memory,
    )
    observation = ExternalTransitionObservation(
        state=torch.randn(2, 2),
        intention=torch.randn(2, 1),
        next_state=torch.randn(2, 2),
    )
    context_key = adapter.encode_observation(observation)
    query.register_slot(10, adapter, route_key=context_key)
    query.register_slot(20, route_key=torch.tensor([0.0, 1.0, 0.0, 0.0]))

    assert not query.uses_trajectory_feature()
    proposal = query.propose_observation(
        observation,
        torch.randn(2, 4),
        (10, 20),
        fallback_query=context_key,
    )
    assert proposal.selected_slot_id == 10
    assert query.configuration()["feature"] == "context_key_v1"


def test_online_router_verified_prior_selection_is_isolated_and_persistent() -> None:
    torch.manual_seed(1213)
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        hidden_width=8,
        capacity=2,
    )
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=6,
        context_width=4,
    )
    source_context = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    source_digest = bank.models[source_context].digest()

    def probe(
        transfer: torch.nn.Module,
        fresh: torch.nn.Module,
        _observation: ExternalTransitionObservation,
    ) -> tuple[float, float]:
        with torch.no_grad():
            for parameter in transfer.parameters():
                parameter.add_(0.01)
            for parameter in fresh.parameters():
                parameter.add_(0.02)
        return 0.1, 0.2

    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=0.0,
        match_margin=0.0,
        admission_observations=1,
        max_contexts=2,
        defer_admission=True,
        candidate_model_families=("nonlinear_mlp_v1",),
        provisional_evidence_policy="streaming_gradient",
        prior_selection_probe=probe,
        prior_selection_probe_updates=1,
        prior_selection_transfer_cost=0.0,
        prior_selection_fresh_cost=1.0,
        prior_selection_cost_weight=0.2,
    )
    result = router.observe(
        ExternalTransitionObservation(
            state=torch.randn(1, 2),
            intention=torch.randn(1, 1),
            next_state=torch.randn(1, 2),
        )
    )

    assert result.status == "staged"
    receipt = router._provisional_candidates[0].prior_selection
    assert receipt is not None
    assert receipt.selected_initialization == "transfer"
    assert receipt.schema.endswith("prior-selection.v2")
    assert receipt.transfer_cost == 0.0
    assert receipt.fresh_cost == 1.0
    assert receipt.cost_weight == 0.2
    assert bank.models[source_context].digest() == source_digest
    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload(),
        prior_selection_probe=probe,
    )
    restored_receipt = restored._provisional_candidates[0].prior_selection
    assert restored_receipt is not None
    assert restored_receipt.selected_model_digest == receipt.selected_model_digest
    assert restored.configuration() == router.configuration()


def test_online_router_learned_cost_ledger_updates_only_after_verified_promotion() -> None:
    torch.manual_seed(1214)
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8, capacity=2)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=6, context_width=4)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    ledger = ExternalRoutedIntentionCostLedger.create(4, initial_cost=0.5)

    def probe(
        _transfer: torch.nn.Module,
        _fresh: torch.nn.Module,
        _observation: ExternalTransitionObservation,
    ) -> tuple[float, float]:
        return 0.1, 0.2

    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=0.0,
        match_margin=0.0,
        admission_observations=1,
        max_contexts=2,
        defer_admission=True,
        candidate_model_families=("nonlinear_mlp_v1",),
        provisional_evidence_policy="streaming_gradient",
        prior_selection_probe=probe,
        prior_selection_cost_ledger=ledger,
    )
    observation = ExternalTransitionObservation(
        state=torch.randn(1, 2),
        intention=torch.randn(1, 1),
        next_state=torch.randn(1, 2),
    )
    result = router.observe(observation)
    assert result.status == "staged"
    prior = router.provisional_prior_selection_at(0)
    assert prior is not None
    assert prior.schema.endswith("prior-selection.v2")
    assert prior.transfer_cost == pytest.approx(0.5)
    assert int(ledger.state.transfer_observations.item()) == 0

    promoted = router.promote_staged_candidate(
        observation,
        lambda _candidate_bank: True,
        prediction_tolerance=1e9,
        prior_selection_observed_cost=0.0,
    )
    assert promoted.accepted
    assert promoted.prior_selection_cost_observation is not None
    assert int(ledger.state.transfer_observations.item()) == 1
    assert bank.context_count == 2

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload(),
        prior_selection_probe=probe,
    )
    assert restored.prior_selection_cost_ledger is not None
    assert int(
        restored.prior_selection_cost_ledger.state.transfer_observations.item()
    ) == 1


def test_learned_transition_route_query_is_persistent_and_opaque() -> None:
    torch.manual_seed(1211)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=6, context_width=4)
    adapter = ExternalTransitionContextAddressAdapter(encoder)
    observation = ExternalTransitionObservation(
        state=torch.randn(3, 2),
        intention=torch.randn(3, 1),
        next_state=torch.randn(3, 2),
        confidence=torch.ones(3),
    )
    scorer = OpaqueCandidateGrowthRouter(width=8, hidden=6)
    with torch.no_grad():
        scorer.score[-1].bias.fill_(0.25)
    query = ExternalTransitionRouteQuery(
        4,
        minimum_score=0.1,
        route_width=8,
        learned_scorer=scorer,
    )
    query.register_slot(
        22,
        adapter,
        route_key=torch.randn(8),
    )
    query.register_slot(
        33,
        adapter,
        route_key=torch.randn(8),
    )
    contexts = torch.randn(2, 4)
    fallback = torch.randn(8)
    proposal = query.propose_observation(
        observation,
        contexts,
        (22, 33),
        fallback_query=fallback,
    )

    restored = ExternalTransitionRouteQuery.from_payload(query.state_payload())
    restored_proposal = restored.propose_observation(
        observation,
        contexts,
        (22, 33),
        fallback_query=fallback,
    )
    assert restored.learned_scorer is not None
    assert restored.configuration() == query.configuration()
    assert restored.digest() == query.digest()
    assert torch.allclose(restored_proposal.scores, proposal.scores)
    assert restored_proposal.selected_slot_id == proposal.selected_slot_id


def test_transition_router_factual_fallback_survives_wrong_route_proposal() -> None:
    torch.manual_seed(1212)
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=6, context_width=4)
    observation = ExternalTransitionObservation(
        state=torch.zeros(2, 2),
        intention=torch.zeros(2, 1),
        next_state=torch.ones(2, 2),
        confidence=torch.ones(2),
    )
    query_vector = encoder.encode_observation(observation)
    first = bank.ensure_context(query_vector)
    second = bank.ensure_context(-query_vector)
    with torch.no_grad():
        for parameter in bank.models[first].parameters():
            parameter.zero_()
        for parameter in bank.models[second].parameters():
            parameter.zero_()
        bank.models[first].network[-1].bias.zero_()
        bank.models[second].network[-1].bias.fill_(1.0)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-6,
        match_margin=1e-6,
        route_query=ExternalTransitionRouteQuery(
            4,
            minimum_score=-1.0,
            route_memory=ExternalTransitionRouteMemory(4),
        ),
    )
    router.route_query.register_slot(  # type: ignore[union-attr]
        bank.slot_id_at(first),
        route_key=query_vector,
    )
    router.route_query.register_slot(  # type: ignore[union-attr]
        bank.slot_id_at(second),
        route_key=-query_vector,
    )

    selected = router._best_slot(observation)

    assert selected is not None
    assert selected[0] == second
    assert router.route_query.route_memory.prototype_count(second) == 2  # type: ignore[union-attr]


def test_transition_model_bank_marks_random_features_replay_free() -> None:
    affine = ExternalTransitionModelBank(
        2,
        1,
        3,
        model_family="affine_sufficient_statistics_v1",
    )
    random_features = ExternalTransitionModelBank(
        2,
        1,
        3,
        model_family="random_feature_sufficient_statistics_v1",
    )
    neural = ExternalTransitionModelBank(2, 1, 3)

    assert affine.replay_free_updates
    assert random_features.replay_free_updates
    assert not neural.replay_free_updates


def test_transition_context_encoder_is_opaque_normalized_and_persistent() -> None:
    torch.manual_seed(1200)
    encoder = ExternalTransitionContextEncoder(
        3,
        2,
        hidden_width=8,
        context_width=5,
    )
    observation = ExternalTransitionObservation(
        state=torch.randn(4, 3),
        intention=torch.randn(4, 2),
        next_state=torch.randn(4, 3),
        confidence=torch.ones(4),
    )

    context = encoder.encode_observation(observation)
    batched = encoder(
        observation.state.unsqueeze(0),
        observation.intention.unsqueeze(0),
        observation.next_state.unsqueeze(0),
        observation.confidence.unsqueeze(0),
    )
    assert context.shape == (5,)
    assert torch.allclose(torch.linalg.vector_norm(context), torch.ones(()))
    assert torch.allclose(context, batched[0])

    left = torch.randn(3, 5)
    right = torch.randn(3, 5)
    assert torch.isfinite(encoder.contrastive_loss(left, right))

    restored = ExternalTransitionContextEncoder.from_payload(encoder.state_payload())
    assert restored.configuration() == encoder.configuration()
    assert restored.digest() == encoder.digest()
    assert torch.equal(restored.encode_observation(observation), context)


def test_transition_context_encoder_copy_on_write_adaptation_is_one_pass() -> None:
    torch.manual_seed(1205)
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=8,
        context_width=5,
    )
    left = [
        ExternalTransitionObservation(
            state=torch.randn(4, 2),
            intention=torch.randn(4, 1),
            next_state=torch.randn(4, 2),
            confidence=torch.ones(4),
        )
        for _ in range(2)
    ]
    right = [
        ExternalTransitionObservation(
            state=observation.state + 0.01 * torch.tanh(observation.state),
            intention=observation.intention,
            next_state=observation.next_state
            + 0.01 * torch.tanh(observation.next_state),
            confidence=observation.confidence,
        )
        for observation in left
    ]
    before_digest = encoder.digest()
    candidate, loss = encoder.copy_on_write_contrastive_step(left, right)

    assert math.isfinite(loss)
    assert encoder.digest() == before_digest
    assert candidate.digest() != before_digest
    restored = ExternalTransitionContextEncoder.from_payload(
        candidate.state_payload()
    )
    assert restored.digest() == candidate.digest()


def test_transition_context_encoder_prefix_alignment_is_copy_on_write() -> None:
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(1206)
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=8,
        context_width=5,
    )

    def observation(seed: int, length: int) -> ExternalTransitionObservation:
        generator = torch.Generator().manual_seed(seed)
        return ExternalTransitionObservation(
            state=torch.randn(length, 2, generator=generator),
            intention=torch.randn(length, 1, generator=generator),
            next_state=torch.randn(length, 2, generator=generator),
            confidence=torch.ones(length),
        )

    prefixes = (
        (observation(1, 1), observation(2, 2)),
        (observation(3, 1), observation(4, 2)),
    )
    full = (observation(5, 3), observation(6, 3))
    before_digest = encoder.digest()

    candidate, loss = encoder.copy_on_write_prefix_alignment_step(prefixes, full)

    assert math.isfinite(loss)
    assert encoder.digest() == before_digest
    assert candidate.digest() != before_digest
    assert (
        candidate.configuration()["prefix_alignment_update"]
        == "copy_on_write_one_pass_v1"
    )
    restored = ExternalTransitionContextEncoder.from_payload(
        candidate.state_payload()
    )
    assert restored.digest() == candidate.digest()
    torch.random.set_rng_state(rng_state)


def test_transition_context_mean_pool_is_permutation_invariant_and_persistent() -> None:
    torch.manual_seed(1200)
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=8,
        context_width=5,
        aggregation="mean_pool",
    )
    observation = ExternalTransitionObservation(
        state=torch.randn(6, 2),
        intention=torch.randn(6, 1),
        next_state=torch.randn(6, 2),
        confidence=torch.rand(6),
    )
    permutation = torch.tensor([5, 1, 3, 0, 4, 2])
    permuted = ExternalTransitionObservation(
        state=observation.state.index_select(0, permutation),
        intention=observation.intention.index_select(0, permutation),
        next_state=observation.next_state.index_select(0, permutation),
        confidence=observation.confidence.index_select(0, permutation),
    )

    assert torch.allclose(
        encoder.encode_observation(observation),
        encoder.encode_observation(permuted),
        atol=1e-6,
    )
    restored = ExternalTransitionContextEncoder.from_payload(encoder.state_payload())
    assert restored.configuration() == encoder.configuration()
    assert restored.digest() == encoder.digest()


def test_transition_context_recency_latest_preserves_actual_latest_evidence() -> None:
    torch.manual_seed(1207)
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=8,
        context_width=5,
        aggregation="recency_weighted_and_latest",
        recency_decay=0.75,
    )
    observation = ExternalTransitionObservation(
        state=torch.randn(5, 2),
        intention=torch.randn(5, 1),
        next_state=torch.randn(5, 2),
        confidence=torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]),
    )
    changed_absent_tail = ExternalTransitionObservation(
        state=observation.state.clone(),
        intention=observation.intention.clone(),
        next_state=observation.next_state.clone(),
        confidence=observation.confidence.clone(),
    )
    changed_absent_tail.next_state[4].add_(100.0)
    changed_latest = ExternalTransitionObservation(
        state=observation.state.clone(),
        intention=observation.intention.clone(),
        next_state=observation.next_state.clone(),
        confidence=observation.confidence.clone(),
    )
    changed_latest.next_state[2].add_(0.5)

    base = encoder.encode_observation(observation)
    absent_tail = encoder.encode_observation(changed_absent_tail)
    latest = encoder.encode_observation(changed_latest)
    restored = ExternalTransitionContextEncoder.from_payload(encoder.state_payload())

    assert torch.equal(base, absent_tail)
    assert not torch.equal(base, latest)
    assert restored.configuration() == encoder.configuration()
    assert restored.digest() == encoder.digest()


def test_context_address_adapter_is_copy_on_write_and_persistent() -> None:
    torch.manual_seed(1203)
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=8,
        context_width=5,
        aggregation="mean_pool",
    )
    adapter = ExternalTransitionContextAddressAdapter(
        encoder,
        learning_rate=0.01,
        adaptation_steps=16,
        anchor_cosine_ceiling=0.2,
    )
    observation = ExternalTransitionObservation(
        state=torch.randn(8, 2),
        intention=torch.randn(8, 1),
        next_state=torch.randn(8, 2),
        confidence=torch.ones(8),
    )
    anchor = adapter.encode_observation(observation).unsqueeze(0)
    before_digest = adapter.digest()
    candidate = adapter.copy_on_write(observation, anchor)

    assert adapter.digest() == before_digest
    assert candidate.version == adapter.version + 1
    assert candidate.parent_digest == before_digest
    assert (
        float(
            adapter.encode_observation(observation).detach()
            @ candidate.encode_observation(observation).detach()
        )
        <= 0.2
    )
    restored = ExternalTransitionContextAddressAdapter.from_payload(
        candidate.state_payload()
    )
    assert restored.configuration() == candidate.configuration()
    assert restored.digest() == candidate.digest()


def test_context_address_adapter_prefix_alignment_is_versioned_and_isolated() -> None:
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(1207)
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=8,
        context_width=5,
    )
    adapter = ExternalTransitionContextAddressAdapter(
        encoder,
        learning_rate=0.01,
        adaptation_steps=2,
    )

    def observation(seed: int, length: int) -> ExternalTransitionObservation:
        generator = torch.Generator().manual_seed(seed)
        return ExternalTransitionObservation(
            state=torch.randn(length, 2, generator=generator),
            intention=torch.randn(length, 1, generator=generator),
            next_state=torch.randn(length, 2, generator=generator),
            confidence=torch.ones(length),
        )

    prefixes = (
        (observation(11, 1), observation(12, 2)),
        (observation(13, 1), observation(14, 2)),
    )
    full = (observation(15, 3), observation(16, 3))
    before_digest = adapter.digest()

    candidate, loss = adapter.copy_on_write_prefix_alignment(prefixes, full)

    assert math.isfinite(loss)
    assert adapter.digest() == before_digest
    assert candidate.digest() != before_digest
    assert candidate.version == adapter.version + 1
    assert candidate.parent_digest == before_digest
    restored = ExternalTransitionContextAddressAdapter.from_payload(
        candidate.state_payload()
    )
    assert restored.digest() == candidate.digest()
    torch.random.set_rng_state(rng_state)


def test_online_router_keeps_address_adapter_copy_isolated_until_promotion() -> None:
    torch.manual_seed(1204)
    bank = ExternalTransitionModelBank(
        2,
        1,
        5,
        model_family="random_feature_sufficient_statistics_v1",
        capacity=2,
    )
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=8,
        context_width=5,
        aggregation="mean_pool",
    )
    adapter = ExternalTransitionContextAddressAdapter(
        encoder,
        learning_rate=0.01,
        adaptation_steps=2,
    )
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=2,
        max_contexts=2,
        defer_admission=True,
        provisional_continuation_tolerance=1e9,
        provisional_evidence_policy="streaming_statistics",
        candidate_model_families=("random_feature_sufficient_statistics_v1",),
        address_adapter=adapter,
    )
    observations = ExternalTransitionObservation(
        state=torch.randn(2, 2),
        intention=torch.randn(2, 1),
        next_state=torch.randn(2, 2),
        confidence=torch.ones(2),
    )

    assert (
        router.observe(
            ExternalTransitionObservation(
                observations.state[:1],
                observations.intention[:1],
                observations.next_state[:1],
                observations.confidence[:1],
            )
        ).status
        == "pending"
    )
    staged = router.observe(
        ExternalTransitionObservation(
            observations.state[1:2],
            observations.intention[1:2],
            observations.next_state[1:2],
            observations.confidence[1:2],
        )
    )
    assert staged.status == "staged"
    assert router.address_adapter.digest() == adapter.digest()
    assert router._provisional_candidates[0].address_adapter is not None
    assert router._provisional_candidates[0].address_adapter.version == 2

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert restored.address_adapter is not None
    assert restored.address_adapter.digest() == router.address_adapter.digest()
    assert (
        restored._provisional_candidates[0].address_adapter is not None
        and restored._provisional_candidates[0].address_adapter.digest()
        == router._provisional_candidates[0].address_adapter.digest()
    )


def test_transition_context_prefix_alignment_supports_variable_evidence() -> None:
    torch.manual_seed(1200)
    prefixes = torch.randn(3, 4, 5)
    full = torch.randn(3, 5)
    loss = ExternalTransitionContextEncoder.prefix_alignment_loss(prefixes, full)
    assert torch.isfinite(loss)
    assert loss.ndim == 0

    with pytest.raises(ValueError, match="at least two regimes"):
        ExternalTransitionContextEncoder.prefix_alignment_loss(
            torch.randn(1, 2, 5), torch.randn(1, 5)
        )


def test_transition_model_learns_from_opaque_observations_without_controller_state() -> (
    None
):
    torch.manual_seed(1201)
    model = ExternalTransitionModel(3, 2, hidden_width=16)
    observation = ExternalTransitionObservation(
        state=torch.randn(8, 3),
        intention=torch.randn(8, 2),
        next_state=torch.randn(8, 3),
        confidence=torch.ones(8),
    )

    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    loss = model.loss(observation)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert model.configuration()["behavior"] == (
        "derived_by_external_search_not_stored_policy_v1"
    )
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    assert model(observation.state, observation.intention).shape == (8, 3)


def test_transition_model_payload_round_trip_preserves_predictions() -> None:
    torch.manual_seed(1202)
    model = ExternalTransitionModel(4, 3, hidden_width=12)
    state = torch.randn(5, 4)
    intention = torch.randn(5, 3)
    expected = model(state, intention)

    restored = ExternalTransitionModel.from_payload(model.state_payload())

    assert restored.configuration() == model.configuration()
    assert restored.digest() == model.digest()
    assert torch.equal(restored(state, intention), expected)


class _AdditiveTransitionModel(ExternalTransitionModel):
    """Deterministic model fixture for planner behavior tests."""

    def __init__(self) -> None:
        super().__init__(1, 1, hidden_width=4)

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(state, intention)
        return state + intention


def test_planner_derives_behavior_by_search_and_accepts_variable_candidates() -> None:
    model = _AdditiveTransitionModel()
    planner = ExternalModelBasedPlanner(model, beam_width=2)
    state = torch.zeros(1, 1)
    goal = torch.full((1, 1), 2.0)

    before = model.digest()
    result = planner.plan(
        state,
        goal,
        torch.tensor([[-1.0], [1.0]]),
        horizon=2,
    )

    assert result.intentions.shape == (1, 2, 1)
    assert torch.equal(result.intentions[0, :, 0], torch.tensor([1.0, 1.0]))
    assert torch.equal(result.predicted_states[0, :, 0], torch.tensor([1.0, 2.0]))
    assert result.scores.item() == 0.0
    assert result.expanded_nodes == 6
    assert model.digest() == before

    shorter = planner.plan(
        state,
        goal,
        torch.tensor([[0.5], [1.0], [2.0]]),
        horizon=1,
    )
    assert shorter.intentions.shape == (1, 1, 1)
    assert shorter.intentions[0, 0, 0].item() == 2.0


def test_planner_derives_signed_external_entry_value_into_behavior() -> None:
    transition_model = _AdditiveTransitionModel()
    entry_model = ExternalSignedEntryValueModel(1, 1, hidden_width=4)
    with torch.no_grad():
        for parameter in entry_model.state_network.parameters():
            parameter.zero_()
        entry_model.entry_projection.weight.fill_(1.0)
    planner = ExternalModelBasedPlanner(
        transition_model,
        beam_width=2,
        entry_value_model=entry_model,
    )
    state = torch.zeros(1, 1)
    goal = torch.zeros(1, 1)
    intentions = torch.tensor([[-1.0], [1.0]])
    entries = torch.tensor([[-1.0], [1.0]])
    transition_digest = transition_model.digest()
    entry_digest = entry_model.digest()

    positive = planner.plan(
        state,
        goal,
        intentions,
        candidate_entries=entries,
        entry_value_weight=1.0,
        horizon=1,
    )
    reversed_entries = planner.plan(
        state,
        goal,
        intentions,
        candidate_entries=-entries,
        entry_value_weight=1.0,
        horizon=1,
    )

    assert positive.intentions[0, 0, 0].item() == 1.0
    assert reversed_entries.intentions[0, 0, 0].item() == -1.0
    assert planner.configuration()["entry_value"] == (
        "external_opaque_entry_value_v1"
    )
    assert transition_model.digest() == transition_digest
    assert entry_model.digest() == entry_digest

    with pytest.raises(ValueError, match="candidate entries"):
        planner.plan(
            state,
            goal,
            intentions,
            entry_value_weight=1.0,
            horizon=1,
        )
    with pytest.raises(ValueError, match="external entry value model"):
        ExternalModelBasedPlanner(transition_model).plan(
            state,
            goal,
            intentions,
            candidate_entries=entries,
            horizon=1,
        )


def test_planner_can_opt_into_goal_progress_heuristic_for_long_horizons() -> None:
    planner = ExternalModelBasedPlanner(_AdditiveTransitionModel(), beam_width=4)
    state = torch.zeros(1, 1)
    goal = torch.full((1, 1), 10.0)
    candidates = torch.tensor([[-1.0], [0.0], [1.0]])

    terminal_only = planner.plan(
        state,
        goal,
        candidates,
        horizon=10,
    )
    heuristic = planner.plan(
        state,
        goal,
        candidates,
        horizon=10,
        goal_progress_weight=1.0,
    )

    # Terminal-only beam search prunes the useful prefix when every
    # intermediate score is tied. The opt-in opaque progress heuristic keeps
    # the goal-directed prefix without changing the planner's default.
    assert not torch.equal(
        terminal_only.intentions[0, :, 0], torch.ones(10)
    )
    assert torch.equal(heuristic.intentions[0, :, 0], torch.ones(10))
    assert heuristic.predicted_states[0, -1, 0].item() == 10.0


def test_planner_rollout_error_measures_recursive_heldout_behavior() -> None:
    planner = ExternalModelBasedPlanner(_AdditiveTransitionModel(), beam_width=1)
    rollout = ExternalTransitionRollout(
        initial_state=torch.tensor([0.0]),
        intentions=torch.tensor([[1.0], [1.0], [1.0]]),
        expected_states=torch.tensor([[1.0], [2.0], [3.0]]),
        confidence=torch.tensor([1.0, 2.0, 1.0]),
    )

    assert planner.rollout_error(rollout) == 0.0

    corrupted = ExternalTransitionRollout(
        initial_state=rollout.initial_state,
        intentions=rollout.intentions,
        expected_states=torch.tensor([[1.0], [2.0], [4.0]]),
    )
    assert planner.rollout_error(corrupted) == pytest.approx(1.0 / 3.0)


def test_planner_rollout_error_rejects_mismatched_horizon() -> None:
    planner = ExternalModelBasedPlanner(_AdditiveTransitionModel(), beam_width=1)
    with pytest.raises(ValueError, match="differ"):
        planner.rollout_error(
            ExternalTransitionRollout(
                initial_state=torch.zeros(1),
                intentions=torch.ones(2, 1),
                expected_states=torch.ones(1, 1),
            )
        )


def test_planner_can_trade_terminal_success_for_lower_opaque_step_cost() -> None:
    planner = ExternalModelBasedPlanner(_AdditiveTransitionModel(), beam_width=3)
    state = torch.zeros(1, 1)
    goal = torch.full((1, 1), 2.0)
    candidates = torch.tensor([[2.0], [1.0], [0.0]])

    terminal_only = planner.plan(
        state,
        goal,
        candidates,
        horizon=2,
        beam_width=3,
    )
    cost_aware = planner.plan(
        state,
        goal,
        candidates,
        horizon=2,
        beam_width=3,
        intention_costs=torch.tensor([5.0, 1.0, 4.0]),
        step_cost_weight=1.0,
    )

    assert torch.equal(
        terminal_only.intentions[0, :, 0], torch.tensor([2.0, 0.0])
    )
    assert torch.equal(cost_aware.intentions[0, :, 0], torch.tensor([1.0, 1.0]))
    assert terminal_only.scores.item() == 0.0
    assert cost_aware.scores.item() == 2.0


def test_planner_supports_per_batch_candidate_sets_without_resize() -> None:
    planner = ExternalModelBasedPlanner(_AdditiveTransitionModel(), beam_width=1)
    result = planner.plan(
        torch.zeros(2, 1),
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[[1.0], [0.0]], [[-1.0], [2.0]]]),
        horizon=1,
    )

    assert torch.equal(result.intentions[:, 0, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(result.predicted_states[:, 0, 0], torch.tensor([1.0, 2.0]))


def test_transition_model_bank_isolates_updates_and_round_trips() -> None:
    torch.manual_seed(1208)
    bank = ExternalTransitionModelBank(3, 2, 4, hidden_width=8)
    source_context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    target_context = torch.tensor([0.0, 1.0, 0.0, 0.0])
    source_index = bank.ensure_context(source_context)
    target_index = bank.ensure_context(
        target_context,
        initialize_from=source_index,
    )
    assert (source_index, target_index) == (0, 1)

    observation = ExternalTransitionObservation(
        state=torch.randn(4, 3),
        intention=torch.randn(4, 2),
        next_state=torch.randn(4, 3),
        confidence=torch.ones(4),
    )
    source_digest = bank.models[source_index].digest()
    optimizer = torch.optim.Adam(bank.models[target_index].parameters(), lr=0.01)
    bank.adaptation_step(
        observation,
        target_context.unsqueeze(0).expand(4, -1),
        optimizer,
    )
    assert bank.models[source_index].digest() == source_digest

    source_state = bank(
        observation.state,
        observation.intention,
        source_context.unsqueeze(0).expand(4, -1),
    )
    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    target_state = restored(
        observation.state,
        observation.intention,
        target_context.unsqueeze(0).expand(4, -1),
    )
    assert restored.context_count == 2
    assert torch.allclose(
        restored(
            observation.state,
            observation.intention,
            source_context.unsqueeze(0).expand(4, -1),
        ),
        source_state,
    )
    assert torch.isfinite(target_state).all()
    count_before_unknown = bank.context_count
    try:
        bank(
            observation.state[:1],
            observation.intention[:1],
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        )
    except KeyError as error:
        assert "ensure_context" in str(error)
    else:
        raise AssertionError("unknown model context must not mutate the bank")
    assert bank.context_count == count_before_unknown


def test_transition_model_bank_round_trip_preserves_learned_context_bytes() -> None:
    bank = ExternalTransitionModelBank(2, 2, 3, hidden_width=8)
    context = torch.nn.functional.normalize(
        torch.tensor([0.1234567, -0.7654321, 0.2345678]),
        dim=0,
    )
    bank.ensure_context(context)

    restored = ExternalTransitionModelBank.from_payload(bank.payload())

    assert restored.digest() == bank.digest()
    assert torch.equal(restored._contexts[0], bank._contexts[0])
    legacy_payload = bank.payload()
    legacy_payload.pop("model_aliases")
    legacy_restored = ExternalTransitionModelBank.from_payload(legacy_payload)
    assert legacy_restored.digest() == bank.digest()
    pre_address_payload = bank.payload()
    pre_address_payload["configuration"].pop("slot_addressing")
    pre_address_payload.pop("slot_ids")
    pre_address_payload.pop("next_slot_id")
    pre_address_payload["sha256"] = bank._legacy_digest()
    pre_address_restored = ExternalTransitionModelBank.from_payload(pre_address_payload)
    assert pre_address_restored.slot_ids == (0,)
    assert pre_address_restored.digest() == bank.digest()


def test_transition_model_bank_growth_is_verified_and_content_preserving() -> None:
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8, capacity=2)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 1.0, 0.0]))
    content_before = bank.content_digest()

    accepted = bank.grow_verified(3, lambda candidate: candidate.context_count == 2)

    assert accepted.accepted
    assert accepted.source_capacity == 2
    assert bank.capacity == 3
    assert bank.content_digest() == content_before
    bank.ensure_context(torch.tensor([0.0, 0.0, 1.0]))
    rejected = bank.grow_verified(4, lambda _candidate: False)
    assert not rejected.accepted
    assert bank.capacity == 3
    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    assert restored.capacity == 3
    assert restored.content_digest() == bank.content_digest()


def test_transition_model_bank_prior_challenger_is_copy_on_write() -> None:
    torch.manual_seed(1210)
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8)
    source_index = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    source_digest = bank.models[source_index].digest()
    observation = ExternalTransitionObservation(
        state=torch.randn(4, 2),
        intention=torch.randn(4, 1),
        next_state=torch.randn(4, 2),
        confidence=torch.ones(4),
    )

    def probe(
        transfer: torch.nn.Module,
        fresh: torch.nn.Module,
        current: ExternalTransitionObservation,
    ) -> tuple[float, float]:
        with torch.no_grad():
            next(iter(fresh.parameters())).add_(0.01)
        return (
            float(transfer.loss(current).detach()),
            float(fresh.loss(current).detach()) + 1.0,
        )

    receipt, selected = bank.select_verified_transfer_prior(
        source_index,
        observation,
        probe,
        probe_updates=4,
    )

    assert isinstance(receipt, ExternalTransitionModelPriorSelectionReceipt)
    assert receipt.selected_initialization == "transfer"
    assert receipt.source_slot_id == bank.slot_id_at(source_index)
    assert receipt.probe_updates == 4
    assert receipt.selected_model_digest == selected.digest()
    assert bank.context_count == 1
    assert bank.models[source_index].digest() == source_digest


def test_transition_model_bank_prior_challenger_accepts_matched_fresh_candidate() -> None:
    torch.manual_seed(1211)
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8)
    source_index = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    fresh = bank.new_model(bank.model_family_at(source_index))
    fresh_initial_digest = fresh.digest()
    observation = ExternalTransitionObservation(
        state=torch.randn(4, 2),
        intention=torch.randn(4, 1),
        next_state=torch.randn(4, 2),
        confidence=torch.ones(4),
    )

    def probe(
        transfer: torch.nn.Module,
        candidate: torch.nn.Module,
        current: ExternalTransitionObservation,
    ) -> tuple[float, float]:
        assert candidate is fresh
        assert candidate.digest() == fresh_initial_digest
        return (float(transfer.loss(current).detach()), 0.0)

    receipt, selected = bank.select_verified_transfer_prior(
        source_index,
        observation,
        probe,
        fresh_candidate=fresh,
    )

    assert receipt.selected_initialization == "fresh"
    assert selected is fresh
    assert fresh.digest() == selected.digest()
    assert bank.context_count == 1


def test_transition_model_bank_prior_challenger_can_include_acquisition_cost() -> None:
    torch.manual_seed(1212)
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8)
    source_index = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    observation = ExternalTransitionObservation(
        state=torch.randn(4, 2),
        intention=torch.randn(4, 1),
        next_state=torch.randn(4, 2),
        confidence=torch.ones(4),
    )

    def probe(
        _transfer: torch.nn.Module,
        _fresh: torch.nn.Module,
        _current: ExternalTransitionObservation,
    ) -> tuple[float, float]:
        return 0.10, 0.05

    receipt, _selected = bank.select_verified_transfer_prior(
        source_index,
        observation,
        probe,
        transfer_cost=0.0,
        fresh_cost=1.0,
        cost_weight=0.20,
    )

    assert receipt.selected_initialization == "transfer"
    assert receipt.schema.endswith("prior-selection.v2")
    assert receipt.transfer_adjusted_error == pytest.approx(0.10)
    assert receipt.fresh_adjusted_error == pytest.approx(0.25)
    assert receipt.transfer_cost == 0.0
    assert receipt.fresh_cost == 1.0
    assert receipt.cost_weight == 0.20


def test_transition_model_bank_eviction_is_verified_and_alias_safe() -> None:
    torch.manual_seed(1209)
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8, capacity=3)
    source = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    redundant = bank.ensure_context(
        torch.tensor([0.0, 1.0, 0.0]),
        initialize_from=source,
    )
    retained = bank.ensure_context(torch.tensor([0.0, 0.0, 1.0]))
    bank.models[retained] = bank.models[source]
    source_digest = bank.models[source].digest()
    redundant_context = bank.context_at(redundant)
    content_before = bank.content_digest()

    def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
        if candidate.context_count == 3:
            return candidate.models[0].digest() == source_digest
        return (
            candidate.context_count == 2
            and candidate.models[0].digest() == source_digest
            and torch.equal(candidate.context_at(1), redundant_context)
        )

    accepted = bank.evict_verified(
        retained,
        retention_probe,
    )

    assert accepted.accepted
    assert accepted.source_context_count == 3
    assert accepted.destination_context_count == 2
    assert accepted.physical_models_before == 2
    assert accepted.physical_models_after == 2
    assert bank.context_count == 2
    assert bank.models[0].digest() == source_digest
    assert bank.content_digest() != content_before
    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    assert restored.digest() == bank.digest()
    non_tail = bank.evict_verified(0, lambda _candidate: True)
    assert not non_tail.accepted
    assert "slot indices" in non_tail.reason
    rejected = bank.evict_verified(1, lambda _candidate: False)
    assert not rejected.accepted
    assert bank.context_count == 2
    assert bank.models[0].digest() == source_digest


def test_transition_model_bank_keeps_logical_addresses_across_middle_eviction() -> None:
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8, capacity=4)
    first = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    middle = bank.ensure_context(torch.tensor([0.0, 1.0, 0.0]))
    last = bank.ensure_context(torch.tensor([0.0, 0.0, 1.0]))
    assert (bank.slot_id_at(first), bank.slot_id_at(middle), bank.slot_id_at(last)) == (
        0,
        1,
        2,
    )

    retained_ids: list[tuple[int, ...]] = []

    def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
        retained_ids.append(candidate.slot_ids)
        return candidate.slot_ids in {(0, 1, 2), (0, 2)}

    receipt = bank.evict_verified_id(1, retention_probe)

    assert receipt.accepted
    assert receipt.evicted_slot_id == 1
    assert bank.slot_ids == (0, 2)
    assert bank.physical_index_for_slot_id(0) == 0
    assert bank.physical_index_for_slot_id(2) == 1
    assert bank.slot_id_at(1) == 2
    with pytest.raises(KeyError):
        bank.physical_index_for_slot_id(1)

    new_index = bank.ensure_context(torch.tensor([0.5, 0.5, 0.5]))
    assert bank.slot_id_at(new_index) == 3
    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    assert restored.slot_ids == bank.slot_ids
    assert restored.digest() == bank.digest()
    assert retained_ids == [(0, 1, 2), (0, 2)]


def test_online_router_repairs_active_physical_index_after_logical_eviction() -> None:
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8, capacity=3)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 1.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 0.0, 1.0]))
    router = ExternalOnlineTransitionContextRouter(
        bank,
        ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=3),
        max_contexts=3,
    )
    router._set_active_slot(2)

    receipt = router.evict_verified_id(
        1,
        lambda candidate: candidate.slot_ids in {(0, 1, 2), (0, 2)},
    )

    assert receipt.accepted
    assert router._active_slot_id == 2
    assert router._active_slot == 1
    assert router.bank.slot_id_at(router._active_slot) == 2


def test_online_router_repairs_reference_after_policy_selected_eviction() -> None:
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8, capacity=3)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 1.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 0.0, 1.0]))
    router = ExternalOnlineTransitionContextRouter(
        bank,
        ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=3),
        max_contexts=3,
    )
    router._set_active_slot(2)
    policy = ExternalTransitionModelLifetimePolicy(3, hidden_width=8)

    proposal, receipt = router.evict_with_lifetime_policy_verified(
        policy,
        torch.tensor([4.0, 1.0, 3.0]),
        torch.tensor([1.0, 4.0, 2.0]),
        torch.tensor([0.2, 0.4, 0.1]),
        torch.tensor([True, False, True]),
        lambda candidate: candidate.slot_ids in {(0, 1, 2), (0, 2)},
    )

    assert proposal.selected_slot_id == 1
    assert receipt is not None and receipt.accepted
    assert router._active_slot_id == 2
    assert router._active_slot == 1


def test_model_based_planner_selects_external_model_by_goal_reachability() -> None:
    bank = ExternalTransitionModelBank(
        1,
        1,
        2,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
    )
    contexts = (
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    )
    state = torch.tensor([[0.0], [0.2], [0.4], [0.1], [0.8], [0.3], [0.6], [0.9]])
    intention = torch.tensor([[0.1], [0.7], [0.2], [0.9], [0.4], [0.8], [0.3], [0.6]])
    for index, context in enumerate(contexts):
        slot = bank.ensure_context(context)
        target = state + intention + float(index * 5)
        observation = ExternalTransitionObservation(state, intention, target)
        context_batch = bank.context_at(slot).unsqueeze(0).expand(8, -1)
        bank.adaptation_step(observation, context_batch, None)
    planner = ExternalModelBasedPlanner(bank, beam_width=1)
    candidate_intentions = torch.ones(1, 1)
    first = planner.select_bank_model(
        bank,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        candidate_intentions,
        horizon=1,
    )
    second = planner.select_bank_model(
        bank,
        torch.zeros(1, 1),
        torch.full((1, 1), 6.0),
        candidate_intentions,
        horizon=1,
    )
    assert first.selected_slot_id == 0
    assert second.selected_slot_id == 1
    assert first.scores[0] < first.scores[1]
    assert second.scores[1] < second.scores[0]


def test_external_model_selection_requires_matching_representation_spaces() -> None:
    bank = ExternalTransitionModelBank(
        1,
        1,
        2,
        hidden_width=4,
        state_space_id="controller-state-v2",
        intention_space_id="controller-intention-v2",
    )
    bank.ensure_context(torch.tensor([1.0, 0.0]))
    planner = ExternalModelBasedPlanner(
        bank,
        beam_width=1,
        state_space_id="controller-state-v1",
        intention_space_id="controller-intention-v2",
    )

    with pytest.raises(ValueError, match="state representation space"):
        planner.select_bank_model(
            bank,
            torch.zeros(1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1),
            horizon=1,
        )


def test_external_model_bank_representation_spaces_round_trip() -> None:
    bank = ExternalTransitionModelBank(
        1,
        1,
        2,
        hidden_width=4,
        state_space_id="state-replacement-v3",
        intention_space_id="intention-replacement-v3",
    )
    bank.ensure_context(torch.tensor([1.0, 0.0]))

    restored = ExternalTransitionModelBank.from_payload(bank.payload())

    assert restored.state_space_id == "state-replacement-v3"
    assert restored.intention_space_id == "intention-replacement-v3"
    assert restored.digest() == bank.digest()
    assert restored.configuration()["representation_space_schema"] == (
        "neural-computer.external-representation-space.v1"
    )


def test_external_model_bank_verified_representation_migration_is_copy_on_write() -> (
    None
):
    source = ExternalTransitionModelBank(
        1,
        1,
        2,
        hidden_width=4,
        state_space_id="state-v1",
        intention_space_id="intention-v1",
    )
    source_index = source.ensure_context(torch.tensor([1.0, 0.0]))
    candidate = ExternalTransitionModelBank(
        1,
        1,
        2,
        hidden_width=4,
        state_space_id="state-v2",
        intention_space_id="intention-v2",
    )
    candidate_index = candidate.ensure_context(torch.tensor([1.0, 0.0]))
    candidate.models[candidate_index].load_state_dict(
        source.models[source_index].state_dict()
    )
    heldout = ExternalTransitionObservation(
        state=torch.tensor([[0.2], [0.8]]),
        intention=torch.tensor([[0.1], [-0.3]]),
        next_state=torch.zeros(2, 1),
    )
    source_digest = source.digest()
    receipt = source.migrate_representation_verified(
        candidate,
        [(source.slot_ids[source_index], heldout)],
        retention_probe=lambda bank: bank.context_count == 1,
    )

    assert receipt.accepted
    assert receipt.max_heldout_difference == 0.0
    assert source.digest() == source_digest
    assert source.state_space_id == "state-v1"
    assert candidate.state_space_id == "state-v2"

    next(candidate.models[candidate_index].parameters()).data.add_(1.0)
    rejected = source.migrate_representation_verified(
        candidate,
        [(source.slot_ids[source_index], heldout)],
        prediction_tolerance=1e-8,
    )
    assert not rejected.accepted
    assert rejected.reason == "held-out transition behavior changed"


def test_transition_model_bank_consolidation_shares_only_equivalent_models() -> None:
    torch.manual_seed(1212)
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8)
    first = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    second = bank.ensure_context(
        torch.tensor([0.0, 1.0, 0.0]),
        initialize_from=first,
    )
    heldout = ExternalTransitionObservation(
        state=torch.randn(5, 2),
        intention=torch.randn(5, 1),
        next_state=torch.randn(5, 2),
    )

    accepted = bank.consolidate_verified(first, second, [heldout])

    assert accepted.accepted
    assert bank.physical_model_count == 1
    assert bank.model_aliases() == [0, 0]
    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    assert restored.physical_model_count == 1
    assert restored.model_aliases() == [0, 0]

    rejected_bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8)
    rejected_first = rejected_bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    rejected_second = rejected_bank.ensure_context(
        torch.tensor([0.0, 1.0, 0.0]),
        initialize_from=rejected_first,
    )
    optimizer = torch.optim.Adam(
        rejected_bank.models[rejected_second].parameters(),
        lr=0.1,
    )
    rejected_observation = ExternalTransitionObservation(
        state=torch.zeros(5, 2),
        intention=torch.zeros(5, 1),
        next_state=torch.ones(5, 2),
    )
    rejected_bank.adaptation_step(
        rejected_observation,
        torch.tensor([0.0, 1.0, 0.0]).expand(5, -1),
        optimizer,
    )
    rejected = rejected_bank.consolidate_verified(
        rejected_first,
        rejected_second,
        [heldout],
        prediction_tolerance=1e-8,
    )

    assert not rejected.accepted
    assert rejected_bank.physical_model_count == 2


def test_transition_model_consolidation_is_copy_on_write_after_later_adaptation() -> None:
    torch.manual_seed(12125)
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8)
    first = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    second = bank.ensure_context(
        torch.tensor([0.0, 1.0, 0.0]),
        initialize_from=first,
    )
    heldout = ExternalTransitionObservation(
        state=torch.randn(5, 2),
        intention=torch.randn(5, 1),
        next_state=torch.randn(5, 2),
    )
    assert bank.consolidate_verified(first, second, [heldout]).accepted
    first_digest = bank.models[first].digest()
    optimizer = torch.optim.Adam(bank.models[second].parameters(), lr=0.1)
    update = ExternalTransitionObservation(
        state=torch.randn(5, 2),
        intention=torch.randn(5, 1),
        next_state=torch.ones(5, 2),
    )

    bank.adaptation_step(
        update,
        bank.context_at(second).unsqueeze(0).expand(5, -1),
        optimizer,
    )

    assert bank.model_aliases() == [0, 1]
    assert bank.models[first].digest() == first_digest
    assert bank.models[second].digest() != first_digest


def test_transition_model_bank_compression_requires_retention_and_round_trips() -> None:
    torch.manual_seed(1213)
    bank = ExternalTransitionModelBank(2, 1, 2, hidden_width=8, capacity=2)
    first = bank.ensure_context(torch.tensor([1.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 1.0]), initialize_from=first)
    observation = ExternalTransitionObservation(
        state=torch.randn(5, 2),
        intention=torch.randn(5, 1),
        next_state=torch.randn(5, 2),
    )

    accepted = bank.compress_verified(
        dtype=torch.float16,
        retention_probe=lambda candidate: torch.isfinite(
            candidate.loss(
                observation,
                torch.tensor([1.0, 0.0]).expand(5, -1),
            )
        ),
    )
    rejected = bank.compress_verified(
        dtype="int4",
        retention_probe=lambda _candidate: False,
    )

    assert accepted.accepted
    assert accepted.compressed_bytes < accepted.source_bytes
    assert not rejected.accepted
    restored = ExternalTransitionModelBank.from_compressed_payload(
        bank.compressed_payload(dtype=torch.float16)
    )
    assert restored.model_aliases() == bank.model_aliases()
    assert restored.context_count == bank.context_count


def test_random_feature_stats_compression_quantizes_solved_predictor() -> None:
    torch.manual_seed(1214)
    bank = ExternalTransitionModelBank(
        2,
        1,
        2,
        model_family="random_feature_sufficient_statistics_v1",
        random_feature_width=16,
        capacity=1,
    )
    context = torch.tensor([1.0, 0.0])
    index = bank.ensure_context(context)
    observation = ExternalTransitionObservation(
        state=torch.randn(8, 2),
        intention=torch.randn(8, 1),
        next_state=torch.randn(8, 2),
    )
    bank.adaptation_step(
        observation,
        context.expand(observation.state.shape[0], -1),
        None,
    )
    baseline = float(
        bank.loss(
            observation,
            context.expand(observation.state.shape[0], -1),
        )
    )

    payload = bank.compressed_payload(dtype="float16_stats")
    restored = ExternalTransitionModelBank.from_compressed_payload(payload)
    restored_loss = float(
        restored.loss(
            observation,
            context.expand(observation.state.shape[0], -1),
        )
    )

    assert index == 0
    assert payload["codec"] == "float16_stats"
    assert restored_loss <= baseline + 1e-3
    assert restored.context_count == bank.context_count
    assert restored.slot_ids == bank.slot_ids
    assert payload["models"][0]["state"]["target_matrix"].dtype == torch.float16


def test_transition_model_bank_selects_smallest_retained_codec() -> None:
    bank = ExternalTransitionModelBank(2, 1, 2, hidden_width=8)
    first = bank.ensure_context(torch.tensor([1.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 1.0]), initialize_from=first)

    selection = bank.select_compression_verified(
        (torch.float16, torch.int8, "int4"),
        retention_probe=lambda candidate: candidate.physical_model_count == 2,
    )

    assert selection.accepted
    assert len(selection.receipts) == 3
    accepted = [receipt for receipt in selection.receipts if receipt.accepted]
    smallest = min(accepted, key=lambda receipt: receipt.compressed_bytes)
    assert selection.selected_codec == smallest.codec


def test_online_transition_context_router_admits_current_bundle_and_persists() -> None:
    torch.manual_seed(1210)
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8)
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=8,
        context_width=4,
    )
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=2,
        conflict_patience=2,
    )
    rows = [
        ExternalTransitionObservation(
            state=torch.tensor([[0.1, 0.2]]),
            intention=torch.tensor([[0.3]]),
            next_state=torch.tensor([[0.7, -0.4]]),
            confidence=torch.ones(1),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[0.2, 0.1]]),
            intention=torch.tensor([[-0.3]]),
            next_state=torch.tensor([[0.4, -0.6]]),
            confidence=torch.ones(1),
        ),
    ]

    first = router.observe(rows[0])
    admitted = router.observe(rows[1])

    assert first.status == "pending"
    assert first.pending_observations == 1
    assert admitted.status == "admitted"
    assert admitted.slot_index == 1
    assert admitted.observation is not None
    assert admitted.observation.state.shape == (2, 2)
    optimizer = torch.optim.Adam(
        router.bank.models[admitted.slot_index].parameters(),
        lr=0.01,
    )
    assert router.adaptation_step(admitted, optimizer) > 0.0

    router.observe(rows[0])
    continuation = router.observe(rows[1])
    assert continuation.status == "continuation"
    assert continuation.slot_index == admitted.slot_index
    assert continuation.observation is not None

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert restored.configuration() == router.configuration()
    assert restored.bank.digest() == router.bank.digest()
    assert restored.context_encoder.digest() == router.context_encoder.digest()
    assert restored._active_slot == router._active_slot
    assert restored._conflict_windows == router._conflict_windows


def test_online_transition_context_router_capacity_guard_does_not_grow_or_write() -> (
    None
):
    torch.manual_seed(1211)
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=2,
        max_contexts=1,
    )
    row = ExternalTransitionObservation(
        state=torch.tensor([[0.1, 0.2]]),
        intention=torch.tensor([[0.3]]),
        next_state=torch.tensor([[0.7, -0.4]]),
    )

    router.observe(row)
    capacity = router.observe(row)

    assert capacity.status == "capacity"
    assert capacity.pending_observations == 0
    assert router.bank.context_count == 1


def test_online_router_stages_candidate_before_heldout_verified_promotion() -> None:
    torch.manual_seed(1212)
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8, capacity=2)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    source_index = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    source_digest = bank.models[source_index].digest()
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=2,
        max_contexts=2,
        defer_admission=True,
    )
    rows = [
        ExternalTransitionObservation(
            state=torch.tensor([[0.1, 0.2]]),
            intention=torch.tensor([[0.3]]),
            next_state=torch.tensor([[0.7, -0.4]]),
            confidence=torch.ones(1),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[0.2, 0.1]]),
            intention=torch.tensor([[-0.3]]),
            next_state=torch.tensor([[0.4, -0.6]]),
            confidence=torch.ones(1),
        ),
    ]

    router.observe(rows[0])
    staged = router.observe(rows[1])
    assert staged.status == "staged"
    assert router.bank.context_count == 1
    assert router.provisional_model is not None
    optimizer = torch.optim.Adam(router.provisional_model.parameters(), lr=0.03)
    for _ in range(80):
        router.adaptation_step(staged, optimizer)

    receipt = router.promote_staged_candidate(
        ExternalTransitionObservation(
            state=torch.cat([row.state for row in rows]),
            intention=torch.cat([row.intention for row in rows]),
            next_state=torch.cat([row.next_state for row in rows]),
            confidence=torch.ones(2),
        ),
        lambda candidate: (
            candidate.context_count == 2
            and candidate.models[0].digest() == source_digest
        ),
        prediction_tolerance=0.05,
    )
    assert receipt.accepted
    assert receipt.slot_index == 1
    assert router.bank.context_count == 2
    assert router.provisional_model is None
    assert router.bank.models[0].digest() == source_digest


def test_online_router_exposes_best_provisional_portfolio_hypothesis() -> None:
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        hidden_width=8,
        capacity=2,
        model_family=EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY,
    )
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        candidate_model_families=(
            EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        ),
        defer_admission=True,
    )
    observation = ExternalTransitionObservation(
        state=torch.tensor([[0.2, -0.4]]),
        intention=torch.tensor([[0.7]]),
        next_state=torch.zeros(1, 2),
    )

    class ConstantModel(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
            del intention
            return torch.full_like(state, self.value)

    router._stage_candidate(
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        observation=observation,
        prior_index=None,
    )
    candidate = router._provisional_candidates[0]
    candidate.model = ConstantModel(1.0)
    candidate.alternatives = {
        EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY: ConstantModel(0.0),
    }

    staged = router._staged_result(observation, candidate_index=0)

    assert staged.prediction_error == pytest.approx(0.0)
    assert (
        router.provisional_model_family_at(0)
        == EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY
    )
    assert router.provisional_model_at(0) is candidate.model
    assert set(candidate.models()) == {
        EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    }


def test_online_router_persists_provisional_evidence_window() -> None:
    torch.manual_seed(1213)
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8, capacity=2)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=2,
        max_contexts=2,
        defer_admission=True,
    )
    rows = [
        ExternalTransitionObservation(
            state=torch.tensor([[0.1, 0.2]]),
            intention=torch.tensor([[0.3]]),
            next_state=torch.tensor([[0.7, -0.4]]),
            confidence=torch.ones(1),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[0.2, 0.1]]),
            intention=torch.tensor([[-0.3]]),
            next_state=torch.tensor([[0.4, -0.6]]),
            confidence=torch.ones(1),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[-0.2, 0.4]]),
            intention=torch.tensor([[0.1]]),
            next_state=torch.tensor([[0.0, 0.6]]),
            confidence=torch.ones(1),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[0.5, -0.1]]),
            intention=torch.tensor([[-0.2]]),
            next_state=torch.tensor([[0.3, -0.5]]),
            confidence=torch.ones(1),
        ),
    ]

    router.observe(rows[0])
    first_staged = router.observe(rows[1])
    assert first_staged.status == "staged"
    router.observe(rows[2])
    second_staged = router.observe(rows[3])
    assert second_staged.status == "staged"
    payload = router.state_payload()
    restored = ExternalOnlineTransitionContextRouter.from_payload(payload)

    assert len(payload["provisional_observations"]) == 2
    assert len(restored._provisional_observations) == 2
    assert restored.provisional_model is not None
    assert restored._provisional_context is not None
    assert restored.state_payload()["bank"]["sha256"] == bank.digest()


def test_online_router_promotes_candidate_with_verified_capacity_growth() -> None:
    torch.manual_seed(1215)
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8, capacity=1)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    source_index = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    source_digest = bank.models[source_index].digest()
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=2,
        max_contexts=2,
        defer_admission=True,
    )
    rows = [
        ExternalTransitionObservation(
            state=torch.tensor([[0.1, 0.2]]),
            intention=torch.tensor([[0.3]]),
            next_state=torch.tensor([[0.7, -0.4]]),
            confidence=torch.ones(1),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[0.2, 0.1]]),
            intention=torch.tensor([[-0.3]]),
            next_state=torch.tensor([[0.4, -0.6]]),
            confidence=torch.ones(1),
        ),
    ]
    router.observe(rows[0])
    staged = router.observe(rows[1])
    optimizer = torch.optim.Adam(router.provisional_model.parameters(), lr=0.03)
    for _ in range(80):
        router.adaptation_step(staged, optimizer)

    receipt = router.promote_staged_candidate(
        ExternalTransitionObservation(
            state=torch.cat([row.state for row in rows]),
            intention=torch.cat([row.intention for row in rows]),
            next_state=torch.cat([row.next_state for row in rows]),
            confidence=torch.ones(2),
        ),
        lambda candidate: (
            candidate.context_count == 2
            and candidate.models[0].digest() == source_digest
        ),
        prediction_tolerance=0.05,
        destination_capacity=2,
    )

    assert receipt.accepted
    assert router.bank.capacity == 2
    assert router.max_contexts == 2
    assert router.bank.context_count == 2
    assert router.bank.models[0].digest() == source_digest


def test_online_router_recycles_verified_tail_capacity_for_next_candidate() -> None:
    torch.manual_seed(1216)
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8, capacity=1)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    source_index = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    source_digest = bank.models[source_index].digest()
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=2,
        max_contexts=2,
        defer_admission=True,
    )
    rows = [
        ExternalTransitionObservation(
            state=torch.tensor([[0.1, 0.2]]),
            intention=torch.tensor([[0.3]]),
            next_state=torch.tensor([[0.7, -0.4]]),
            confidence=torch.ones(1),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[0.2, 0.1]]),
            intention=torch.tensor([[-0.3]]),
            next_state=torch.tensor([[0.4, -0.6]]),
            confidence=torch.ones(1),
        ),
    ]

    router.observe(rows[0])
    staged_first = router.observe(rows[1])
    first_optimizer = torch.optim.Adam(router.provisional_model.parameters(), lr=0.03)
    for _ in range(80):
        router.adaptation_step(staged_first, first_optimizer)
    first_receipt = router.promote_staged_candidate(
        ExternalTransitionObservation(
            state=torch.cat([row.state for row in rows]),
            intention=torch.cat([row.intention for row in rows]),
            next_state=torch.cat([row.next_state for row in rows]),
            confidence=torch.ones(2),
        ),
        lambda candidate: (
            candidate.context_count == 2
            and candidate.models[0].digest() == source_digest
        ),
        prediction_tolerance=0.05,
        destination_capacity=2,
    )
    assert first_receipt.accepted
    assert router._active_slot == 1

    evicted = router.evict_verified(
        1,
        lambda candidate: candidate.models[0].digest() == source_digest,
    )
    assert evicted.accepted
    assert router.bank.context_count == 1
    assert router._active_slot is None

    router.observe(rows[0])
    staged_second = router.observe(rows[1])
    second_optimizer = torch.optim.Adam(router.provisional_model.parameters(), lr=0.03)
    for _ in range(80):
        router.adaptation_step(staged_second, second_optimizer)
    second_receipt = router.promote_staged_candidate(
        ExternalTransitionObservation(
            state=torch.cat([row.state for row in rows]),
            intention=torch.cat([row.intention for row in rows]),
            next_state=torch.cat([row.next_state for row in rows]),
            confidence=torch.ones(2),
        ),
        lambda candidate: (
            candidate.context_count == 2
            and candidate.models[0].digest() == source_digest
        ),
        prediction_tolerance=0.05,
    )
    assert second_receipt.accepted
    assert router.bank.context_count == 2
    assert router.bank.capacity == 2
    assert router.bank.models[0].digest() == source_digest


def test_online_router_isolates_alternating_provisional_candidates() -> None:
    torch.manual_seed(1214)
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8, capacity=3)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    source_index = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    source_digest = bank.models[source_index].digest()
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e-8,
        admission_observations=2,
        max_contexts=3,
        defer_admission=True,
    )
    stream_a = [
        ExternalTransitionObservation(
            state=torch.tensor([[0.1, 0.2]]),
            intention=torch.tensor([[0.3]]),
            next_state=torch.tensor([[0.7, -0.4]]),
            confidence=torch.ones(1),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[0.2, 0.1]]),
            intention=torch.tensor([[-0.3]]),
            next_state=torch.tensor([[0.4, -0.6]]),
            confidence=torch.ones(1),
        ),
    ]
    stream_b = [
        ExternalTransitionObservation(
            state=torch.tensor([[0.1, 0.2]]),
            intention=torch.tensor([[0.3]]),
            next_state=torch.tensor([[10.7, -10.4]]),
            confidence=torch.ones(1),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[0.2, 0.1]]),
            intention=torch.tensor([[-0.3]]),
            next_state=torch.tensor([[10.4, -10.6]]),
            confidence=torch.ones(1),
        ),
    ]

    router.observe(stream_a[0])
    staged_a = router.observe(stream_a[1])
    assert staged_a.status == "staged"
    candidate_a_digest = router._provisional_candidates[0].model.digest()
    router.observe(stream_b[0])
    staged_b = router.observe(stream_b[1])
    assert staged_b.status == "staged"
    assert staged_b.slot_index == 1
    assert router.provisional_candidate_count == 2
    assert router._provisional_candidates[0].model.digest() == candidate_a_digest

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert restored.provisional_candidate_count == 2
    assert [
        len(candidate.observations) for candidate in restored._provisional_candidates
    ] == [1, 1]

    optimizer = torch.optim.Adam(
        router._provisional_candidates[1].model.parameters(), lr=0.03
    )
    for _ in range(120):
        router.adaptation_step(staged_b, optimizer)
    receipt = router.promote_staged_candidate(
        ExternalTransitionObservation(
            state=torch.cat([row.state for row in stream_b]),
            intention=torch.cat([row.intention for row in stream_b]),
            next_state=torch.cat([row.next_state for row in stream_b]),
            confidence=torch.ones(2),
        ),
        lambda candidate: (
            candidate.context_count == 2
            and candidate.models[0].digest() == source_digest
        ),
        prediction_tolerance=0.05,
        candidate_index=1,
    )
    assert receipt.accepted
    assert router.bank.context_count == 2
    assert router.provisional_candidate_count == 1
    assert router._provisional_candidates[0].model.digest() == candidate_a_digest


def test_online_transition_context_router_growth_updates_capacity_atomically() -> None:
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8, capacity=1)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=2,
        max_contexts=1,
    )

    receipt = router.grow_verified(2, lambda candidate: candidate.context_count == 1)

    assert receipt.accepted
    assert router.max_contexts == 2
    assert router.bank.capacity == 2


def test_transition_observation_rejects_mismatched_batch_and_nonfinite_values() -> None:
    observation = ExternalTransitionObservation(
        state=torch.zeros(2, 3),
        intention=torch.zeros(1, 2),
        next_state=torch.zeros(2, 3),
    )
    try:
        observation.validate(state_width=3, intention_width=2)
    except ValueError as error:
        assert "batch" in str(error)
    else:
        raise AssertionError("expected transition batch validation")

    bad = ExternalTransitionObservation(
        state=torch.full((1, 3), float("nan")),
        intention=torch.zeros(1, 2),
        next_state=torch.zeros(1, 3),
    )
    try:
        bad.validate(state_width=3, intention_width=2)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("expected transition finiteness validation")


def test_append_only_transition_memory_retains_disjoint_contextual_dynamics() -> None:
    memory = ExternalTransitionMemory(1, 1, context_width=1)
    state = torch.tensor([[0.0], [1.0]])
    intention = torch.tensor([[1.0], [1.0]])
    source = ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=torch.tensor([[1.0], [2.0]]),
    )
    target = ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=torch.tensor([[-1.0], [0.0]]),
    )

    memory.write(source, context=torch.ones(2, 1))
    source_before, source_hits = memory.predict_with_hit(
        state, intention, context=torch.ones(2, 1)
    )
    memory.write(target, context=-torch.ones(2, 1))
    source_after, source_hits_after = memory.predict_with_hit(
        state, intention, context=torch.ones(2, 1)
    )
    target_after, target_hits = memory.predict_with_hit(
        state, intention, context=-torch.ones(2, 1)
    )

    assert memory.record_count == 4
    assert source_hits.all() and source_hits_after.all() and target_hits.all()
    assert torch.equal(source_before, source_after)
    assert torch.equal(source_after, source.next_state)
    assert torch.equal(target_after, target.next_state)


def test_factored_transition_freezes_base_and_persists_context_residuals() -> None:
    model = ExternalFactoredTransitionModel(
        1,
        1,
        1,
        hidden_width=8,
    )
    source = ExternalTransitionObservation(
        state=torch.tensor([[0.0], [1.0]]),
        intention=torch.ones(2, 1),
        next_state=torch.tensor([[1.0], [2.0]]),
    )
    target = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=torch.tensor([[-1.0], [0.0]]),
    )
    model.freeze_base()
    base_before = model.base.digest()
    model.write_residual(source, context=torch.ones(1))
    model.write_residual(target, context=-torch.ones(1))

    source_prediction = model.predict_with_context(
        source.state,
        source.intention,
        torch.ones(2, 1),
    )
    target_prediction = model.predict_with_context(
        target.state,
        target.intention,
        -torch.ones(2, 1),
    )
    assert torch.allclose(source_prediction, source.next_state)
    assert torch.allclose(target_prediction, target.next_state)
    assert model.base.digest() == base_before
    assert model.base_frozen
    assert model.residual_record_count == 4

    restored = ExternalFactoredTransitionModel.from_payload(model.state_payload())
    assert restored.digest() == model.digest()
    assert torch.allclose(
        restored.predict_with_context(
            target.state,
            target.intention,
            -torch.ones(2, 1),
        ),
        target.next_state,
    )


def test_factored_transition_persists_a_replaceable_affine_base() -> None:
    base = ExternalAffineTransitionStatistics(1, 1, ridge=0.01)
    base.observe(
        ExternalTransitionObservation(
            state=torch.tensor([[0.0], [1.0]]),
            intention=torch.ones(2, 1),
            next_state=torch.tensor([[1.0], [2.0]]),
        )
    )
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        residual_mode="exact_residual_memory_v1",
        base_model=base,
    )
    model.freeze_base()
    probe_state = torch.tensor([[0.25], [0.75]])
    probe_intention = torch.ones(2, 1)
    before = model(probe_state, probe_intention)

    payload = model.state_payload()
    assert payload["base_payload"]["schema"] == base.schema
    restored = ExternalFactoredTransitionModel.from_payload(payload)

    assert isinstance(restored.base, ExternalAffineTransitionStatistics)
    assert restored.configuration()["base_model_schema"] == base.schema
    assert restored.base.digest() == base.digest()
    assert restored.digest() == model.digest()
    assert torch.allclose(restored(probe_state, probe_intention), before)


def test_learned_factored_residual_generalizes_while_base_stays_frozen() -> None:
    torch.manual_seed(1921)
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        hidden_width=16,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_hidden_width=32,
        residual_learning_rate=0.01,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    base_before = model.base.digest()
    context = torch.tensor([1.0, 0.0])
    train_state = torch.linspace(-1.0, 1.0, 8).unsqueeze(-1)
    training = ExternalTransitionObservation(
        state=train_state,
        intention=torch.ones(8, 1),
        next_state=torch.sin(train_state * 2.0) + train_state,
    )
    heldout_state = torch.tensor([[-0.85], [-0.35], [0.15], [0.65]])
    heldout = ExternalTransitionObservation(
        state=heldout_state,
        intention=torch.ones(4, 1),
        next_state=torch.sin(heldout_state * 2.0) + heldout_state,
    )
    loss, updates = model.fit_residual(
        training,
        context=context,
        updates=1_000,
    )

    error = float(
        model.loss(
            heldout,
            context=context.unsqueeze(0).expand(heldout.state.shape[0], -1),
        ).detach()
    )
    assert updates == 1_000
    assert loss < 1e-5
    assert error < 1e-3
    assert model.base.digest() == base_before
    assert model.base_frozen
    assert model.residual_context_count == 1

    restored = ExternalFactoredTransitionModel.from_payload(model.state_payload())
    assert restored.digest() == model.digest()
    assert torch.allclose(
        restored.predict_with_context(
            heldout.state,
            heldout.intention,
            context.unsqueeze(0).expand(heldout.state.shape[0], -1),
        ),
        model.predict_with_context(
            heldout.state,
            heldout.intention,
            context.unsqueeze(0).expand(heldout.state.shape[0], -1),
        ),
    )


def test_factored_random_feature_residual_is_replay_free_and_persistent() -> None:
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        hidden_width=8,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_random_feature_width=32,
        residual_random_feature_seed=17,
        residual_ridge=0.01,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    base_before = model.base.digest()
    context = torch.tensor([1.0, 0.0])
    first = ExternalTransitionObservation(
        state=torch.tensor([[-1.0], [0.0]]),
        intention=torch.ones(2, 1),
        next_state=torch.tensor([[-0.5], [0.5]]),
    )
    second = ExternalTransitionObservation(
        state=torch.tensor([[0.5], [1.0]]),
        intention=torch.ones(2, 1),
        next_state=torch.tensor([[1.0], [1.5]]),
    )
    assert model.fit_residual(first, context=context, updates=100)[1] == 1
    assert model.fit_residual(second, context=context, updates=100)[1] == 1
    assert model.residual_bank is not None
    assert model.residual_bank.models[0].sample_count.item() == 4
    assert model.base.digest() == base_before

    restored = ExternalFactoredTransitionModel.from_payload(model.state_payload())
    assert restored.digest() == model.digest()
    assert restored.residual_bank is not None
    assert restored.residual_bank.models[0].sample_count.item() == 4

    reparameterized = model.reparameterized_residual_ridge(context, 0.1)
    assert reparameterized.base.digest() == base_before
    assert reparameterized.residual_bank is not None
    assert reparameterized.residual_bank.models[0].ridge == 0.1
    restored_reparameterized = ExternalFactoredTransitionModel.from_payload(
        reparameterized.state_payload()
    )
    assert restored_reparameterized.digest() == reparameterized.digest()


def test_factored_router_owns_verified_growth_compression_and_stable_eviction() -> None:
    torch.manual_seed(777)
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        hidden_width=8,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_random_feature_width=32,
        residual_random_feature_seed=777,
        residual_ridge=0.01,
        residual_capacity=1,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    router = ExternalFactoredTransitionRouter(
        model,
        ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2),
        max_contexts=1,
        match_tolerance=0.005,
        match_margin=0.0,
        quarantine_capacity=4,
    )

    def observation(offset: float, values: torch.Tensor) -> ExternalTransitionObservation:
        return ExternalTransitionObservation(
            state=values,
            intention=torch.ones_like(values),
            next_state=0.2 * values + offset,
        )

    def rows(item: ExternalTransitionObservation) -> list[ExternalTransitionObservation]:
        return [
            ExternalTransitionObservation(
                state=item.state[index : index + 1],
                intention=item.intention[index : index + 1],
                next_state=item.next_state[index : index + 1],
            )
            for index in range(item.state.shape[0])
        ]

    source = observation(0.0, torch.tensor([[-1.0], [0.0], [1.0]]))
    source_heldout = observation(0.0, torch.tensor([[-0.5], [0.5]]))
    target = observation(1.0, torch.tensor([[-1.0], [0.0], [1.0]]))
    target_heldout = observation(1.0, torch.tensor([[-0.5], [0.5]]))
    assert router.route_bundle(rows(source)).status == "staged"
    assert router.promote_staged_candidate(source_heldout, lambda _candidate: True).accepted

    grown = router.grow_verified(2, lambda candidate: candidate.context_count == 1)
    assert grown.accepted
    assert router.max_contexts == 2
    assert router.model.residual_capacity == 2
    assert router.route_bundle(rows(target)).status == "staged"
    assert router.promote_staged_candidate(target_heldout, lambda _candidate: True).accepted

    partial_digest = router.digest()
    partial = router.route_partial_bundle(
        rows(source)[:1],
        match_tolerance=0.1,
        contradiction_tolerance=0.1,
        match_margin=0.0,
    )
    contradictory = router.route_partial_bundle(
        rows(source)[:1] + rows(target)[:1],
        min_match_fraction=0.5,
        match_tolerance=0.1,
        contradiction_tolerance=0.1,
        match_margin=0.0,
    )
    empty = router.route_partial_bundle([])
    assert partial.status == "matched" and partial.slot_id == 0
    assert contradictory.status == "ambiguous"
    assert empty.status == "ambiguous"
    assert router.digest() == partial_digest
    quarantined = router.quarantine_partial_bundle(rows(source)[:1] + rows(target)[:1])
    assert quarantined.accepted
    assert router.quarantined_observations == 2
    assert router.quarantined_bundles == 2
    restored_quarantine = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload()
    )
    assert restored_quarantine.quarantined_observations == 2
    assert restored_quarantine.quarantined_bundles == 2
    assert len(router.peek_quarantine()) == 2
    assert router.resolve_quarantine(
        match_tolerance=0.1,
        contradiction_tolerance=0.1,
        match_margin=0.0,
    ) == (0, 1)
    assert router.quarantined_observations == 0
    corrupted = ExternalTransitionObservation(
        state=source.state[:1],
        intention=source.intention[:1],
        next_state=source.next_state[:1] + 5.0,
    )
    assert router.quarantine_partial_bundle((corrupted,)).accepted
    assert router.resolve_quarantine(
        match_tolerance=0.1,
        contradiction_tolerance=0.1,
        match_margin=0.0,
    ) == ()
    drained = router.drain_quarantine()
    assert len(drained) == 1
    assert all(item.state.shape[0] == 1 for item in drained)
    assert router.quarantined_observations == 0

    compressed = router.select_compression_verified(
        ["float16_stats"],
        retention_probe=lambda candidate: candidate.context_count == 2,
    )
    assert compressed.accepted
    contextual_statistics = ExternalContextualTransitionEvidenceStatistics(1, 2)
    for context in router.contexts:
        contextual_statistics.observe(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.ones(1),
            context,
        )
    router.evidence_evaluator = contextual_statistics
    evicted = router.evict_verified_id(
        0,
        lambda candidate: candidate.slot_ids in {(0, 1), (1,)},
    )
    assert evicted.accepted
    assert router.slot_ids == (1,)
    assert contextual_statistics.context_count == 1


def test_factored_router_persists_optional_verified_address_proposal() -> None:
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(778)
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        hidden_width=8,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_random_feature_width=16,
        residual_capacity=1,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    adapter = ExternalTransitionContextAddressAdapter(
        encoder,
        learning_rate=0.01,
        adaptation_steps=1,
    )
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        max_contexts=1,
        match_tolerance=0.05,
        match_margin=0.01,
        address_adapter=adapter,
        route_query=ExternalTransitionRouteQuery(2),
    )
    source = ExternalTransitionObservation(
        state=torch.tensor([[-1.0], [1.0]]),
        intention=torch.ones(2, 1),
        next_state=torch.tensor([[-0.5], [0.5]]),
    )

    assert router.route_bundle((source,)).status == "staged"
    assert router.promote_staged_candidate(source, lambda _candidate: True).accepted
    assert router.route_query is not None
    assert router.route_query._slot_route_keys.keys() == {0}

    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    assert restored.digest() == router.digest()
    assert restored.address_adapter is not None
    assert restored.route_query is not None
    assert restored.route_query._slot_route_keys.keys() == {0}
    torch.random.set_rng_state(rng_state)


def test_factored_router_prefix_address_update_isolated_from_factual_memory() -> None:
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(779)
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        hidden_width=8,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_random_feature_width=16,
        residual_capacity=2,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    adapter = ExternalTransitionContextAddressAdapter(encoder, adaptation_steps=1)
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        max_contexts=2,
        match_tolerance=0.05,
        match_margin=0.01,
        address_adapter=adapter,
        route_query=ExternalTransitionRouteQuery(2),
    )
    source = ExternalTransitionObservation(
        state=torch.tensor([[-1.0], [1.0]]),
        intention=torch.ones(2, 1),
        next_state=torch.tensor([[-0.5], [0.5]]),
    )
    target = ExternalTransitionObservation(
        state=torch.tensor([[-1.0], [1.0]]),
        intention=torch.ones(2, 1),
        next_state=torch.tensor([[0.5], [1.5]]),
    )
    assert router.route_bundle((source,)).status == "staged"
    assert router.promote_staged_candidate(source, lambda _candidate: True).accepted
    assert router.route_bundle((target,)).status == "staged"
    assert router.promote_staged_candidate(target, lambda _candidate: True).accepted

    probe = router.request_disambiguation_probe(
        source,
        torch.tensor([[0.0], [1.0]]),
        candidate_slot_ids=(0, 1),
    )
    assert probe.candidate_slot_ids == (0, 1)
    assert probe.predicted_next_states.shape == (2, 2, 1)
    assert probe.selected_intention_index in {0, 1}

    before_digest = router.digest()
    candidate, loss = router.copy_on_write_prefix_address_update(
        {
            0: (source,),
            1: (target,),
        },
        {
            0: source,
            1: target,
        },
    )

    assert math.isfinite(loss)
    assert router.digest() == before_digest
    assert candidate.digest() != before_digest
    assert candidate.model.digest() == router.model.digest()
    assert candidate.slot_ids == router.slot_ids == (0, 1)
    restored = ExternalFactoredTransitionRouter.from_payload(
        candidate.state_payload()
    )
    assert restored.digest() == candidate.digest()
    torch.random.set_rng_state(rng_state)


def test_factored_partial_route_uses_address_only_for_close_factual_ties() -> None:
    torch.manual_seed(780)
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_EXACT_RESIDUAL_MODE,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    adapter = ExternalTransitionContextAddressAdapter(encoder)
    route_memory = ExternalTransitionRouteMemory(2)
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        max_contexts=2,
        match_tolerance=0.01,
        match_margin=0.01,
        address_adapter=adapter,
        route_query=ExternalTransitionRouteQuery(
            2,
            minimum_score=0.0,
            route_memory=route_memory,
        ),
    )
    source = ExternalTransitionObservation(
        state=torch.tensor([[0.0], [0.0]]),
        intention=torch.tensor([[0.0], [1.0]]),
        next_state=torch.tensor([[0.0], [0.0]]),
    )
    target = ExternalTransitionObservation(
        state=torch.tensor([[1.0], [0.0]]),
        intention=torch.tensor([[0.0], [1.0]]),
        next_state=torch.tensor([[1.0], [10.0]]),
    )
    assert router.route_bundle((source,)).status == "staged"
    assert router.promote_staged_candidate(source, lambda _candidate: True).accepted
    assert router.route_bundle((target,)).status == "staged"
    assert router.promote_staged_candidate(target, lambda _candidate: True).accepted

    tie = ExternalTransitionObservation(
        state=torch.tensor([[0.0]]),
        intention=torch.tensor([[0.0]]),
        next_state=torch.tensor([[0.0]]),
    )
    query = adapter.encode_observation(tie)
    route_memory.unregister_slot(0)
    route_memory.unregister_slot(1)
    route_memory.register_slot(0, prototype=query)
    route_memory.register_slot(1, prototype=query)
    assert router.route_partial_bundle((tie,)).status == "ambiguous"

    route_memory.unregister_slot(0)
    route_memory.unregister_slot(1)
    route_memory.register_slot(0, prototype=query)
    route_memory.register_slot(1, prototype=-query)
    before_digest = router.digest()
    resolved = router.route_partial_sequence(((tie,), (tie,)))
    assert resolved.status == "matched"
    assert resolved.slot_id == 0
    assert router.digest() == before_digest

    contradictory = ExternalTransitionObservation(
        state=tie.state,
        intention=tie.intention,
        next_state=torch.tensor([[5.0]]),
    )
    assert router.route_partial_bundle((contradictory,)).status == "ambiguous"


def test_factored_partial_sequence_horizon_decay_preserves_bound_identity() -> None:
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_EXACT_RESIDUAL_MODE,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        match_tolerance=0.1,
        match_margin=0.0,
    )
    router._contexts = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ]
    router._slot_ids = [0, 1]

    close_stable_tie = False

    def predict(
        state: torch.Tensor,
        _intention: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        slot_zero = context[:, :1] > 0.5
        slot_zero_error = torch.where(
            state < 2.0,
            torch.zeros_like(state),
            torch.full_like(state, 0.09),
        )
        slot_one_error = torch.where(
            state < 2.0,
            torch.full_like(state, 0.04),
            torch.where(
                torch.tensor(close_stable_tie, device=state.device),
                torch.full_like(state, 0.1),
                torch.zeros_like(state),
            ),
        )
        return torch.where(slot_zero, slot_zero_error, slot_one_error)

    model.predict_with_context = predict  # type: ignore[method-assign]
    rows = tuple(
        ExternalTransitionObservation(
            state=torch.tensor([[float(index)]]),
            intention=torch.zeros(1, 1),
            next_state=torch.zeros(1, 1),
        )
        for index in range(4)
    )
    before_digest = router.digest()
    unweighted = router.route_partial_sequence(
        (rows[:2], rows[2:]),
        match_tolerance=0.1,
        confirmation_bundles=2,
    )
    assert unweighted.status == "ambiguous"

    weighted = router.route_partial_sequence(
        (rows[:2], rows[2:]),
        match_tolerance=0.1,
        confirmation_bundles=2,
        horizon_decay=0.2,
    )
    assert weighted.status == "matched"
    assert weighted.slot_id == 0
    assert router.digest() == before_digest

    close_stable_tie = True
    router.match_margin = 0.01
    stable = router.route_partial_sequence(
        (rows[:2], rows[2:]),
        match_tolerance=0.1,
        confirmation_bundles=2,
        stable_identity_confirmation=True,
    )
    assert stable.status == "matched"
    assert stable.slot_id == 0
    assert router.digest() == before_digest

    contradictory = rows[-1]
    contradictory = ExternalTransitionObservation(
        state=contradictory.state,
        intention=contradictory.intention,
        next_state=torch.ones(1, 1),
    )
    assert (
        router.route_partial_sequence(
            (rows[:2], rows[2:3], (contradictory,)),
            match_tolerance=0.1,
            confirmation_bundles=2,
            horizon_decay=0.2,
        ).status
        == "ambiguous"
    )


def test_factored_disambiguation_probe_resolves_an_opaque_factual_tie() -> None:
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(2001)
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_EXACT_RESIDUAL_MODE,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    router = ExternalFactoredTransitionRouter(
        model,
        ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2),
        max_contexts=2,
        match_tolerance=0.01,
        match_margin=0.01,
    )
    source = ExternalTransitionObservation(
        state=torch.tensor([[0.0], [0.0]]),
        intention=torch.tensor([[0.0], [1.0]]),
        next_state=torch.tensor([[0.0], [0.0]]),
    )
    target = ExternalTransitionObservation(
        state=torch.tensor([[1.0], [0.0]]),
        intention=torch.tensor([[0.0], [1.0]]),
        next_state=torch.tensor([[1.0], [10.0]]),
    )
    assert router.route_bundle((source,)).status == "staged"
    assert router.promote_staged_candidate(source, lambda _candidate: True).accepted
    assert router.route_bundle((target,)).status == "staged"
    assert router.promote_staged_candidate(
        ExternalTransitionObservation(
            state=target.state[:1],
            intention=target.intention[:1],
            next_state=target.next_state[:1],
        ),
        lambda _candidate: True,
        prediction_tolerance=0.01,
    ).accepted

    tie = ExternalTransitionObservation(
        state=torch.tensor([[0.0]]),
        intention=torch.tensor([[0.0]]),
        next_state=torch.tensor([[0.0]]),
    )
    assert router.route_partial_bundle((tie,)).status == "ambiguous"
    probe = router.request_disambiguation_probe(
        tie,
        torch.tensor([[0.0], [1.0]]),
        candidate_slot_ids=(0, 1),
    )
    assert probe.selected_intention_index == 1
    assert probe.disagreement_scores[1] > probe.disagreement_scores[0]

    before_probe_sequence = router.digest()
    probe_sequence = router.request_disambiguation_probe_sequence(
        tie,
        torch.tensor([[0.0], [1.0]]),
        candidate_slot_ids=(0, 1),
        probe_state=torch.tensor([[0.0]]),
        horizon=2,
        beam_width=4,
    )
    assert probe_sequence.selected_intentions.shape == (2, 1)
    assert probe_sequence.candidate_slot_ids == (0, 1)
    assert probe_sequence.predicted_next_states.shape == (2, 2, 1)
    assert router.digest() == before_probe_sequence

    observed_probe = ExternalTransitionObservation(
        state=torch.tensor([[0.0]]),
        intention=probe.selected_intention.unsqueeze(0),
        next_state=torch.tensor([[10.0]]),
    )
    resolved = router.route_partial_bundle((observed_probe,))
    assert resolved.status == "matched"
    assert resolved.slot_id == 1
    torch.random.set_rng_state(rng_state)


def test_factored_router_auto_grows_on_verified_novel_bundle() -> None:
    model = ExternalFactoredTransitionModel(1, 1, 2, hidden_width=8)
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        max_contexts=1,
        auto_grow=True,
        match_tolerance=1e-6,
        match_margin=0.0,
    )

    def rows(next_value: float) -> list[ExternalTransitionObservation]:
        return [
            ExternalTransitionObservation(
                state=torch.tensor([[0.0]]),
                intention=torch.tensor([[1.0]]),
                next_state=torch.tensor([[next_value]]),
            ),
            ExternalTransitionObservation(
                state=torch.tensor([[1.0]]),
                intention=torch.tensor([[1.0]]),
                next_state=torch.tensor([[next_value + 1.0]]),
            ),
        ]

    source = rows(0.0)
    target = rows(10.0)
    assert router.route_bundle(source).status == "staged"
    assert router.promote_staged_candidate(
        source[0],
        lambda candidate: candidate.residual_record_count == 2,
        prediction_tolerance=1e-6,
        heldout_rollout=ExternalTransitionRollout(
            initial_state=torch.tensor([0.0]),
            intentions=torch.ones(1, 1),
            expected_states=torch.tensor([[0.0]]),
        ),
        rollout_error_tolerance=1e-6,
    ).accepted
    assert router.max_contexts == 1

    before_target = router.model.digest()
    assert router.route_bundle(target).status == "staged"
    target_receipt = router.promote_staged_candidate(
        target[0],
        lambda candidate: candidate.residual_record_count == 4,
        prediction_tolerance=1e-6,
        heldout_rollout=ExternalTransitionRollout(
            initial_state=torch.tensor([0.0]),
            intentions=torch.ones(1, 1),
            expected_states=torch.tensor([[999.0]]),
        ),
        rollout_error_tolerance=1e-6,
    )
    assert not target_receipt.accepted
    assert target_receipt.heldout_rollout_error is not None
    assert "recursive" in target_receipt.reason
    assert router.max_contexts == 1
    assert router.model.digest() == before_target

    assert router.route_bundle(target).status == "staged"
    target_receipt = router.promote_staged_candidate(
        target[0],
        lambda candidate: candidate.residual_record_count == 4,
        prediction_tolerance=1e-6,
        heldout_rollout=ExternalTransitionRollout(
            initial_state=torch.tensor([0.0]),
            intentions=torch.ones(1, 1),
            expected_states=torch.tensor([[10.0]]),
        ),
        rollout_error_tolerance=1e-6,
    )
    assert target_receipt.accepted
    assert "capacity growth" in target_receipt.reason
    assert router.max_contexts == 2
    assert router.model.digest() != before_target
    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    assert restored.auto_grow
    assert restored.max_contexts == 2


def test_factored_reliability_gate_vetoes_corruption_but_preserves_growth_path() -> None:
    model = ExternalFactoredTransitionModel(
        1,
        1,
        2,
        hidden_width=8,
        residual_capacity=2,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    reliability = ExternalTransitionEvidenceStatistics(
        1,
        bin_count=8,
        error_scale=0.1,
        prior_count=0.01,
    )
    reliability.observe(
        torch.tensor([[1.0], [1.0]]),
        torch.tensor([[1.0], [1.0]]),
        torch.ones(2),
    )
    reliability.observe(
        torch.tensor([[1.0], [1.0]]),
        torch.tensor([[1.12], [1.12]]),
        torch.zeros(2),
    )
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        max_contexts=2,
        match_tolerance=0.02,
        match_margin=0.0,
        quarantine_capacity=4,
        evidence_evaluator=reliability,
        evidence_threshold=0.9,
        evidence_gate_min_evidence=1,
        committed_evidence_gate=True,
    )

    source = ExternalTransitionObservation(
        state=torch.tensor([[-1.0], [1.0]]),
        intention=torch.ones(2, 1),
        next_state=torch.tensor([[-0.5], [0.5]]),
    )
    heldout = source
    assert router.route_bundle((source,)).status == "staged"
    assert router.promote_staged_candidate(heldout, lambda _candidate: True).accepted

    corrupted = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state + 0.12,
    )
    vetoed = router.route_bundle((corrupted,))
    assert vetoed.status == "reliability_veto"
    assert vetoed.quarantine_accepted is True
    assert not router.candidate_active
    assert router.slot_ids == (0,)
    assert router.quarantined_observations == 2

    assert router.route_bundle((corrupted,)).quarantine_accepted is True
    saturated = router.route_bundle((corrupted,))
    assert saturated.status == "reliability_veto"
    assert saturated.quarantine_accepted is False
    assert router.quarantined_observations == 4

    novel = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state + 1.0,
    )
    assert router.route_bundle((novel,)).status == "staged"
    assert router.candidate_active

    restored = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=reliability
    )
    assert restored.digest() == router.digest()


def test_factored_contextual_reliability_isolates_opaque_slots() -> None:
    model = ExternalFactoredTransitionModel(1, 1, 2, hidden_width=8)
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        max_contexts=2,
        match_tolerance=0.02,
        match_margin=0.0,
        quarantine_capacity=2,
    )
    source_a = ExternalTransitionObservation(
        state=torch.tensor([[-1.0], [-0.5]]),
        intention=torch.ones(2, 1),
        next_state=torch.tensor([[-0.5], [0.0]]),
    )
    source_b = ExternalTransitionObservation(
        state=torch.tensor([[0.5], [1.0]]),
        intention=torch.ones(2, 1),
        next_state=torch.tensor([[1.0], [1.5]]),
    )
    assert router.route_bundle((source_a,)).status == "staged"
    assert router.promote_staged_candidate(source_a, lambda _candidate: True).accepted
    assert router.route_bundle((source_b,)).status == "staged"
    assert router.promote_staged_candidate(source_b, lambda _candidate: True).accepted

    evaluator = ExternalContextualEvidenceCalibrator(
        ExternalTransitionEvidenceEvaluator(1, hidden_width=4),
        context_width=2,
    )
    first_context = router.contexts[0]
    second_context = router.contexts[1]
    first_index = evaluator.ensure_context(first_context)
    second_index = evaluator.ensure_context(second_context)
    with torch.no_grad():
        evaluator.calibrators[first_index].bias.fill_(10.0)
        evaluator.calibrators[second_index].bias.fill_(-10.0)
    router.evidence_evaluator = evaluator
    router.committed_evidence_gate = True

    source_a_drift = ExternalTransitionObservation(
        state=source_a.state,
        intention=source_a.intention,
        next_state=source_a.next_state + 0.01,
    )
    source_b_drift = ExternalTransitionObservation(
        state=source_b.state,
        intention=source_b.intention,
        next_state=source_b.next_state + 0.01,
    )
    allowed = router.route_bundle((source_a_drift,))
    vetoed = router.route_bundle((source_b_drift,))

    assert allowed.status == "matched"
    assert allowed.slot_id == router.slot_ids[0]
    assert vetoed.status == "reliability_veto"
    assert vetoed.quarantine_accepted is True
    assert router.quarantined_observations == source_b.state.shape[0]


def test_contextual_evidence_statistics_isolate_and_persist_contexts() -> None:
    statistics = ExternalContextualTransitionEvidenceStatistics(
        1,
        2,
        bin_count=8,
        error_scale=0.1,
        prior_count=0.01,
    )
    prediction = torch.zeros(2, 1)
    observed = torch.full((2, 1), 0.01)
    contexts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    statistics.observe(
        prediction,
        observed,
        torch.tensor([0.0, 1.0]),
        contexts,
    )

    scores = statistics.score(prediction, observed, None, contexts)
    assert scores[0] < 0.0
    assert scores[1] > 0.0
    statistics.evict_context(contexts[0])
    assert statistics.context_count == 1
    assert torch.equal(
        statistics.score(prediction[1:], observed[1:], None, contexts[1:]),
        scores[1:],
    )
    restored = ExternalContextualTransitionEvidenceStatistics.from_payload(
        statistics.payload()
    )
    assert restored.digest() == statistics.digest()
    assert torch.equal(
        restored.score(prediction[1:], observed[1:], None, contexts[1:]),
        scores[1:],
    )


def test_contextual_evidence_statistics_adapts_reversal_with_local_decay() -> None:
    statistics = ExternalContextualTransitionEvidenceStatistics(
        1,
        2,
        bin_count=8,
        error_scale=0.1,
        prior_count=0.01,
        count_decay=0.1,
    )
    prediction = torch.zeros(1, 1)
    observed = torch.full((1, 1), 0.01)
    context = torch.tensor([1.0, 0.0])
    statistics.observe(prediction, observed, torch.ones(1), context)
    assert statistics.score(prediction, observed, None, context).item() > 0.0
    statistics.observe(prediction, observed, torch.zeros(1), context)
    assert statistics.score(prediction, observed, None, context).item() < 0.0
    assert statistics.configuration()["count_decay"] == 0.1


def test_factored_router_resolves_quarantine_into_isolated_candidate() -> None:
    model = ExternalFactoredTransitionModel(1, 1, 2, hidden_width=8)
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        max_contexts=1,
        quarantine_capacity=2,
    )
    anchor = ExternalTransitionObservation(
        state=torch.tensor([[0.0]]),
        intention=torch.tensor([[1.0]]),
        next_state=torch.tensor([[1.0]]),
    )
    assert router.quarantine_partial_bundle((anchor,)).accepted
    live_digest = router.model.digest()
    assert router.route_bundle((anchor,)).status == "staged"
    assert router.candidate_active
    assert router.resolve_quarantine_to_candidate(match_tolerance=1e-6) == 1
    assert router.quarantined_observations == 0
    assert router.candidate_active
    assert router.model.digest() == live_digest


def test_factored_router_stages_promotes_and_reuses_opaque_context() -> None:
    model = ExternalFactoredTransitionModel(1, 1, 2, hidden_width=8)
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        admission_observations=2,
        match_tolerance=1e-6,
        match_margin=0.0,
    )
    first = ExternalTransitionObservation(
        state=torch.tensor([[0.0]]),
        intention=torch.tensor([[1.0]]),
        next_state=torch.tensor([[1.0]]),
    )
    second = ExternalTransitionObservation(
        state=torch.tensor([[1.0]]),
        intention=torch.tensor([[1.0]]),
        next_state=torch.tensor([[2.0]]),
    )
    assert router.observe(first).status == "pending"
    staged = router.observe(second)
    assert staged.status == "staged"
    assert router.candidate_active
    receipt = router.promote_staged_candidate(
        first,
        lambda candidate: candidate.residual_record_count == 2,
        prediction_tolerance=1e-6,
    )
    assert receipt.accepted
    assert receipt.slot_id == 0
    assert router.observe(first).status == "matched"

    unseen = ExternalTransitionObservation(
        state=torch.tensor([[2.0]]),
        intention=torch.tensor([[1.0]]),
        next_state=torch.tensor([[3.0]]),
    )
    updated = router.update_bound_slot(
        0,
        unseen,
        lambda candidate: candidate.residual_record_count == 3,
        heldout=second,
        prediction_tolerance=1e-6,
    )
    assert updated.accepted
    assert updated.slot_id == 0
    assert router.observe(unseen).status == "matched"

    before_rejected_update = router.model.digest()
    rejected = router.update_bound_slot(
        0,
        ExternalTransitionObservation(
            state=torch.tensor([[3.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[999.0]]),
        ),
        lambda _candidate: False,
        heldout=second,
    )
    assert not rejected.accepted
    assert router.model.digest() == before_rejected_update

    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    assert restored.digest() == router.digest()
    assert restored.slot_ids == (0,)


def test_factored_router_routes_opaque_bundles_atomically() -> None:
    model = ExternalFactoredTransitionModel(1, 1, 2, hidden_width=8)
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        admission_observations=2,
        match_tolerance=1e-6,
        match_margin=0.0,
    )
    source_rows = [
        ExternalTransitionObservation(
            state=torch.tensor([[0.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[1.0]]),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[1.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[2.0]]),
        ),
    ]
    assert router.route_bundle(source_rows).status == "staged"
    router.promote_staged_candidate(
        source_rows[0],
        lambda candidate: candidate.residual_record_count == 2,
        prediction_tolerance=1e-6,
    )

    conflicting_rows = [
        ExternalTransitionObservation(
            state=torch.tensor([[0.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[10.0]]),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[1.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[11.0]]),
        ),
    ]
    staged = router.route_bundle(conflicting_rows)
    assert staged.status == "staged"
    assert staged.pending_observations == 0
    assert router.candidate_active
    assert router.pending_observations == 0


def test_factored_router_accumulates_partial_evidence_without_forcing_identity() -> None:
    torch.manual_seed(4001)
    model = ExternalFactoredTransitionModel(1, 1, 2, hidden_width=8)
    model.freeze_base()
    encoder = ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2)
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        max_contexts=2,
        match_tolerance=0.05,
        match_margin=0.01,
    )
    source = (
        ExternalTransitionObservation(
            state=torch.tensor([[0.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[1.0]]),
        ),
        ExternalTransitionObservation(
            state=torch.tensor([[1.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[2.0]]),
        ),
    )
    target = (
        source[0],
        ExternalTransitionObservation(
            state=torch.tensor([[1.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[3.0]]),
        ),
    )
    assert router.route_bundle(source).status == "staged"
    assert router.promote_staged_candidate(
        source[0],
        lambda _candidate: True,
        prediction_tolerance=0.05,
    ).accepted
    assert router.route_bundle(target).status == "staged"
    assert router.promote_staged_candidate(
        target[0],
        lambda _candidate: True,
        prediction_tolerance=0.05,
    ).accepted

    digest_before = router.digest()
    ambiguous = router.route_partial_bundle((target[0],))
    assert ambiguous.status == "ambiguous"
    resolved = router.route_partial_sequence(
        ((target[0],), (target[1],)),
    )
    assert resolved.status == "matched"
    assert resolved.slot_id == 1
    single_bundle = router.route_partial_sequence((tuple(target),))
    assert single_bundle.status == "ambiguous"
    assert router.digest() == digest_before


def test_factored_router_learns_external_residual_functions_without_base_updates() -> (
    None
):
    torch.manual_seed(1922)
    model = ExternalFactoredTransitionModel(
        1,
        1,
        4,
        hidden_width=16,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_hidden_width=32,
        residual_learning_rate=0.01,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    base_before = model.base.digest()
    encoder = ExternalTransitionContextEncoder(
        1,
        1,
        hidden_width=16,
        context_width=4,
    )
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        match_tolerance=0.05,
        match_margin=0.001,
        residual_adaptation_updates=300,
    )
    train_state = torch.linspace(-1.0, 1.0, 8).unsqueeze(-1)
    heldout_state = torch.tensor([[-0.85], [-0.35], [0.15], [0.65]])

    def observation(
        state: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> ExternalTransitionObservation:
        return ExternalTransitionObservation(
            state=state,
            intention=torch.ones(state.shape[0], 1),
            next_state=transform(state),
        )

    source = observation(train_state, lambda state: torch.sin(state * 2.0) + state)
    source_heldout = observation(
        heldout_state,
        lambda state: torch.sin(state * 2.0) + state,
    )
    target = observation(train_state, lambda state: torch.cos(state * 2.0) + state)
    target_heldout = observation(
        heldout_state,
        lambda state: torch.cos(state * 2.0) + state,
    )

    def rows(item: ExternalTransitionObservation) -> list[ExternalTransitionObservation]:
        return [
            ExternalTransitionObservation(
                state=item.state[index : index + 1],
                intention=item.intention[index : index + 1],
                next_state=item.next_state[index : index + 1],
            )
            for index in range(item.state.shape[0])
        ]

    assert router.route_bundle(rows(source)).status == "staged"
    source_receipt = router.promote_staged_candidate(
        source_heldout,
        lambda candidate: candidate.residual_context_count == 1,
        prediction_tolerance=0.01,
    )
    assert source_receipt.accepted
    source_slot = source_receipt.slot_id
    assert source_slot == 0

    source_context = router.contexts[0]
    assert router.route_bundle(rows(target)).status == "staged"
    target_receipt = router.promote_staged_candidate(
        target_heldout,
        lambda candidate: float(
            candidate.loss(
                source_heldout,
                context=source_context.unsqueeze(0).expand(
                    source_heldout.state.shape[0], -1
                ),
            ).detach()
        )
        < 0.01,
        prediction_tolerance=0.01,
    )
    assert target_receipt.accepted
    assert target_receipt.slot_id == 1
    assert router.route_bundle(rows(source)).slot_id == source_slot
    assert router.route_bundle(rows(target)).slot_id == target_receipt.slot_id
    assert model.base.digest() == base_before

    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    assert restored.digest() == router.digest()
    assert restored.route_bundle(rows(source)).slot_id == source_slot
    assert restored.route_bundle(rows(target)).slot_id == target_receipt.slot_id


def test_goal_evaluator_learns_scalar_verifier_without_latent_distance() -> None:
    torch.manual_seed(1203)
    evaluator = ExternalGoalEvaluator(2, hidden_width=16)
    state = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    goal = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    outcome = torch.tensor([1.0, 0.0, 0.0, 1.0])
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=0.05)

    for _ in range(250):
        optimizer.zero_grad()
        loss = evaluator.loss(state, goal, outcome)
        loss.backward()
        optimizer.step()

    probability = torch.sigmoid(evaluator(state, goal))
    assert probability[[0, 3]].min().item() > 0.99
    assert probability[[1, 2]].max().item() < 0.01


def test_goal_evaluator_payload_is_versioned_and_exact() -> None:
    torch.manual_seed(1215)
    evaluator = ExternalGoalEvaluator(2, hidden_width=8)
    payload = evaluator.state_payload()
    restored = ExternalGoalEvaluator.from_payload(payload)
    state = torch.randn(4, 2)
    goal = torch.randn(4, 2)

    assert restored.digest() == evaluator.digest()
    assert torch.equal(restored(state, goal), evaluator(state, goal))

    corrupted = dict(payload)
    corrupted["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        ExternalGoalEvaluator.from_payload(corrupted)


def test_planner_accepts_contextual_append_only_transition_memory() -> None:
    memory = ExternalTransitionMemory(1, 1, context_width=1)
    memory.write(
        ExternalTransitionObservation(
            state=torch.tensor([[0.0], [0.0]]),
            intention=torch.tensor([[-1.0], [1.0]]),
            next_state=torch.tensor([[-1.0], [1.0]]),
        ),
        context=torch.ones(2, 1),
    )
    result = ExternalModelBasedPlanner(memory).plan(
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.tensor([[-1.0], [1.0]]),
        horizon=1,
        transition_context=torch.ones(1, 1),
    )

    assert torch.equal(result.intentions[0, 0], torch.ones(1))
    assert torch.equal(result.predicted_states[0, 0], torch.ones(1))


def test_bound_transition_memory_keeps_context_stable_for_iterative_planning() -> None:
    memory = ExternalTransitionMemory(1, 1, context_width=1)
    memory.write(
        ExternalTransitionObservation(
            state=torch.tensor([[0.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[1.0]]),
        ),
        context=torch.ones(1, 1),
    )
    original_context = torch.ones(1)
    bound = memory.bind_context(original_context)
    original_context.fill_(-1.0)

    assert isinstance(bound, ExternalBoundTransitionModel)
    assert bound.configuration()["binding"] == (
        "single_opaque_context_for_iterative_execution_v1"
    )
    prediction, hit = bound.predict_with_hit(
        torch.zeros(1, 1),
        torch.ones(1, 1),
    )
    assert torch.equal(prediction, torch.ones(1, 1))
    assert torch.equal(hit, torch.ones(1, dtype=torch.bool))

    result = ExternalModelBasedPlanner(bound).plan(
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.tensor([[1.0]]),
        horizon=1,
        require_known=True,
    )
    assert torch.equal(result.predicted_states[0, 0], torch.ones(1))


def test_fail_closed_planning_rejects_unknown_transition_in_later_prefix() -> None:
    memory = ExternalTransitionMemory(1, 1, context_width=1)
    memory.write(
        ExternalTransitionObservation(
            state=torch.tensor([[0.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[1.0]]),
        ),
        context=torch.ones(1, 1),
    )
    planner = ExternalModelBasedPlanner(memory)

    with pytest.raises(LookupError, match="unknown|verified transition"):
        planner.plan(
            torch.zeros(1, 1),
            torch.tensor([[2.0]]),
            torch.tensor([[1.0]]),
            horizon=2,
            transition_context=torch.ones(1, 1),
            require_known=True,
        )


def test_fail_closed_rollout_exposes_corrupted_factual_memory() -> None:
    memory = ExternalTransitionMemory(1, 1, context_width=1)
    memory.write(
        ExternalTransitionObservation(
            state=torch.tensor([[0.0]]),
            intention=torch.tensor([[1.0]]),
            next_state=torch.tensor([[9.0]]),
        ),
        context=torch.ones(1, 1),
    )
    planner = ExternalModelBasedPlanner(memory)
    error = planner.rollout_error(
        ExternalTransitionRollout(
            initial_state=torch.tensor([0.0]),
            intentions=torch.tensor([[1.0]]),
            expected_states=torch.tensor([[1.0]]),
        ),
        transition_context=torch.ones(1, 1),
        require_known=True,
    )
    assert error == pytest.approx(64.0)


def test_planner_accepts_runtime_sized_opaque_goal_sets() -> None:
    planner = ExternalModelBasedPlanner(_AdditiveTransitionModel(), beam_width=1)
    result = planner.plan(
        torch.zeros(1, 1),
        torch.tensor([[[1.0], [3.0]]]),
        torch.tensor([[-1.0], [1.0], [2.0]]),
        horizon=1,
    )

    assert result.intentions.shape == (1, 1, 1)
    assert result.predicted_states.shape == (1, 1, 1)
    assert result.intentions[0, 0, 0].item() == 1.0
    assert result.predicted_states[0, 0, 0].item() == 1.0
    assert result.scores.item() == 0.0


def test_planner_goal_set_uses_learned_verifier_as_existential_predicate() -> None:
    torch.manual_seed(1214)
    evaluator = ExternalGoalEvaluator(1, hidden_width=8)
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=0.05)
    states = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    goals = torch.tensor([[1.0], [1.0], [3.0], [3.0]])
    outcomes = torch.tensor([0.0, 1.0, 0.0, 1.0])
    for _ in range(250):
        optimizer.zero_grad()
        loss = evaluator.loss(states, goals, outcomes)
        loss.backward()
        optimizer.step()

    planner = ExternalModelBasedPlanner(
        _AdditiveTransitionModel(),
        beam_width=1,
        goal_evaluator=evaluator,
    )
    result = planner.plan(
        torch.zeros(1, 1),
        torch.tensor([[[1.0], [3.0]]]),
        torch.tensor([[1.0], [2.0]]),
        horizon=1,
    )

    assert result.intentions[0, 0, 0].item() == 1.0
    assert result.scores.item() < -0.9


def test_planner_selects_active_intention_that_disambiguates_model_slots() -> None:
    bank = ExternalTransitionModelBank(
        1,
        1,
        2,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
    )
    contexts = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    for index, context in enumerate(contexts):
        slot = bank.ensure_context(context)
        state = torch.tensor([[0.0], [0.0]])
        intention = torch.tensor([[0.0], [1.0]])
        next_state = (
            torch.zeros_like(intention) if index == 0 else intention.clone()
        )
        bank.adaptation_step(
            ExternalTransitionObservation(state, intention, next_state),
            bank.context_at(slot).unsqueeze(0).expand(2, -1),
            None,
        )

    planner = ExternalModelBasedPlanner(bank)
    result = planner.select_disambiguating_intention(
        bank,
        torch.zeros(1, 1),
        torch.tensor([[0.0], [1.0]]),
    )

    assert result.selected_intention_index == 1
    assert result.selected_intention.item() == 1.0
    assert result.disagreement_scores[1] > result.disagreement_scores[0]
    assert result.predicted_next_states.shape == (2, 2, 1)

    utility_memory = ExternalTransitionProbeUtilityMemory(key_width=4)
    utility_memory.observe(result.utility_profiles[1].unsqueeze(0), utility=1.0)
    utility_result = planner.select_disambiguating_intention(
        bank,
        torch.zeros(1, 1),
        torch.tensor([[0.0], [1.0]]),
        utility_memory=utility_memory,
    )
    assert utility_result.selected_intention_index == 1
    assert utility_result.utility_scores is not None
    assert utility_result.utility_scores[1] > utility_result.utility_scores[0]

    confident_utility = ExternalTransitionProbeUtilityMemory(key_width=4)
    assert result.utility_profiles is not None
    for _ in range(20):
        confident_utility.observe(result.utility_profiles[0].unsqueeze(0), utility=1.0)
        confident_utility.observe(result.utility_profiles[1].unsqueeze(0), utility=0.0)
    overridden = planner.select_disambiguating_intention(
        bank,
        torch.zeros(1, 1),
        torch.tensor([[0.0], [1.0]]),
        utility_memory=confident_utility,
    )
    assert overridden.selected_intention_index == 0
    assert overridden.utility_confidence_scores is not None
    assert overridden.utility_confidence_scores[0] > 0.9

    contextual_utility = ExternalTransitionProbeContextualUtilityMemory(
        intention_width=1,
        context_width=3,
        intention_merge_cosine=0.99,
        context_merge_cosine=0.99,
        context_kernel_floor=0.75,
    )
    for _ in range(20):
        contextual_utility.observe(
            torch.tensor([[0.0], [1.0]]),
            result.utility_profiles[:, 1:],
            utility=torch.tensor([1.0, 0.0]),
        )
    contextual_result = planner.select_disambiguating_intention(
        bank,
        torch.zeros(1, 1),
        torch.tensor([[0.0], [1.0]]),
        utility_memory=contextual_utility,
    )
    assert contextual_result.selected_intention_index == 0
    assert contextual_result.utility_confidence_scores is not None
    assert contextual_result.utility_confidence_scores[0] > 0.9


def test_transition_support_statistics_calibrate_opaque_leverage_without_replay() -> None:
    support = ExternalTransitionSupportStatistics(
        bin_count=8,
        leverage_scale=10.0,
    )
    support.observe(torch.ones(8), torch.ones(8))
    support.observe(torch.full((8,), 100.0), torch.zeros(8))

    scores = support(torch.tensor([1.0, 100.0]))
    assert scores[0] > scores[1]
    assert support.observation_count.item() == 16

    restored = ExternalTransitionSupportStatistics.from_payload(support.payload())
    assert restored.digest() == support.digest()
    assert torch.allclose(restored(torch.tensor([1.0, 100.0])), scores)

    keyed = ExternalTransitionSupportStatistics(
        bin_count=8,
        leverage_scale=10.0,
        slot_capacity=4,
    )
    keyed.observe(
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        slot_ids=(10, 20),
    )
    assert keyed.slot_ids.tolist()[:2] == [10, 20]
    assert keyed.slot_observation_counts.tolist()[:2] == [2, 2]
    keyed_restored = ExternalTransitionSupportStatistics.from_payload(keyed.payload())
    assert keyed_restored.digest() == keyed.digest()

    planner = ExternalModelBasedPlanner(_AdditiveTransitionModel())
    assert planner.configuration()["policy"] == "none_behavior_derived_at_inference_v1"


def test_probe_utility_memory_learns_scalar_resolution_without_replay() -> None:
    memory = ExternalTransitionProbeUtilityMemory(2, merge_cosine=0.99)
    memory.observe(torch.tensor([[1.0, 0.0]]), utility=1.0)
    memory.observe(torch.tensor([[0.0, 1.0]]), utility=0.0)
    memory.observe(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        utility=torch.tensor([0.0, 1.0]),
        outcome_mask=torch.tensor([False, True]),
    )

    scores = memory.scores(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    assert scores[0] > scores[1]
    assert scores[2].item() == pytest.approx(0.5)
    assert memory.observed_outcome_count == 3

    restored = ExternalTransitionProbeUtilityMemory.from_payload(memory.payload())
    assert restored.content_digest() == memory.content_digest()
    assert torch.allclose(restored.scores(torch.tensor([[1.0, 0.0], [0.0, 1.0]])), scores[:2])


def test_contextual_probe_utility_transfers_only_across_related_opaque_contexts() -> None:
    memory = ExternalTransitionProbeContextualUtilityMemory(
        intention_width=2,
        context_width=2,
        intention_merge_cosine=0.99,
        context_merge_cosine=0.99,
        context_kernel_floor=0.70,
    )
    intention = torch.tensor([[1.0, 0.0]])
    context = torch.tensor([[1.0, 0.0]])
    for _ in range(8):
        memory.observe(intention, context, utility=1.0)

    scores, confidence = memory.scores_and_confidence(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        torch.tensor(
            [
                [0.98, 0.20],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )
    assert scores[0] > 0.80
    assert confidence[0] > 0.70
    assert scores[1].item() == pytest.approx(0.5)
    assert confidence[1].item() == pytest.approx(0.0)
    assert scores[2].item() == pytest.approx(0.5)
    assert confidence[2].item() == pytest.approx(0.0)

    memory.observe(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[0.98, 0.20], [0.0, 1.0]]),
        utility=torch.tensor([0.0, 0.0]),
        outcome_mask=torch.tensor([False, False]),
    )
    assert memory.observed_outcome_count == 8
    restored = ExternalTransitionProbeContextualUtilityMemory.from_payload(
        memory.payload()
    )
    assert restored.content_digest() == memory.content_digest()
    restored_scores, restored_confidence = restored.scores_and_confidence(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[0.98, 0.20]])
    )
    assert torch.allclose(restored_scores, scores[:1])
    assert torch.allclose(restored_confidence, confidence[:1])


def test_planner_selects_a_fixed_opaque_probe_sequence() -> None:
    bank = ExternalTransitionModelBank(
        1,
        1,
        2,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
    )
    for index, context in enumerate(
        (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    ):
        slot = bank.ensure_context(context)
        state = torch.tensor([[0.0], [0.0]])
        intention = torch.tensor([[0.0], [1.0]])
        next_state = (
            torch.zeros_like(intention) if index == 0 else intention.clone()
        )
        bank.adaptation_step(
            ExternalTransitionObservation(state, intention, next_state),
            bank.context_at(slot).unsqueeze(0).expand(2, -1),
            None,
        )

    planner = ExternalModelBasedPlanner(bank)
    before = bank.digest()
    result = planner.select_disambiguating_intention_sequence(
        bank,
        torch.zeros(1, 1),
        torch.tensor([[0.0], [1.0]]),
        horizon=2,
        beam_width=4,
    )

    assert result.selected_intentions.shape == (2, 1)
    assert result.selected_intention_indices.shape == (2,)
    assert result.predicted_next_states.shape == (2, 2, 1)
    assert torch.isfinite(result.disagreement_scores).all()
    assert bank.digest() == before


def test_online_router_requests_read_only_probe_for_ambiguous_evidence() -> None:
    bank = ExternalTransitionModelBank(
        1,
        1,
        2,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
    )
    for index, context in enumerate(
        (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    ):
        slot = bank.ensure_context(context)
        state = torch.tensor([[0.0], [0.0]])
        intention = torch.tensor([[0.0], [1.0]])
        next_state = (
            torch.zeros_like(intention) if index == 0 else intention.clone()
        )
        bank.adaptation_step(
            ExternalTransitionObservation(state, intention, next_state),
            bank.context_at(slot).unsqueeze(0).expand(2, -1),
            None,
        )
    router = ExternalOnlineTransitionContextRouter(
        bank,
        ExternalTransitionContextEncoder(1, 1, hidden_width=8, context_width=2),
        match_margin=0.01,
    )
    ambiguous = ExternalTransitionObservation(
        state=torch.tensor([[0.0]]),
        intention=torch.tensor([[0.0]]),
        next_state=torch.tensor([[0.0]]),
    )
    before = bank.digest()
    probe = router.request_disambiguation_probe(
        ambiguous,
        torch.tensor([[0.0], [1.0]]),
    )

    assert probe.selected_intention.item() == 1.0
    assert probe.candidate_slot_ids == (0, 1)
    assert bank.digest() == before
    sequence = router.request_disambiguation_probe_sequence(
        ambiguous,
        torch.tensor([[0.0], [1.0]]),
        horizon=2,
        beam_width=4,
    )
    assert sequence.selected_intentions.shape == (2, 1)
    assert sequence.candidate_slot_ids == (0, 1)
    assert bank.digest() == before
    assert router.configuration()["active_probe"] == (
        "read_only_uncertainty_weighted_model_disagreement_request_v1"
    )
    assert router.configuration()["active_probe_sequence"] == (
        "read_only_beam_search_uncertainty_weighted_probe_sequence_v1"
    )


def test_context_resolver_reuses_consistent_facts_and_allocates_new_regime() -> None:
    memory = ExternalTransitionMemory(1, 1, context_width=2)
    resolver = ExternalContextAddressResolver(2, address_seed=1204)
    source = ExternalTransitionObservation(
        state=torch.tensor([[0.0], [1.0]]),
        intention=torch.tensor([[1.0], [1.0]]),
        next_state=torch.tensor([[1.0], [2.0]]),
    )
    reversal = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=torch.tensor([[-1.0], [0.0]]),
    )

    first = resolver.resolve(source, memory)
    memory.write(source, context=first.context.expand(2, -1))
    source_again = resolver.resolve(source, memory)
    second = resolver.resolve(reversal, memory)
    memory.write(reversal, context=second.context.expand(2, -1))
    reversal_again = resolver.resolve(reversal, memory)
    restored = ExternalContextAddressResolver.from_payload(resolver.payload())

    assert not first.reused
    assert source_again.reused
    assert not second.reused
    assert reversal_again.reused
    assert resolver.context_count == 2
    assert torch.allclose(restored.addresses(), resolver.addresses())


def test_online_context_resolver_accumulates_interleaved_evidence_without_early_writes() -> (
    None
):
    memory = ExternalTransitionMemory(1, 1, context_width=2)
    resolver = ExternalOnlineContextAddressResolver(
        2,
        address_seed=1205,
        admission_observations=3,
        contradiction_observations=2,
    )

    def row(position: float, next_position: float) -> ExternalTransitionObservation:
        return ExternalTransitionObservation(
            state=torch.tensor([[position]]),
            intention=torch.ones(1, 1),
            next_state=torch.tensor([[next_position]]),
        )

    stream_a = torch.tensor([1.0, 0.0])
    stream_b = torch.tensor([0.0, 1.0])
    a1 = resolver.observe(row(0.0, 1.0), stream_a, memory)
    b1 = resolver.observe(row(0.0, -1.0), stream_b, memory)
    a2 = resolver.observe(row(1.0, 2.0), stream_a, memory)
    b2 = resolver.observe(row(1.0, 0.0), stream_b, memory)
    assert a1.status == "uncertain" and b1.status == "uncertain"
    assert a2.status == "uncertain" and b2.status == "uncertain"
    assert memory.record_count == 0

    a3 = resolver.observe(row(2.0, 3.0), stream_a, memory)
    b3 = resolver.observe(row(2.0, 1.0), stream_b, memory)

    assert a3.status == "admitted" and b3.status == "admitted"
    assert memory.record_count == 6

    duplicate = resolver.observe(row(0.0, 1.0), torch.tensor([1.0, 1.0]), memory)
    assert duplicate.status == "reused"
    assert duplicate.committed_observations == 0
    assert memory.record_count == 6

    reversal_1 = resolver.observe(row(0.0, -1.0), stream_a, memory)
    reversal_2 = resolver.observe(row(1.0, 0.0), stream_a, memory)
    assert reversal_1.status == "conflict"
    assert reversal_1.committed_observations == 0
    assert reversal_2.status == "admitted"
    assert memory.record_count == 8

    restored = ExternalOnlineContextAddressResolver.from_payload(resolver.payload())
    assert restored.context_count == resolver.context_count == 3
    assert restored.pending_observations(stream_a) == 0

    pending_memory = ExternalTransitionMemory(1, 1, context_width=2)
    pending_resolver = ExternalOnlineContextAddressResolver(
        2, address_seed=1206, admission_observations=3
    )
    pending_resolver.observe(row(0.0, 1.0), stream_a, pending_memory)
    pending_resolver.observe(row(1.0, 2.0), stream_a, pending_memory)
    resumed = ExternalOnlineContextAddressResolver.from_payload(
        pending_resolver.payload()
    )
    resumed_result = resumed.observe(row(2.0, 3.0), stream_a, pending_memory)
    assert resumed_result.status == "admitted"
    assert pending_memory.record_count == 3


def test_online_context_resolver_reactivates_retained_version_after_reversal_cycle() -> None:
    memory = ExternalTransitionMemory(1, 1, context_width=2)
    resolver = ExternalOnlineContextAddressResolver(
        2,
        address_seed=1207,
        admission_observations=3,
        contradiction_observations=2,
    )

    def row(position: float, next_position: float) -> ExternalTransitionObservation:
        return ExternalTransitionObservation(
            state=torch.tensor([[position]]),
            intention=torch.ones(1, 1),
            next_state=torch.tensor([[next_position]]),
        )

    stream = torch.tensor([1.0, 0.0])
    source_rows = (row(0.0, 1.0), row(1.0, 2.0), row(2.0, 3.0))
    for item in source_rows:
        resolution = resolver.observe(item, stream, memory)
    assert resolution.status == "admitted"
    source_context = resolution.context.clone()
    source_context_count = resolver.context_count
    source_record_count = memory.record_count

    reversal_rows = (row(0.0, -1.0), row(1.0, 0.0))
    assert resolver.observe(reversal_rows[0], stream, memory).status == "conflict"
    reversal = resolver.observe(reversal_rows[1], stream, memory)
    assert reversal.status == "admitted"
    reversal_context = reversal.context.clone()
    assert resolver.context_count == source_context_count + 1
    assert memory.record_count == source_record_count + 2

    # Returning to the old regime reactivates its retained address rather
    # than allocating A-v2.  The first row is enough to identify the old
    # factual version because it is already committed and exact.
    restored_source = resolver.observe(source_rows[0], stream, memory)
    assert restored_source.status == "reused"
    assert torch.allclose(restored_source.context, source_context)
    assert resolver.context_count == source_context_count + 1
    assert memory.record_count == source_record_count + 2
    assert resolver.observe(source_rows[1], stream, memory).status == "reused"

    # The new version is also retained and can be reactivated without a
    # write, proving that both sides of the reversal remain addressable.
    restored_reversal = resolver.observe(reversal_rows[0], stream, memory)
    assert restored_reversal.status == "reused"
    assert torch.allclose(restored_reversal.context, reversal_context)
    assert resolver.context_count == source_context_count + 1
    assert memory.record_count == source_record_count + 2

    restored = ExternalOnlineContextAddressResolver.from_payload(resolver.payload())
    assert restored.context_count == resolver.context_count
    assert torch.allclose(restored.addresses(), resolver.addresses())
    assert restored.observe(source_rows[0], stream, memory).status == "reused"
    assert restored.observe(reversal_rows[0], stream, memory).status == "reused"


def test_transition_evidence_evaluator_has_versioned_scalar_outcome_boundary() -> None:
    evaluator = ExternalTransitionEvidenceEvaluator(3, hidden_width=8)
    prediction = torch.zeros(4, 3)
    observed = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    outcomes = torch.tensor([1.0, 1.0, 0.0, 0.0])
    logits = evaluator(prediction, observed, torch.ones(4))

    assert logits.shape == (4,)
    assert torch.isfinite(evaluator.loss(prediction, observed, outcomes))
    assert evaluator.configuration()["behavior"] == "read_only_consistency_gate_v1"


def test_transition_evidence_statistics_learns_once_and_persists_without_rows() -> None:
    statistics = ExternalTransitionEvidenceStatistics(
        2,
        bin_count=8,
        error_scale=0.1,
    )
    prediction = torch.zeros(4, 2)
    observed = torch.tensor([[0.0, 0.0], [0.01, 0.0], [0.5, 0.0], [0.6, 0.0]])
    outcomes = torch.tensor([1.0, 1.0, 0.0, 0.0])
    statistics.observe(prediction, observed, outcomes, torch.ones(4))

    assert int(statistics.observation_count) == 4
    assert torch.isfinite(statistics(prediction, observed)).all()
    assert torch.isfinite(statistics.loss(prediction, observed, outcomes))
    payload = statistics.payload()
    assert "observations" not in payload
    restored = ExternalTransitionEvidenceStatistics.from_payload(payload)
    assert restored.digest() == statistics.digest()
    assert torch.equal(
        restored(prediction, observed),
        statistics(prediction, observed),
    )


def test_contextual_evidence_calibration_isolated_and_persistent() -> None:
    evaluator = ExternalTransitionEvidenceEvaluator(2, hidden_width=8)
    calibrator = ExternalContextualEvidenceCalibrator(
        evaluator,
        3,
        prior_strength=0.0,
    )
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    source_index = calibrator.ensure_context(source)
    target_index = calibrator.ensure_context(target)
    assert (source_index, target_index) == (0, 1)

    with torch.no_grad():
        calibrator.calibrators[target_index].bias.fill_(2.0)
    prediction = torch.zeros(2, 2)
    observed = torch.ones(2, 2)
    contexts = torch.stack((source, target))
    before = calibrator(prediction, observed, torch.ones(2), contexts)
    assert before[1] > before[0]

    source_digest = calibrator.calibrators[source_index].digest()
    payload = calibrator.payload()
    restored = ExternalContextualEvidenceCalibrator.from_payload(
        payload,
        evaluator=evaluator,
    )
    assert restored.context_count == 2
    assert torch.allclose(
        restored(prediction, observed, torch.ones(2), contexts), before
    )
    assert restored.calibrators[source_index].digest() == source_digest


def test_online_resolver_passes_candidate_context_to_contextual_calibrator() -> None:
    memory = ExternalTransitionMemory(1, 1, context_width=3)
    resolver = ExternalOnlineContextAddressResolver(
        3,
        address_seed=1207,
        admission_observations=2,
    )
    row = ExternalTransitionObservation(
        state=torch.tensor([[0.0]]),
        intention=torch.tensor([[1.0]]),
        next_state=torch.tensor([[1.0]]),
    )
    stream_a = torch.tensor([1.0, 0.0, 0.0])
    stream_b = torch.tensor([0.0, 1.0, 0.0])
    resolver.observe(row, stream_a, memory)
    admitted = resolver.observe(row, stream_a, memory)
    assert admitted.status == "admitted"

    evaluator = ExternalTransitionEvidenceEvaluator(1, hidden_width=8)
    calibrator = ExternalContextualEvidenceCalibrator(evaluator, 3)
    address = admitted.context
    assert address is not None
    slot = calibrator.ensure_context(address)
    with torch.no_grad():
        calibrator.calibrators[slot].bias.fill_(10.0)
    resolver.evidence_evaluator = calibrator
    reused = resolver.observe(row, stream_b, memory)
    assert reused.status == "reused"
    assert reused.committed_observations == 0
    assert memory.record_count == 1


def test_signed_entry_value_factorizes_salience_and_polarity() -> None:
    torch.manual_seed(1208)
    model = ExternalSignedEntryValueModel(3, 2, hidden_width=8)
    state = torch.randn(5, 3)
    entry = torch.randn(5, 2)
    salience = model.state_salience(state)
    polarity = model.entry_polarity(entry)
    logits = model(state, entry)

    assert bool((salience > 0.0).all())
    assert torch.allclose(model.state_salience(state), salience)
    assert torch.allclose(model.entry_polarity(-entry), -polarity)
    assert torch.allclose(model(state, -entry), -logits, atol=1e-6, rtol=1e-6)

    payload = model.state_payload()
    restored = ExternalSignedEntryValueModel.from_payload(payload)
    assert restored.digest() == model.digest()
    assert torch.allclose(restored(state, entry), logits)
    corrupt = dict(payload)
    corrupt["state"] = dict(payload["state"])
    corrupt["state"]["entry_projection.weight"] = (
        corrupt["state"]["entry_projection.weight"].clone()
    )
    corrupt["state"]["entry_projection.weight"][0, 0] += 0.1
    with pytest.raises(ValueError, match="checksum"):
        ExternalSignedEntryValueModel.from_payload(corrupt)
