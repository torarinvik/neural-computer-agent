from __future__ import annotations

import pytest
import torch

from neural_computer import (
    ExternalAffineTransitionStatistics,
    ExternalOnlineTransitionContextRouter,
    ExternalRandomFeatureTransitionStatistics,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionModelLifetimePolicy,
    ExternalTransitionObservation,
)


def _affine_observation(rows: int = 8) -> ExternalTransitionObservation:
    state = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) / 7.0
    intention = torch.arange(rows, dtype=torch.float32).reshape(rows, 1) / 5.0
    features = torch.cat((state, intention, torch.ones(rows, 1)), dim=-1)
    weights = torch.tensor(
        [[1.0, 0.2], [-0.3, 0.8], [0.7, -1.1], [0.4, -0.6]]
    )
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=features @ weights,
        confidence=torch.ones(rows),
    )


class _DeterministicEvidenceGate(torch.nn.Module):
    """Test-only replaceable evaluator with an opaque factual boundary."""

    def __init__(self, state_width: int) -> None:
        super().__init__()
        self.state_width = state_width

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": "test.deterministic-evidence-gate.v1",
            "state_width": self.state_width,
        }

    def forward(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        hit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hit
        error = (prediction - observed).square().mean(dim=-1)
        return 8.0 - 100.0 * error


def test_lifetime_policy_is_permutation_equivariant_and_masks_protected_slots() -> None:
    torch.manual_seed(1501)
    policy = ExternalTransitionModelLifetimePolicy(4, hidden_width=8)
    contexts = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    slot_ids = (10, 20, 30)
    usage = torch.tensor([1.0, 4.0, 2.0])
    age = torch.tensor([3.0, 1.0, 8.0])
    error = torch.tensor([0.2, 0.4, 0.1])
    protected = torch.tensor([False, True, False])

    proposal = policy.propose(contexts, slot_ids, usage, age, error, protected)
    permutation = torch.tensor([2, 0, 1])
    permuted = policy.propose(
        contexts.index_select(0, permutation),
        tuple(slot_ids[index] for index in permutation.tolist()),
        usage.index_select(0, permutation),
        age.index_select(0, permutation),
        error.index_select(0, permutation),
        protected.index_select(0, permutation),
    )

    original_scores = dict(zip(proposal.eligible_slot_ids, proposal.scores.tolist()))
    permuted_scores = dict(zip(permuted.eligible_slot_ids, permuted.scores.tolist()))
    assert original_scores.keys() == permuted_scores.keys()
    for slot_id, score in original_scores.items():
        assert score == pytest.approx(permuted_scores[slot_id])
    assert proposal.selected_slot_id != 20
    all_protected = policy.propose(
        contexts,
        slot_ids,
        usage,
        age,
        error,
        torch.ones(3, dtype=torch.bool),
    )
    assert all_protected.selected_slot_id is None


def test_lifetime_policy_learns_one_verifier_bit_and_persists_exactly() -> None:
    torch.manual_seed(1502)
    policy = ExternalTransitionModelLifetimePolicy(3, hidden_width=8)
    contexts = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    before = policy.digest()
    loss = policy.adaptation_step(
        contexts,
        (0, 1),
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 1.0]),
        torch.tensor([0.1, 0.2]),
        selected_slot_id=1,
        verifier_accepted=False,
    )
    assert loss > 0.0
    assert policy.digest() != before
    restored = ExternalTransitionModelLifetimePolicy.from_payload(
        policy.state_payload()
    )
    assert restored.digest() == policy.digest()
    assert torch.equal(
        restored(
            contexts,
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 1.0]),
            torch.tensor([0.1, 0.2]),
        ),
        policy(
            contexts,
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 1.0]),
            torch.tensor([0.1, 0.2]),
        ),
    )


def test_lifetime_policy_transaction_is_verifier_gated_and_uses_stable_ids() -> None:
    torch.manual_seed(1503)
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8, capacity=3)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 1.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 0.0, 1.0]))
    policy = ExternalTransitionModelLifetimePolicy(3, hidden_width=8)
    metadata = {
        "usage": torch.tensor([4.0, 1.0, 3.0]),
        "age": torch.tensor([1.0, 4.0, 2.0]),
        "prediction_error": torch.tensor([0.2, 0.4, 0.1]),
    }
    bank_digest = bank.digest()
    proposal, rejected = policy.evict_verified(
        bank,
        **metadata,
        protected=torch.tensor([True, False, True]),
        retention_probe=lambda _candidate: False,
    )
    assert proposal.selected_slot_id == 1
    assert rejected is not None and not rejected.accepted
    assert bank.digest() == bank_digest

    proposal, accepted = policy.evict_verified(
        bank,
        **metadata,
        protected=torch.tensor([True, False, True]),
        retention_probe=lambda candidate: candidate.slot_ids in {(0, 1, 2), (0, 2)},
    )
    assert proposal.selected_slot_id == 1
    assert accepted is not None and accepted.accepted
    assert accepted.evicted_slot_id == 1
    assert bank.slot_ids == (0, 2)


def test_bank_owns_lifetime_telemetry_and_persists_it_through_logical_eviction() -> None:
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8, capacity=3)
    for context in (
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    ):
        bank.ensure_context(context)

    bank.record_lifetime_observation(0, 0.8)
    bank.record_lifetime_observation(0, 0.4)
    bank.record_lifetime_observation(1, 0.2)
    telemetry = bank.lifetime_telemetry()
    assert telemetry.slot_ids == (0, 1, 2)
    assert telemetry.usage.tolist() == [2.0, 1.0, 0.0]
    assert telemetry.age.tolist() == [1.0, 0.0, 3.0]
    assert telemetry.prediction_error.tolist() == pytest.approx([0.7, 0.2, 0.0])

    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    restored_telemetry = restored.lifetime_telemetry()
    assert restored_telemetry.logical_clock == telemetry.logical_clock
    assert torch.equal(restored_telemetry.usage, telemetry.usage)
    assert torch.equal(restored_telemetry.age, telemetry.age)
    assert torch.equal(restored_telemetry.prediction_error, telemetry.prediction_error)

    policy = ExternalTransitionModelLifetimePolicy(3, hidden_width=8)
    proposal, receipt = policy.evict_from_bank_verified(
        bank,
        protected=torch.tensor([True, False, True]),
        retention_probe=lambda candidate: candidate.slot_ids in {(0, 1, 2), (0, 2)},
    )
    assert proposal.selected_slot_id == 1
    assert receipt is not None and receipt.accepted
    assert bank.slot_ids == (0, 2)
    surviving = bank.lifetime_telemetry()
    assert surviving.slot_ids == (0, 2)
    assert surviving.usage.tolist() == [2.0, 0.0]


def test_bank_adaptation_updates_lifetime_telemetry_without_replay() -> None:
    bank = ExternalTransitionModelBank(
        2,
        1,
        3,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
    )
    context = torch.tensor([1.0, 0.0, 0.0])
    bank.ensure_context(context)
    observation = _affine_observation(rows=4)
    bank.adaptation_step(
        observation,
        context.unsqueeze(0).expand(observation.state.shape[0], -1),
        None,
    )
    telemetry = bank.lifetime_telemetry()
    assert telemetry.usage.tolist() == [4.0]
    assert telemetry.logical_clock == 1
    assert float(telemetry.prediction_error[0]) > 0.0


def test_lifetime_telemetry_batches_share_one_logical_timestamp() -> None:
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 1.0, 0.0]))
    bank.record_lifetime_observations((0, 1), (0.1, 0.2))
    telemetry = bank.lifetime_telemetry()
    assert telemetry.logical_clock == 1
    assert telemetry.usage.tolist() == [1.0, 1.0]
    assert telemetry.age.tolist() == [0.0, 0.0]
    bank.record_lifetime_observation(0, 0.3)
    assert bank.lifetime_telemetry().age.tolist() == [0.0, 1.0]


def test_query_conditioned_eviction_preserves_aligned_slot_under_capacity_pressure() -> None:
    torch.manual_seed(1603)
    bank = ExternalTransitionModelBank(2, 1, 3, hidden_width=8, capacity=3)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 1.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 0.0, 1.0]))
    policy = ExternalTransitionModelLifetimePolicy(3, hidden_width=8)
    query = torch.tensor([0.0, 1.0, 0.0])
    proposal = policy.propose_from_query(
        bank,
        query,
        torch.zeros(3, dtype=torch.bool),
        relevance_weight=100.0,
    )
    assert proposal.selected_slot_id != 1
    proposal, receipt = policy.evict_from_bank_query_verified(
        bank,
        query,
        torch.zeros(3, dtype=torch.bool),
        lambda candidate: candidate.slot_ids in {(0, 1, 2), tuple(
            slot_id for slot_id in (0, 1, 2) if slot_id != proposal.selected_slot_id
        )},
        relevance_weight=100.0,
        update=False,
    )
    assert receipt is not None and receipt.accepted
    assert 1 in bank.slot_ids
def test_affine_statistics_is_a_bank_fast_path_without_optimizer_replay() -> None:
    observation = _affine_observation()
    bank = ExternalTransitionModelBank(
        2,
        1,
        3,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=1,
    )
    context = torch.tensor([1.0, 0.0, 0.0])
    bank.ensure_context(context)
    loss_before = bank.adaptation_step(
        observation,
        context.unsqueeze(0).expand(observation.state.shape[0], -1),
        None,
    )
    assert loss_before > 0.0
    assert int(bank.models[0].sample_count) == observation.state.shape[0]
    assert float(bank.loss(observation, context.unsqueeze(0).expand(8, -1))) < 1e-7

    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    assert restored.configuration()["model_family"] == "affine_sufficient_statistics_v1"
    assert restored.content_digest() == bank.content_digest()


def test_nonlinear_bank_can_update_through_memory_boundary_without_optimizer() -> None:
    torch.manual_seed(1302)
    bank = ExternalTransitionModelBank(
        2,
        1,
        3,
        adaptation_learning_rate=0.03,
    )
    context = torch.tensor([1.0, 0.0, 0.0])
    bank.ensure_context(context)
    observation = ExternalTransitionObservation(
        state=torch.randn(4, 2),
        intention=torch.randn(4, 1),
        next_state=torch.randn(4, 2),
    )
    before = bank.models[0].digest()
    bank.adaptation_step(
        observation,
        context.unsqueeze(0).expand(4, -1),
        None,
    )
    assert bank.models[0].digest() != before


def test_random_feature_statistics_learns_nonlinear_transitions_one_pass() -> None:
    for seed in (1401, 1402):
        torch.manual_seed(seed)
        model = ExternalRandomFeatureTransitionStatistics(
            2,
            1,
            feature_width=128,
            ridge=1e-4,
            seed=17,
        )
        state = torch.rand(128, 2) * 2.0 - 1.0
        intention = torch.rand(128, 1) * 2.0 - 1.0
        next_state = torch.cat(
            (
                torch.sin(2.0 * state[:, 0:1] + intention),
                state[:, 0:1] * state[:, 1:2] + intention.square(),
            ),
            dim=-1,
        )
        train = ExternalTransitionObservation(
            state=state[:64],
            intention=intention[:64],
            next_state=next_state[:64],
        )
        heldout = ExternalTransitionObservation(
            state=state[64:],
            intention=intention[64:],
            next_state=next_state[64:],
        )
        model.observe(train)
        assert int(model.sample_count) == 64
        assert float(model.loss(heldout)) < 0.02
        restored = ExternalRandomFeatureTransitionStatistics.from_payload(
            model.state_payload()
        )
        assert restored.digest() == model.digest()
        assert torch.allclose(restored(state[64:], intention[64:]), model(state[64:], intention[64:]))


def test_random_feature_growth_preserves_old_predictions_without_replay() -> None:
    torch.manual_seed(1403)
    model = ExternalRandomFeatureTransitionStatistics(
        2,
        1,
        feature_width=16,
        ridge=1e-4,
        seed=19,
    )
    state = torch.rand(32, 2) * 2.0 - 1.0
    intention = torch.rand(32, 1) * 2.0 - 1.0
    next_state = torch.cat(
        (
            torch.sin(2.0 * state[:, 0:1] + intention),
            state[:, 0:1] * state[:, 1:2] + intention.square(),
        ),
        dim=-1,
    )
    model.observe(ExternalTransitionObservation(state, intention, next_state))
    probe_state = torch.rand(16, 2) * 2.0 - 1.0
    probe_intention = torch.rand(16, 1) * 2.0 - 1.0
    before = model(probe_state, probe_intention).detach().clone()
    sample_count = int(model.sample_count)
    receipt = model.grow_features_verified(
        32,
        lambda candidate: torch.allclose(
            candidate(probe_state, probe_intention),
            before,
            atol=2e-5,
            rtol=0.0,
        ),
    )
    assert receipt.accepted
    assert model.feature_width == 32
    assert int(model.sample_count) == sample_count
    assert torch.allclose(
        model(probe_state, probe_intention),
        before,
        atol=2e-5,
        rtol=0.0,
    )
    digest = model.digest()
    rejected = model.grow_features_verified(64, lambda _candidate: False)
    assert not rejected.accepted
    assert model.feature_width == 32
    assert model.digest() == digest
    restored = ExternalRandomFeatureTransitionStatistics.from_payload(
        model.state_payload()
    )
    assert restored.digest() == model.digest()


def test_router_stages_and_promotes_affine_candidate_without_optimizer() -> None:
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=1,
    )
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=4,
        max_contexts=1,
        defer_admission=True,
    )
    observation = _affine_observation(4)
    rows = [
        ExternalTransitionObservation(
            state=observation.state[row : row + 1],
            intention=observation.intention[row : row + 1],
            next_state=observation.next_state[row : row + 1],
            confidence=torch.ones(1),
        )
        for row in range(4)
    ]
    for row in rows[:3]:
        assert router.observe(row).status == "pending"
    staged = router.observe(rows[3])
    assert staged.status == "staged"
    assert router.adaptation_step(staged, None) > 0.0
    assert int(router.provisional_model_at(0).sample_count) == 4
    restored_router = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert int(restored_router.provisional_model_at(0).sample_count) == 4

    heldout = _affine_observation(4)
    receipt = router.promote_staged_candidate(
        heldout,
        lambda candidate: candidate.context_count == 1,
        prediction_tolerance=1e-7,
    )
    assert receipt.accepted
    assert router.bank.models[0].sample_count.item() == 4


def test_bank_selects_smallest_verified_model_family_without_mutating_candidates() -> None:
    torch.manual_seed(1303)
    observation = _affine_observation(12)
    heldout = _affine_observation(4)
    bank = ExternalTransitionModelBank(2, 1, 3)
    affine = bank.new_model("affine_sufficient_statistics_v1")
    for row in range(observation.state.shape[0]):
        affine.observe(
            ExternalTransitionObservation(
                state=observation.state[row : row + 1],
                intention=observation.intention[row : row + 1],
                next_state=observation.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
    nonlinear = bank.new_model("nonlinear_mlp_v1")
    affine_digest = affine.digest()
    nonlinear_digest = nonlinear.digest()

    selection = bank.select_model_family_verified(
        {
            "affine_sufficient_statistics_v1": affine,
            "nonlinear_mlp_v1": nonlinear,
        },
        heldout,
        prediction_tolerance=1e-6,
    )
    assert selection.accepted
    assert selection.selected_family == "affine_sufficient_statistics_v1"
    assert affine.digest() == affine_digest
    assert nonlinear.digest() == nonlinear_digest


def test_mixed_bank_keeps_affine_and_nonlinear_slots_independently() -> None:
    torch.manual_seed(1304)
    bank = ExternalTransitionModelBank(
        2,
        1,
        3,
        model_family="mixed_verified_v1",
        capacity=2,
    )
    affine_index = bank.ensure_context(
        torch.tensor([1.0, 0.0, 0.0]),
        model_family="affine_sufficient_statistics_v1",
    )
    nonlinear_index = bank.ensure_context(
        torch.tensor([0.0, 1.0, 0.0]),
        model_family="nonlinear_mlp_v1",
    )
    affine_observation = _affine_observation(8)
    nonlinear_observation = ExternalTransitionObservation(
        state=torch.randn(4, 2),
        intention=torch.randn(4, 1),
        next_state=torch.randn(4, 2),
    )
    observation = ExternalTransitionObservation(
        state=torch.cat((affine_observation.state, nonlinear_observation.state)),
        intention=torch.cat(
            (affine_observation.intention, nonlinear_observation.intention)
        ),
        next_state=torch.cat(
            (affine_observation.next_state, nonlinear_observation.next_state)
        ),
        confidence=torch.ones(12),
    )
    contexts = torch.cat(
        (
            torch.tensor([1.0, 0.0, 0.0]).expand(8, -1),
            torch.tensor([0.0, 1.0, 0.0]).expand(4, -1),
        )
    )
    optimizer = torch.optim.Adam(bank.models[nonlinear_index].parameters(), lr=0.01)
    bank.adaptation_step(observation, contexts, {nonlinear_index: optimizer})
    assert bank.model_family_at(affine_index) == "affine_sufficient_statistics_v1"
    assert bank.model_family_at(nonlinear_index) == "nonlinear_mlp_v1"
    assert int(bank.models[affine_index].sample_count) == 8

    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    assert restored.model_family == "mixed_verified_v1"
    assert restored.model_family_at(affine_index) == "affine_sufficient_statistics_v1"
    assert restored.model_family_at(nonlinear_index) == "nonlinear_mlp_v1"
    assert restored.content_digest() == bank.content_digest()


def test_replay_free_statistics_slots_do_not_copy_prior_regime_evidence() -> None:
    bank = ExternalTransitionModelBank(
        2,
        1,
        3,
        model_family="random_feature_sufficient_statistics_v1",
        capacity=2,
    )
    source = bank.ensure_context(torch.tensor([1.0, 0.0, 0.0]))
    observation = _affine_observation(1)
    bank.adaptation_step(
        observation,
        bank.context_at(source).unsqueeze(0),
        None,
    )
    target = bank.ensure_context(
        torch.tensor([0.0, 1.0, 0.0]),
        initialize_from=source,
    )
    assert int(bank.models[source].sample_count) == 1
    assert int(bank.models[target].sample_count) == 0


def test_mixed_router_adapts_both_families_and_promotes_verified_winner() -> None:
    torch.manual_seed(1305)
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family="mixed_verified_v1",
        affine_ridge=1e-7,
        capacity=1,
    )
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=8,
        max_contexts=1,
        defer_admission=True,
    )
    observation = _affine_observation(8)
    rows = [
        ExternalTransitionObservation(
            state=observation.state[row : row + 1],
            intention=observation.intention[row : row + 1],
            next_state=observation.next_state[row : row + 1],
            confidence=torch.ones(1),
        )
        for row in range(8)
    ]
    for row in rows[:-1]:
        assert router.observe(row).status == "pending"
    staged = router.observe(rows[-1])
    assert staged.status == "staged"
    assert set(router._provisional_candidates[0].models()) == {
        "nonlinear_mlp_v1",
        "affine_sufficient_statistics_v1",
        "random_feature_sufficient_statistics_v1",
    }
    router.adaptation_step(staged, None)
    assert int(
        router._provisional_candidates[0].alternatives[
            "affine_sufficient_statistics_v1"
        ].sample_count
    ) == 8

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert set(restored._provisional_candidates[0].models()) == {
        "nonlinear_mlp_v1",
        "affine_sufficient_statistics_v1",
        "random_feature_sufficient_statistics_v1",
    }
    receipt = router.promote_staged_candidate(
        _affine_observation(4),
        lambda candidate: candidate.context_count == 1
        and candidate.model_family_at(0) == "affine_sufficient_statistics_v1",
        prediction_tolerance=1e-6,
    )
    assert receipt.accepted
    assert router.bank.model_family_at(0) == "affine_sufficient_statistics_v1"


def test_streaming_statistics_candidate_does_not_retain_raw_evidence() -> None:
    torch.manual_seed(1306)
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=2,
    )
    source_context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    source_index = bank.ensure_context(source_context)
    source = _affine_observation(8)
    bank.adaptation_step(
        source,
        source_context.unsqueeze(0).expand(source.state.shape[0], -1),
        None,
    )
    target = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state * 1.7,
        confidence=source.confidence,
    )
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=2,
        max_contexts=2,
        defer_admission=True,
        provisional_evidence_policy="streaming_statistics",
    )
    for row in range(target.state.shape[0]):
        result = router.observe(
            ExternalTransitionObservation(
                state=target.state[row : row + 1],
                intention=target.intention[row : row + 1],
                next_state=target.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)

    assert router.provisional_candidate_count == 1
    assert router.provisional_evidence_count(0) == target.state.shape[0]
    assert router._provisional_observations == []
    payload = router.state_payload()
    assert payload["configuration"]["provisional_evidence_policy"] == (
        "streaming_statistics"
    )
    assert payload["provisional_observations"] == []
    assert payload["provisional_candidates"][0]["observations"] == []
    assert payload["provisional_candidates"][0]["evidence_count"] == 8

    restored = ExternalOnlineTransitionContextRouter.from_payload(payload)
    assert restored.provisional_evidence_count(0) == 8
    assert restored._provisional_observations == []
    assert restored.provisional_model is not None
    assert int(restored.provisional_model.sample_count) == 8
    assert restored.state_payload()["bank"]["sha256"] == bank.digest()
    assert source_index == 0


def test_streaming_statistics_isolates_two_interleaved_candidates() -> None:
    torch.manual_seed(1307)
    affine_family = "affine_sufficient_statistics_v1"
    random_feature_family = "random_feature_sufficient_statistics_v1"
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family="mixed_verified_v1",
        affine_ridge=1e-7,
        random_feature_width=64,
        random_feature_seed=17,
        capacity=3,
    )
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-6,
        match_margin=1e-4,
        continuation_tolerance=1e-6,
        provisional_continuation_tolerance=0.05,
        provisional_match_margin=0.05,
        admission_observations=4,
        max_contexts=3,
        defer_admission=True,
        candidate_model_families=(affine_family, random_feature_family),
        provisional_evidence_policy="streaming_statistics",
    )
    source = _affine_observation(64)
    for row in range(source.state.shape[0]):
        result = router.observe(
            ExternalTransitionObservation(
                state=source.state[row : row + 1],
                intention=source.intention[row : row + 1],
                next_state=source.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)
    source_receipt = router.promote_staged_candidate(
        source,
        lambda candidate: candidate.context_count == 1,
        prediction_tolerance=1e-6,
    )
    assert source_receipt.accepted
    committed_source_context = router.bank.context_at(0)

    target_a = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state * 2.0,
        confidence=source.confidence,
    )
    target_b = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state * -1.0,
        confidence=source.confidence,
    )
    for chunk in range(16):
        for observation in (target_a, target_b):
            for row in range(chunk * 4, chunk * 4 + 4):
                result = router.observe(
                    ExternalTransitionObservation(
                        state=observation.state[row : row + 1],
                        intention=observation.intention[row : row + 1],
                        next_state=observation.next_state[row : row + 1],
                        confidence=torch.ones(1),
                    )
                )
                if result.status == "staged":
                    router.adaptation_step(result, None, replay_evidence=False)

    assert router.provisional_candidate_count == 2
    assert [router.provisional_evidence_count(index) for index in range(2)] == [64, 64]
    assert all(
        not candidate.observations
        for candidate in router._provisional_candidates
    )
    candidate_contexts = [
        router.provisional_context_at(index) for index in range(2)
    ]
    payload = router.state_payload()
    assert all(
        not candidate["observations"]
        for candidate in payload["provisional_candidates"]
    )
    router.provisional_continuation_tolerance = 100.0
    ambiguous = None
    midpoint_state = target_a.state[:4]
    midpoint_intention = target_a.intention[:4]
    candidate_predictions = [
        candidate.model(midpoint_state, midpoint_intention)
        for candidate in router._provisional_candidates
    ]
    midpoint = ExternalTransitionObservation(
        state=midpoint_state,
        intention=midpoint_intention,
        next_state=sum(candidate_predictions) / len(candidate_predictions),
        confidence=torch.ones(4),
    )
    for row in range(4):
        ambiguous = router.observe(
            ExternalTransitionObservation(
                state=midpoint.state[row : row + 1],
                intention=midpoint.intention[row : row + 1],
                next_state=midpoint.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
    assert ambiguous is not None and ambiguous.status == "ambiguous"
    assert [router.provisional_evidence_count(index) for index in range(2)] == [64, 64]

    def retained(candidate: ExternalTransitionModelBank) -> bool:
        return (
            candidate.context_count == 2
            and float(
                    candidate.loss(
                        source,
                        committed_source_context.unsqueeze(0).expand(
                            source.state.shape[0], -1
                        ),
                    )
            ) < 1e-6
        )

    first = router.promote_staged_candidate(
        target_a,
        retained,
        prediction_tolerance=1e-6,
        candidate_index=0,
    )
    assert first.accepted
    second = router.promote_staged_candidate(
        target_b,
        lambda candidate: candidate.context_count == 3,
        prediction_tolerance=1e-6,
        candidate_index=0,
    )
    assert second.accepted
    assert router.bank.context_count == 3
    assert router.bank.model_family_at(1) == affine_family
    assert router.bank.model_family_at(2) == affine_family
    assert router.bank.context_at(1).shape == candidate_contexts[0].shape
    assert router.bank.context_at(2).shape == candidate_contexts[1].shape
    assert random_feature_family in router.configuration()["candidate_model_families"]


def test_ambiguous_streaming_evidence_is_quarantined_resolved_and_persisted() -> None:
    torch.manual_seed(1308)
    affine_family = "affine_sufficient_statistics_v1"
    random_feature_family = "random_feature_sufficient_statistics_v1"
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family="mixed_verified_v1",
        affine_ridge=1e-7,
        random_feature_width=64,
        random_feature_seed=19,
        capacity=3,
    )
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-6,
        match_margin=1e-4,
        continuation_tolerance=1e-6,
        provisional_continuation_tolerance=0.05,
        provisional_match_margin=0.05,
        admission_observations=2,
        max_contexts=3,
        defer_admission=True,
        candidate_model_families=(affine_family, random_feature_family),
        provisional_evidence_policy="streaming_statistics",
        ambiguous_evidence_policy="quarantine",
        quarantine_capacity=2,
    )
    source = _affine_observation(16)
    for row in range(source.state.shape[0]):
        result = router.observe(
            ExternalTransitionObservation(
                state=source.state[row : row + 1],
                intention=source.intention[row : row + 1],
                next_state=source.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)
    source_receipt = router.promote_staged_candidate(
        source,
        lambda candidate: candidate.context_count == 1,
        prediction_tolerance=1e-6,
    )
    assert source_receipt.accepted
    committed_source_context = router.bank.context_at(0)

    target_a = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state * 2.0,
        confidence=source.confidence,
    )
    target_b = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state * -1.0,
        confidence=source.confidence,
    )
    for observation in (target_a, target_b):
        for row in range(target_a.state.shape[0]):
            result = router.observe(
                ExternalTransitionObservation(
                    state=observation.state[row : row + 1],
                    intention=observation.intention[row : row + 1],
                    next_state=observation.next_state[row : row + 1],
                    confidence=torch.ones(1),
                )
            )
            if result.status == "staged":
                router.adaptation_step(result, None, replay_evidence=False)

    assert router.provisional_candidate_count == 2
    assert [router.provisional_evidence_count(index) for index in range(2)] == [
        16,
        16,
    ]
    router.provisional_match_margin = 1000.0
    provisional_tolerance = router.provisional_continuation_tolerance
    router.provisional_continuation_tolerance = 1e9
    ambiguous = None
    midpoint_state = target_a.state[:2]
    midpoint_intention = target_a.intention[:2]
    midpoint_predictions = [
        candidate.model(midpoint_state, midpoint_intention)
        for candidate in router._provisional_candidates
    ]
    midpoint = ExternalTransitionObservation(
        state=midpoint_state,
        intention=midpoint_intention,
        next_state=sum(midpoint_predictions) / len(midpoint_predictions),
        confidence=torch.ones(2),
    )
    for row in range(2):
        ambiguous = router.observe(
            ExternalTransitionObservation(
                state=midpoint.state[row : row + 1],
                intention=midpoint.intention[row : row + 1],
                next_state=midpoint.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
    router.provisional_continuation_tolerance = provisional_tolerance
    assert ambiguous is not None and ambiguous.status == "ambiguous"
    assert router.quarantined_observations == 2
    rejected = router.promote_staged_candidate(
        target_a,
        lambda candidate: candidate.context_count == 2,
        prediction_tolerance=1e-6,
        candidate_index=0,
    )
    assert not rejected.accepted
    assert "quarantined" in rejected.reason

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert restored.quarantined_observations == 2
    assert restored.provisional_evidence_count(0) == 16
    restored.provisional_match_margin = 0.05
    resolved = None
    for row in range(2, 4):
        resolved = restored.observe(
            ExternalTransitionObservation(
                state=target_a.state[row : row + 1],
                intention=target_a.intention[row : row + 1],
                next_state=target_a.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
        if resolved.status == "staged":
            restored.adaptation_step(resolved, None, replay_evidence=False)
    assert resolved is not None and resolved.status == "staged"
    assert restored.quarantined_observations == 0
    assert restored.provisional_evidence_count(0) == 20
    assert restored._provisional_candidates[0].deferred_observations == []
    assert all(
        not candidate.observations
        for candidate in restored._provisional_candidates
    )

    def retained(candidate: ExternalTransitionModelBank) -> bool:
        return (
            candidate.context_count == 2
            and float(
                candidate.loss(
                    source,
                    committed_source_context.unsqueeze(0).expand(
                        source.state.shape[0], -1
                    ),
                )
            )
            < 1e-6
        )

    promoted = restored.promote_staged_candidate(
        target_a,
        retained,
        prediction_tolerance=0.02,
        candidate_index=0,
    )
    assert promoted.accepted
    payload = restored.state_payload()
    assert payload["ambiguous_quarantine"] == []
    assert payload["configuration"]["ambiguous_evidence_policy"] == "quarantine"
    assert random_feature_family in restored.configuration()["candidate_model_families"]


def test_streaming_router_uses_external_evidence_gate_for_corruption() -> None:
    torch.manual_seed(1309)
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=2,
    )
    router = ExternalOnlineTransitionContextRouter(
        bank,
        ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4),
        match_tolerance=1e-6,
        match_margin=1e-4,
        continuation_tolerance=1e-6,
        provisional_continuation_tolerance=0.05,
        admission_observations=2,
        max_contexts=2,
        defer_admission=True,
        provisional_evidence_policy="streaming_statistics",
        evidence_evaluator=_DeterministicEvidenceGate(2),
        evidence_threshold=0.9,
        evidence_gate_min_evidence=8,
    )
    source = _affine_observation(8)
    for row in range(source.state.shape[0]):
        result = router.observe(
            ExternalTransitionObservation(
                state=source.state[row : row + 1],
                intention=source.intention[row : row + 1],
                next_state=source.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)
    assert router.promote_staged_candidate(
        source,
        lambda candidate: candidate.context_count == 1,
        prediction_tolerance=1e-6,
    ).accepted

    target = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state * 2.0,
        confidence=source.confidence,
    )
    for row in range(target.state.shape[0]):
        result = router.observe(
            ExternalTransitionObservation(
                state=target.state[row : row + 1],
                intention=target.intention[row : row + 1],
                next_state=target.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)
    assert router.provisional_candidate_count == 1
    before = router.provisional_evidence_count(0)

    corrupted = ExternalTransitionObservation(
        state=target.state[:2],
        intention=target.intention[:2],
        next_state=target.next_state[:2] + 3.0,
        confidence=torch.ones(2),
    )
    result = None
    for row in range(2):
        result = router.observe(
            ExternalTransitionObservation(
                state=corrupted.state[row : row + 1],
                intention=corrupted.intention[row : row + 1],
                next_state=corrupted.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
    assert result is not None and result.status == "capacity"
    assert router.provisional_evidence_count(0) == before
    payload = router.state_payload()
    with pytest.raises(ValueError, match="external evidence evaluator"):
        ExternalOnlineTransitionContextRouter.from_payload(payload)
    restored = ExternalOnlineTransitionContextRouter.from_payload(
        payload,
        evidence_evaluator=router.evidence_evaluator,
    )
    assert restored.configuration()["evidence_threshold"] == pytest.approx(0.9)
    assert restored.configuration()["evidence_gate_min_evidence"] == 8
    assert restored.provisional_evidence_count(0) == before


def test_affine_transition_statistics_learns_and_persists_one_pass() -> None:
    torch.manual_seed(1301)
    model = ExternalAffineTransitionStatistics(2, 1, ridge=1e-7)
    state = torch.randn(8, 2)
    intention = torch.randn(8, 1)
    features = torch.cat((state, intention, torch.ones(8, 1)), dim=-1)
    true_weights = torch.tensor(
        [
            [1.0, 0.2],
            [-0.3, 0.8],
            [0.7, -1.1],
            [0.4, -0.6],
        ]
    )
    next_state = features @ true_weights
    observation = ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=next_state,
        confidence=torch.ones(8),
    )

    for row in range(state.shape[0]):
        model.observe(
            ExternalTransitionObservation(
                state=observation.state[row : row + 1],
                intention=observation.intention[row : row + 1],
                next_state=observation.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
    )

    assert int(model.sample_count) == 8
    assert float(model.loss(observation)) < 1e-7
    heldout_state = torch.randn(4, 2)
    heldout_intention = torch.randn(4, 1)
    heldout_features = torch.cat(
        (heldout_state, heldout_intention, torch.ones(4, 1)), dim=-1
    )
    heldout = ExternalTransitionObservation(
        state=heldout_state,
        intention=heldout_intention,
        next_state=heldout_features @ true_weights,
    )
    assert float(model.loss(heldout)) < 1e-6
    restored = ExternalAffineTransitionStatistics.from_payload(model.state_payload())
    assert restored.digest() == model.digest()
    assert torch.allclose(
        restored(state, intention),
        model(state, intention),
        atol=1e-7,
        rtol=0.0,
    )


def test_affine_transition_statistics_rejects_checksum_corruption() -> None:
    model = ExternalAffineTransitionStatistics(1, 1)
    model.observe(
        ExternalTransitionObservation(
            state=torch.tensor([[1.0]]),
            intention=torch.tensor([[2.0]]),
            next_state=torch.tensor([[3.0]]),
        )
    )
    payload = model.state_payload()
    payload["state"]["normal_matrix"][0, 0] += 1.0

    try:
        ExternalAffineTransitionStatistics.from_payload(payload)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("expected affine statistics checksum rejection")
