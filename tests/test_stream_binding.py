from __future__ import annotations

import torch

from experiments.external_learned_binding_factual_consolidation.train import (
    run as run_factual_consolidation_pressure_test,
)
from experiments.external_learned_binding_factual_growth.train import (
    run as run_factual_growth_pressure_test,
)
from experiments.external_learned_stream_binding.train import run as run_pressure_test
from experiments.external_learned_stream_binding_factual_lifecycle.train import (
    run as run_factual_lifecycle_pressure_test,
)
from experiments.external_learned_stream_binding_lifecycle.train import (
    run as run_lifecycle_pressure_test,
)
from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    ExternalLearnedMultiStreamTransitionContextRouter,
    ExternalMultiStreamTransitionContextRouter,
    ExternalOnlineStreamBindingMemory,
    ExternalOnlineTransitionContextRouter,
    ExternalStreamBindingLifecyclePolicy,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)


def _fixture(
    seed: int = 501,
    stream_count: int = 3,
) -> tuple[ExternalTransitionContextEncoder, list[ExternalTransitionObservation]]:
    torch.manual_seed(seed)
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=16,
        context_width=4,
    )
    observations: list[ExternalTransitionObservation] = []
    for stream in range(stream_count):
        state = torch.randn(6, 2)
        state[:, 1] += stream * 5.0
        intention = torch.randn(6, 1)
        next_state = state + intention * torch.tensor([0.2 + stream, 1.0])
        observations.append(
            ExternalTransitionObservation(
                state,
                intention,
                next_state,
                torch.ones(6),
            )
        )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    for update in range(160):
        left: list[torch.Tensor] = []
        right: list[torch.Tensor] = []
        for stream, observation in enumerate(observations):
            left_index = (update + stream) % 6
            right_index = (update * 3 + stream + 1) % 6
            left.append(_row(observation, left_index, encoder))
            right.append(_row(observation, right_index, encoder))
        loss = encoder.contrastive_loss(torch.stack(left), torch.stack(right))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    encoder.eval()
    return encoder, observations


def _row(
    observation: ExternalTransitionObservation,
    index: int,
    encoder: ExternalTransitionContextEncoder | None = None,
) -> ExternalTransitionObservation | torch.Tensor:
    row = ExternalTransitionObservation(
        observation.state[index : index + 1],
        observation.intention[index : index + 1],
        observation.next_state[index : index + 1],
        observation.confidence[index : index + 1]
        if observation.confidence is not None
        else None,
    )
    return row if encoder is None else encoder.encode_observation(row)


def test_external_binding_learns_anonymous_tracks_and_persists() -> None:
    encoder, observations = _fixture()
    memory = ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=4,
        max_streams=3,
        match_tolerance=0.55,
        new_track_tolerance=0.7,
        match_margin=0.05,
    )
    assert all(not parameter.requires_grad for parameter in memory.encoder.parameters())
    assignments: dict[int, int] = {}
    results = []
    for step in range(6):
        for stream in range(3):
            result = memory.observe(
                _row(observations[stream], step),
                timestamp=step + stream * 0.1,
            )
            assert result.status in {"new", "matched"}
            assert result.track_id is not None
            previous = assignments.setdefault(stream, result.track_id)
            assert result.track_id == previous
            results.append(result)

    assert memory.stream_count == 3
    assert len(set(assignments.values())) == 3
    assert all(memory.track_state(track_id)["mean_delay"] is not None for track_id in memory.track_ids)

    memory.observe_verifier_outcome(results[-1], 1.0)
    memory.observe_verifier_outcome(results[-1], 0.0)
    assert memory.track_state(results[-1].track_id)["reliability"] == 0.5

    restored = ExternalOnlineStreamBindingMemory.from_payload(memory.state_payload())
    assert restored.digest() == memory.digest()
    assert restored.track_ids == memory.track_ids


def test_learned_binding_routes_without_caller_stream_keys() -> None:
    encoder, observations = _fixture(502)
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        capacity=3,
    )
    single = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=2,
        max_contexts=3,
        defer_admission=True,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
    )
    router = ExternalLearnedMultiStreamTransitionContextRouter(
        ExternalOnlineStreamBindingMemory(
            encoder,
            window_capacity=4,
            max_streams=3,
            match_tolerance=0.55,
            new_track_tolerance=0.7,
            match_margin=0.05,
        ),
        ExternalMultiStreamTransitionContextRouter(single, stream_key_width=4),
    )

    route_results = []
    for row_index in range(2):
        for stream in range(3):
            route_results.append(
                router.observe(
                    _row(observations[stream], row_index),
                    timestamp=row_index + stream * 0.1,
                )
            )

    assert all(result.routing is not None for result in route_results)
    assert router.stream_count == 3
    assert router.router.stream_count == 3
    assert all(result.binding.stream_key is not None for result in route_results)
    assert len({result.binding.track_id for result in route_results}) == 3

    restored = ExternalLearnedMultiStreamTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert restored.digest() == router.digest()

    corrupted = router.state_payload()
    corrupted["binding"]["tracks"][0]["stream_key"][0] += 0.01
    try:
        ExternalLearnedMultiStreamTransitionContextRouter.from_payload(corrupted)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("corrupted learned binding state was accepted")


def test_learned_binding_pressure_test_passes() -> None:
    report = run_pressure_test(2301)
    assert report["promoted"] is True
    assert report["gates"]["learned_assignment_consistent"] is True
    assert report["gates"]["learned_beats_fresh"] is True
    assert report["gates"]["binding_encoder_frozen"] is True


def test_open_set_arrival_is_quarantined_until_retention_safe_admission() -> None:
    encoder, observations = _fixture(503, stream_count=4)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    for update in range(180):
        left = []
        right = []
        for stream, observation in enumerate(observations):
            left.append(
                encoder.encode_observation(
                    _row(observation, (update + stream) % 6)
                )
            )
            right.append(
                encoder.encode_observation(
                    _row(observation, (update * 3 + stream + 1) % 6)
                )
            )
        loss = encoder.contrastive_loss(torch.stack(left), torch.stack(right))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    encoder.eval()
    memory = ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=4,
        max_streams=3,
        provisional_capacity=2,
        match_tolerance=0.55,
        new_track_tolerance=0.7,
        provisional_tolerance=0.7,
        match_margin=0.05,
    )
    for stream in range(3):
        assert memory.observe(_row(observations[stream], 0)).status == "new"

    provisional_results = [
        memory.observe(_row(observations[3], index), timestamp=10 + index)
        for index in range(6)
    ]
    assert all(result.status == "provisional" for result in provisional_results)
    provisional_id = provisional_results[0].provisional_id
    assert provisional_id is not None
    assert all(result.provisional_id == provisional_id for result in provisional_results)
    assert memory.track_ids == (0, 1, 2)
    assert memory.provisional_ids == (provisional_id,)

    rejected = memory.promote_provisional_track(provisional_id, lambda _: False)
    assert not rejected.accepted
    assert rejected.reason == "live_capacity_full"
    assert memory.provisional_ids == (provisional_id,)

    rejected_retirement = memory.retire_verified_track(1, lambda _: False)
    assert not rejected_retirement.accepted
    assert memory.track_ids == (0, 1, 2)

    retired = memory.retire_verified_track(1, lambda candidate: candidate.stream_count == 2)
    assert retired.accepted
    rejected = memory.promote_provisional_track(provisional_id, lambda _: False)
    assert not rejected.accepted
    assert rejected.reason == "retention_probe_rejected"
    promoted = memory.promote_provisional_track(
        provisional_id,
        lambda candidate: candidate.stream_count == 3,
    )
    assert promoted.accepted
    assert promoted.track_id is not None
    assert memory.track_ids == (0, 2, promoted.track_id)
    assert memory.provisional_ids == ()

    restored = ExternalOnlineStreamBindingMemory.from_payload(memory.state_payload())
    assert restored.digest() == memory.digest()


def test_atomic_replacement_keeps_state_on_rejection() -> None:
    encoder, observations = _fixture(504, stream_count=4)
    memory = ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=4,
        max_streams=3,
        provisional_capacity=2,
        match_tolerance=0.55,
        new_track_tolerance=0.7,
        provisional_tolerance=0.7,
        match_margin=0.05,
    )
    for stream in range(3):
        assert memory.observe(_row(observations[stream], 0)).status == "new"
    provisional = memory.observe(_row(observations[3], 0))
    assert provisional.provisional_id is not None
    pairs, features = memory.lifecycle_candidate_features()
    assert len(pairs) == 3
    assert features.shape == (3, 2 * encoder.context_width + 11)
    before = memory.digest()
    rejected_scalar = memory.replace_on_verifier_outcome(
        provisional.provisional_id,
        1,
        0.0,
    )
    assert not rejected_scalar.accepted
    assert rejected_scalar.reason == "verifier_outcome_rejected"
    assert memory.digest() == before
    rejected = memory.replace_verified_track_with_provisional(
        provisional.provisional_id,
        1,
        lambda _: False,
    )
    assert not rejected.accepted
    assert rejected.reason == "retention_probe_rejected"
    assert memory.digest() == before
    accepted = memory.replace_verified_track_with_provisional(
        provisional.provisional_id,
        1,
        lambda candidate: candidate.stream_count == 3
        and candidate.provisional_count == 0,
    )
    assert accepted.accepted
    assert accepted.track_id is not None
    assert memory.track_ids == (0, 2, accepted.track_id)
    assert memory.provisional_ids == ()


def test_lifecycle_policy_learns_from_outcome_and_persists() -> None:
    encoder, observations = _fixture(505, stream_count=4)
    memory = ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=4,
        max_streams=3,
        provisional_capacity=2,
        match_tolerance=0.55,
        new_track_tolerance=0.7,
        provisional_tolerance=0.7,
        match_margin=0.05,
    )
    for stream in range(3):
        assert memory.observe(_row(observations[stream], 0)).status == "new"
    provisional = memory.observe(_row(observations[3], 0))
    assert provisional.provisional_id is not None
    policy = ExternalStreamBindingLifecyclePolicy(
        encoder.context_width,
        hidden_width=8,
        learning_rate=0.02,
    )
    proposal = policy.propose(memory, sample=True, generator=torch.Generator().manual_seed(8))
    assert proposal.selected_provisional_id == provisional.provisional_id
    assert proposal.selected_track_id is not None
    assert 0.0 < proposal.selected_propensity <= 1.0
    before = policy.digest()
    loss = policy.adaptation_step(proposal, 1.0)
    assert loss >= 0.0
    assert policy.digest() != before
    restored = ExternalStreamBindingLifecyclePolicy.from_payload(policy.state_payload())
    assert restored.digest() == policy.digest()


def test_outcome_trained_lifecycle_pressure_test_passes() -> None:
    report = run_lifecycle_pressure_test(2401)
    assert report["promoted"] is True
    assert report["gates"]["learned_safe_policy"] is True
    assert report["gates"]["contradiction_prefers_hold"] is True
    assert report["gates"]["outcome_shuffle_control_lower"] is True


def test_joint_binding_factual_lifecycle_pressure_test_passes() -> None:
    report = run_factual_lifecycle_pressure_test(2501)
    assert report["promoted"] is True
    assert report["gates"]["learned_joint_proposal_correct"] is True
    assert report["gates"]["wrong_heldout_rejection_atomic"] is True
    assert report["gates"]["sibling_factual_slot_retained"] is True
    assert report["gates"]["drift_does_not_mutate_factual_bank"] is True
    assert "conflict" in report["drift"]["statuses"]
    assert all(status == "matched" for status in report["drift"]["binding_statuses"])


def test_joint_binding_factual_growth_pressure_test_passes() -> None:
    report = run_factual_growth_pressure_test(2601)
    assert report["promoted"] is True
    assert report["gates"]["learned_beats_fresh_control"] is True
    assert report["gates"]["learned_beats_shuffled_control"] is True
    assert report["gates"]["binding_capacity_grew"] is True
    assert report["gates"]["factual_capacity_grew"] is True
    assert report["gates"]["source_factual_slot_retained"] is True
    assert report["gates"]["new_slots_routed"] is True
    assert report["gates"]["joint_persistence_exact"] is True
    assert report["gates"]["controller_frozen_unchanged"] is True


def test_learned_binding_factual_consolidation_pressure_test_passes() -> None:
    report = run_factual_consolidation_pressure_test(2701)
    assert report["promoted"] is True
    assert report["gates"]["distinct_rejection_atomic"] is True
    assert report["gates"]["mutating_probe_rejection_atomic"] is True
    assert report["gates"]["equivalent_consolidation_committed"] is True
    assert report["gates"]["slot_addresses_retained"] is True
    assert report["gates"]["physical_model_count_reduced"] is True
    assert report["gates"]["exact_persistence"] is True


def test_joint_binding_factual_growth_is_atomic_and_persistent() -> None:
    encoder, observations = _fixture(506, stream_count=3)
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family="mixed_verified_v1",
        affine_ridge=1e-7,
        random_feature_width=32,
        random_feature_seed=7,
        capacity=1,
    )
    single = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=2,
        max_contexts=1,
        defer_admission=True,
        continuation_tolerance=0.5,
        candidate_model_families=(
            EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        ),
        provisional_evidence_policy="streaming_statistics",
    )
    router = ExternalLearnedMultiStreamTransitionContextRouter(
        ExternalOnlineStreamBindingMemory(
            encoder,
            window_capacity=4,
            max_streams=1,
            provisional_capacity=2,
            match_tolerance=0.55,
            new_track_tolerance=0.7,
            provisional_tolerance=0.7,
            match_margin=0.05,
        ),
        ExternalMultiStreamTransitionContextRouter(single, stream_key_width=4),
    )
    staged = None
    for row_index in range(2):
        result = router.observe(_row(observations[0], row_index))
        if result.routing is not None and result.routing.result.status == "staged":
            staged = result
            router.adaptation_step(result, None, replay_evidence=False)
    assert staged is not None
    source_receipt = router.promote_staged_candidate(
        staged,
        _row(observations[0], 2),
        lambda candidate: candidate.context_count == 1,
        prediction_tolerance=100.0,
    )
    assert source_receipt.accepted

    provisional = [router.observe(_row(observations[1], row)) for row in range(2)]
    assert all(result.binding.status == "provisional" for result in provisional)
    policy = ExternalStreamBindingLifecyclePolicy(4, hidden_width=16, learning_rate=0.02)
    proposal = policy.propose(router.binding, sample=False)
    before_binding = router.binding.digest()
    before_router = router.digest()
    rejected = router.grow_with_factual_candidate(
        proposal,
        _row(observations[1], 2),
        0.0,
        prediction_tolerance=100.0,
    )
    assert not rejected.accepted
    assert rejected.reason == "verifier_outcome_rejected"
    assert router.binding.digest() == before_binding
    assert router.digest() == before_router

    accepted = router.grow_with_factual_candidate(
        proposal,
        _row(observations[1], 2),
        1.0,
        prediction_tolerance=100.0,
    )
    assert accepted.accepted
    assert accepted.track_id is not None
    assert accepted.slot_id is not None
    assert router.binding.max_streams == 2
    assert router.binding.stream_count == 2
    assert router.router.bank.capacity == 2
    assert router.router.bank.context_count == 2
    assert router.binding.provisional_count == 0
    assert router.router.bank.model_family_at(0) in {
        EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    }
    restored = ExternalLearnedMultiStreamTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert restored.digest() == router.digest()

    first_key = router.binding.track_state(0)["stream_key"]
    second_key = router.binding.track_state(1)["stream_key"]
    before_consolidation_binding = router.binding.digest()
    before_consolidation_router = router.digest()
    rejected_consolidation = router.consolidate_factual_slots_verified(
        first_key,
        second_key,
        [_row(observations[0], 2)],
        prediction_tolerance=1e-8,
    )
    assert not rejected_consolidation.accepted
    assert router.binding.digest() == before_consolidation_binding
    assert router.digest() == before_consolidation_router

    assert router.router.bank.model_family_at(0) == router.router.bank.model_family_at(1)
    router.router.bank.models[1].load_state_dict(
        router.router.bank.models[0].state_dict()
    )

    before_mutating_probe_router = router.digest()

    def mutate_consolidation_candidate(
        candidate: ExternalLearnedMultiStreamTransitionContextRouter,
    ) -> bool:
        candidate.binding.max_streams += 1
        return True

    rejected_mutating_probe = router.consolidate_factual_slots_verified(
        first_key,
        second_key,
        [_row(observations[0], 2)],
        prediction_tolerance=1e-8,
        retention_probe=mutate_consolidation_candidate,
    )
    assert not rejected_mutating_probe.accepted
    assert router.digest() == before_mutating_probe_router

    accepted_consolidation = router.consolidate_factual_slots_verified(
        first_key,
        second_key,
        [_row(observations[0], 2)],
        prediction_tolerance=1e-8,
    )
    assert accepted_consolidation.accepted
    assert accepted_consolidation.physical_models_before == 2
    assert accepted_consolidation.physical_models_after == 1
    assert accepted_consolidation.first_slot_id != accepted_consolidation.second_slot_id
    assert router.binding.digest() == before_consolidation_binding
    assert router.router.bound_slot_id(first_key) == accepted_consolidation.first_slot_id
    assert router.router.bound_slot_id(second_key) == accepted_consolidation.second_slot_id
    consolidated_restored = (
        ExternalLearnedMultiStreamTransitionContextRouter.from_payload(
            router.state_payload()
        )
    )
    assert consolidated_restored.digest() == router.digest()
