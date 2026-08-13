import pytest
import torch

from neural_computer import (
    CapabilityEvictionObservation,
    ExternalCapabilityEvictionPolicy,
    ExternalFastWeightPlasticity,
    ExternalFastWeightState,
    ExternalMemoryEvictionPolicy,
    ExternalMemoryWritePolicy,
    ExternalOutcomeCreditPlasticity,
    ExternalOutcomeCreditState,
    ExternalOutcomeProgramCapacityGrowthReceipt,
    ExternalOutcomeProgramCellBank,
    ExternalOutcomeProgramCellSelectionReceipt,
    ExternalOutcomeProgramPriorSelectionReceipt,
    ExternalOutcomeProgramRouter,
    ExternalOutcomeProgramRouterState,
    ExternalOutcomeValueBaseline,
    ExternalOutcomeValueState,
    GatedResidualCapabilityEvictionPolicyBank,
    MemoryEvictionObservation,
    MemoryWriteObservation,
)


def test_fast_weight_plasticity_learns_while_state_is_external() -> None:
    rule = ExternalFastWeightPlasticity(key_width=4, value_width=2, hidden=8)
    state = rule.initial_state(2)
    queries = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    values = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    next_state = rule.update(
        state,
        queries,
        values,
        torch.tensor([1.0, 0.0]),
    )

    read = rule.read(next_state, queries)
    assert read[0].abs().min() > 0.9
    assert torch.equal(read[1], torch.zeros(2))
    assert next_state.updates.tolist() == [1, 1]
    assert rule.configuration()["update_rule"] == (
        "outcome_gated_normalized_delta_fast_weight_v1"
    )


def test_fast_weight_plasticity_preserves_old_state_on_missing_evidence() -> None:
    rule = ExternalFastWeightPlasticity(key_width=3, value_width=2, hidden=8)
    state = rule.initial_state(2)
    queries = torch.eye(3)[:2]
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mastered = rule.update(state, queries, values, torch.ones(2))
    old_weights = mastered.weights[0].clone()
    updated = rule.update(
        mastered,
        queries,
        torch.tensor([[-1.0, -1.0], [0.25, 0.75]]),
        torch.ones(2),
        present=torch.tensor([False, True]),
    )

    assert torch.equal(updated.weights[0], old_weights)
    assert not torch.equal(updated.weights[1], mastered.weights[1])


def test_fast_weight_plasticity_state_round_trips_as_tensor_payload() -> None:
    rule = ExternalFastWeightPlasticity(key_width=3, value_width=2, hidden=8)
    state = rule.update(
        rule.initial_state(1),
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[0.5, -0.5]]),
        torch.ones(1),
    )
    restored = rule.state_from_payload(rule.state_payload(state))

    assert isinstance(restored, ExternalFastWeightState)
    assert torch.equal(restored.weights, state.weights)
    assert torch.equal(restored.updates, state.updates)


def test_external_outcome_credit_assigns_delayed_feedback_to_external_policy() -> None:
    rule = ExternalOutcomeCreditPlasticity(
        feature_width=3,
        action_count=2,
        initial_learning_rate=0.2,
        initial_trace_decay=0.8,
    )
    state = rule.initial_state(1)
    features = torch.tensor([[1.0, 0.0, 0.0]])
    state = rule.record_decision(
        state,
        features,
        torch.tensor([0]),
        torch.tensor([0.5]),
    )
    assert bool(torch.any(state.eligibility != 0.0))
    updated = rule.apply_feedback(
        state,
        torch.ones(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )

    assert bool(torch.any(updated.policy != 0.0))
    assert torch.equal(updated.eligibility, torch.zeros_like(updated.eligibility))
    assert updated.decisions.tolist() == [1]
    assert updated.feedbacks.tolist() == [1]
    assert rule.configuration()["update_rule"] == (
        "importance_weighted_delayed_policy_gradient_v1"
    )


def test_external_outcome_credit_supports_independent_batched_trajectories() -> None:
    rule = ExternalOutcomeCreditPlasticity(
        feature_width=3,
        action_count=2,
        initial_learning_rate=0.2,
        initial_trace_decay=0.8,
    )
    state = rule.initial_state(4)
    state = rule.record_decision(
        state,
        torch.eye(3).repeat(2, 1)[:4],
        torch.tensor([0, 1, 0, 1]),
        torch.full((4,), 0.5),
    )
    updated = rule.apply_feedback(
        state,
        torch.ones(4),
        terminal=torch.ones(4, dtype=torch.bool),
    )

    assert updated.policy.shape == (4, 3, 2)
    assert bool(torch.isfinite(updated.policy).all())
    assert bool(torch.any(updated.policy != 0.0))


def test_external_outcome_credit_supports_full_information_counterfactual_feedback() -> None:
    rule = ExternalOutcomeCreditPlasticity(
        feature_width=3,
        action_count=3,
        initial_learning_rate=0.5,
        initial_baseline_rate=0.1,
    )
    state = rule.record_decision(
        rule.initial_state(1),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([1]),
        torch.tensor([1.0 / 3.0]),
    )
    updated = rule.apply_counterfactual_feedback(
        state,
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
    )

    assert int(updated.policy[0, 0].argmax()) == 0
    assert torch.equal(updated.eligibility, torch.zeros_like(updated.eligibility))
    assert updated.decisions.tolist() == [4]
    assert updated.feedbacks.tolist() == [3]
    assert rule.configuration()["counterfactual_update"] == (
        "mean_full_information_policy_gradient_v1"
    )


def test_external_outcome_credit_counterfactual_missing_evidence_is_exact_noop() -> None:
    rule = ExternalOutcomeCreditPlasticity(feature_width=2, action_count=3)
    state = rule.apply_counterfactual_feedback(
        rule.initial_state(1),
        torch.tensor([[1.0, -1.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        present=torch.zeros(1, dtype=torch.bool),
    )

    initial = rule.initial_state(1)
    assert torch.equal(state.policy, initial.policy)
    assert torch.equal(state.eligibility, initial.eligibility)
    assert torch.equal(state.baseline, initial.baseline)
    assert torch.equal(state.decisions, initial.decisions)
    assert torch.equal(state.feedbacks, initial.feedbacks)


def test_external_outcome_credit_missing_feedback_preserves_external_state() -> None:
    rule = ExternalOutcomeCreditPlasticity(feature_width=2, action_count=2)
    state = rule.record_decision(
        rule.initial_state(1),
        torch.tensor([[1.0, -1.0]]),
        torch.tensor([1]),
        torch.tensor([0.5]),
    )
    unchanged = rule.apply_feedback(
        state,
        torch.ones(1),
        present=torch.zeros(1, dtype=torch.bool),
        terminal=torch.ones(1, dtype=torch.bool),
    )

    assert torch.equal(unchanged.policy, state.policy)
    assert torch.equal(unchanged.eligibility, state.eligibility)
    assert torch.equal(unchanged.baseline, state.baseline)
    assert torch.equal(unchanged.feedbacks, state.feedbacks)


def test_external_outcome_credit_state_round_trips_as_tensor_payload() -> None:
    rule = ExternalOutcomeCreditPlasticity(feature_width=2, action_count=3)
    state = rule.record_decision(
        rule.initial_state(2),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 2]),
        torch.tensor([1.0 / 3.0, 1.0 / 3.0]),
    )
    state = rule.apply_feedback(state, torch.tensor([1.0, 0.0]))
    restored = rule.state_from_payload(rule.state_payload(state))

    assert isinstance(restored, ExternalOutcomeCreditState)
    assert torch.equal(restored.policy, state.policy)
    assert torch.equal(restored.eligibility, state.eligibility)
    assert torch.equal(restored.baseline, state.baseline)
    assert torch.equal(restored.decisions, state.decisions)
    assert torch.equal(restored.feedbacks, state.feedbacks)


def test_external_outcome_credit_masks_unadmitted_actions() -> None:
    rule = ExternalOutcomeCreditPlasticity(feature_width=2, action_count=3)
    state = rule.initial_state(1)
    mask = torch.tensor([[True, False, False]])
    logits = rule.logits(state, torch.tensor([[1.0, 0.0]]), action_mask=mask)

    assert int(logits.argmax(dim=-1).item()) == 0
    with pytest.raises(ValueError, match="masked out"):
        rule.record_decision(
            state,
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([1]),
            torch.ones(1),
            action_mask=mask,
        )


def test_external_outcome_program_router_appends_and_round_trips() -> None:
    router = ExternalOutcomeProgramRouter(
        feature_width=3,
        program_capacity=3,
        initial_programs=1,
        initial_learning_rate=0.2,
        initial_trace_decay=0.8,
    )
    state = router.initial_state(1)
    state = router.append_program(state)
    choice, propensity = router.sample_program(
        state,
        torch.tensor([[1.0, 0.0, 0.0]]),
        exploration=0.0,
    )
    state = router.record_decision(
        state,
        torch.tensor([[1.0, 0.0, 0.0]]),
        choice,
        propensity,
    )
    state = router.apply_feedback(
        state,
        torch.ones(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    restored = router.state_from_payload(router.state_payload(state))

    assert isinstance(restored, ExternalOutcomeProgramRouterState)
    assert restored.active_programs == 2
    assert torch.equal(restored.credit.policy, state.credit.policy)
    assert torch.equal(restored.credit.feedbacks, state.credit.feedbacks)
    assert router.action_mask(restored).tolist() == [[True, True, False]]


def test_external_outcome_program_router_applies_counterfactual_route_credit() -> None:
    router = ExternalOutcomeProgramRouter(
        feature_width=2,
        program_capacity=4,
        initial_programs=3,
        initial_learning_rate=0.5,
        initial_trace_decay=0.0,
    )
    state = router.apply_counterfactual_feedback(
        router.initial_state(1),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )

    assert int(router.logits(state, torch.tensor([[1.0, 0.0]])).argmax()) == 0
    assert state.credit.decisions.tolist() == [3]
    assert state.credit.feedbacks.tolist() == [3]
    assert torch.equal(
        state.credit.policy[..., 3],
        torch.zeros_like(state.credit.policy[..., 3]),
    )


def test_external_outcome_program_router_can_activate_a_conservative_new_route() -> None:
    router = ExternalOutcomeProgramRouter(
        feature_width=2,
        program_capacity=3,
        initial_programs=2,
    )
    state = router.initial_state(1)
    policy = state.credit.policy.clone()
    policy[..., 0] = 2.0
    policy[..., 1] = -0.5
    state = ExternalOutcomeProgramRouterState(
        credit=ExternalOutcomeCreditState(
            policy=policy,
            eligibility=state.credit.eligibility,
            baseline=state.credit.baseline,
            decisions=state.credit.decisions,
            feedbacks=state.credit.feedbacks,
        ),
        active_programs=state.active_programs,
    )

    conservative = router.append_program(state, initialization="conservative")

    assert torch.all(conservative.credit.policy[..., 2] < -0.5)
    assert torch.equal(conservative.credit.policy[..., :2], policy[..., :2])


def test_external_outcome_program_router_growth_is_transactional_and_retains_policy() -> None:
    router = ExternalOutcomeProgramRouter(
        feature_width=3,
        program_capacity=2,
        initial_programs=2,
        initial_learning_rate=0.2,
        initial_trace_decay=0.8,
    )
    state = router.record_decision(
        router.initial_state(1),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([1]),
        torch.tensor([0.5]),
    )
    state = router.apply_feedback(
        state,
        torch.ones(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    features = torch.tensor([[1.0, 0.0, 0.0]])
    old_logits = router.logits(state, features)
    old_policy = state.credit.policy.clone()
    calls: list[tuple[int, int]] = []

    def retain(candidate: ExternalOutcomeProgramRouter, candidate_state: object) -> bool:
        assert isinstance(candidate_state, ExternalOutcomeProgramRouterState)
        calls.append((candidate.program_capacity, candidate_state.credit.policy.shape[-1]))
        return bool(torch.allclose(candidate.logits(candidate_state, features)[..., :2], old_logits))

    receipt, grown = router.grow_capacity_verified(state, 4, retain)

    assert isinstance(receipt, ExternalOutcomeProgramCapacityGrowthReceipt)
    assert receipt.accepted
    assert receipt.source_capacity == 2
    assert receipt.destination_capacity == 4
    assert calls == [(2, 2), (4, 4)]
    assert router.program_capacity == 4
    assert grown.active_programs == 2
    assert torch.equal(grown.credit.policy[..., :2], old_policy)
    assert torch.equal(grown.credit.policy[..., 2:], torch.zeros(1, 3, 2))
    assert torch.allclose(router.logits(grown, features)[..., :2], old_logits)

    restored = router.state_from_payload(router.state_payload(grown))
    assert torch.equal(restored.credit.policy, grown.credit.policy)
    assert router.action_mask(restored).tolist() == [[True, True, False, False]]


def test_external_outcome_program_router_growth_rejection_does_not_mutate_capacity() -> None:
    router = ExternalOutcomeProgramRouter(
        feature_width=2,
        program_capacity=2,
        initial_programs=1,
    )
    state = router.initial_state(1)
    old_policy = state.credit.policy.clone()
    calls = 0

    def reject_expanded(
        _candidate: ExternalOutcomeProgramRouter,
        _candidate_state: ExternalOutcomeProgramRouterState,
    ) -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    receipt, unchanged = router.grow_capacity_verified(state, 3, reject_expanded)

    assert not receipt.accepted
    assert calls == 2
    assert router.program_capacity == 2
    assert unchanged is state
    assert torch.equal(unchanged.credit.policy, old_policy)


def test_external_outcome_program_router_prior_challenger_is_copy_on_write() -> None:
    router = ExternalOutcomeProgramRouter(
        feature_width=2,
        program_capacity=2,
        initial_programs=2,
    )
    state = router.record_decision(
        router.initial_state(1),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1]),
        torch.tensor([0.5]),
    )
    state = router.apply_feedback(
        state,
        torch.ones(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    source_policy = state.credit.policy.clone()
    source_capacity = router.program_capacity

    def probe(
        transfer_router: ExternalOutcomeProgramRouter,
        transfer_state: ExternalOutcomeProgramRouterState,
        fresh_router: ExternalOutcomeProgramRouter,
        fresh_state: ExternalOutcomeProgramRouterState,
    ) -> tuple[
        float,
        float,
        ExternalOutcomeProgramRouterState,
        ExternalOutcomeProgramRouterState,
    ]:
        del transfer_router, fresh_router
        return 0.25, 0.75, transfer_state, fresh_state

    receipt, selected_router, selected_state = router.select_verified_transfer_prior(
        state,
        3,
        3,
        probe,
        probe_updates=4,
    )

    assert isinstance(receipt, ExternalOutcomeProgramPriorSelectionReceipt)
    assert receipt.selected_initialization == "transfer"
    assert receipt.source_active_programs == 2
    assert receipt.destination_capacity == 3
    assert receipt.destination_active_programs == 3
    assert receipt.probe_updates == 4
    assert selected_router.program_capacity == 3
    assert selected_state.active_programs == 3
    assert router.program_capacity == source_capacity
    assert torch.equal(state.credit.policy, source_policy)
    assert receipt.source_state_digest == router._state_digest(state)
    assert receipt.selected_state_digest == selected_router._state_digest(selected_state)


def test_external_outcome_program_router_protected_prefix_is_retention_safe() -> None:
    router = ExternalOutcomeProgramRouter(
        feature_width=2,
        program_capacity=3,
        initial_programs=3,
        initial_learning_rate=0.5,
        initial_trace_decay=0.0,
    )
    state = router.record_decision(
        router.initial_state(1),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([2]),
        torch.tensor([0.5]),
    )
    old_policy = state.credit.policy.clone()
    updated = router.apply_feedback(
        state,
        torch.ones(1),
        terminal=torch.ones(1, dtype=torch.bool),
        protected_programs=2,
    )

    assert torch.equal(updated.credit.policy[..., :2], old_policy[..., :2])
    assert not torch.equal(updated.credit.policy[..., 2:], old_policy[..., 2:])


def test_external_outcome_program_cell_bank_selection_is_copy_on_write() -> None:
    bank = ExternalOutcomeProgramCellBank(
        feature_width=2,
        program_capacity=2,
        context_width=3,
        initial_programs=2,
    )
    router, state = bank.new_cell_candidate(active_programs=2)
    cell_id = bank.append_cell(torch.tensor([1.0, 0.0, 0.0]), router, state)
    before_digest = bank.content_digest()

    def probe(
        candidate_router: ExternalOutcomeProgramRouter,
        candidate_state: ExternalOutcomeProgramRouterState,
    ) -> tuple[float, ExternalOutcomeProgramRouterState]:
        del candidate_router
        return 0.1, candidate_state

    receipt, selected_router, selected_state = bank.select_verified_cell(
        torch.tensor([0.0, 1.0, 0.0]),
        probe,
        match_threshold=0.2,
        probe_updates=3,
    )

    assert isinstance(receipt, ExternalOutcomeProgramCellSelectionReceipt)
    assert receipt.reused
    assert receipt.selected_cell_id == cell_id
    assert receipt.selected_cell_index == 0
    assert selected_router is not None
    assert selected_state is not None
    assert bank.content_digest() == before_digest
    bank.commit_state(0, selected_state)
    assert bank.cell_ids == (cell_id,)
    restored = ExternalOutcomeProgramCellBank.from_payload(bank.payload())
    assert restored.content_digest() == bank.content_digest()
    assert restored.cell_ids == bank.cell_ids


def test_external_outcome_value_baseline_learns_and_round_trips() -> None:
    critic = ExternalOutcomeValueBaseline(
        feature_width=3,
        initial_learning_rate=0.2,
        initial_trace_decay=0.8,
    )
    state = critic.initial_state(1)
    prediction, state = critic.record_decision(
        state,
        torch.tensor([[1.0, 0.0, 0.0]]),
    )
    updated = critic.apply_feedback(
        state,
        torch.ones(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    restored = critic.state_from_payload(critic.state_payload(updated))

    assert torch.allclose(prediction, torch.full((1,), 0.5))
    assert bool(torch.any(updated.weights != 0.0))
    assert torch.equal(updated.eligibility, torch.zeros_like(updated.eligibility))
    assert isinstance(restored, ExternalOutcomeValueState)
    assert torch.equal(restored.weights, updated.weights)
    assert torch.equal(restored.bias, updated.bias)
    assert torch.equal(restored.feedbacks, updated.feedbacks)


def test_external_outcome_value_baseline_missing_feedback_is_no_write() -> None:
    critic = ExternalOutcomeValueBaseline(
        feature_width=2,
        initial_learning_rate=0.2,
        initial_trace_decay=0.8,
    )
    state = critic.initial_state(1)
    _, state = critic.record_decision(state, torch.tensor([[1.0, 0.0]]))
    missing = critic.apply_feedback(
        state,
        torch.ones(1),
        present=torch.zeros(1, dtype=torch.bool),
        terminal=torch.ones(1, dtype=torch.bool),
    )

    assert torch.equal(missing.weights, state.weights)
    assert torch.equal(missing.eligibility, state.eligibility)
    assert torch.equal(missing.bias, state.bias)
    assert torch.equal(missing.prediction_trace, state.prediction_trace)
    assert torch.equal(missing.trace_mass, state.trace_mass)
    assert torch.equal(missing.feedbacks, state.feedbacks)


def test_external_outcome_credit_accepts_external_value_baseline_override() -> None:
    rule = ExternalOutcomeCreditPlasticity(feature_width=2, action_count=2)
    state = rule.record_decision(
        rule.initial_state(1),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([0.5]),
    )
    updated = rule.apply_feedback(
        state,
        torch.ones(1),
        baseline_override=torch.zeros(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )

    assert bool(torch.any(updated.policy != 0.0))


def _observation(batch: int = 3) -> MemoryWriteObservation:
    return MemoryWriteObservation(
        event=torch.randn(batch, 4),
        hidden=torch.randn(batch, 4),
        workspace_read=torch.randn(batch, 4),
        query_key=torch.randn(batch, 4),
        write_value=torch.randn(batch, 4),
        controller_write_proposal=torch.rand(batch),
        controller_write_context=torch.randn(batch, 8),
        controller_write_relevance=torch.rand(batch),
        memory_read_value=torch.randn(batch, 4),
        memory_read_hit=torch.tensor([True, False, True])[:batch],
        action=torch.randn(batch, 2),
        reward=torch.randn(batch),
        propensity=torch.rand(batch),
        has_feedback=torch.ones(batch),
    )


def test_external_memory_writer_has_stable_opaque_boundary() -> None:
    policy = ExternalMemoryWritePolicy(
        event_width=4,
        hidden_width=4,
        workspace_width=4,
        key_width=4,
        value_width=4,
        memory_read_width=4,
        action_width=2,
        controller_write_context_width=8,
        controller_write_relevance_width=1,
    )

    observation = _observation()
    probability = policy(observation)
    adapted_value = policy.adapt_value(observation)

    assert probability.shape == (3,)
    assert adapted_value.shape == (3, 4)
    assert torch.allclose(adapted_value, observation.write_value)
    assert bool(torch.all((probability >= 0.0) & (probability <= 1.0)))
    assert policy.configuration()["schema"] == (
        "neural-computer.external-memory-write-policy.v11"
    )


def test_external_memory_writer_can_train_without_controller_parameters() -> None:
    policy = ExternalMemoryWritePolicy(
        event_width=4,
        hidden_width=4,
        workspace_width=4,
        key_width=4,
        value_width=4,
        memory_read_width=4,
        action_width=2,
        controller_write_context_width=8,
        controller_write_relevance_width=1,
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    loss = -torch.log(policy(_observation())).mean()
    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, policy.parameters(), strict=True)
    )


def test_external_memory_writer_rejects_nonfinite_observations() -> None:
    policy = ExternalMemoryWritePolicy(
        event_width=4,
        hidden_width=4,
        workspace_width=4,
        key_width=4,
        value_width=4,
        memory_read_width=4,
        action_width=2,
        controller_write_context_width=8,
        controller_write_relevance_width=1,
    )
    observation = _observation()
    observation = MemoryWriteObservation(
        **{
            **observation.__dict__,
            "event": torch.full((3, 4), float("nan")),
        }
    )

    try:
        policy(observation)
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite memory observations must be rejected")


def test_external_memory_eviction_policy_scores_opaque_candidates() -> None:
    policy = ExternalMemoryEvictionPolicy(
        event_width=4,
        hidden_width=4,
        workspace_width=4,
        key_width=4,
        value_width=4,
        memory_read_width=4,
        action_width=2,
        controller_write_context_width=8,
        controller_write_relevance_width=1,
        candidate_key_width=4,
        candidate_value_width=4,
    )
    write = _observation()
    observation = MemoryEvictionObservation(
        write=write,
        candidate_key=torch.randn(3, 4),
        candidate_value=torch.randn(3, 4),
        candidate_strength=torch.rand(3),
        candidate_timestamp=torch.rand(3),
        candidate_occupied=torch.ones(3),
    )
    scores = policy(observation)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    assert torch.equal(scores, torch.zeros_like(scores))
    assert policy.configuration()["schema"] == (
        "neural-computer.external-memory-eviction-policy.v1"
    )


def test_external_capability_eviction_policy_ranks_variable_opaque_bank() -> None:
    policy = ExternalCapabilityEvictionPolicy(
        context_width=4,
        candidate_width=6,
        hidden=8,
    )
    context = torch.randn(3, 4)
    candidates = torch.randn(3, 5, 6)
    scores = policy.score_candidates(context, candidates)
    assert scores.shape == (3, 5)
    assert torch.isfinite(scores).all()
    direct = policy(
        CapabilityEvictionObservation(
            context=context,
            candidate=candidates[:, 0],
        )
    )
    assert torch.allclose(scores[:, 0], direct)
    assert policy.configuration()["schema"] == (
        "neural-computer.external-capability-eviction-policy.v1"
    )


def test_external_capability_eviction_policy_learns_from_scalar_pairwise_signal() -> None:
    policy = ExternalCapabilityEvictionPolicy(
        context_width=2,
        candidate_width=3,
        hidden=8,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    context = torch.zeros(32, 2)
    candidates = torch.zeros(32, 2, 3)
    candidates[:, 0, 0] = 1.0
    candidates[:, 1, 0] = -1.0
    for _ in range(40):
        scores = policy.score_candidates(context, candidates)
        loss = torch.nn.functional.softplus(-(scores[:, 1] - scores[:, 0])).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    scores = policy.score_candidates(context[:1], candidates[:1])
    assert float(scores[0, 1].detach()) > float(scores[0, 0].detach())


def test_gated_residual_eviction_bank_isolates_and_activates_maintenance_slots() -> None:
    base = ExternalCapabilityEvictionPolicy(
        context_width=4,
        candidate_width=6,
        hidden=8,
    )
    bank = GatedResidualCapabilityEvictionPolicyBank(
        base,
        context_width=4,
        candidate_width=6,
        max_slots=2,
    )
    key_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    key_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert bank.add_slot(key_a) == 0
    assert bank.add_slot(key_b) == 1
    context = key_a.unsqueeze(0)
    candidates = torch.randn(1, 3, 6)
    optimizer = torch.optim.Adam(bank.trainable_parameters(0), lr=0.02)
    selected_before = bank.score_candidates(context, candidates)
    for _ in range(8):
        scores = bank.residual_slots[0](
            torch.cat((context[:, None, :].expand(-1, 3, -1), candidates), dim=-1)
        ).squeeze(-1)
        selected = int(scores.argmax())
        bank.adaptation_step(
            context,
            candidates,
            0,
            selected,
            1.0,
            optimizer=optimizer,
        )
    bank.activate_slot(0)
    selected_after = bank.score_candidates(context, candidates)
    expected_after = bank._normalized_prior(
        base.score_candidates(context, candidates)
    )
    residual = bank.residual_slots[0](
        torch.cat((context[:, None, :].expand(-1, 3, -1), candidates), dim=-1)
    ).squeeze(-1)
    expected_after = expected_after + residual

    assert torch.allclose(
        bank.route_scores(torch.stack((key_a, key_b))),
        torch.eye(2),
    )
    assert torch.equal(selected_before, base.score_candidates(context, candidates))
    assert torch.allclose(selected_after, expected_after)
    assert not torch.equal(selected_after, selected_before)
    bank.freeze_slot(0)
    with pytest.raises(RuntimeError, match="frozen"):
        bank.trainable_parameters(0)
