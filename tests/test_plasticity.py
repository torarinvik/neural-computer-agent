import torch

from neural_computer import (
    CapabilityEvictionObservation,
    ExternalFastWeightPlasticity,
    ExternalCapabilityEvictionPolicy,
    ExternalMemoryEvictionPolicy,
    ExternalMemoryWritePolicy,
    ExternalFastWeightState,
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
