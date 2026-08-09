import torch

from neural_computer import (
    ExternalContextAddressResolver,
    ExternalContextualEvidenceCalibrator,
    ExternalGoalEvaluator,
    ExternalModelBasedPlanner,
    ExternalOnlineContextAddressResolver,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionEvidenceEvaluator,
    ExternalTransitionMemory,
    ExternalTransitionModel,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)


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


def test_transition_model_learns_from_opaque_observations_without_controller_state() -> None:
    torch.manual_seed(1201)
    model = ExternalTransitionModel(3, 2, hidden_width=16)
    observation = ExternalTransitionObservation(
        state=torch.randn(8, 3),
        intention=torch.randn(8, 2),
        next_state=torch.randn(8, 3),
        confidence=torch.ones(8),
    )

    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
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
        admission_observations=2,
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

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert restored.configuration() == router.configuration()
    assert restored.bank.digest() == router.bank.digest()
    assert restored.context_encoder.digest() == router.context_encoder.digest()


def test_online_transition_context_router_capacity_guard_does_not_grow_or_write() -> None:
    torch.manual_seed(1211)
    bank = ExternalTransitionModelBank(2, 1, 4, hidden_width=8)
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    bank.ensure_context(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
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


def test_online_context_resolver_accumulates_interleaved_evidence_without_early_writes() -> None:
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
