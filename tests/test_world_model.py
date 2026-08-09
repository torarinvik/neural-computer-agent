import torch

from neural_computer import (
    ExternalGoalEvaluator,
    ExternalModelBasedPlanner,
    ExternalTransitionMemory,
    ExternalTransitionModel,
    ExternalTransitionObservation,
)


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
