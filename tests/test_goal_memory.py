import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    ExternalGoalFragmentCandidate,
    ExternalGoalFragmentMemory,
    ExternalGoalFragmentStager,
    ExternalModelBasedPlanner,
    ExternalTransitionObservation,
    PersistentOpaqueContextRouteEvidence,
    PolicyFreeAmodalRuntime,
)


class _AdditiveModel(nn.Module):
    state_width = 2
    intention_width = 2

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        return state + intention


class _RuntimeAdditiveModel(nn.Module):
    state_width = 12
    intention_width = 2

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, :2] = result[:, :2] + intention
        return result


def test_goal_fragment_memory_is_opaque_copy_on_write_and_persistent() -> None:
    memory = ExternalGoalFragmentMemory(3)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    masks = torch.tensor([[True, False, True]])
    source_digest = memory.digest()

    rejected = memory.admit_verified(
        values,
        masks,
        lambda _candidate: False,
    )
    assert not rejected.accepted
    assert memory.fragment_count == 0
    assert memory.digest() == source_digest

    accepted = memory.admit_verified(
        values,
        masks,
        lambda candidate: candidate.fragment_count == 1,
    )
    assert accepted.accepted
    assert accepted.fragment_id == 0
    assert memory.fragment_count == 1

    restored = ExternalGoalFragmentMemory.from_payload(memory.state_payload())
    assert restored.digest() == memory.digest()
    proposed = restored.propose((0,), composition="intersection", batch_size=2)
    assert proposed.values.shape == (2, 1, 3)
    assert proposed.masks.dtype is torch.bool
    assert proposed.fragment_ids == (0,)


def test_goal_fragment_memory_supports_per_batch_opaque_selection() -> None:
    memory = ExternalGoalFragmentMemory(3)
    memory.append(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([True, False, False]))
    memory.append(torch.tensor([0.0, 1.0, 0.0]), torch.tensor([False, True, False]))

    proposed = memory.propose_per_batch(
        torch.tensor([[0], [1]]),
        batch_size=2,
        composition="intersection",
    )

    assert proposed.values.shape == (2, 1, 3)
    assert proposed.fragment_ids == ()
    assert torch.equal(
        proposed.values[:, 0, :],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert proposed.composition == "intersection"


def test_policy_free_runtime_routes_goal_fragments_from_opaque_context_evidence() -> None:
    torch.manual_seed(73)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )
    event = [AmodalEvent(torch.randn(1, 4))]
    preview, _ = runtime.step_events(event, state, feedback)
    context = preview.controller.state_representation.detach()
    goal_memory = ExternalGoalFragmentMemory(12)
    positive_goal = context[0].clone()
    positive_goal[0] += 1.0
    negative_goal = context[0].clone()
    negative_goal[0] -= 1.0
    goal_mask = torch.tensor(
        [True, False, False, False, False, False, False, False, False, False, False, False]
    )
    goal_memory.append(positive_goal, goal_mask)
    goal_memory.append(negative_goal, goal_mask)
    route = PersistentOpaqueContextRouteEvidence(
        width=12,
        mastery_threshold=0.75,
        min_mastery_observations=2,
        reversal_patience=2,
    )
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_RuntimeAdditiveModel(), beam_width=2),
        goal_memory=goal_memory,
        goal_route_evidence=route,
    )
    controller_before = {
        name: value.detach().clone() for name, value in controller.state_dict().items()
    }
    policy_free.observe_goal_fragment_route(context, 1, 1.0)
    policy_free.observe_goal_fragment_route(context, 1, 1.0)

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        None,
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
        horizon=1,
    )

    assert output.goal_fragment_indices is not None
    assert output.goal_fragment_indices.tolist() == [[1]]
    assert torch.allclose(output.intention.payload, torch.tensor([[-1.0, 0.0]]))
    assert torch.allclose(
        output.planning.predicted_states[0, -1, 0], negative_goal[0]
    )
    assert route.has_context(context[0])
    assert route.preferred_slots(context).tolist() == [1]
    transition = policy_free.transition_observation(
        output,
        output,
        confidence=torch.ones(1),
    )
    assert isinstance(transition, ExternalTransitionObservation)
    assert torch.equal(transition.state, output.state)
    assert torch.equal(transition.intention, output.intention.payload)
    assert torch.equal(transition.next_state, output.state)
    for name, value in controller.state_dict().items():
        assert torch.equal(value, controller_before[name])

    # A fresh opaque context has no evidence and must use append-order fallback.
    fresh_event = [AmodalEvent(torch.randn(1, 4))]
    fresh_preview, _ = runtime.step_events(fresh_event, state, feedback)
    fresh_context = fresh_preview.controller.state_representation.detach()
    fresh_output, _ = policy_free.step_events(
        fresh_event,
        state,
        feedback,
        None,
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
        horizon=1,
    )
    assert fresh_output.goal_fragment_indices.tolist() == [[0]]
    assert route.preferred_slots(fresh_context).tolist() == [0]
    assert not route.has_context(fresh_context[0])

    # Reversal evidence clears the protected preference without touching the core.
    policy_free.observe_goal_fragment_route(context, 1, 0.0)
    policy_free.observe_goal_fragment_route(context, 1, 0.0)
    reversed_output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        None,
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
        horizon=1,
    )
    assert reversed_output.goal_fragment_indices.tolist() == [[0]]
    assert torch.allclose(reversed_output.intention.payload, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(
        reversed_output.planning.predicted_states[0, -1, 0], positive_goal[0]
    )
    assert route.payload()["contexts"]

    memory_payload = policy_free.goal_memory_state_payload()
    route_payload = policy_free.goal_route_state_payload()
    policy_free.load_goal_memory_state_payload(memory_payload)
    policy_free.load_goal_route_state_payload(route_payload)
    assert (
        policy_free.goal_memory_state_payload()["sha256"] == memory_payload["sha256"]
    )
    assert policy_free.goal_route_state_payload() == route_payload
    for name, value in controller.state_dict().items():
        assert torch.equal(value, controller_before[name])


def test_intersection_goal_fragments_require_all_puzzle_pieces() -> None:
    memory = ExternalGoalFragmentMemory(2)
    memory.append(torch.tensor([1.0, 0.0]), torch.tensor([True, False]))
    memory.append(torch.tensor([0.0, 1.0]), torch.tensor([False, True]))
    fragments = memory.propose((0, 1), composition="intersection")
    planner = ExternalModelBasedPlanner(_AdditiveModel(), beam_width=5)

    result = planner.plan(
        torch.zeros(1, 2),
        None,
        torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        ),
        horizon=2,
        beam_width=5,
        goal_fragments=fragments,
    )

    assert torch.allclose(result.predicted_states[0, -1], torch.ones(2))
    assert result.candidate_indices is not None
    assert set(result.candidate_indices[0].tolist()) == {1, 2}


def test_policy_free_runtime_can_read_destinations_from_external_goal_memory() -> None:
    torch.manual_seed(71)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )
    event = [AmodalEvent(torch.randn(1, 4))]
    preview, _ = runtime.step_events(event, state, feedback)
    goal_memory = ExternalGoalFragmentMemory(12)
    goal_memory.append(
        preview.controller.state_representation[0],
        torch.ones(12, dtype=torch.bool),
    )
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_RuntimeAdditiveModel(), beam_width=2),
        goal_memory=goal_memory,
    )

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        None,
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        horizon=1,
        goal_fragment_indices=(0,),
        goal_composition="intersection",
    )

    assert output.goal_state is None
    assert output.goal_fragments is not None
    assert output.goal_fragments.composition == "intersection"
    assert output.goal_fragments.fragment_ids == (0,)
    assert torch.allclose(output.intention.payload, torch.zeros(1, 2))


def test_goal_fragment_stager_keeps_only_scalar_evidence_and_promotes_copy_on_write() -> (
    None
):
    candidate = ExternalGoalFragmentCandidate(
        torch.tensor([1.0, -1.0]),
        torch.tensor([True, True]),
    )
    stager = ExternalGoalFragmentStager(
        2,
        threshold=0.75,
        min_observations=3,
        min_stable_observations=2,
    )
    digest = candidate.digest(state_width=2)
    for outcome in (0.8, 0.9, 0.8, 0.9):
        receipt = stager.observe(candidate, outcome)
    assert receipt.ready
    assert receipt.candidate_digest == digest
    assert receipt.stable_observations == 2

    payload = stager.state_payload()
    assert "outcomes" not in repr(payload)
    restored = ExternalGoalFragmentStager.from_payload(payload)
    assert restored.observation(digest) == stager.observation(digest)

    memory = ExternalGoalFragmentMemory(2)
    planner = ExternalModelBasedPlanner(_AdditiveModel(), beam_width=2)

    def retained(proposed: ExternalGoalFragmentMemory) -> bool:
        result = planner.plan(
            torch.zeros(1, 2),
            None,
            torch.tensor([[1.0, -1.0], [0.0, 0.0]]),
            horizon=1,
            goal_fragments=proposed.propose((0,)),
        )
        return bool(
            torch.allclose(result.predicted_states[0, -1], torch.tensor([1.0, -1.0]))
        )

    admission = restored.admit_verified(
        memory,
        digest,
        retained,
    )
    assert admission.accepted
    assert memory.fragment_count == 1
    assert restored.pending_count == 0


def test_goal_fragment_stager_rejects_unstable_or_shuffled_evidence() -> None:
    good = ExternalGoalFragmentCandidate(
        torch.tensor([1.0, 0.0]),
        torch.tensor([True, False]),
    )
    bad = ExternalGoalFragmentCandidate(
        torch.tensor([0.0, 1.0]),
        torch.tensor([False, True]),
    )
    stager = ExternalGoalFragmentStager(
        2,
        threshold=0.75,
        min_observations=3,
        min_stable_observations=2,
    )
    for outcome in (1.0, 0.0, 1.0, 1.0):
        stager.observe(good, outcome)
    for outcome in (0.0, 1.0, 0.0, 1.0):
        stager.observe(bad, outcome)
    assert not stager.observation(good.digest(state_width=2)).ready
    assert not stager.observation(bad.digest(state_width=2)).ready

    memory = ExternalGoalFragmentMemory(2)
    rejected = stager.admit_verified(
        memory,
        good.digest(state_width=2),
        lambda _candidate: True,
    )
    assert not rejected.accepted
    assert memory.fragment_count == 0


def test_policy_free_runtime_stages_goal_from_external_scalar_outcomes() -> None:
    torch.manual_seed(72)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    goal_memory = ExternalGoalFragmentMemory(12)
    goal_stager = ExternalGoalFragmentStager(
        12,
        threshold=0.75,
        min_observations=2,
        min_stable_observations=1,
    )
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_RuntimeAdditiveModel(), beam_width=2),
        goal_memory=goal_memory,
        goal_stager=goal_stager,
    )
    controller_before = {
        name: value.detach().clone()
        for name, value in controller.state_dict().items()
    }
    state = runtime.initial_state(1, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )
    preview, _ = runtime.step_events(
        [AmodalEvent(torch.randn(1, 4))], state, feedback
    )
    candidate = policy_free.goal_fragment_candidate_from_controller_output(
        preview.controller
    )
    receipt = policy_free.observe_goal_fragment_controller_output(
        preview.controller, 1.0
    )
    assert receipt.candidate_digest == candidate.digest(state_width=12)
    policy_free.observe_goal_fragment(candidate, 1.0)
    admission = policy_free.admit_goal_fragment_verified(
        candidate.digest(state_width=12),
        lambda proposed: proposed.fragment_count == 1,
    )
    assert admission.accepted
    assert goal_memory.fragment_count == 1
    for name, value in controller.state_dict().items():
        assert torch.equal(value, controller_before[name])
