import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    ExternalGoalFragmentMemory,
    ExternalModelBasedPlanner,
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
