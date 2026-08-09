import pytest
import torch

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    RuntimeMigrationExample,
)


def _feedback(batch: int, width: int) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch, width),
        reward=torch.zeros(batch),
        propensity=torch.ones(batch),
        has_feedback=torch.zeros(batch),
    )


def _runtime(
    controller: AmodalCognitiveController,
    *,
    event_space_id: str,
    state_space_id: str,
    intention_space_id: str,
) -> AmodalControllerRuntime:
    return AmodalControllerRuntime(
        controller,
        event_space_id=event_space_id,
        state_space_id=state_space_id,
        intention_space_id=intention_space_id,
    )


def test_runtime_migration_verifies_paired_frontend_and_controller_replacement() -> None:
    torch.manual_seed(2301)
    source_controller = AmodalCognitiveController(
        width=8,
        workspace_slots=2,
        intention_width=4,
        feedback_width=3,
        event_window_capacity=4,
    )
    candidate_controller = AmodalCognitiveController(
        width=8,
        workspace_slots=2,
        intention_width=4,
        feedback_width=3,
        event_window_capacity=4,
    )
    candidate_controller.load_state_dict(source_controller.state_dict())
    source = _runtime(
        source_controller,
        event_space_id="frontend-v1",
        state_space_id="controller-state-v1",
        intention_space_id="intention-v1",
    )
    candidate = _runtime(
        candidate_controller,
        event_space_id="frontend-v2",
        state_space_id="controller-state-v2",
        intention_space_id="intention-v2",
    )
    source_state = source.initial_state(1, device="cpu")
    candidate_state = candidate.initial_state(1, device="cpu")
    event = AmodalEvent(
        torch.randn(1, 8),
        timestamp=torch.tensor([1.0]),
        confidence=torch.ones(1),
    )
    example = RuntimeMigrationExample(
        source_events=[event],
        target_events=[AmodalEvent(
            event.payload.clone(),
            timestamp=event.timestamp.clone(),
            confidence=event.confidence.clone(),
        )],
        source_state=source_state,
        target_state=candidate_state,
        feedback=_feedback(1, 3),
    )
    source_digest = source._controller_digest(source)

    receipt = source.migrate_controller_verified(
        candidate,
        [example],
        retention_probe=lambda runtime: runtime.intention_width == 4,
    )

    assert receipt.accepted
    assert receipt.max_intention_difference == 0.0
    assert receipt.max_execution_difference == 0.0
    assert receipt.max_continuation_difference == 0.0
    assert source._controller_digest(source) == source_digest

    next(candidate.controller.parameters()).data.add_(1.0)
    rejected = source.migrate_controller_verified(candidate, [example])

    assert not rejected.accepted
    assert rejected.reason == "candidate changed held-out controller behavior"


def test_runtime_migration_rejects_memoryful_probe_and_space_mismatch() -> None:
    source = _runtime(
        AmodalCognitiveController(width=8, workspace_slots=2, intention_width=4, feedback_width=3),
        event_space_id="events-v1",
        state_space_id="state-v1",
        intention_space_id="intent-v1",
    )
    candidate = _runtime(
        AmodalCognitiveController(width=8, workspace_slots=2, intention_width=4, feedback_width=3),
        event_space_id="events-v2",
        state_space_id="state-v2",
        intention_space_id="intent-v2",
    )
    candidate_state = candidate.initial_state(1, device="cpu")
    example = RuntimeMigrationExample(
        source_events=[AmodalEvent(torch.zeros(1, 8))],
        target_events=[AmodalEvent(torch.zeros(1, 8))],
        source_state=source.initial_state(1, device="cpu"),
        target_state=candidate_state,
        feedback=_feedback(1, 3),
    )

    # The method's structural contract remains explicit even when the source
    # and target IDs are otherwise valid: a memoryful runtime is not silently
    # treated as a pure controller probe.
    candidate.memory = object()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="memory-free"):
        source.migrate_controller_verified(candidate, [example])
