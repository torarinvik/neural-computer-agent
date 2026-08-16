from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    ControllerFeedback,
    ExternalModelBasedPlanner,
    LiveActionReceipt,
    LiveOutcomeEvent,
    PolicyFreeAmodalLiveMachine,
    PolicyFreeAmodalRuntime,
    ResolvedLiveOutcome,
)


class _AdditiveFactualModel(nn.Module):
    state_width = 12
    intention_width = 2

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, :2] += intention
        return result


class _Decision:
    def __init__(self, action: torch.Tensor, propensity: torch.Tensor) -> None:
        self.action = action
        self.propensity = propensity


class _OpaqueDecoder:
    intention_width = 2

    def decide(self, intention, *, sample: bool = True):
        del sample
        return _Decision(
            intention.payload.detach().clone(),
            torch.ones(intention.payload.shape[0]),
        )


def _feedback() -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, 2),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )


def _machine() -> tuple[PolicyFreeAmodalLiveMachine, AmodalEventCollection]:
    torch.manual_seed(71)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=2,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_AdditiveFactualModel(), beam_width=4),
    )
    state = runtime.initial_state(1, device="cpu")
    event = AmodalEventCollection.from_events(
        [AmodalEvent(torch.randn(1, 4))], width=4
    )
    preview, _ = runtime.step_events(event, state, _feedback())
    goal = preview.controller.state_representation.detach().clone()
    goal[:, 0] += 1.0
    machine = PolicyFreeAmodalLiveMachine(
        policy_free,
        _OpaqueDecoder(),
        goal_state=goal,
        candidate_intentions=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        ),
        output_key="opaque",
    )
    return machine, event


def test_policy_free_machine_emits_planned_intention_and_consumes_receipt_outcome() -> None:
    machine, events = _machine()
    first = machine.tick(events, (), now=0.0, elapsed=0.0)
    assert len(first) == 1
    assert first[0].output_key == "opaque"
    assert first[0].credit_state is not None

    receipt = LiveActionReceipt(
        receipt_id=1,
        action=first[0].action,
        propensity=first[0].propensity,
        output_key="opaque",
        emitted_at=0.0,
        model_version=0,
    )
    outcome = LiveOutcomeEvent(
        receipt_id=1,
        reward=torch.tensor([1.0]),
        present=torch.tensor([True]),
        observed_at=1.0,
    )
    second = machine.tick(
        events,
        (ResolvedLiveOutcome(outcome, receipt, first[0]),),
        now=1.0,
        elapsed=1.0,
    )
    assert len(second) == 1
    assert machine.last_output is not None
    assert machine.configuration()["controller_input"] == (
        "amodal_event_collection_only"
    )


def test_policy_free_machine_requires_a_protocol_to_feedback_encoder() -> None:
    machine, events = _machine()
    machine.feedback_encoder = lambda action: action[:, :1]
    first = machine.tick(events, (), now=0.0, elapsed=0.0)[0]
    receipt = LiveActionReceipt(
        receipt_id=1,
        action=first.action,
        propensity=first.propensity,
        output_key="opaque",
        emitted_at=0.0,
        model_version=0,
    )
    outcome = LiveOutcomeEvent(
        receipt_id=1,
        reward=torch.tensor([1.0]),
        present=torch.tensor([True]),
        observed_at=1.0,
    )
    try:
        machine.tick(
            events,
            (ResolvedLiveOutcome(outcome, receipt, first),),
            now=1.0,
            elapsed=1.0,
        )
    except ValueError as error:
        assert "feedback encoder" in str(error)
    else:
        raise AssertionError("protocol action bypassed the feedback encoder")


def test_production_slice_has_no_navigation_oracle_dependency() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/neural_computer/navigation_runtime.py"
    ).read_text()
    for forbidden in (
        "verifier.place",
        "config.rule",
        "cluster_of_place",
        "scoring oracle",
    ):
        assert forbidden not in source
