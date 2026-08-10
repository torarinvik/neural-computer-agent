import pytest
import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    ExternalIntentionRepertoire,
    ExternalModelBasedPlanner,
    PolicyFreeAmodalRuntime,
)


class _AdditiveFactualModel(nn.Module):
    state_width = 12
    intention_width = 2

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, :2] = result[:, :2] + intention
        return result


class _EchoDecoder(nn.Module):
    def forward(self, intention):
        return intention.payload


def _feedback() -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )


def test_repertoire_grows_from_opaque_outcomes_without_reward_ranking() -> None:
    repertoire = ExternalIntentionRepertoire(2)
    first = repertoire.observe(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        utility=torch.tensor([1.0, 0.0]),
        propensity=torch.tensor([0.5, 1.0]),
        timestamp=torch.tensor([7, 8]),
    )
    assert first.added == (True, True)
    assert repertoire.record_count == 2

    duplicate = repertoire.observe(
        torch.tensor([[2.0, 0.0]]),
        utility=0.25,
        propensity=0.25,
        timestamp=9,
    )
    assert duplicate.added == (False,)
    assert repertoire.record_count == 2
    stats = repertoire.statistics()
    assert stats["attempts"].tolist() == [2, 1]
    assert stats["outcome_counts"].tolist() == [2, 1]
    assert torch.allclose(
        stats["inverse_propensity_utility_sums"],
        torch.tensor([3.0, 0.0], dtype=torch.float64),
    )

    proposal = repertoire.propose(torch.tensor([0.0, -1.0]))
    assert proposal.intentions.shape == (1, 3, 2)
    assert proposal.source_indices == (-1, 0, 1)
    assert proposal.exploration_mask.tolist() == [[True, False, False]]
    assert torch.allclose(proposal.propensities, torch.full((1, 3), 1 / 3))

    restored = ExternalIntentionRepertoire.from_payload(repertoire.payload())
    assert restored.content_digest() == repertoire.content_digest()
    corrupt = repertoire.payload()
    corrupt["intentions"] = corrupt["intentions"].clone()
    corrupt["intentions"][0, 0] += 0.1
    with pytest.raises(ValueError, match="checksum"):
        ExternalIntentionRepertoire.from_payload(corrupt)


def test_verified_intention_admission_is_copy_on_write_and_retention_safe() -> None:
    repertoire = ExternalIntentionRepertoire(2)
    repertoire.observe(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    candidate_intention = torch.tensor([0.5, 0.5])

    def accept(candidate: ExternalIntentionRepertoire) -> bool:
        candidate.observe(candidate_intention, utility=1.0, propensity=1.0)
        return True

    accepted = repertoire.admit_verified(candidate_intention, accept)
    assert accepted.accepted
    assert accepted.entry_index == 2
    assert repertoire.record_count == 3
    assert accepted.destination_digest == repertoire.content_digest()

    rejected_digest = repertoire.content_digest()
    rejected = repertoire.admit_verified(
        torch.tensor([0.25, -0.75]),
        lambda candidate: False,
    )
    assert not rejected.accepted
    assert rejected.source_digest == rejected.destination_digest == rejected_digest
    assert repertoire.content_digest() == rejected_digest

    def mutating_verifier(candidate: ExternalIntentionRepertoire) -> bool:
        candidate.observe(torch.tensor([1.0, 0.0]), utility=0.0)
        return True

    mutation_rejected = repertoire.admit_verified(
        torch.tensor([-0.5, 0.5]),
        mutating_verifier,
    )
    assert not mutation_rejected.accepted
    assert mutation_rejected.source_digest == rejected_digest
    assert repertoire.content_digest() == rejected_digest


def test_policy_free_runtime_can_source_candidates_from_external_repertoire() -> None:
    torch.manual_seed(19)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    before = {
        name: value.detach().clone() for name, value in controller.state_dict().items()
    }
    runtime = AmodalControllerRuntime(controller)
    runtime.register_decoder("echo", _EchoDecoder())
    repertoire = ExternalIntentionRepertoire(2)
    repertoire.observe(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    )
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_AdditiveFactualModel(), beam_width=8),
        intention_repertoire=repertoire,
    )
    state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    event = [AmodalEvent(torch.randn(1, 4))]
    preview, _ = runtime.step_events(event, state, feedback)
    goal = preview.controller.state_representation.detach().clone()
    goal[:, :2] += 1.0

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        horizon=2,
        beam_width=8,
        goal_progress_weight=1.0,
    )
    assert output.proposal is not None
    assert output.proposal.intentions.shape[1] == 4
    assert not bool(output.proposal.exploration_mask.any())
    assert torch.allclose(output.planning.predicted_states[0, -1], goal[0])
    assert torch.allclose(output.decoded["echo"], output.intention.payload)
    assert policy_free.configuration()["candidate_intentions"] == (
        "external_append_only_intention_repertoire_v1"
    )

    receipt = policy_free.observe_intention(
        output.intention,
        utility=1.0,
        propensity=1.0 / output.proposal.intentions.shape[1],
    )
    assert receipt.outcome_observed
    for name, value in controller.state_dict().items():
        assert torch.equal(value, before[name])


def test_policy_free_runtime_requires_candidates_or_repertoire() -> None:
    runtime = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=4,
            workspace_slots=2,
            intention_width=2,
            feedback_width=3,
        )
    )
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_AdditiveFactualModel()),
    )
    state = runtime.initial_state(1, device="cpu")
    preview, _ = runtime.step_events([AmodalEvent(torch.randn(1, 4))], state, _feedback())
    with pytest.raises(ValueError, match="candidate intentions"):
        policy_free.step_events(
            [AmodalEvent(torch.randn(1, 4))],
            state,
            _feedback(),
            preview.controller.state_representation[:, :12],
            horizon=1,
        )
