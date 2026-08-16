from __future__ import annotations

import pytest
import torch

from neural_computer import (
    ExternalCausalIdentityArtifact,
    ExternalCausalIdentityAssignment,
)


def _bound_histories() -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
    )
    deltas = torch.zeros(1, 6, 2, 2)
    deltas[0, :, 0, 0] = 2.0 * actions[0, :, 0] + actions[0, :, 1]
    deltas[0, :, 1, 0] = torch.tensor([0.0, 1.0, -1.0, 0.0, 1.0, -1.0])
    events = torch.cat(
        [torch.zeros(1, 1, 2, 2), deltas.cumsum(dim=1)], dim=1
    )
    return events, actions


def test_causal_identity_artifact_prefers_action_dependent_bound_track() -> None:
    events, actions = _bound_histories()
    evidence = ExternalCausalIdentityArtifact().evidence(events, actions)

    assert evidence.shape == (1, 2)
    assert evidence[0, 0] > evidence[0, 1]
    assignment = ExternalCausalIdentityAssignment(margin=0.2).resolve(evidence)
    assert assignment.selected_slot.tolist() == [0]
    assert not bool(assignment.abstained[0])


def test_causal_identity_artifact_abstains_when_action_has_no_variation() -> None:
    events, actions = _bound_histories()
    constant_actions = torch.ones_like(actions)
    evidence = ExternalCausalIdentityArtifact().evidence(events, constant_actions)

    assert torch.equal(evidence, torch.zeros_like(evidence))
    assignment = ExternalCausalIdentityAssignment(margin=0.2).resolve(evidence)
    assert bool(assignment.abstained[0])


def test_identity_assignment_abstains_when_margin_is_high_but_evidence_is_weak() -> None:
    assignment = ExternalCausalIdentityAssignment(
        margin=0.1, minimum_evidence=0.2
    ).resolve(torch.tensor([[0.15, 0.0]]))

    assert bool(assignment.abstained[0])


@pytest.mark.parametrize(
    ("events", "actions", "message"),
    [
        (
            torch.zeros(1, 3, 2, 2),
            torch.zeros(1, 2, 2),
            "shorter",
        ),
        (
            torch.zeros(1, 4, 2, 2),
            torch.zeros(1, 4, 2),
            "align",
        ),
        (
            torch.zeros(1, 4, 2),
            torch.zeros(1, 3, 2),
            "shape",
        ),
    ],
)
def test_causal_identity_artifact_rejects_unbound_histories(
    events: torch.Tensor, actions: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ExternalCausalIdentityArtifact().evidence(events, actions)
