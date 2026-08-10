import torch

from neural_computer import (
    ExternalProgramArtifact,
    ExternalProgramCandidateSearch,
    ExternalProgramCandidateSearchState,
)


def _artifact(codes: torch.Tensor) -> ExternalProgramArtifact:
    return ExternalProgramArtifact(
        codes=codes,
        interpreter_schema="neural-computer.external-register.v4",
        execution_schema="neural-computer.external-register-read-execute.v1",
    )


def test_candidate_search_generates_and_adopts_a_verified_structural_program() -> None:
    atom_zero = torch.tensor([[1.0, 0.0, 0.0]])
    atom_one = torch.tensor([[0.0, 1.0, 0.0]])
    target = _artifact(torch.cat((atom_zero, atom_one), dim=0))
    parent = _artifact(atom_zero)
    search = ExternalProgramCandidateSearch(
        3,
        instruction_bank=torch.cat((atom_zero, atom_one), dim=0),
        max_program_length=2,
    )
    state = search.initial_state()
    generator = torch.Generator().manual_seed(31)

    accepted = None
    for _ in range(128):
        proposal = search.propose(state, parent, generator=generator)
        outcomes = (
            torch.ones(8)
            if proposal.artifact.digest() == target.digest()
            else torch.zeros(8)
        )
        feedback = search.record_outcomes(
            state,
            proposal,
            outcomes,
            min_observations=8,
            min_stable_observations=8,
        )
        state = feedback.state
        if feedback.receipt.accepted:
            accepted = feedback
            break

    assert accepted is not None
    assert accepted.proposal.artifact.digest() == target.digest()
    assert accepted.state.accepted == 1
    assert accepted.state.proposals == state.proposals
    assert "outcomes" not in accepted.state.payload()


def test_candidate_search_state_round_trip_contains_only_aggregate_evidence() -> None:
    search = ExternalProgramCandidateSearch(2)
    state = search.initial_state()
    restored = ExternalProgramCandidateSearchState.from_payload(state.payload())

    assert torch.equal(restored.reward_totals, state.reward_totals)
    assert torch.equal(restored.reward_counts, state.reward_counts)
    assert "outcomes" not in state.payload()
    assert restored.proposals == 0


def test_candidate_search_respects_length_bounds_and_preserves_parent() -> None:
    parent = _artifact(torch.eye(2))
    search = ExternalProgramCandidateSearch(
        2,
        instruction_bank=torch.eye(2),
        min_program_length=2,
        max_program_length=2,
    )
    state = search.initial_state()
    probabilities = search.proposal_probabilities(state, parent)

    assert probabilities[1].item() == 0.0
    assert probabilities[2].item() == 0.0
    assert probabilities[3].item() > 0.0
    proposal = search.propose(
        state,
        parent,
        generator=torch.Generator().manual_seed(9),
    )
    assert proposal.artifact.program_length == parent.program_length
    assert proposal.parent_digest == parent.digest()
    assert torch.equal(parent.codes, torch.eye(2))
