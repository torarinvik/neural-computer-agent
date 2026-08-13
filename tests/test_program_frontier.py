import torch

from neural_computer import (
    ExternalProgramArtifact,
    ExternalProgramCandidateSearch,
    ExternalProgramHypothesisFrontier,
    ExternalProgramHypothesisFrontierState,
)


def _artifact(codes: torch.Tensor) -> ExternalProgramArtifact:
    return ExternalProgramArtifact(
        codes=codes,
        interpreter_schema="neural-computer.external-register.v4",
        execution_schema="neural-computer.external-register-read-execute.v1",
    )


def _prefix_quality(artifact: ExternalProgramArtifact) -> float:
    atoms = {
        (1.0, 0.0, 0.0): 0,
        (0.0, 1.0, 0.0): 1,
        (0.0, 0.0, 1.0): 2,
    }
    target = (0, 1, 2)
    prefix = 0
    for row, expected in zip(artifact.codes, target, strict=False):
        if tuple(float(value) for value in row) not in atoms:
            break
        if atoms[tuple(float(value) for value in row)] != expected:
            break
        prefix += 1
    return prefix / len(target)


def test_frontier_finds_a_multi_step_program_and_keeps_the_root() -> None:
    atoms = torch.eye(3)
    root = _artifact(atoms[0:1])
    target = _artifact(atoms)
    search = ExternalProgramCandidateSearch(
        3,
        instruction_bank=atoms,
        max_program_length=3,
        exploration=0.5,
        temperature=0.5,
    )
    frontier = ExternalProgramHypothesisFrontier(
        search,
        beam_width=4,
        max_depth=3,
        minimum_quality=0.0,
        parent_temperature=0.15,
    )
    state = frontier.initial_state(root, root_quality=_prefix_quality(root))
    generator = torch.Generator().manual_seed(47)

    accepted = False
    for _ in range(512):
        proposal = frontier.propose(state, generator=generator)
        quality = _prefix_quality(proposal.artifact)
        state, feedback = frontier.record_outcomes(
            state,
            proposal,
            torch.full((8,), quality),
            threshold=0.95,
            min_observations=8,
            min_stable_observations=8,
        )
        if feedback.receipt.accepted:
            accepted = True
            assert proposal.artifact.digest() == target.digest()
            break

    assert accepted
    assert state.root_digest == root.digest()
    assert any(
        hypothesis.artifact.digest() == root.digest()
        for hypothesis in state.hypotheses
    )
    assert any(
        hypothesis.artifact.digest() == target.digest()
        and hypothesis.depth == 2
        for hypothesis in state.hypotheses
    )


def test_frontier_payload_round_trip_preserves_opaque_hypotheses_without_outcomes() -> None:
    root = _artifact(torch.tensor([[1.0, 0.0]]))
    frontier = ExternalProgramHypothesisFrontier(
        ExternalProgramCandidateSearch(
            2,
            instruction_bank=torch.eye(2),
            max_program_length=2,
        )
    )
    state = frontier.initial_state(root)
    proposal = frontier.propose(
        state,
        generator=torch.Generator().manual_seed(5),
    )
    state, _ = frontier.record_outcomes(state, proposal, torch.zeros(4))
    payload = state.payload()
    restored = ExternalProgramHypothesisFrontierState.from_payload(payload)

    assert restored.digest() == state.digest()
    assert "outcomes" not in payload
    assert restored.root_digest == root.digest()
    assert restored.evaluations == 1
