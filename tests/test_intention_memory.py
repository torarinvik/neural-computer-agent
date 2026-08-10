import torch

from neural_computer import (
    ExternalOutcomeIntentionGenerator,
    ExternalOutcomeIntentionMemory,
)


def _memory() -> tuple[ExternalOutcomeIntentionMemory, object]:
    torch.manual_seed(4401)
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=4,
            intention_width=2,
            hidden_width=8,
            initial_learning_rate=0.2,
            initial_baseline_rate=0.05,
            noise_scale=0.35,
            initial_parameter_scale=0.05,
        )
    )
    return memory, memory.initial_state(3)


def test_memory_cells_are_independent_from_controller_batch_and_delayed_credit() -> None:
    memory, state = _memory()
    context = torch.zeros(1, 4)
    proposal = memory.propose(state, context)
    assert proposal.intentions.shape == (1, 3, 2)
    assert proposal.log_propensities.shape == (1, 3)

    before = state
    state = memory.record_decision(state, proposal, torch.tensor([1]))
    later_proposal = memory.propose(state, context)
    state = memory.record_decision(state, later_proposal, torch.tensor([0]))
    state = memory.apply_feedback(
        state,
        later_proposal,
        torch.tensor([0]),
        torch.zeros(1),
    )
    state = memory.apply_feedback(
        state,
        proposal,
        torch.tensor([1]),
        torch.ones(1),
    )

    assert state.decisions.tolist() == [1, 1, 0]
    assert state.feedbacks.tolist() == [1, 1, 0]
    assert not torch.equal(state.output_weights[0], before.output_weights[0])
    assert not torch.equal(state.output_weights[1], before.output_weights[1])
    assert later_proposal.intentions.shape == (1, 3, 2)


def test_memory_protected_cell_missing_feedback_and_reload_are_exact() -> None:
    memory, state = _memory()
    state = memory.protect(state, [0])
    context = torch.randn(1, 4)
    proposal = memory.propose(state, context)
    before = state
    state = memory.record_decision(
        state,
        proposal,
        torch.tensor([0]),
        present=torch.tensor([False]),
    )
    state = memory.apply_feedback(
        state,
        proposal,
        torch.tensor([0]),
        torch.zeros(1),
        present=torch.tensor([False]),
    )
    assert torch.equal(state.input_weights, before.input_weights)
    assert torch.equal(state.output_weights, before.output_weights)
    assert state.decisions.tolist() == [0, 0, 0]
    assert state.feedbacks.tolist() == [0, 0, 0]

    payload = memory.state_payload(state)
    restored = memory.state_from_payload(payload)
    assert torch.equal(restored.input_weights, state.input_weights)
    assert torch.equal(restored.output_weights, state.output_weights)
    assert torch.equal(restored.protected, state.protected)


def test_memory_sparse_proposal_credits_physical_cell_id() -> None:
    memory, state = _memory()
    context = torch.randn(1, 4)
    before = state
    proposal = memory.propose(state, context, cell_indices=[2])

    assert proposal.intentions.shape == (1, 1, 2)
    assert proposal.cell_indices == (2,)
    state = memory.record_decision(state, proposal, torch.tensor([2]))
    state = memory.apply_feedback(state, proposal, torch.tensor([2]), torch.ones(1))

    assert state.decisions.tolist() == [0, 0, 1]
    assert state.feedbacks.tolist() == [0, 0, 1]
    assert torch.equal(state.output_weights[0], before.output_weights[0])
    assert torch.equal(state.output_weights[1], before.output_weights[1])
    assert not torch.equal(state.output_weights[2], before.output_weights[2])
