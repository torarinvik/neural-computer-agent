import torch

from neural_computer import (
    ExternalOutcomeIntentionGenerator,
    ExternalOutcomeIntentionMemory,
    ExternalOutcomeIntentionRouter,
)


def _router() -> tuple[ExternalOutcomeIntentionRouter, object]:
    torch.manual_seed(5511)
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=4,
            intention_width=2,
            hidden_width=8,
            initial_learning_rate=0.1,
            initial_baseline_rate=0.05,
            noise_scale=0.3,
            initial_parameter_scale=0.05,
        )
    )
    router = ExternalOutcomeIntentionRouter(
        memory,
        initial_learning_rate=0.1,
        exploration_bonus=0.8,
    )
    return router, router.initial_state(2)


def test_router_selects_one_cell_and_credits_only_that_cell() -> None:
    router, state = _router()
    context = torch.zeros(1, 4)
    proposal = router.propose(state, context)
    selected = int(proposal.selected_cells.item())
    before = state
    state = router.record_decision(state, proposal)
    later = router.propose(state, context)
    state = router.apply_feedback(
        state,
        later,
        torch.zeros(1),
        present=torch.tensor([False]),
    )
    state = router.apply_feedback(state, proposal, torch.ones(1))

    assert proposal.candidates.intentions.shape == (1, 2, 2)
    assert proposal.selected_intentions.shape == (1, 2)
    assert state.routing_decisions.tolist() == [1 if selected == 0 else 0, 1 if selected == 1 else 0]
    assert int(state.routing_feedbacks.sum()) == 1
    assert not torch.equal(
        state.cells.output_weights[selected], before.cells.output_weights[selected]
    )
    other = 1 - selected
    assert torch.equal(state.cells.output_weights[other], before.cells.output_weights[other])


def test_router_explores_appended_cells_protects_content_and_reloads() -> None:
    router, state = _router()
    state = router.protect(state, [0])
    before = state
    state, new_cell = router.append_cell(state, source_cell=0)
    proposal = router.propose(state, torch.randn(1, 4))
    state = router.apply_feedback(
        state,
        proposal,
        torch.zeros(1),
        present=torch.tensor([False]),
    )

    assert new_cell == 2
    assert state.cells.baseline.shape == (3,)
    assert torch.equal(state.cells.output_weights[0], before.cells.output_weights[0])
    assert torch.equal(state.cells.output_weights[1], before.cells.output_weights[1])
    payload = router.state_payload(state)
    restored = router.state_from_payload(payload)
    assert torch.equal(restored.routing_keys, state.routing_keys)
    assert torch.equal(restored.cells.input_weights, state.cells.input_weights)
