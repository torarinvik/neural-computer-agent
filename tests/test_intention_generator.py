import torch

from neural_computer import (
    ExternalOutcomeIntentionGenerator,
    ExternalOutcomeIntentionGeneratorState,
)


def _run_generator(*, shuffled: bool) -> tuple[ExternalOutcomeIntentionGenerator, ExternalOutcomeIntentionGeneratorState, torch.Tensor]:
    torch.manual_seed(17)
    generator = ExternalOutcomeIntentionGenerator(
        context_width=2,
        intention_width=2,
        hidden_width=16,
        initial_learning_rate=0.15,
        initial_baseline_rate=0.05,
        noise_scale=0.4,
    )
    state = generator.initial_state(1)
    context = torch.tensor([[1.0, 0.0]])
    hidden_target = torch.tensor([1.0, -1.0])

    def verifier(intention: torch.Tensor) -> torch.Tensor:
        distance = (intention - hidden_target).square().sum(dim=-1)
        return torch.exp(-distance / 1.5).clamp(0.0, 1.0)

    for _ in range(800):
        proposal = generator.propose(state, context)
        outcome = verifier(proposal.intentions)
        if shuffled:
            outcome = torch.rand_like(outcome)
        state = generator.record_decision(state, proposal)
        state = generator.apply_feedback(
            state,
            outcome,
            terminal=torch.ones(1, dtype=torch.bool),
        )
    return generator, state, context


def test_outcome_only_generator_discovers_new_opaque_intention_content() -> None:
    generator, state, context = _run_generator(shuffled=False)
    mean = generator.mean(state, context)
    target = torch.tensor([[1.0, -1.0]])
    utility = torch.exp(-(mean - target).square().sum(dim=-1) / 1.5)

    assert float(utility.item()) > 0.7
    assert state.decisions.tolist() == [800]
    assert state.feedbacks.tolist() == [800]


def test_outcome_shuffling_breaks_generator_discovery() -> None:
    generator, state, context = _run_generator(shuffled=True)
    mean = generator.mean(state, context)
    target = torch.tensor([[1.0, -1.0]])
    utility = torch.exp(-(mean - target).square().sum(dim=-1) / 1.5)

    assert float(utility.item()) < 0.2


def test_generator_missing_feedback_is_a_no_op_and_protected_cells_retain() -> None:
    generator, state, context = _run_generator(shuffled=False)
    proposal = generator.propose(state, context)
    recorded = generator.record_decision(state, proposal)
    unchanged = generator.apply_feedback(
        recorded,
        torch.ones(1),
        present=torch.zeros(1, dtype=torch.bool),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    assert all(
        torch.equal(getattr(unchanged, name), getattr(recorded, name))
        for name in (
            "input_weights",
            "input_bias",
            "output_weights",
            "output_bias",
            "input_weight_eligibility",
            "input_bias_eligibility",
            "output_weight_eligibility",
            "output_bias_eligibility",
            "baseline",
            "decisions",
            "feedbacks",
        )
    )

    protected = generator.protect(state, [0])
    protected_before = protected
    proposal = generator.propose(protected, context)
    protected = generator.record_decision(protected, proposal)
    protected = generator.apply_feedback(
        protected,
        torch.zeros(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    assert torch.equal(protected.input_weights, protected_before.input_weights)
    assert torch.equal(protected.output_weights, protected_before.output_weights)
    assert torch.equal(protected.baseline, protected_before.baseline)
    assert protected.decisions.tolist() == [801]
    assert protected.feedbacks.tolist() == [801]


def test_generator_state_round_trips_exactly() -> None:
    generator, state, context = _run_generator(shuffled=False)
    proposal = generator.propose(state, context)
    state = generator.record_decision(state, proposal)
    payload = generator.state_payload(state)
    restored = generator.state_from_payload(payload)

    assert isinstance(restored, ExternalOutcomeIntentionGeneratorState)
    for name in (
        "input_weights",
        "input_bias",
        "output_weights",
        "output_bias",
        "input_weight_eligibility",
        "input_bias_eligibility",
        "output_weight_eligibility",
        "output_bias_eligibility",
        "baseline",
        "decisions",
        "feedbacks",
        "protected",
    ):
        assert torch.equal(getattr(restored, name), getattr(state, name))


def test_generator_appends_copy_on_write_cell_without_touching_old_cell() -> None:
    generator, state, context = _run_generator(shuffled=False)
    protected = generator.protect(state, [0])
    grown, new_cell = generator.append_cell(protected, source_cell=0)
    assert new_cell == 1
    assert grown.baseline.shape == (2,)
    assert torch.equal(grown.output_weights[0], protected.output_weights[0])
    assert torch.equal(grown.output_weights[1], protected.output_weights[0])
    assert not bool(grown.protected[1])
    new_before = grown.output_weights[1].clone()

    contexts = context.expand(2, -1).clone()
    proposal = generator.propose(grown, contexts)
    grown = generator.record_decision(grown, proposal)
    grown = generator.apply_feedback(
        grown,
        torch.tensor([0.0, 1.0]),
        terminal=torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(grown.output_weights[0], protected.output_weights[0])
    assert not torch.equal(grown.output_weights[1], new_before)


def test_masked_generator_keeps_missing_values_out_of_value_credit() -> None:
    generator = ExternalOutcomeIntentionGenerator(
        context_width=4,
        intention_width=2,
        hidden_width=8,
        initial_trace_decay=0.5,
        context_masking=True,
    )
    state = generator.initial_state(1)
    context = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, False, True, False]])
    proposal = generator.propose(
        state,
        context,
        context_mask=mask,
        generator=torch.Generator().manual_seed(19),
    )

    assert proposal.features.shape == (1, 9)
    assert torch.equal(
        proposal.features,
        torch.tensor([[1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),
    )
    state = generator.record_decision(state, proposal)
    assert torch.equal(state.input_weight_eligibility[:, :, 1], torch.zeros(1, 8))
    assert torch.equal(state.input_weight_eligibility[:, :, 3], torch.zeros(1, 8))

    restored = generator.state_from_payload(generator.state_payload(state))
    assert torch.equal(restored.input_weights, state.input_weights)
    assert restored.input_weights.shape[-1] == 9
