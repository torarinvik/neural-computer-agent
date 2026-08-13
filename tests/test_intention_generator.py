import torch

from neural_computer import (
    EXTERNAL_OUTCOME_INTENTION_GENERATOR_SCHEMA_V1,
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
            "context_residual_weights",
            "input_weight_eligibility",
            "input_bias_eligibility",
            "output_weight_eligibility",
            "output_bias_eligibility",
            "context_residual_eligibility",
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
        "context_residual_weights",
        "input_weight_eligibility",
        "input_bias_eligibility",
        "output_weight_eligibility",
        "output_bias_eligibility",
        "context_residual_eligibility",
        "baseline",
        "decisions",
        "feedbacks",
        "protected",
    ):
        assert torch.equal(getattr(restored, name), getattr(state, name))

    legacy = generator.state_payload(state)
    legacy["schema"] = EXTERNAL_OUTCOME_INTENTION_GENERATOR_SCHEMA_V1
    legacy["configuration"].pop("mask_stable_content")
    legacy["configuration"].pop("factorized_context_residual")
    legacy.pop("context_residual_weights")
    legacy.pop("context_residual_eligibility")
    migrated = generator.state_from_payload(legacy)
    assert torch.equal(
        migrated.context_residual_weights,
        torch.zeros_like(state.context_residual_weights),
    )
    assert torch.equal(
        migrated.input_weights,
        state.input_weights,
    )

    upgraded_generator = ExternalOutcomeIntentionGenerator(
        context_width=2,
        intention_width=2,
        hidden_width=16,
        factorized_context_residual=True,
    )
    upgraded = upgraded_generator.state_from_payload(legacy)
    assert torch.equal(
        upgraded.context_residual_weights,
        torch.zeros_like(upgraded.context_residual_weights),
    )


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

    learned_mask_weights = state.input_weights.clone()
    learned_mask_weights[:, :, 4:8] = 1.0
    state = ExternalOutcomeIntentionGeneratorState(
        input_weights=learned_mask_weights,
        input_bias=state.input_bias,
        output_weights=state.output_weights,
        output_bias=state.output_bias,
        input_weight_eligibility=state.input_weight_eligibility,
        input_bias_eligibility=state.input_bias_eligibility,
        output_weight_eligibility=state.output_weight_eligibility,
        output_bias_eligibility=state.output_bias_eligibility,
        context_residual_weights=state.context_residual_weights,
        context_residual_eligibility=state.context_residual_eligibility,
        baseline=state.baseline,
        decisions=state.decisions,
        feedbacks=state.feedbacks,
        protected=state.protected,
    )
    grown, new_cell = generator.append_cell(state, source_cell=0)
    assert new_cell == 1
    assert torch.equal(grown.input_weights[new_cell, :, 4:8], torch.zeros(8, 4))


def test_mask_stable_content_disconnects_mask_from_mutable_hidden_program() -> None:
    generator = ExternalOutcomeIntentionGenerator(
        context_width=4,
        intention_width=2,
        hidden_width=8,
        context_masking=True,
        mask_stable_content=True,
    )
    state = generator.initial_state(1)
    context = torch.tensor([[1.0, 0.0, 3.0, 0.0]])
    first_mask = torch.tensor([[True, False, True, False]])
    second_mask = torch.tensor([[True, True, True, False]])
    first_mean = generator.mean(state, context, context_mask=first_mask)
    second_mean = generator.mean(state, context, context_mask=second_mask)

    assert torch.equal(first_mean, second_mean)
    proposal = generator.propose(
        state,
        context,
        context_mask=first_mask,
        generator=torch.Generator().manual_seed(73),
    )
    state = generator.record_decision(state, proposal)
    assert torch.equal(
        state.input_weight_eligibility[:, :, 4:8],
        torch.zeros(1, 8, 4),
    )
    restored = generator.state_from_payload(generator.state_payload(state))
    assert restored.input_weights.shape[-1] == 9


def test_factorized_residual_is_value_only_and_updates_as_external_state() -> None:
    generator = ExternalOutcomeIntentionGenerator(
        context_width=3,
        intention_width=2,
        hidden_width=8,
        context_masking=True,
        mask_stable_content=True,
        factorized_context_residual=True,
        initial_learning_rate=0.2,
    )
    state = generator.initial_state(1)
    state.context_residual_weights[0, 0, 0] = 0.75
    state.context_residual_weights[0, 0, -1] = -0.25
    context = torch.tensor([[2.0, 0.0, -1.0]])
    first_mask = torch.tensor([[True, False, True]])
    second_mask = torch.tensor([[True, True, True]])

    first_mean = generator.mean(state, context, context_mask=first_mask)
    second_mean = generator.mean(state, context, context_mask=second_mask)
    assert torch.equal(first_mean, second_mean)

    proposal = generator.propose(
        state,
        context,
        context_mask=first_mask,
        generator=torch.Generator().manual_seed(91),
    )
    state = generator.record_decision(state, proposal)
    updated = generator.apply_feedback(
        state,
        torch.ones(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    assert not torch.equal(
        updated.context_residual_weights,
        torch.zeros_like(updated.context_residual_weights),
    )
    assert torch.equal(
        updated.context_residual_eligibility,
        torch.zeros_like(updated.context_residual_eligibility),
    )
