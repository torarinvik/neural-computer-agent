from __future__ import annotations

import pytest
import torch

from neural_computer import (
    AdaptiveOnlineEpisodicRelationReader,
    EpisodicContextEncoder,
    EpisodicCreditHead,
    EpisodicIntentAdapter,
    ExternalCapabilityPipeline,
    ExternalCapabilityProgram,
    IntentEvent,
    OnlineEpisodicRelationReader,
    episodic_context_contrastive_loss,
    paired_event_credit_loss,
)


def test_episodic_context_encoder_masks_padding_and_normalizes_context() -> None:
    encoder = EpisodicContextEncoder(4, 3, hidden=8, context_width=6)
    events = torch.randn(2, 4, 4)
    actions = torch.randn(2, 4, 3)
    outcomes = torch.randn(2, 4)
    present = torch.tensor([[True, True, False, False], [True, True, True, False]])

    output = encoder(events, actions, outcomes, present)

    assert output.context.shape == (2, 6)
    assert torch.allclose(output.context.norm(dim=-1), torch.ones(2))
    assert torch.equal(output.credit_weights[~present], torch.zeros(3))
    assert torch.allclose(
        (output.credit_weights * present).sum(dim=-1),
        torch.ones(2),
    )


def test_episodic_context_contrastive_loss_has_gradient() -> None:
    left = torch.randn(4, 8, requires_grad=True)
    right = torch.randn(4, 8)

    loss = episodic_context_contrastive_loss(left, right)
    loss.backward()

    assert loss.ndim == 0
    assert left.grad is not None
    assert torch.isfinite(left.grad).all()


def test_external_credit_head_can_be_trained_without_changing_context_encoder() -> None:
    encoder = EpisodicContextEncoder(4, 2, hidden=8, context_width=6)
    head = EpisodicCreditHead(hidden=8, context_width=6)
    events = torch.randn(2, 3, 4)
    actions = torch.randn(2, 3, 2)
    outcomes = torch.zeros(2, 3)
    present = torch.ones(2, 3, dtype=torch.bool)
    output = encoder(events, actions, outcomes, present)
    before = output.context.detach().clone()

    logits = head(output.sequence.detach(), output.context.detach(), outcomes, present)
    loss, _ = paired_event_credit_loss(
        logits,
        torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            ]
        ),
        present=present,
    )
    loss.backward()

    assert torch.equal(before, output.context)
    assert head.network[-1].weight.grad is not None


def test_online_context_step_matches_full_episode_prefix() -> None:
    encoder = EpisodicContextEncoder(4, 2, hidden=8, context_width=6)
    events = torch.randn(2, 3, 4)
    actions = torch.randn(2, 3, 2)
    outcomes = torch.randn(2, 3)
    full = encoder(events, actions, outcomes)
    state = encoder.initial_state(2, device="cpu")
    online = None
    for index in range(3):
        online, state = encoder.step(
            events[:, index],
            actions[:, index],
            outcomes[:, index],
            state,
        )
    assert online is not None
    assert torch.allclose(online.context, full.context)


def test_episodic_intent_adapter_is_behavior_preserving_at_initialization() -> None:
    adapter = EpisodicIntentAdapter(context_width=6, intention_width=4, hidden=8)
    intention = IntentEvent(torch.randn(2, 4))
    adapted = adapter(intention, torch.randn(2, 6))
    assert torch.equal(adapted.payload, intention.payload)
    assert adapter.configuration()["schema"] == (
        "neural-computer.episodic-intent-adapter.v1"
    )


def test_external_capability_program_keeps_state_outside_controller() -> None:
    program = ExternalCapabilityProgram(
        event_width=4,
        action_width=2,
        intention_width=6,
        context_hidden=8,
        context_width=5,
        adapter_hidden=7,
    )
    intention = IntentEvent(torch.randn(3, 6))
    state = program.initial_state(3, device="cpu")
    adapted, next_state = program.step(
        event=torch.randn(3, 4),
        action=torch.zeros(3, 2),
        outcome=torch.zeros(3),
        intention=intention,
        state=state,
    )

    assert torch.equal(adapted.payload, intention.payload)
    assert next_state.context.shape == (3, 8)
    assert program.configuration()["schema"] == (
        "neural-computer.external-capability.v1"
    )


def test_external_capability_pipeline_keeps_program_states_independent() -> None:
    first = ExternalCapabilityProgram(
        event_width=4,
        action_width=2,
        intention_width=6,
        context_hidden=8,
        context_width=5,
        adapter_hidden=7,
    )
    second = ExternalCapabilityProgram(
        event_width=4,
        action_width=2,
        intention_width=6,
        context_hidden=9,
        context_width=5,
        adapter_hidden=7,
    )
    pipeline = ExternalCapabilityPipeline((first, second))
    intention = IntentEvent(torch.randn(3, 6))
    event = torch.randn(3, 4)
    action = torch.zeros(3, 2)
    outcome = torch.zeros(3)
    state = pipeline.initial_state(3, device="cpu")

    adapted, next_state = pipeline.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=state,
    )
    first_intention, first_state = first.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=state.programs[0],
    )
    expected, second_state = second.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=first_intention,
        state=state.programs[1],
    )

    assert torch.equal(adapted.payload, expected.payload)
    assert torch.equal(next_state.programs[0].context, first_state.context)
    assert torch.equal(next_state.programs[1].context, second_state.context)
    assert pipeline.configuration()["program_count"] == 2
    assert pipeline.configuration()["program_schemas"] == (
        "neural-computer.external-capability.v1",
        "neural-computer.external-capability.v1",
    )


def test_empty_external_capability_pipeline_is_identity() -> None:
    pipeline = ExternalCapabilityPipeline(
        event_width=4,
        action_width=2,
        intention_width=6,
    )
    intention = IntentEvent(torch.randn(3, 6))
    adapted, next_state = pipeline.step(
        event=torch.randn(3, 4),
        action=torch.zeros(3, 2),
        outcome=torch.zeros(3),
        intention=intention,
        state=pipeline.initial_state(3, device="cpu"),
    )

    assert torch.equal(adapted.payload, intention.payload)
    assert next_state.programs == ()
    assert pipeline.configuration()["program_count"] == 0


def test_online_relation_reader_returns_external_content_age_context() -> None:
    reader = OnlineEpisodicRelationReader(
        event_width=4,
        action_width=2,
        memory_capacity=3,
        context_width=6,
        hidden=8,
    )
    state = reader.initial_state(2, device="cpu")
    context, state = reader.step(
        torch.randn(2, 4),
        torch.zeros(2, 2),
        torch.zeros(2),
        state,
    )
    assert context.shape == (2, 6)
    assert state.events.shape == (2, 3, 4)
    assert bool(state.present[:, -1].all())
    assert reader.configuration()["schema"] == (
        "neural-computer.online-episodic-relation-reader.v1"
    )


def test_adaptive_relation_reader_scores_each_external_row() -> None:
    reader = AdaptiveOnlineEpisodicRelationReader(
        event_width=4,
        action_width=2,
        memory_capacity=5,
        context_width=6,
        hidden=8,
    )
    state = reader.initial_state(2, device="cpu")
    context, state = reader.step(
        torch.randn(2, 4),
        torch.zeros(2, 2),
        torch.zeros(2),
        state,
    )
    assert context.shape == (2, 6)
    assert state.events.shape == (2, 5, 4)
    assert reader.configuration()["schema"] == (
        "neural-computer.adaptive-online-episodic-relation-reader.v1"
    )


def test_adaptive_relation_reader_can_expand_without_losing_shared_weights() -> None:
    reader = AdaptiveOnlineEpisodicRelationReader(
        event_width=4,
        action_width=2,
        memory_capacity=5,
        context_width=6,
        hidden=8,
    )
    before = {
        name: value.detach().clone()
        for name, value in reader.named_parameters()
        if name != "age_embedding"
    }

    expanded = reader.expand_capacity(7)

    assert expanded.memory_capacity == 7
    assert expanded.age_embedding.shape == (7, 8)
    for name, value in before.items():
        assert torch.equal(value, dict(expanded.named_parameters())[name])


def test_paired_event_credit_loss_returns_detached_counterfactual_advantage() -> None:
    logits = torch.zeros(2, 3, requires_grad=True)
    utilities = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        ]
    )

    loss, advantage = paired_event_credit_loss(logits, utilities)
    loss.backward()

    assert advantage.tolist() == [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]
    assert not advantage.requires_grad
    assert logits.grad is not None


def test_episodic_context_rejects_mismatched_trajectory_shapes() -> None:
    encoder = EpisodicContextEncoder(4, 2)
    with pytest.raises(ValueError, match="share batch and time"):
        encoder(torch.randn(2, 3, 4), torch.randn(2, 2, 2), torch.randn(2, 3))
