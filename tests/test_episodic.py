from __future__ import annotations

import pytest
import torch

from neural_computer import (
    AdaptiveOnlineEpisodicRelationReader,
    AppendOnlyLearnedComputeCandidateScreen,
    EpisodicContextEncoder,
    EpisodicCreditHead,
    EpisodicIntentAdapter,
    ExternalCapabilityComposition,
    ExternalCapabilityPipeline,
    ExternalCapabilityProgram,
    ExternalCapabilityResidualComputeBank,
    ExternalCapabilityReusableComputeLibrary,
    ExternalCapabilitySharedResidualBank,
    ExternalComputeCandidateScreen,
    IntentEvent,
    LearnedComputeCandidateScreen,
    OnlineEpisodicRelationReader,
    episodic_context_contrastive_loss,
    paired_event_credit_loss,
    select_reusable_compute_slot,
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
    assert torch.allclose(adapted.payload, intention.payload)
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


def test_shared_residual_bank_grows_without_changing_old_slot_or_state() -> None:
    bank = ExternalCapabilitySharedResidualBank(
        event_width=4,
        action_width=2,
        intention_width=6,
        slot_count=1,
        context_hidden=8,
        context_width=5,
        adapter_hidden=7,
    )
    old_shared = {
        name: value.detach().clone()
        for name, value in bank.shared_context_encoder.state_dict().items()
    }
    old_residual = {
        name: value.detach().clone()
        for name, value in bank.residual_slots[0].state_dict().items()
    }

    assert bank.add_slot() == 1
    assert bank.slot_count == 2
    assert all(
        torch.equal(value, bank.shared_context_encoder.state_dict()[name])
        for name, value in old_shared.items()
    )
    assert all(
        torch.equal(value, bank.residual_slots[0].state_dict()[name])
        for name, value in old_residual.items()
    )

    state = bank.initial_state(2, device="cpu")
    kwargs = {
        "event": torch.randn(2, 4),
        "action": torch.zeros(2, 2),
        "outcome": torch.zeros(2),
        "intention": IntentEvent(torch.randn(2, 6)),
        "state": state,
    }
    slot_adapted, slot_next = bank.step_slot(
        0,
        kwargs["event"],
        kwargs["action"],
        kwargs["outcome"],
        intention=kwargs["intention"],
        state=state.programs[0],
    )
    adapted, next_state = bank.step(slot_index=0, **kwargs)

    assert torch.equal(slot_adapted.payload, adapted.payload)
    assert torch.equal(slot_next.context, next_state.programs[0].context)
    assert torch.equal(adapted.payload, kwargs["intention"].payload)
    assert not torch.equal(next_state.programs[0].context, state.programs[0].context)
    assert torch.equal(next_state.programs[1].context, state.programs[1].context)
    assert bank.configuration()["schema"] == (
        "neural-computer.external-capability-shared-residual.v1"
    )
    bank.freeze_shared_base()
    assert all(
        not parameter.requires_grad
        for parameter in bank.shared_context_encoder.parameters()
    )
    assert all(parameter.requires_grad for parameter in bank.residual_slots[1].parameters())
    bank.freeze_slot(0)
    assert all(
        not parameter.requires_grad for parameter in bank.residual_slots[0].parameters()
    )


def test_residual_compute_bank_adds_local_recurrent_capacity() -> None:
    bank = ExternalCapabilityResidualComputeBank(
        event_width=4,
        action_width=2,
        intention_width=6,
        slot_count=1,
        shared_context_hidden=8,
        shared_context_width=5,
        residual_context_hidden=3,
        residual_context_width=2,
        adapter_hidden=7,
    )
    old_shared = {
        name: value.detach().clone()
        for name, value in bank.shared_context_encoder.state_dict().items()
    }
    old_slot = {
        name: value.detach().clone()
        for name, value in bank.residual_slots[0].state_dict().items()
    }
    assert bank.add_slot() == 1
    assert all(
        torch.equal(value, bank.shared_context_encoder.state_dict()[name])
        for name, value in old_shared.items()
    )
    assert all(
        torch.equal(value, bank.residual_slots[0].state_dict()[name])
        for name, value in old_slot.items()
    )
    state = bank.initial_state(2, device="cpu")
    event = torch.randn(2, 4)
    action = torch.zeros(2, 2)
    outcome = torch.zeros(2)
    intention = IntentEvent(torch.randn(2, 6))
    adapted, next_state = bank.step_slot(
        slot_index=0,
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=state.programs[0],
    )
    assert adapted.payload.shape == (2, 6)
    assert next_state.context.shape == (2, 11)
    assert not torch.equal(next_state.context, state.programs[0].context)
    bank.freeze_shared_base()
    bank.freeze_slot(0)
    assert all(
        not parameter.requires_grad
        for parameter in bank.shared_context_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in bank.residual_slots[0].parameters()
    )
    assert bank.configuration()["schema"] == (
        "neural-computer.external-capability-residual-compute.v1"
    )


def test_reusable_compute_library_shares_physical_compute_with_isolated_bindings() -> None:
    library = ExternalCapabilityReusableComputeLibrary(
        event_width=4,
        action_width=2,
        intention_width=6,
        compute_slot_count=1,
        binding_compute_slots=(0,),
        shared_context_hidden=8,
        shared_context_width=5,
        residual_context_hidden=3,
        residual_context_width=2,
        adapter_hidden=7,
    )
    old_compute = {
        name: value.detach().clone()
        for name, value in library.compute_slots[0].state_dict().items()
    }
    assert library.add_binding(0) == 1
    assert library.slot_count == 2
    assert library.compute_slot_count == 1
    assert library.binding_compute_slots == (0, 0)
    assert all(
        torch.equal(value, library.compute_slots[0].state_dict()[name])
        for name, value in old_compute.items()
    )
    state = library.initial_state(2, device="cpu")
    kwargs = {
        "event": torch.randn(2, 4),
        "action": torch.zeros(2, 2),
        "outcome": torch.zeros(2),
        "intention": IntentEvent(torch.randn(2, 6)),
    }
    first, first_state = library.step_binding(
        binding_index=0,
        state=state.programs[0],
        **kwargs,
    )
    second, second_state = library.step_binding(
        binding_index=1,
        state=state.programs[1],
        **kwargs,
    )
    assert torch.equal(first.payload, second.payload)
    assert not torch.equal(first_state.context, state.programs[0].context)
    assert not torch.equal(second_state.context, state.programs[1].context)
    library.freeze_shared_base()
    library.freeze_compute_slot(0)
    library.freeze_binding(0)
    assert all(
        not parameter.requires_grad
        for parameter in library.shared_context_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in library.compute_slots[0].parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in library.binding_adapters[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in library.binding_adapters[1].parameters()
    )
    assert library.configuration()["schema"] == (
        "neural-computer.external-capability-reusable-compute.v1"
    )
    library.remove_binding(1)
    assert library.slot_count == 1
    assert library.compute_slot_count == 1


def test_reusable_compute_admission_requires_every_fresh_probe() -> None:
    reuse = select_reusable_compute_slot(
        {3: (0.82, 0.91), 1: (0.84, 0.84)}, threshold=0.8
    )
    assert reuse.action == "reuse"
    assert reuse.compute_slot_index == 1
    reject = select_reusable_compute_slot({0: (0.8, 0.74)}, threshold=0.8)
    assert reject.action == "grow"
    assert reject.compute_slot_index is None
    empty = select_reusable_compute_slot({}, threshold=0.8)
    assert empty.action == "grow"
    assert empty.reason == "no_compute_candidates"


def test_compute_candidate_screen_orders_from_learned_event_outcomes_only() -> None:
    screen = ExternalComputeCandidateScreen(width=4)
    assert screen.add_candidate() == 0
    assert screen.add_candidate() == 1
    assert screen.add_candidate() == 2
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])

    assert screen.order(query) == (0, 1, 2)
    screen.observe(query, 2, 1.0)

    assert screen.order(query) == (2, 1, 0)
    unseen_query = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert screen.order(unseen_query) == (2, 1, 0)
    screen.observe(unseen_query, 1, 1.0)
    assert screen.order(query) == (2, 1, 0)
    assert screen.order(unseen_query) == (1, 2, 0)
    assert screen.configuration()["role"] == "order_only_fresh_admission_required"


def test_compute_candidate_screen_reloads_without_semantic_metadata() -> None:
    screen = ExternalComputeCandidateScreen(width=3, matching_tolerance=1e-3)
    screen.add_candidate()
    screen.add_candidate()
    query = torch.tensor([0.0, 1.0, 0.0])
    screen.observe(query, 1, 1.0)

    restored = ExternalComputeCandidateScreen.from_payload(screen.payload())

    assert restored.payload() == screen.payload()
    assert restored.order(query) == (1, 0)
    assert "task" not in restored.payload()
    assert "label" not in restored.payload()


def test_learned_compute_screen_is_neutral_and_permutation_equivariant() -> None:
    screen = LearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    query = torch.randn(2, 4)
    keys = torch.randn(3, 3)
    permutation = torch.tensor([2, 0, 1])

    scores = screen(query, keys)
    permuted_scores = screen(query, keys[permutation])

    assert torch.equal(scores, torch.zeros_like(scores))
    assert torch.allclose(permuted_scores, scores[:, permutation])
    assert screen.order(query[0], keys) == (0, 1, 2)
    assert screen.configuration()["role"] == "order_only_fresh_admission_required"


def test_learned_compute_screen_ranking_loss_uses_only_scalar_outcomes() -> None:
    screen = LearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    query = torch.randn(2, 4)
    keys = torch.randn(3, 3)
    outcomes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    screen.enable()
    loss, informative = screen.outcome_ranking_loss(query, keys, outcomes)
    loss.backward()

    assert loss.ndim == 0
    assert informative == 4
    assert any(parameter.grad is not None for parameter in screen.parameters())


def test_learned_compute_screen_calibrates_a_single_attempted_candidate() -> None:
    screen = LearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    query = torch.randn(4, 4)
    keys = torch.randn(1, 3)
    attempted = torch.zeros(4, dtype=torch.long)
    outcomes = torch.tensor([1.0, 0.0, 1.0, 0.0])

    screen.enable()
    loss, informative = screen.outcome_calibration_loss(
        query,
        keys,
        attempted,
        outcomes,
    )
    loss.backward()

    assert loss.ndim == 0
    assert informative == 4
    assert any(parameter.grad is not None for parameter in screen.parameters())


def test_learned_compute_screen_state_round_trips() -> None:
    torch.manual_seed(11)
    screen = LearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    restored = LearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    restored.load_state_dict(screen.state_dict(), strict=True)
    query = torch.randn(2, 4)
    keys = torch.randn(3, 3)

    assert torch.equal(screen(query, keys), restored(query, keys))
    assert screen.configuration() == restored.configuration()


def test_append_only_learned_screen_preserves_base_until_failure() -> None:
    torch.manual_seed(19)
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    screen.enable_base()
    base_before = {
        name: value.detach().clone()
        for name, value in screen.base_screen.state_dict().items()
    }
    assert screen.append_extension(2) == 0
    screen.enable_extension(0)
    query = torch.randn(3, 4)
    base_keys = torch.randn(3, 3)
    extension_keys = torch.randn(2, 3)
    base_scores = screen.base_screen(query, base_keys)

    cold = screen(query, base_keys, extension_keys)
    assert torch.equal(cold[:, :3], base_scores)
    assert torch.equal(cold.argmax(dim=-1), base_scores.argmax(dim=-1))

    failed = screen(
        query,
        base_keys,
        extension_keys,
        failed_extensions=torch.ones(3, 1, dtype=torch.bool),
    )
    assert torch.equal(failed[:, :3], base_scores)
    expected_extension = base_scores.max(dim=-1).values.unsqueeze(1) + screen.extensions[
        0
    ](query, extension_keys)
    assert torch.allclose(failed[:, 3:], expected_extension)
    assert all(
        torch.equal(value, screen.base_screen.state_dict()[name])
        for name, value in base_before.items()
    )


def test_append_only_learned_screen_can_copy_base_as_independent_prior() -> None:
    torch.manual_seed(21)
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1,),
    )
    screen.enable_base()
    base_before = {
        name: value.detach().clone()
        for name, value in screen.base_screen.state_dict().items()
    }

    screen.initialize_extension_from_base(0)

    assert not bool(screen.extensions[0].enabled.item())
    assert all(
        torch.equal(value, screen.extensions[0].state_dict()[name])
        for name, value in base_before.items()
        if name != "enabled"
    )
    with torch.no_grad():
        screen.extensions[0].query_projection[0].bias.add_(1.0)
    assert all(
        torch.equal(value, screen.base_screen.state_dict()[name])
        for name, value in base_before.items()
    )


def test_append_only_learned_screen_state_round_trips() -> None:
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1, 2),
    )
    screen.enable_base()
    screen.enable_extension(1)
    restored = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1, 2),
    )
    restored.load_state_dict(screen.state_dict(), strict=True)
    assert screen.configuration() == restored.configuration()
    query = torch.randn(2, 4)
    base_keys = torch.randn(2, 3)
    extension_keys = torch.randn(3, 3)
    failures = torch.tensor([[False, True], [True, False]])
    assert torch.equal(
        screen(query, base_keys, extension_keys, failures),
        restored(query, base_keys, extension_keys, failures),
    )


def test_append_only_learned_screen_cannot_skip_an_unfailed_prior_stage() -> None:
    torch.manual_seed(23)
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1, 1),
    )
    screen.enable_base()
    screen.enable_extension(0)
    screen.enable_extension(1)
    with torch.no_grad():
        screen.extensions[0].router.query_encoder[-1].bias.zero_()
        screen.extensions[0].router.key_encoder[-1].bias.zero_()
        screen.extensions[1].router.query_encoder[-1].bias.fill_(3.0)
        screen.extensions[1].router.key_encoder[-1].bias.fill_(3.0)
    query = torch.randn(2, 4)
    base_keys = torch.randn(2, 3)
    extension_keys = torch.randn(2, 3)

    skipped = screen(
        query,
        base_keys,
        extension_keys,
        failed_extensions=torch.tensor([[False, True], [False, True]]),
    )

    assert bool((skipped.argmax(dim=-1) < 2).all())


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


def test_external_capability_composition_routes_external_slots_and_keeps_identity() -> None:
    programs = tuple(
        ExternalCapabilityProgram(
            event_width=4,
            action_width=2,
            intention_width=6,
            context_hidden=8,
            context_width=5,
            adapter_hidden=7,
        )
        for _ in range(2)
    )
    composition = ExternalCapabilityComposition(
        programs,
        composition_steps=2,
        router_hidden=9,
    )
    intention = IntentEvent(torch.randn(3, 6))
    state = composition.initial_state(3, device="cpu")
    adapted, next_state = composition.step(
        event=torch.randn(3, 4),
        action=torch.zeros(3, 2),
        outcome=torch.zeros(3),
        intention=intention,
        state=state,
    )

    assert torch.allclose(adapted.payload, intention.payload)
    assert len(next_state.programs) == 2
    assert all(item.context.shape == (3, 8) for item in next_state.programs)
    assert composition.configuration()["composition_steps"] == 2
    assert composition.configuration()["routing"] == (
        "learned_event_conditioned_soft_slot_binding_v1"
    )


def test_external_capability_composition_accepts_opaque_slot_binding() -> None:
    programs = tuple(
        ExternalCapabilityProgram(
            event_width=4,
            action_width=2,
            intention_width=6,
            context_hidden=8,
            context_width=5,
            adapter_hidden=7,
        )
        for _ in range(2)
    )
    composition = ExternalCapabilityComposition(programs, composition_steps=1)
    state = composition.initial_state(2, device="cpu")
    kwargs = {
        "event": torch.randn(2, 4),
        "action": torch.zeros(2, 2),
        "outcome": torch.zeros(2),
        "intention": IntentEvent(torch.randn(2, 6)),
        "state": state,
    }

    adapted, next_state = composition.step(
        **kwargs,
        slot_mask=torch.tensor([[True, False], [True, False]]),
    )

    assert adapted.payload.shape == (2, 6)
    assert torch.equal(next_state.programs[1].context, state.programs[1].context)
    assert composition.configuration()["binding"] == (
        "optional_opaque_external_slot_mask_v1"
    )
    assert composition.configuration()["execution"] == "masked_sparse_active_slots_v1"
    mixed_state = composition.initial_state(2, device="cpu")
    _, mixed_next_state = composition.step(
        **{**kwargs, "state": mixed_state},
        slot_mask=torch.tensor([[True, False], [False, True]]),
    )
    assert torch.equal(
        mixed_next_state.programs[0].context[1], mixed_state.programs[0].context[1]
    )
    assert torch.equal(
        mixed_next_state.programs[1].context[0], mixed_state.programs[1].context[0]
    )
    with pytest.raises(ValueError, match="at least one slot"):
        composition.step(**kwargs, slot_mask=torch.zeros(2, 2, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        composition.step(**kwargs, slot_mask=torch.ones(2, 2))


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


def test_external_capability_pipeline_can_hide_events_after_first_program() -> None:
    def make_programs() -> tuple[ExternalCapabilityProgram, ...]:
        return tuple(
            ExternalCapabilityProgram(
                event_width=4,
                action_width=2,
                intention_width=6,
                context_hidden=8,
                context_width=5,
                adapter_hidden=7,
            )
            for _ in range(2)
        )

    visible = ExternalCapabilityPipeline(make_programs())
    hidden = ExternalCapabilityPipeline(
        make_programs(),
        hide_downstream_events=True,
    )
    intention = IntentEvent(torch.randn(3, 6))
    event = torch.randn(3, 4)
    action = torch.zeros(3, 2)
    outcome = torch.zeros(3)
    visible_state = visible.initial_state(3, device="cpu")
    hidden_state = hidden.initial_state(3, device="cpu")
    _, visible_next = visible.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=visible_state,
    )
    _, hidden_next = hidden.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=hidden_state,
    )

    assert not torch.equal(
        visible_next.programs[1].context,
        hidden_next.programs[1].context,
    )
    assert hidden.configuration()["event_visibility"] == "head_only"


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
