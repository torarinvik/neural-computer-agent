from dataclasses import replace

import torch

from neural_computer import (
    EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V1,
    EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V2,
    EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V3,
    EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V4,
    ExternalOutcomeIntentionGenerator,
    ExternalOutcomeIntentionMemory,
    ExternalOutcomeIntentionRouter,
    ExternalRoutedIntentionPriorSelectionReceipt,
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
    context = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
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

    assert proposal.candidates.intentions.shape == (1, 1, 2)
    assert proposal.candidates.cell_indices == (selected,)
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
    assert torch.equal(
        restored.retention_context_prototypes,
        state.retention_context_prototypes,
    )

    legacy_payload = dict(payload)
    v2_payload = dict(payload)
    v2_payload["schema"] = EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V2
    v2_restored = router.state_from_payload(v2_payload)
    assert torch.equal(v2_restored.retention_mastered, state.retention_mastered)

    legacy_payload["schema"] = EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V1
    for name in (
        "retention_observations",
        "retention_successes",
        "retention_prefix_minima",
        "retention_reversal_streaks",
        "retention_reversal_counts",
        "retention_mastered",
        "retention_context_prototypes",
        "retention_context_masses",
    ):
        legacy_payload.pop(name)
    migrated = router.state_from_payload(legacy_payload)
    assert migrated.retention_observations.shape == state.retention_observations.shape
    assert int(migrated.retention_observations.sum()) == 0


def test_router_prior_challenger_isolated_and_selects_higher_verifier_score() -> None:
    router, state = _router()
    source_digest = router._state_digest(state)

    def probe(
        transfer_router,
        transfer_state,
        transfer_cell,
        fresh_router,
        fresh_state,
        fresh_cell,
    ):
        assert transfer_cell == fresh_cell == 2
        assert transfer_state.cells.baseline.shape == (3,)
        assert fresh_state.cells.baseline.shape == (3,)
        assert transfer_router is not router
        assert fresh_router is not router
        return 0.8, 0.2, transfer_state, fresh_state

    receipt, selected_router, selected_state, selected_cell = (
        router.select_verified_transfer_prior(state, 0, probe, probe_updates=5)
    )

    assert isinstance(receipt, ExternalRoutedIntentionPriorSelectionReceipt)
    assert receipt.selected_initialization == "transfer"
    assert receipt.transfer_probe_score == 0.8
    assert receipt.fresh_probe_score == 0.2
    assert receipt.probe_updates == 5
    assert selected_cell == 2
    assert selected_state.cells.baseline.shape == (3,)
    assert router._state_digest(state) == source_digest
    assert router.state_payload(state)["routing_keys"].shape == (2, 4)
    assert selected_router.state_payload(selected_state)["routing_keys"].shape == (3, 4)


def test_router_prior_challenger_can_reject_copy_and_select_fresh_state() -> None:
    router, state = _router()
    source_digest = router._state_digest(state)

    def probe(*_args):
        transfer_state = _args[1]
        fresh_state = _args[4]
        return 0.1, 0.9, transfer_state, fresh_state

    receipt, selected_router, selected_state, selected_cell = (
        router.select_verified_transfer_prior(state, 0, probe)
    )

    assert receipt.selected_initialization == "fresh"
    assert selected_cell == receipt.fresh_cell == 2
    assert selected_router is not router
    assert selected_state.cells.baseline.shape == (3,)
    assert router._state_digest(state) == source_digest


def test_router_batches_sparse_union_and_credits_physical_cells() -> None:
    router, state = _router()
    state, _ = router.append_cell(state)
    context = torch.randn(4, 4)
    proposal = router.propose(state, context)
    selected = proposal.selected_cells.tolist()

    assert proposal.candidates.cell_indices == tuple(sorted(set(selected)))
    assert all(
        int(cell_index) in proposal.candidates.cell_indices for cell_index in selected
    )
    state = router.record_decision(state, proposal)
    state = router.apply_feedback(state, proposal, torch.ones(4))
    assert int(state.routing_decisions.sum()) == 4
    assert int(state.routing_feedbacks.sum()) == 4


def test_router_gives_unqualified_cells_an_exploration_floor() -> None:
    router, state = _router()
    state = router.protect(state, [0])
    state, new_cell = router.append_cell(state)
    proposal = router.propose(state, torch.zeros(1, 4))

    assert new_cell == 2
    assert float(proposal.route_probabilities[0, new_cell]) >= 0.25
    assert proposal.candidates.cell_indices == (new_cell,)


def test_router_retention_auto_protects_and_releases_with_hysteresis() -> None:
    torch.manual_seed(7711)
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=4,
            intention_width=2,
            hidden_width=8,
            noise_scale=0.3,
        )
    )
    router = ExternalOutcomeIntentionRouter(
        memory,
        mastery_threshold=0.75,
        min_mastery_feedbacks=3,
        reversal_threshold=0.25,
        reversal_patience=3,
    )
    state = router.initial_state(1)
    context = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    def observe(value: float) -> None:
        nonlocal state
        proposal = router.propose(state, context)
        state = router.record_decision(state, proposal)
        state = router.apply_feedback(state, proposal, torch.tensor([value]))

    for _ in range(3):
        observe(1.0)
    assert bool(state.cells.protected[0])
    assert bool(state.retention_mastered[0])
    protected_weights = state.cells.output_weights.clone()

    for _ in range(3):
        observe(0.0)
    assert not bool(state.cells.protected[0])
    assert not bool(state.retention_mastered[0])
    assert int(state.retention_reversal_counts[0]) == 1
    assert torch.equal(state.cells.output_weights, protected_weights)

    for _ in range(3):
        observe(1.0)
    assert bool(state.cells.protected[0])
    assert int(state.retention_observations[0]) == 3


def test_router_heldout_verifier_can_protect_without_mutating_learning_state() -> None:
    router, state = _router()
    context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    before = router.state_payload(state)

    state, receipt = router.verify_and_protect(
        state,
        0,
        context,
        [1.0] * 8,
        floor=0.9,
    )

    assert receipt.accepted
    assert receipt.reason == "heldout_prefix_floor_passed"
    assert bool(state.cells.protected[0])
    assert bool(state.retention_mastered[0])
    assert int(state.retention_observations.sum()) == int(
        before["retention_observations"].sum()
    )
    assert torch.equal(state.cells.input_weights, before["cells"]["input_weights"])
    assert torch.equal(state.routing_keys, before["routing_keys"])

    rejected_state, rejected = router.verify_and_protect(
        state,
        0,
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        [1.0] * 8,
        floor=0.9,
    )
    assert not rejected.accepted
    assert rejected.reason == "heldout_context_not_relevant"
    assert torch.equal(rejected_state.cells.input_weights, state.cells.input_weights)


def test_router_protection_freezes_address_until_reversal() -> None:
    router, state = _router()
    state = router.protect(state, [0])
    context = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    random_source = torch.Generator().manual_seed(19)
    for _ in range(100):
        proposal = router.propose(state, context, generator=random_source)
        if int(proposal.selected_cells.item()) == 0:
            break
    else:
        raise AssertionError("protected cell was never sampled")
    before_keys = state.routing_keys.clone()
    before_bias = state.routing_bias.clone()
    before_decisions = state.routing_decisions.clone()
    state = router.record_decision(state, proposal)
    state = router.apply_feedback(state, proposal, torch.ones(1))

    assert torch.equal(state.routing_keys, before_keys)
    assert torch.equal(state.routing_bias, before_bias)
    assert torch.equal(state.routing_decisions, before_decisions)
    assert int(state.routing_feedbacks[0]) == 1


def test_verified_context_prototype_restores_address_for_protected_cell() -> None:
    router, state = _router()
    context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state, receipt = router.verify_and_protect(
        state,
        0,
        context,
        [1.0] * 8,
        floor=0.9,
    )
    proposal = router.propose(state, context.unsqueeze(0))

    assert receipt.accepted
    assert float(proposal.route_probabilities[0, 0]) > 0.65


def test_router_tracks_partial_retention_without_learning_missing_dimensions() -> None:
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=4,
            intention_width=2,
            hidden_width=8,
            context_masking=True,
        )
    )
    router = ExternalOutcomeIntentionRouter(
        memory,
        exploration_bonus=0.0,
        unqualified_cell_probability=0.0,
    )
    state = router.initial_state(1)
    context = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, False, True, False]])
    proposal = router.propose(
        state,
        context,
        context_mask=mask,
        generator=torch.Generator().manual_seed(23),
    )
    assert state.routing_keys.shape == (1, 8)
    assert proposal.route_key_gradients.shape == (1, 1, 8)
    state = router.record_decision(state, proposal)
    state = router.apply_feedback(state, proposal, torch.ones(1))
    selected = int(proposal.selected_cells.item())

    assert torch.equal(
        state.retention_context_observed_masses[selected],
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    assert torch.equal(
        state.retention_context_mask_profiles[selected],
        torch.tensor([0.25, 0.0, 0.25, 0.0]),
    )
    assert torch.equal(
        state.retention_context_prototypes[selected],
        torch.tensor([1.0, 0.0, 3.0, 0.0]),
    )
    payload = router.state_payload(state)
    restored = router.state_from_payload(payload)
    assert torch.equal(
        restored.retention_context_observed_masses,
        state.retention_context_observed_masses,
    )
    assert torch.equal(
        restored.retention_context_mask_profiles,
        state.retention_context_mask_profiles,
    )

    legacy = dict(payload)
    legacy["schema"] = EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V3
    legacy.pop("retention_context_observed_masses")
    migrated = router.state_from_payload(legacy)
    assert torch.equal(
        migrated.retention_context_observed_masses,
        state.retention_context_masses.unsqueeze(-1).expand_as(
            state.retention_context_observed_masses
        ),
    )
    v4_payload = dict(payload)
    v4_payload["schema"] = EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V4
    v4_payload.pop("retention_context_mask_profiles")
    v4_migrated = router.state_from_payload(v4_payload)
    assert torch.equal(
        v4_migrated.retention_context_mask_profiles,
        (state.retention_context_observed_masses > 0.0).to(
            dtype=state.retention_context_mask_profiles.dtype
        ),
    )


def test_masked_copy_on_write_neutralizes_source_unobserved_dimensions() -> None:
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=4,
            intention_width=2,
            hidden_width=8,
            context_masking=True,
            initial_parameter_scale=0.05,
        )
    )
    router = ExternalOutcomeIntentionRouter(memory, exploration_bonus=0.0)
    state = router.initial_state(1)
    context = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, False, True, False]])
    proposal = router.propose(state, context, context_mask=mask)
    state = router.record_decision(state, proposal)
    state = router.apply_feedback(state, proposal, torch.ones(1))

    source_input_weights = state.cells.input_weights[0].clone()
    source_routing_key = state.routing_keys[0].clone()
    assert bool(source_input_weights[:, 1].abs().sum() > 0.0)
    assert bool(source_input_weights[:, 3].abs().sum() > 0.0)
    assert bool(source_routing_key[1].abs() > 0.0)
    assert bool(source_routing_key[3].abs() > 0.0)

    state, child = router.append_cell(state, source_cell=0)
    assert torch.equal(
        state.cells.input_weights[child, :, [0, 2]],
        source_input_weights[:, [0, 2]],
    )
    assert torch.equal(
        state.cells.input_weights[child, :, [1, 3]],
        torch.zeros(8, 2),
    )
    assert torch.equal(state.routing_keys[child, [0, 2]], source_routing_key[[0, 2]])
    assert torch.equal(state.routing_keys[child, [1, 3]], torch.zeros(2))
    state, exposed_child = router.append_cell(
        state,
        source_cell=0,
        context_mask=torch.tensor([True, True, True, False]),
    )
    assert torch.equal(
        state.cells.input_weights[exposed_child, :, 1],
        torch.zeros(8),
    )
    assert torch.equal(
        state.cells.input_weights[exposed_child, :, 3],
        torch.zeros(8),
    )


def test_context_mask_profile_routes_to_matching_evidence_variant() -> None:
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=4,
            intention_width=2,
            hidden_width=8,
            context_masking=True,
        )
    )
    router = ExternalOutcomeIntentionRouter(
        memory,
        exploration_bonus=0.0,
        unqualified_cell_probability=0.0,
        context_mask_profile_scale=6.0,
    )
    state = router.initial_state(2)
    state = replace(
        state,
        routing_keys=torch.zeros_like(state.routing_keys),
        retention_context_mask_profiles=torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
        ),
    )
    proposal = router.propose(
        state,
        torch.zeros(1, 4),
        context_mask=torch.tensor([[True, False, True, False]]),
    )

    assert float(proposal.route_probabilities[0, 0]) > float(
        proposal.route_probabilities[0, 1]
    )


def test_masked_reversal_quarantines_instead_of_mutating_protected_cell() -> None:
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=4,
            intention_width=2,
            hidden_width=8,
            context_masking=True,
        )
    )
    router = ExternalOutcomeIntentionRouter(
        memory,
        exploration_bonus=0.0,
        mastery_threshold=0.75,
        min_mastery_feedbacks=3,
        reversal_threshold=0.25,
        reversal_patience=3,
    )
    state = router.initial_state(1)
    context = torch.tensor([[1.0, 0.0, 3.0, 0.0]])
    mask = torch.tensor([[True, False, True, False]])

    def observe(outcome: float) -> None:
        nonlocal state
        proposal = router.propose(state, context, context_mask=mask)
        state = router.record_decision(state, proposal)
        state = router.apply_feedback(
            state,
            proposal,
            torch.tensor([outcome]),
        )

    for _ in range(3):
        observe(1.0)
    before = state.cells.output_weights.clone()
    for _ in range(3):
        observe(0.0)

    assert bool(state.cells.protected[0])
    assert bool(state.retention_mastered[0])
    assert int(state.retention_reversal_counts[0]) == 1
    assert torch.equal(state.cells.output_weights, before)
