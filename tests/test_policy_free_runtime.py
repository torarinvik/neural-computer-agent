from dataclasses import replace

import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    ExternalBoundTransitionModel,
    ExternalControllerEventWindowStateAdapter,
    ExternalControllerStateAdapter,
    ExternalControllerTrajectoryQueryAdapter,
    ExternalEntryBindingRepertoire,
    ExternalEntryRepertoire,
    ExternalIntentionRepertoire,
    ExternalModelBasedPlanner,
    ExternalOutcomeIntentionGenerator,
    ExternalOutcomeIntentionMemory,
    ExternalOutcomeIntentionRouter,
    ExternalSignedEntryValueModel,
    ExternalTransitionMemory,
    ExternalTransitionObservation,
    PolicyFreeAmodalRuntime,
)


class _AdditiveFactualModel(nn.Module):
    state_width = 12
    intention_width = 2

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, :2] = result[:, :2] + intention
        return result


class _EchoDecoder(nn.Module):
    def forward(self, intention):
        return intention.payload


def _feedback() -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )


def test_policy_free_runtime_decodes_planner_intention_not_controller_preference() -> None:
    torch.manual_seed(12)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    runtime.register_decoder("echo", _EchoDecoder())
    state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    event = [AmodalEvent(torch.randn(1, 4))]

    preview, _ = runtime.step_events(event, state, feedback)
    current_state = preview.controller.state_representation.detach()
    goal = current_state.clone()
    goal[:, 0:2] += 1.0
    candidate_intentions = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )

    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_AdditiveFactualModel(), beam_width=4),
    )
    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        candidate_intentions,
        horizon=2,
        beam_width=4,
    )

    assert output.planning.intentions.shape == (1, 2, 2)
    assert torch.allclose(output.planning.predicted_states[0, -1], goal[0])
    assert torch.allclose(output.intention.payload, output.planning.intentions[:, 0])
    assert torch.allclose(output.decoded["echo"], output.intention.payload)
    assert policy_free.configuration()["behavior"] == (
        "factual_model_search_no_stored_policy_v1"
    )


def test_policy_free_runtime_can_fail_closed_on_bound_memory_reads() -> None:
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller.eval()
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    event = [AmodalEvent(torch.randn(1, 4))]
    adapter = ExternalControllerStateAdapter(12, 4)

    preview, _ = runtime.step_events(event, state, feedback)
    model_state = adapter(preview.controller).detach()
    goal = model_state + 1.0
    memory = ExternalTransitionMemory(4, 2, context_width=1)
    memory.write(
        ExternalTransitionObservation(
            state=model_state,
            intention=torch.tensor([[1.0, 0.0]]),
            next_state=goal,
        ),
        context=torch.ones(1, 1),
    )
    bound = ExternalBoundTransitionModel(memory, torch.ones(1))
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(bound, beam_width=2),
        state_adapter=adapter,
    )

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        horizon=1,
        require_known=True,
    )

    assert torch.equal(output.planning.candidate_indices, torch.zeros(1, 1, dtype=torch.long))
    assert torch.equal(output.intention.payload, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(output.planning.predicted_states[:, -1], goal)
    assert policy_free.configuration()["unknown_handling"] == (
        "caller_opt_in_fail_closed_transition_reads_v1"
    )


def test_policy_free_router_can_use_trajectory_statistics_without_changing_planner_state() -> None:
    torch.manual_seed(121)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    event = [AmodalEvent(torch.randn(1, 4))]
    route_adapter = ExternalControllerTrajectoryQueryAdapter(4)
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=route_adapter.query_width,
            intention_width=2,
            hidden_width=8,
        )
    )
    router = ExternalOutcomeIntentionRouter(memory)
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_AdditiveFactualModel(), beam_width=4),
        intention_router=router,
        route_query_adapter=route_adapter,
    )
    preview, _ = runtime.step_events(event, state, feedback)
    goal = preview.controller.state_representation.detach().clone()

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        horizon=1,
        beam_width=4,
        intention_router_state=router.initial_state(1),
    )

    assert output.intention_routing is not None
    assert output.state.shape == (1, 12)
    assert output.intention_routing.candidates.features.shape == (
        1,
        route_adapter.query_width + 1,
    )
    assert policy_free.configuration()["route_query_adapter"]["statistics"] == (
        "masked_mean_and_max_v1"
    )


def test_trajectory_query_adapter_can_preserve_causal_order_at_memory_boundary() -> None:
    torch.manual_seed(122)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    output, next_state = runtime.step_events(
        [AmodalEvent(torch.randn(1, 4))],
        state,
        _feedback(),
    )
    payload = torch.zeros_like(next_state.event_window.payload)
    payload[:, 0, 0] = 1.0
    payload[:, 1, 0] = 3.0
    present = torch.zeros_like(next_state.event_window.present)
    present[:, :2] = True
    ordered_state = replace(
        next_state,
        event_window=replace(
            next_state.event_window,
            payload=payload,
            present=present,
        ),
    )
    reversed_payload = payload.clone()
    reversed_payload[:, 0] = payload[:, 1]
    reversed_payload[:, 1] = payload[:, 0]
    reversed_state = replace(
        next_state,
        event_window=replace(
            next_state.event_window,
            payload=reversed_payload,
            present=present,
        ),
    )

    compatibility = ExternalControllerTrajectoryQueryAdapter(4)
    causal = ExternalControllerTrajectoryQueryAdapter(
        4,
        trajectory_statistics="recency_weighted_and_latest_v1",
        recency_decay=0.75,
    )
    compatibility_first = compatibility(output.controller, ordered_state)
    compatibility_second = compatibility(output.controller, reversed_state)
    causal_first = causal(output.controller, ordered_state)
    causal_second = causal(output.controller, reversed_state)

    assert torch.allclose(compatibility_first, compatibility_second)
    assert not torch.allclose(causal_first, causal_second)
    assert causal.configuration()["statistics"] == (
        "recency_weighted_and_latest_v1"
    )
    assert causal.configuration()["recency_decay"] == 0.75

    gapped_payload = torch.zeros_like(payload)
    gapped_payload[:, 0, 0] = 1.0
    gapped_payload[:, 2, 0] = 3.0
    gapped_present = torch.zeros_like(present)
    gapped_present[:, (0, 2)] = True
    gapped_state = replace(
        next_state,
        event_window=replace(
            next_state.event_window,
            payload=gapped_payload,
            present=gapped_present,
        ),
    )
    gapped_query = causal(output.controller, gapped_state)
    assert torch.allclose(gapped_query[:, -4:], gapped_payload[:, 2])


def test_policy_free_runtime_passes_external_entries_into_factual_search() -> None:
    torch.manual_seed(13)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    event = [AmodalEvent(torch.randn(1, 4))]
    preview, _ = runtime.step_events(event, state, feedback)
    goal = preview.controller.state_representation.detach().clone()
    intentions = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    entries = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    entry_model = ExternalSignedEntryValueModel(12, 2, hidden_width=4)
    with torch.no_grad():
        for parameter in entry_model.state_network.parameters():
            parameter.zero_()
        entry_model.entry_projection.weight.zero_()
        entry_model.entry_projection.weight[0, 0] = 1.0
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(
            _AdditiveFactualModel(),
            beam_width=4,
            entry_value_model=entry_model,
        ),
    )

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        intentions,
        candidate_entries=entries,
        entry_value_weight=1.0,
        horizon=1,
        beam_width=4,
    )

    assert torch.allclose(output.intention.payload, intentions[0:1])
    assert policy_free.configuration()["planner"]["entry_value"] == (
        "external_opaque_entry_value_v1"
    )


def test_policy_free_runtime_retrieves_entries_from_external_repertoire() -> None:
    torch.manual_seed(14)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    event = [AmodalEvent(torch.randn(1, 4))]
    preview, _ = runtime.step_events(event, state, feedback)
    goal = preview.controller.state_representation.detach().clone()
    intentions = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    entry_repertoire = ExternalEntryRepertoire(2)
    entry_repertoire.observe(
        torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )
    )
    entry_model = ExternalSignedEntryValueModel(12, 2, hidden_width=4)
    with torch.no_grad():
        for parameter in entry_model.state_network.parameters():
            parameter.zero_()
        entry_model.entry_projection.weight.zero_()
        entry_model.entry_projection.weight[0, 0] = 1.0
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(
            _AdditiveFactualModel(),
            beam_width=4,
            entry_value_model=entry_model,
        ),
        entry_repertoire=entry_repertoire,
    )

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        intentions,
        entry_value_weight=1.0,
        horizon=1,
        beam_width=4,
    )

    assert output.entry_proposal is not None
    assert output.entry_proposal.source_indices == (0, 1, 2, 3)
    assert torch.allclose(output.intention.payload, intentions[0:1])
    receipt = policy_free.observe_entry(
        output.entry_proposal.entries[0],
        utility=1.0,
        propensity=0.25,
    )
    assert receipt.outcome_observed


def test_policy_free_runtime_retrieves_atomic_intention_entry_bindings() -> None:
    torch.manual_seed(15)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    event = [AmodalEvent(torch.randn(1, 4))]
    preview, _ = runtime.step_events(event, state, feedback)
    goal = preview.controller.state_representation.detach().clone()
    bindings = ExternalEntryBindingRepertoire(2, 2)
    bindings.observe(
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        ),
        torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        ),
    )
    entry_model = ExternalSignedEntryValueModel(12, 2, hidden_width=4)
    with torch.no_grad():
        for parameter in entry_model.state_network.parameters():
            parameter.zero_()
        entry_model.entry_projection.weight.zero_()
        entry_model.entry_projection.weight[0, 0] = 1.0
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(
            _AdditiveFactualModel(),
            beam_width=4,
            entry_value_model=entry_model,
        ),
        entry_binding_repertoire=bindings,
    )

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        entry_value_weight=1.0,
        horizon=1,
        beam_width=4,
    )

    assert output.binding_proposal is not None
    assert output.binding_proposal.source_indices == (0, 1, 2, 3)
    assert torch.allclose(
        output.intention.payload,
        output.binding_proposal.intentions[0:1],
    )
    receipt = policy_free.observe_entry_binding(
        output.binding_proposal.intentions[0],
        output.binding_proposal.entries[0],
        utility=1.0,
        propensity=0.25,
    )
    assert receipt.outcome_observed
    consolidation = policy_free.consolidate_entry_binding_verified(
        (0, 1),
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.5, 0.5]),
        lambda candidate: candidate.logical_ids == (2, 3, 0),
    )
    assert consolidation.accepted
    assert bindings.resolve_logical_id(1) == 0


def test_policy_free_runtime_requires_model_state_width_to_match_adapter() -> None:
    runtime = AmodalControllerRuntime(
        AmodalCognitiveController(
            width=4,
            workspace_slots=2,
            intention_width=2,
            feedback_width=3,
        )
    )
    planner = ExternalModelBasedPlanner(_AdditiveFactualModel())
    adapter = ExternalControllerStateAdapter(11, 12)
    # A supplied adapter cannot quietly consume a different controller
    # representation.  The default adapter may project to a different model
    # state width, but its input contract remains exact.
    try:
        PolicyFreeAmodalRuntime(runtime, planner, state_adapter=adapter)
    except ValueError as error:
        assert "adapter input width" in str(error)
    else:
        raise AssertionError("mismatched policy-free model width was accepted")


def test_event_window_state_adapter_preserves_opaque_state_width_and_history() -> None:
    torch.manual_seed(18)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    output, next_state = runtime.step_events(
        [AmodalEvent(torch.randn(1, 4))],
        state,
        _feedback(),
    )
    adapter = ExternalControllerEventWindowStateAdapter(
        controller.width,
        state_width=controller.width * 3,
        window_gain=0.1,
    )
    adapted = adapter(output.controller, next_state)
    assert adapted.shape == output.controller.state_representation.shape
    assert torch.isfinite(adapted).all()
    assert not torch.allclose(adapted, output.controller.state_representation)
    assert adapter.configuration()["input"] == (
        "opaque_controller_state_plus_bounded_event_window_v1"
    )


def test_event_window_state_adapter_supports_recency_latest_statistics() -> None:
    torch.manual_seed(19)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    output, next_state = runtime.step_events(
        [AmodalEvent(torch.randn(1, 4))],
        state,
        _feedback(),
    )
    adapter = ExternalControllerEventWindowStateAdapter(
        controller.width,
        state_width=controller.width * 3,
        window_gain=0.05,
        window_statistics="recency_weighted_and_latest_v1",
        recency_decay=0.75,
    )
    adapted = adapter(output.controller, next_state)
    assert adapted.shape == output.controller.state_representation.shape
    assert torch.isfinite(adapted).all()
    assert adapter.configuration()["statistics"] == (
        "recency_weighted_and_latest_v1"
    )
    assert adapter.configuration()["recency_decay"] == 0.75


def test_policy_free_runtime_injects_external_generated_candidate_and_feedback() -> None:
    torch.manual_seed(16)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    event = [AmodalEvent(torch.randn(1, 4))]
    preview, _ = runtime.step_events(event, state, feedback)
    goal = preview.controller.state_representation.detach().clone()
    goal[:, :2] += torch.tensor([0.5, 0.5])

    generator = ExternalOutcomeIntentionGenerator(
        context_width=12,
        intention_width=2,
        hidden_width=8,
        noise_scale=0.4,
        context_masking=True,
    )
    generator_state = generator.initial_state(1)
    repertoire = ExternalIntentionRepertoire(2)
    repertoire.observe(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_AdditiveFactualModel(), beam_width=4),
        intention_repertoire=repertoire,
        intention_generator=generator,
    )

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        horizon=1,
        beam_width=4,
        generator_state=generator_state,
        intention_context_mask=torch.tensor(
            [[True, True, True, True, True, True, False, False, False, False, False, False]]
        ),
    )

    assert output.intention_generation is not None
    assert output.intention_generation.intentions.shape == (1, 2)
    assert torch.equal(
        output.intention_generation.features[0, 12:24],
        torch.tensor([1.0] * 6 + [0.0] * 6),
    )
    assert output.planning.expanded_nodes > 0
    assert output.planning.candidate_indices is not None
    assert output.planning.candidate_indices.shape == (1, 1)
    assert policy_free.configuration()["candidate_intentions"] == (
        "external_verified_repertoire_plus_outcome_generator_v1"
    )
    generator_state = policy_free.record_intention_generation_decision(
        generator_state,
        output.intention_generation,
    )
    generator_state = policy_free.apply_intention_generation_feedback(
        generator_state,
        output.intention_generation,
        torch.ones(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    assert generator_state.decisions.tolist() == [1]
    assert generator_state.feedbacks.tolist() == [1]


def test_policy_free_runtime_queries_memory_cells_independently_from_batch() -> None:
    torch.manual_seed(17)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    event = [AmodalEvent(torch.randn(1, 4))]
    preview, _ = runtime.step_events(event, state, feedback)
    goal = preview.controller.state_representation.detach().clone()
    goal[:, :2] += torch.tensor([0.5, 0.5])
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=12,
            intention_width=2,
            hidden_width=8,
            noise_scale=0.4,
        )
    )
    memory_state = memory.initial_state(3)
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_AdditiveFactualModel(), beam_width=4),
        intention_memory=memory,
    )

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        horizon=1,
        beam_width=4,
        intention_memory_state=memory_state,
    )

    assert output.intention_memory_generation is not None
    assert output.intention_memory_generation.intentions.shape == (1, 3, 2)
    assert output.planning.candidate_indices is not None
    selected = output.planning.candidate_indices[:, 0]
    assert selected.shape == (1,)
    assert int(selected.item()) in (0, 1, 2)
    assert policy_free.configuration()["candidate_intentions"] == (
        "outcome_intention_memory_candidates_v1"
    )
    memory_state = policy_free.record_intention_memory_decision(
        memory_state,
        output.intention_memory_generation,
        selected,
    )
    memory_state = policy_free.apply_intention_memory_feedback(
        memory_state,
        output.intention_memory_generation,
        selected,
        torch.ones(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    assert int(memory_state.decisions.sum()) == 1
    assert int(memory_state.feedbacks.sum()) == 1


def test_policy_free_runtime_uses_learned_router_without_caller_cell_selection() -> None:
    torch.manual_seed(18)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=3,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    event = [AmodalEvent(torch.randn(1, 4))]
    preview, _ = runtime.step_events(event, state, feedback)
    goal = preview.controller.state_representation.detach().clone()
    goal[:, :2] += torch.tensor([0.5, 0.5])
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=12,
            intention_width=2,
            hidden_width=8,
            noise_scale=0.4,
        )
    )
    router = ExternalOutcomeIntentionRouter(memory, exploration_bonus=0.8)
    routed_state = router.initial_state(3)
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_AdditiveFactualModel(), beam_width=4),
        intention_router=router,
    )

    output, _ = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        horizon=1,
        beam_width=4,
        intention_router_state=routed_state,
    )

    assert output.intention_routing is not None
    assert output.intention_routing.candidates.intentions.shape == (1, 1, 2)
    assert output.intention_routing.candidates.cell_indices == (
        int(output.intention_routing.selected_cells.item()),
    )
    assert output.intention_routing.selected_intentions.shape == (1, 2)
    assert output.planning.candidate_indices is not None
    assert output.planning.candidate_indices.tolist() == [[0]]
    assert policy_free.configuration()["candidate_intentions"] == (
        "learned_opaque_intention_router_candidate_v1"
    )
    routed_state = policy_free.record_intention_routing_decision(
        routed_state,
        output.intention_routing,
    )
    routed_state = policy_free.apply_intention_routing_feedback(
        routed_state,
        output.intention_routing,
        torch.ones(1),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    assert int(routed_state.routing_decisions.sum()) == 1
    assert int(routed_state.routing_feedbacks.sum()) == 1
