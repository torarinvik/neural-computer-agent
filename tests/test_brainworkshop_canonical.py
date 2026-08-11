from __future__ import annotations

import pytest
import torch

from experiments.brainworkshop_canonical import (
    BrainWorkshopEventEncoder,
    CanonicalBrainWorkshopAgent,
    NBackVerifier,
    train_reward_only,
)
from experiments.brainworkshop_canonical.goal_conditioned_planning import (
    run_goal_conditioned_planning_audit,
)
from experiments.brainworkshop_canonical.goal_fragment_staging import (
    run_goal_fragment_staging_audit,
)
from experiments.brainworkshop_canonical.nonstationary_goal_conditioned_planning import (
    run_nonstationary_goal_conditioned_planning_audit,
)
from experiments.brainworkshop_canonical.replay_free_transition_acquisition import (
    _route_rollout,
    run_nonstationary_transition_retention_audit,
    run_online_transition_discovery_audit,
    run_replay_free_transition_acquisition_audit,
)
from neural_computer import (
    AdaptiveOnlineEpisodicRelationReader,
    ExternalTransitionRollout,
    RetentionPolicyConfig,
)


def test_nback_verifier_hides_target_and_scores_only_after_warmup() -> None:
    verifier = NBackVerifier(batch_size=2, n_back=2, steps=4, seed=17)
    verifier.reset()
    first = verifier.score(torch.zeros(2, dtype=torch.long))
    second = verifier.score(torch.zeros(2, dtype=torch.long))
    third = verifier.score(torch.zeros(2, dtype=torch.long))
    assert not bool(first.eligible.any())
    assert not bool(second.eligible.any())
    assert bool(third.eligible.all())
    assert verifier.done is False


def test_nback_verifier_can_expose_a_rendered_cue_without_task_metadata() -> None:
    verifier = NBackVerifier(
        batch_size=2,
        n_back=2,
        steps=4,
        symbol_count=4,
        cue_symbol=4,
        seed=17,
    )
    verifier.reset()

    assert torch.equal(verifier.observation(), torch.full((2,), 4))
    cue_score = verifier.score(torch.zeros(2, dtype=torch.long))
    assert not bool(cue_score.eligible.any())
    assert verifier.eligible_trials == 2
    assert verifier.steps == 5


def test_nback_targets_are_balanced_and_time_shuffle_preserves_balance() -> None:
    for time_shuffle in (False, True):
        verifier = NBackVerifier(
            batch_size=3,
            n_back=2,
            steps=6,
            seed=17,
            time_shuffle=time_shuffle,
        )
        verifier.reset()
        rewards = []
        while not verifier.done:
            rewards.append(verifier.score(torch.zeros(3, dtype=torch.long)).reward)
        observed = torch.stack(rewards, dim=1)[:, 2:].sum(dim=1)
        assert torch.equal(observed, torch.full((3,), 2.0))


def test_event_encoder_is_a_learned_frontend() -> None:
    encoder = BrainWorkshopEventEncoder(symbol_count=4, event_width=6)
    encoded = encoder(torch.tensor([0, 3], dtype=torch.long))
    assert encoded.shape == (2, 6)
    assert encoder.configuration()["schema"] == (
        "neural-computer.brainworkshop-event-encoder.v1"
    )


def test_canonical_rollout_uses_keypress_and_retention_boundaries() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=12,
        intention_width=5,
        feedback_width=4,
        retention_config=RetentionPolicyConfig(
            mastery_threshold=0.0,
            min_mastery_observations=1,
        ),
        seed=17,
    )
    verifier = NBackVerifier(batch_size=3, n_back=2, steps=5, seed=23)
    rollout = agent.rollout(verifier, sample=False)
    assert rollout.events.shape == (3, 5, 12)
    assert rollout.actions.shape == (3, 5)
    assert rollout.rewards.shape == (3, 5)
    assert rollout.eligible.shape == (3, 5)
    assert rollout.context.shape == (3, 12)
    assert torch.all((rollout.propensities > 0.0) & (rollout.propensities <= 1.0))
    assert len(agent.retention.payload()["records"]) == 1
    assert agent.retention.status(agent.capability_address).protected


def test_reward_only_pilot_freezes_shared_controller_and_replays_nothing() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        seed=17,
    )
    history = train_reward_only(
        agent,
        n_back=2,
        updates=2,
        batch_size=4,
        steps=4,
        seed=31,
    )
    assert len(history) == 2
    assert all(row.replayed_examples == 0 for row in history)
    assert all(not parameter.requires_grad for parameter in agent.controller.parameters())
    assert any(parameter.requires_grad for parameter in agent.intent_adapter.parameters())
    assert len(agent.retention.payload()["records"]) == 0


def test_goal_fragment_staging_uses_real_verifier_bits_without_controller_updates() -> None:
    report = run_goal_fragment_staging_audit(
        seed=17,
        updates=2,
        batch_size=3,
        steps=4,
        staging_lifetimes=2,
    )

    assert report.status == "staging_boundary_only"
    assert report.unique_verifier_bits == 12
    assert report.unique_logical_lifetimes == 6
    assert report.optimizer_updates == 2
    assert report.replayed_examples == 0
    assert report.controller_unchanged
    assert not report.missing_evidence_accepted
    assert not report.fresh_candidate_accepted
    assert not report.inverted_outcome_accepted
    assert not report.reversal_accepted


def test_admitted_goal_fragment_changes_frozen_core_downstream_planning() -> None:
    report = run_goal_conditioned_planning_audit(seed=93)

    assert report.status == "goal_conditioned_external_planning_boundary"
    assert report.controller_unchanged
    assert report.replay_free_bank
    assert report.goal_fragment_admitted
    assert report.goal_fragment_used
    assert report.trained_planner_improved_over_fresh
    assert report.goal_horizon == 2
    assert report.trained_terminal_error < report.fresh_terminal_error
    assert report.transition_rows_consumed_once == 18
    assert report.unique_verifier_bits == 16
    assert report.optimizer_updates == 0
    assert report.replayed_examples == 0
    assert report.missing_evidence_rejected
    assert report.corrupted_goal_rejected


def test_nonstationary_goal_planning_retains_source_and_beats_fresh_target() -> None:
    report = run_nonstationary_goal_conditioned_planning_audit(seed=93)

    assert report.status == "nonstationary_goal_conditioned_external_planning_boundary"
    assert report.controller_unchanged
    assert report.replay_free_bank
    assert report.source_slot_byte_stable
    assert report.target_goal_fragment_admitted
    assert report.target_goal_fragment_used
    assert report.target_planner_improved_over_fresh
    assert report.source_error_after_target == report.source_error_before_target
    assert report.trained_target_terminal_error < report.fresh_target_terminal_error
    assert report.goal_horizon == 2
    assert report.transition_rows_consumed_once == 36
    assert report.unique_verifier_bits == 28
    assert report.optimizer_updates == 0
    assert report.replayed_examples == 0
    assert report.missing_evidence_rejected


def test_rendered_transition_acquisition_improves_heldout_error_without_replay() -> None:
    report = run_replay_free_transition_acquisition_audit(
        seed=91,
        steps=5,
        training_lifetimes=2,
    )

    assert report.status == "rendered_replay_free_transition_boundary"
    assert report.controller_unchanged
    assert report.replay_free_bank
    assert report.model_improved_on_heldout_rollout
    assert report.replayed_examples == 0
    assert report.optimizer_updates == 0
    assert report.transition_rows_consumed_once == 10
    assert report.external_sample_count == 10
    assert report.fresh_sample_count == 0


def test_nonstationary_transition_learning_retains_source_slot_without_replay() -> None:
    report = run_nonstationary_transition_retention_audit(
        seed=92,
        steps=5,
        source_training_lifetimes=1,
        target_training_lifetimes=1,
    )

    assert report.status == "nonstationary_replay_free_transition_boundary"
    assert report.controller_unchanged
    assert report.replay_free_bank
    assert report.source_slot_byte_stable
    assert report.target_model_improved_on_heldout
    assert report.source_heldout_error_after_target == (
        report.source_heldout_error_before_target
    )
    assert report.replayed_examples == 0
    assert report.external_slot_count == 2


def test_online_transition_discovery_grows_target_slot_without_replay() -> None:
    report = run_online_transition_discovery_audit(seed=93)

    assert report.status == "online_replay_free_transition_discovery_boundary"
    assert report.controller_unchanged
    assert report.replay_free_bank
    assert report.source_slot_byte_stable
    assert report.target_context_discovered
    assert report.target_route_recovered
    assert report.target_model_improved_on_heldout
    assert report.target_promotion_accepted
    assert report.source_heldout_error_after_target == (
        report.source_heldout_error_before_target
    )
    assert report.replayed_examples == 0
    assert report.transition_rows_consumed_once == 24
    assert report.external_slot_count == 2
    assert report.target_discovery_status == "staged"
    assert report.target_continuation_status == "staged"
    assert report.target_heldout_status == "continuation"


def test_recency_window_transition_discovery_is_an_explicit_transfer_mode() -> None:
    report = run_online_transition_discovery_audit(
        seed=91,
        window_statistics="recency_weighted_and_latest_v1",
        window_gain=0.05,
        recency_decay=0.75,
        goal_conditioned=True,
        prior_selection_transfer_cost=0.0,
        prior_selection_fresh_cost=1.0,
        prior_selection_cost_weight=0.2,
    )

    assert report.status == "online_replay_free_transition_discovery_boundary"
    assert report.window_statistics == "recency_weighted_and_latest_v1"
    assert report.window_gain == 0.05
    assert report.recency_decay == 0.75
    assert report.target_route_recovered
    assert report.target_model_improved_on_heldout
    assert report.goal_conditioned
    assert report.target_goal_fragment_admitted
    assert report.target_goal_fragment_used
    assert report.target_goal_planner_improved_over_fresh
    assert report.target_goal_horizon == 2
    assert report.target_goal_missing_evidence_rejected
    assert report.trained_target_goal_error < report.fresh_target_goal_error
    assert report.prior_selection_cost_aware
    assert report.replayed_examples == 0


def test_online_transition_discovery_can_learn_external_selection_cost() -> None:
    report = run_online_transition_discovery_audit(
        seed=91,
        window_statistics="recency_weighted_and_latest_v1",
        window_gain=0.05,
        goal_conditioned=True,
        learned_prior_selection_cost=True,
    )

    assert report.status == "online_replay_free_transition_discovery_boundary"
    assert report.prior_selection_cost_ledger_used
    assert report.prior_selection_cost_observed
    assert report.prior_selection_cost_aware

    rejected = run_online_transition_discovery_audit(
        seed=92,
        window_statistics="recency_weighted_and_latest_v1",
        window_gain=0.05,
        goal_conditioned=True,
        learned_prior_selection_cost=True,
    )
    assert rejected.status.endswith("_failed")
    assert rejected.goal_conditioned
    assert rejected.target_goal_horizon == 2
    assert rejected.prior_selection_cost_ledger_used
    assert not rejected.prior_selection_cost_observed

    adaptive = run_online_transition_discovery_audit(
        seed=91,
        window_statistics="recency_weighted_and_latest_v1",
        window_gain=0.05,
        goal_conditioned=True,
        adaptive_address=True,
    )
    assert adaptive.adaptive_address
    assert adaptive.status == "online_replay_free_transition_discovery_boundary"


def test_transition_discovery_firewall_skips_committed_slot_updates() -> None:
    class RouterProbe:
        def __init__(self) -> None:
            self.statuses = iter(("matched", "staged"))
            self.adapted: list[str] = []

        def observe(self, _observation: object) -> object:
            return type("Result", (), {"status": next(self.statuses)})()

        def adaptation_step(
            self,
            result: object,
            _optimizer: object,
            *,
            replay_evidence: bool,
        ) -> float:
            assert replay_evidence is False
            self.adapted.append(result.status)
            return 0.0

    rollout = ExternalTransitionRollout(
        initial_state=torch.zeros(2),
        intentions=torch.zeros(2, 1),
        expected_states=torch.zeros(2, 2),
    )
    router = RouterProbe()
    result = _route_rollout(router, rollout, adapt=True, adapt_committed=False)

    assert result.status == "staged"
    assert router.adapted == ["staged"]


def test_relation_reader_can_replace_gru_context_in_canonical_runner() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    rollout = agent.rollout(NBackVerifier(batch_size=2, n_back=2, steps=4, seed=29))
    assert rollout.context.shape == (2, 8)
    assert agent.reader_kind == "relation"


def test_appended_slot_uses_shared_bus_and_exact_mixture_propensity() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    slot = agent.add_relation_capability(n_back=3, seed=23)
    rollout = agent.rollout(
        NBackVerifier(batch_size=3, n_back=3, steps=5, seed=29),
        sample=True,
        record_retention=False,
        exploration_probability=0.5,
    )
    assert slot == 1
    assert "keypress_extension_1" in agent.runtime.output_bus.decoders
    assert rollout.selected_slots.shape == (3, 5)
    assert torch.all((rollout.propensities > 0.0) & (rollout.propensities <= 1.0))
    forced = agent.rollout(
        NBackVerifier(batch_size=3, n_back=3, steps=5, seed=31),
        sample=False,
        record_retention=False,
        forced_slot=slot,
    )
    assert torch.equal(forced.selected_slots, torch.ones_like(forced.selected_slots))


def test_adaptive_slot_is_provisioned_without_a_task_horizon() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )

    slot = agent.add_adaptive_relation_capability(memory_capacity=5, seed=23)

    extension = agent.extensions[slot - 1]
    assert extension.memory_capacity == 5
    assert not hasattr(extension, "n_back")
    assert isinstance(extension.reader, AdaptiveOnlineEpisodicRelationReader)


def test_adaptive_slot_can_grow_its_external_window() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    slot = agent.add_adaptive_relation_capability(memory_capacity=5, seed=23)
    controller_before = {
        name: value.detach().clone()
        for name, value in agent.controller.state_dict().items()
    }

    agent.expand_adaptive_relation_capability(slot, memory_capacity=6)

    assert agent.extensions[slot - 1].memory_capacity == 6
    assert agent.extensions[slot - 1].reader.memory_capacity == 6
    for name, value in agent.controller.state_dict().items():
        assert torch.equal(value, controller_before[name])


def test_failed_adaptive_growth_resets_only_the_unmastered_external_slot() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    slot = agent.add_adaptive_relation_capability(memory_capacity=5, seed=23)
    extension_before = {
        name: value.detach().clone()
        for name, value in agent.extensions[slot - 1].state_dict().items()
    }
    decoder_before = {
        name: value.detach().clone()
        for name, value in agent.extension_decoder(slot).state_dict().items()
    }
    controller_before = {
        name: value.detach().clone()
        for name, value in agent.controller.state_dict().items()
    }

    agent.expand_adaptive_relation_capability(
        slot,
        memory_capacity=6,
        reset_failed_reader=True,
        reset_seed=71,
    )

    extension_after = agent.extensions[slot - 1]
    decoder_after = agent.extension_decoder(slot)
    assert extension_after.memory_capacity == 6
    assert extension_after.reader.memory_capacity == 6
    assert any(
        not torch.equal(value, extension_after.state_dict()[name])
        for name, value in extension_before.items()
    )
    assert any(
        not torch.equal(value, decoder_after.state_dict()[name])
        for name, value in decoder_before.items()
    )
    for name, value in agent.controller.state_dict().items():
        assert torch.equal(value, controller_before[name])


def test_unprotected_capability_replacement_clears_stale_routes_and_protects_mastery() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    slot = agent.add_adaptive_relation_capability(memory_capacity=5, seed=23)
    cue_event = agent.runtime.encoders["stimulus"](torch.tensor([3]))[0]
    for _ in range(2):
        agent.context_route_evidence.observe(cue_event, slot, 0.0)
    receipt = agent.replace_unprotected_adaptive_relation_capability(
        slot,
        memory_capacity=6,
        seed=71,
    )

    assert receipt["evicted_protected"] is False
    assert receipt["evicted_route_protected"] is False
    assert agent.context_route_evidence.protected_slots() == (False, False)
    assert agent.context_route_evidence.preferred_order(cue_event) == (0, 1)

    for _ in range(8):
        agent.retention.observe(agent.capability_address_for(slot), 1.0)
    with pytest.raises(ValueError, match="protected capability"):
        agent.replace_unprotected_adaptive_relation_capability(
            slot,
            memory_capacity=7,
            seed=72,
        )


def test_persistent_route_evidence_selects_an_opaque_slot_without_core_changes() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    slot = agent.add_relation_capability(n_back=3, seed=23)
    controller_before = {
        name: value.detach().clone()
        for name, value in agent.controller.state_dict().items()
    }

    for _ in range(agent.route_evidence.min_mastery_observations):
        agent.route_evidence.observe(slot, 1.0)
    rollout = agent.rollout(
        NBackVerifier(batch_size=3, n_back=3, steps=5, seed=37),
        sample=False,
        record_retention=False,
        persistent_route=True,
    )

    assert torch.equal(
        rollout.selected_slots[:, 0],
        torch.full((3,), slot, dtype=torch.long),
    )
    assert agent.route_evidence.status().preferred_slot == slot
    for name, value in agent.controller.state_dict().items():
        assert torch.equal(value, controller_before[name])


def test_context_route_uses_a_learned_event_cue_and_keeps_core_opaque() -> None:
    agent = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    slot = agent.add_relation_capability(n_back=3, seed=23)
    encoder = agent.runtime.encoders["stimulus"]
    cue_event = encoder(torch.tensor([4], dtype=torch.long))[0]
    for _ in range(agent.context_route_evidence.min_mastery_observations):
        agent.context_route_evidence.observe(cue_event, slot, 1.0)
    controller_before = {
        name: value.detach().clone()
        for name, value in agent.controller.state_dict().items()
    }

    rollout = agent.rollout(
        NBackVerifier(
            batch_size=3,
            n_back=3,
            steps=5,
            symbol_count=4,
            cue_symbol=4,
            seed=37,
        ),
        sample=False,
        record_retention=False,
        context_route=True,
    )

    assert torch.equal(
        rollout.selected_slots[:, 0],
        torch.full((3,), slot, dtype=torch.long),
    )
    for name, value in agent.controller.state_dict().items():
        assert torch.equal(value, controller_before[name])


def test_route_state_round_trips_without_loading_controller_weights() -> None:
    agent = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    slot = agent.add_relation_capability(n_back=3, seed=23)
    cue_event = agent.runtime.encoders["stimulus"](
        torch.tensor([4], dtype=torch.long)
    )[0]
    for _ in range(agent.route_evidence.min_mastery_observations):
        agent.route_evidence.observe(slot, 1.0)
        agent.context_route_evidence.observe(cue_event, slot, 1.0)
    payload = agent.route_state_payload()

    restored = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    restored.add_relation_capability(n_back=3, seed=23)
    controller_before = {
        name: value.detach().clone()
        for name, value in restored.controller.state_dict().items()
    }
    restored.load_route_state_payload(payload)

    assert restored.route_state_payload() == payload
    rollout = restored.rollout(
        NBackVerifier(
            batch_size=2,
            n_back=3,
            steps=5,
            symbol_count=4,
            cue_symbol=4,
            seed=41,
        ),
        sample=False,
        record_retention=False,
        context_route=True,
    )
    assert torch.equal(
        rollout.selected_slots[:, 0],
        torch.full((2,), slot, dtype=torch.long),
    )
    for name, value in restored.controller.state_dict().items():
        assert torch.equal(value, controller_before[name])
