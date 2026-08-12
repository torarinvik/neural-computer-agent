from __future__ import annotations

import argparse

import pytest
import torch

from experiments.brainworkshop_canonical import (
    BrainWorkshopEventEncoder,
    CanonicalBrainWorkshopAgent,
    NBackVerifier,
    train_reward_only,
)
from experiments.brainworkshop_canonical.cross_family_rule_growth import (
    CrossFamilyVerifier,
)
from experiments.brainworkshop_canonical.cross_family_rule_growth import (
    run as run_cross_family_rule_growth,
)
from experiments.brainworkshop_canonical.environment import NBackVerifierStep
from experiments.brainworkshop_canonical.external_compute_growth import (
    run as run_external_compute_growth,
)
from experiments.brainworkshop_canonical.external_compute_open_growth import (
    run as run_external_compute_open_growth,
)
from experiments.brainworkshop_canonical.external_compute_route import (
    run as run_external_compute_route,
)
from experiments.brainworkshop_canonical.external_compute_route_bank import (
    run as run_external_compute_route_bank,
)
from experiments.brainworkshop_canonical.external_compute_route_reversal import (
    run as run_external_compute_route_reversal,
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
    ExternalWorkingMemoryCell,
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


def test_cross_family_verifier_generates_generic_deeper_nback_targets() -> None:
    for family, depth in (
        ("nback2", 2),
        ("nback3", 3),
        ("nback4", 4),
        ("nback5", 5),
        ("nback8", 8),
    ):
        verifier = CrossFamilyVerifier(
            family=family,
            batch_size=5,
            steps=depth + 4,
            cue_symbol=4,
            seed=17,
        )
        verifier.reset()
        observations = [verifier.observation()]
        scores = []
        while not verifier.done:
            action = torch.zeros(5, dtype=torch.long)
            scores.append(verifier.score(action))
            if not verifier.done:
                observations.append(verifier.observation())

        symbols = torch.stack(observations[1:], dim=1)
        eligible = torch.stack([step.eligible for step in scores], dim=1)
        rewards = torch.stack([step.reward for step in scores], dim=1)
        expected = symbols[:, depth:] == symbols[:, :-depth]
        assert torch.equal(
            eligible[:, depth + 1 :],
            torch.ones_like(eligible[:, depth + 1 :], dtype=torch.bool),
        )
        # Action zero is the negative answer, so its reward is the inverse
        # of the private equality target.
        assert torch.equal(rewards[:, depth + 1 :] > 0.5, ~expected)


def test_heldout_rule_growth_smoke_preserves_external_boundary(tmp_path) -> None:
    from experiments.brainworkshop_canonical.heldout_rule_growth import run

    report = run(
        argparse.Namespace(
            report_out=tmp_path / "heldout-rule-growth.json",
            seed=17,
            source_updates=1,
            target_updates=1,
            batch_size=2,
            steps=6,
            calibration_lifetimes=1,
            discovery_lifetimes=1,
            retention_lifetimes=1,
            learning_rate=1e-2,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-heldout-rule-growth.v1"
    )
    assert report["claim_boundary"].endswith(
        "bounded rule growth, not general continual learning."
    )
    gates = report["gates"]
    assert gates["controller_unchanged"]
    assert gates["encoder_unchanged"]
    assert gates["route_reload_exact"]
    assert gates["incompatible_route_representation_rejected"]
    assert report["accounting"]["replayed_examples"] == 0
    assert report["accounting"]["optimizer_updates"] == 4


def test_event_encoder_is_a_learned_frontend() -> None:
    encoder = BrainWorkshopEventEncoder(symbol_count=4, event_width=6)
    encoded = encoder(torch.tensor([0, 3], dtype=torch.long))
    assert encoded.shape == (2, 6)
    assert encoder.configuration()["schema"] == (
        "neural-computer.brainworkshop-event-encoder.v1"
    )


def test_external_compute_growth_smoke_keeps_the_frozen_core_boundary(tmp_path) -> None:
    report = run_external_compute_growth(
        argparse.Namespace(
            report_out=tmp_path / "external-compute-growth.json",
            seed=17,
            source_updates=2,
            target_updates=2,
            fresh_updates=2,
            batch_size=2,
            steps=6,
            retention_lifetimes=1,
            learning_rate=1e-2,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-compute-growth.v1"
    )
    assert report["gates"]["source_file_unchanged"]
    assert report["gates"]["frozen_controller"]
    assert report["gates"]["frozen_event_encoder"]
    assert report["accounting"]["replayed_examples"] == 0
    assert report["accounting"]["optimizer_updates"] == 4


def test_external_compute_route_smoke_uses_content_addressed_outcome_evidence(
    tmp_path,
) -> None:
    report = run_external_compute_route(
        argparse.Namespace(
            report_out=tmp_path / "external-compute-route.json",
            seed=17,
            source_updates=2,
            target_updates=2,
            route_updates=4,
            route_calibration_lifetimes=8,
            batch_size=32,
            retention_lifetimes=1,
            learning_rate=1e-2,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-compute-route.v2"
    )
    assert report["gates"]["protected_source_context_unchanged"]
    assert report["gates"]["route_reload_exact"]
    assert report["gates"]["frozen_controller"]
    assert report["gates"]["frozen_event_encoder"]
    assert report["accounting"]["replayed_examples"] == 0


def test_external_compute_route_bank_smoke_preserves_append_only_file_boundaries(
    tmp_path,
) -> None:
    report = run_external_compute_route_bank(
        argparse.Namespace(
            report_out=tmp_path / "external-compute-route-bank.json",
            seed=17,
            slot_count=4,
            file_updates=1,
            route_updates=1,
            route_calibration_lifetimes=8,
            batch_size=32,
            retention_lifetimes=1,
            learning_rate=1e-2,
            basis_hidden=32,
            final_family="switch_binary",
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-compute-route-bank.v1"
    )
    assert report["gates"]["prior_files_unchanged_after_growth"]
    assert report["gates"]["frozen_controller"]
    assert report["gates"]["frozen_event_encoder"]
    assert report["accounting"]["replayed_examples"] == 0


def test_external_compute_route_reversal_smoke_preserves_file_boundaries(
    tmp_path,
) -> None:
    report = run_external_compute_route_reversal(
        argparse.Namespace(
            report_out=tmp_path / "external-compute-route-reversal.json",
            seed=17,
            source_updates=2,
            target_updates=2,
            route_updates=4,
            calibration_lifetimes=8,
            transition_batches=8,
            batch_size=32,
            retention_lifetimes=1,
            learning_rate=1e-2,
            reversal_threshold=0.65,
            reversal_patience=4,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-compute-route-reversal.v1"
    )
    assert report["gates"]["old_file_unchanged"]
    assert report["gates"]["replacement_file_unchanged_during_reversal"]
    assert report["gates"]["frozen_controller"]
    assert report["gates"]["frozen_event_encoder"]
    assert report["gates"]["zero_replayed_examples"]
    assert report["transition"][-1]["slot_0_fraction"] == 0.5
    assert report["transition"][-1]["slot_1_fraction"] == 0.5


def test_external_compute_open_growth_smoke_rejects_unmastered_source_cleanly(
    tmp_path,
) -> None:
    report = run_external_compute_open_growth(
        argparse.Namespace(
            report_out=tmp_path / "external-compute-open-growth.json",
            seed=17,
            target_file_count=2,
            candidate_budget=2,
            file_updates=2,
            route_updates=4,
            route_calibration_lifetimes=8,
            transition_batches=4,
            batch_size=32,
            retention_lifetimes=1,
            learning_rate=1e-2,
            reversal_threshold=0.65,
            reversal_patience=4,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-compute-open-growth.v1"
    )
    assert report["status"] == "rejected"
    assert report["architecture"]["accepted_file_count"] == 0
    assert report["gates"]["rejected_candidate_not_promoted"]
    assert report["gates"]["frozen_controller"]
    assert report["gates"]["frozen_event_encoder"]
    assert report["gates"]["zero_replayed_examples"]


def test_external_temporal_offset_growth_smoke_keeps_addressing_external(
    tmp_path,
) -> None:
    from experiments.brainworkshop_canonical.external_temporal_offset_growth import (
        run,
    )

    report = run(
        argparse.Namespace(
            report_out=tmp_path / "temporal-offset-growth.json",
            seed=17,
            updates=2,
            batch_size=2,
            steps=14,
            retention_lifetimes=1,
            learning_rate=3e-3,
            entropy_weight=0.01,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-temporal-offset-growth.v1"
    )
    assert report["gates"]["old_file_unchanged"]
    assert report["gates"]["frozen_controller"]
    assert report["gates"]["frozen_event_encoder"]
    assert report["accounting"]["replayed_examples"] == 0


def test_external_temporal_context_route_growth_smoke_composes_external_addresses(
    tmp_path,
) -> None:
    from experiments.brainworkshop_canonical.external_temporal_context_route_growth import (
        run,
    )

    report = run(
        argparse.Namespace(
            report_out=tmp_path / "temporal-context-route-growth.json",
            seed=17,
            file_updates=2,
            route_updates=2,
            route_calibration_lifetimes=1,
            batch_size=2,
            steps=14,
            retention_lifetimes=1,
            learning_rate=3e-3,
            entropy_weight=0.01,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-temporal-context-route-growth.v1"
    )
    assert report["gates"]["route_reload_exact"]
    assert report["gates"]["frozen_controller"]
    assert report["gates"]["frozen_event_encoder"]
    assert report["accounting"]["replayed_examples"] == 0


def test_external_temporal_query_address_growth_smoke_freezes_readout_on_growth(
    tmp_path,
) -> None:
    from experiments.brainworkshop_canonical.external_temporal_query_address_growth import (
        run,
    )

    report = run(
        argparse.Namespace(
            report_out=tmp_path / "temporal-query-address-growth.json",
            seed=17,
            source_updates=2,
            target_updates=2,
            route_calibration_lifetimes=1,
            batch_size=2,
            data_steps=14,
            retention_lifetimes=1,
            learning_rate=3e-3,
            entropy_weight=0.01,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-temporal-query-address-growth.v1"
    )
    assert report["gates"]["readout_frozen_during_growth"]
    assert report["gates"]["route_reload_exact"]
    assert report["gates"]["controller_frozen"]
    assert report["accounting"]["replayed_examples"] == 0


def test_external_temporal_content_retrieval_growth_smoke_preserves_memory_contract(
    tmp_path,
) -> None:
    from experiments.brainworkshop_canonical.external_temporal_content_retrieval_growth import (
        run,
    )

    report = run(
        argparse.Namespace(
            report_out=tmp_path / "temporal-content-retrieval-growth.json",
            seed=17,
            source_updates=2,
            target_updates=2,
            route_calibration_lifetimes=1,
            batch_size=2,
            data_steps=14,
            retention_lifetimes=1,
            learning_rate=3e-3,
            entropy_weight=0.01,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-temporal-content-retrieval-growth.v1"
    )
    assert report["gates"]["two_routes_written"]
    assert report["gates"]["clear_memory_removes_hits"]
    assert report["gates"]["reload_preserves_noisy_routes"]
    assert report["gates"]["corruption_rejected"]
    assert report["gates"]["controller_frozen"]
    assert report["gates"]["event_encoder_frozen"]
    assert report["gates"]["zero_replayed_examples"]


def test_external_temporal_verified_compaction_growth_smoke_is_verifier_gated(
    tmp_path,
) -> None:
    from experiments.brainworkshop_canonical.external_temporal_verified_compaction_growth import (
        run,
    )

    report = run(
        argparse.Namespace(
            report_out=tmp_path / "temporal-verified-compaction-growth.json",
            seed=17,
            source_updates=2,
            target_updates=2,
            route_calibration_lifetimes=1,
            batch_size=2,
            data_steps=14,
            retention_lifetimes=1,
            learning_rate=3e-3,
            entropy_weight=0.01,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-temporal-verified-compaction-growth.v1"
    )
    assert report["gates"]["bad_compaction_rejected"]
    assert report["gates"]["bad_compaction_did_not_mutate_source"]
    assert report["gates"]["good_compaction_verified"]
    assert report["gates"]["one_redundant_row_removed"]
    assert report["gates"]["reload_preserves_compacted_routes"]
    assert report["gates"]["corruption_rejected"]
    assert report["gates"]["stale_compaction_rejected"]
    assert report["gates"]["controller_frozen"]
    assert report["gates"]["event_encoder_frozen"]
    assert report["gates"]["zero_replayed_examples"]


def test_external_temporal_learned_compaction_growth_smoke_transfers_selector(
    tmp_path,
) -> None:
    from experiments.brainworkshop_canonical.external_temporal_learned_compaction_growth import (
        run,
    )

    report = run(
        argparse.Namespace(
            report_out=tmp_path / "temporal-learned-compaction-growth.json",
            seed=17,
            policy_updates=2,
            policy_batch_size=2,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-temporal-learned-compaction-growth.v1"
    )
    assert report["gates"]["learned_policy_selects_redundant_pair"]
    assert report["gates"]["learned_beats_untrained_on_permutations"]
    assert report["gates"]["verifier_accepts_selected_compaction"]
    assert report["gates"]["reload_preserves_live_routes"]
    assert report["gates"]["corruption_rejected"]
    assert report["gates"]["controller_frozen"]
    assert report["gates"]["event_encoder_frozen"]
    assert report["gates"]["zero_replayed_examples"]


def test_external_temporal_capacity_schedule_smoke_preserves_distinct_routes(
    tmp_path,
) -> None:
    from experiments.brainworkshop_canonical.external_temporal_capacity_schedule_growth import (
        run,
    )

    report = run(
        argparse.Namespace(
            report_out=tmp_path / "temporal-capacity-schedule-growth.json",
            seed=17,
            policy_updates=64,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-temporal-capacity-schedule-growth.v1"
    )
    assert report["gates"]["forward_retains_all_distinct_routes"]
    assert report["gates"]["reversed_retains_all_distinct_routes"]
    assert report["gates"]["forward_reload_exact"]
    assert report["gates"]["reversed_reload_exact"]
    assert report["gates"]["fixed_external_budget"]
    assert report["gates"]["controller_frozen"]
    assert report["gates"]["event_encoder_frozen"]
    assert report["gates"]["zero_replayed_examples"]


def test_external_temporal_shared_basis_compression_smoke_preserves_routes(
    tmp_path,
) -> None:
    from experiments.brainworkshop_canonical.external_temporal_shared_basis_compression_growth import (
        run,
    )

    report = run(
        argparse.Namespace(
            report_out=tmp_path / "temporal-shared-basis-compression-growth.json",
            seed=17,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-temporal-shared-basis-compression-growth.v1"
    )
    assert report["gates"]["forward_rank_one_rejected"]
    assert report["gates"]["forward_rank_one_non_mutating"]
    assert report["gates"]["forward_rank_two_accepted"]
    assert report["gates"]["forward_storage_reduced"]
    assert report["gates"]["forward_reload_routes"]
    assert report["gates"]["reversed_routes_after"]
    assert report["gates"]["corruption_rejected"]
    assert report["gates"]["controller_frozen"]
    assert report["gates"]["event_encoder_frozen"]
    assert report["gates"]["zero_replayed_examples"]


def test_external_temporal_shared_basis_policy_smoke_transfers_growth(
    tmp_path,
) -> None:
    from experiments.brainworkshop_canonical.external_temporal_shared_basis_policy_growth import (
        run,
    )

    report = run(
        argparse.Namespace(
            report_out=tmp_path / "temporal-shared-basis-policy-growth.json",
            seed=17,
            policy_updates=600,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-external-temporal-shared-basis-policy-growth.v1"
    )
    assert report["status"] == "promoted_shared_basis_policy_growth"
    assert report["gates"]["trained_beats_fresh"]
    assert report["gates"]["forward_initial_rank_2"]
    assert report["gates"]["forward_successor_rank_4"]
    assert report["gates"]["forward_old_retained_after_growth"]
    assert report["gates"]["forward_new_retained_after_growth"]
    assert report["gates"]["reversed_all_routes"]
    assert report["gates"]["corruption_rejected"]
    assert report["gates"]["controller_frozen"]
    assert report["gates"]["zero_replayed_examples"]


def test_binary_switch_family_has_a_valid_chance_baseline() -> None:
    verifier = CrossFamilyVerifier(
        family="switch_binary",
        batch_size=2048,
        steps=14,
        cue_symbol=11,
        seed=17,
    )
    verifier.reset()
    while not verifier.done:
        verifier.score(torch.zeros(2048, dtype=torch.long))

    assert abs(float(verifier._targets.float().mean()) - 0.5) < 0.04


def test_odd_symbol_parity_family_is_balanced_and_distinct() -> None:
    even = CrossFamilyVerifier(
        family="symbol_parity",
        batch_size=2048,
        steps=14,
        cue_symbol=10,
        seed=17,
    )
    odd = CrossFamilyVerifier(
        family="symbol_parity_odd",
        batch_size=2048,
        steps=14,
        cue_symbol=10,
        seed=17,
    )
    even.reset()
    odd.reset()

    assert torch.equal(even._symbols, odd._symbols)
    assert torch.equal(even._targets, ~odd._targets)
    assert abs(float(odd._targets.float().mean()) - 0.5) < 0.04


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


def test_streaming_gradient_external_memory_is_accounted_separately() -> None:
    report = run_online_transition_discovery_audit(
        seed=91,
        window_statistics="recency_weighted_and_latest_v1",
        window_gain=0.05,
        external_memory_update_mode="streaming_gradient",
    )

    assert report.external_memory_update_mode == "streaming_gradient"
    assert report.external_memory_optimizer_updates == 2
    assert report.controller_unchanged
    assert report.replayed_examples == 0


def test_active_discovery_probe_improves_the_external_evidence_boundary() -> None:
    report = run_online_transition_discovery_audit(
        seed=91,
        window_statistics="recency_weighted_and_latest_v1",
        window_gain=0.05,
        goal_conditioned=True,
        prior_selection_fresh_cost=1.0,
        prior_selection_cost_weight=0.2,
        discovery_probe_mode="active",
    )

    assert report.status == "online_replay_free_transition_discovery_boundary"
    assert report.discovery_probe_mode == "active"
    assert report.discovery_probe_rows == 6
    assert report.controller_unchanged
    assert report.source_slot_byte_stable
    assert report.replayed_examples == 0
    assert report.external_memory_optimizer_updates == 0


def test_active_discovery_reports_a_changed_target_regime() -> None:
    report = run_online_transition_discovery_audit(
        seed=91,
        window_statistics="recency_weighted_and_latest_v1",
        window_gain=0.05,
        goal_conditioned=True,
        prior_selection_fresh_cost=1.0,
        prior_selection_cost_weight=0.2,
        discovery_probe_mode="active",
        target_n_back=4,
        target_cue_symbol=5,
    )

    assert report.target_n_back == 4
    assert report.target_cue_symbol == 5
    assert report.discovery_probe_mode == "active"
    assert report.routing_match_tolerance == 0.01
    assert report.source_slot_byte_stable
    assert report.controller_unchanged
    assert report.replayed_examples == 0


def test_active_discovery_handles_a_disappearing_provisional_candidate() -> None:
    report = run_online_transition_discovery_audit(
        seed=84,
        window_statistics="masked_mean_and_max_v1",
        window_gain=0.05,
        goal_conditioned=True,
        prior_selection_fresh_cost=1.0,
        prior_selection_cost_weight=0.2,
        discovery_probe_mode="active",
        routing_match_tolerance=0.02,
        target_n_back=5,
        target_cue_symbol=6,
    )

    assert report.target_promotion_accepted is False
    assert report.target_promotion_reason == "no provisional target candidate was staged"
    assert report.controller_unchanged
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

    promoted = run_online_transition_discovery_audit(
        seed=92,
        window_statistics="recency_weighted_and_latest_v1",
        window_gain=0.05,
        goal_conditioned=True,
        learned_prior_selection_cost=True,
    )
    assert promoted.status == "online_replay_free_transition_discovery_boundary"
    assert promoted.goal_conditioned
    assert promoted.target_goal_horizon == 2
    assert promoted.prior_selection_cost_ledger_used
    assert promoted.prior_selection_cost_observed

    rejected = run_online_transition_discovery_audit(
        seed=93,
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


def test_canonical_runner_can_use_versioned_external_working_memory_cell() -> None:
    cell = ExternalWorkingMemoryCell(
        event_width=8,
        action_width=2,
        memory_capacity=4,
        context_width=8,
        hidden=16,
    )
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        working_memory_cell=cell,
        seed=17,
    )

    rollout = agent.rollout(
        NBackVerifier(batch_size=2, n_back=2, steps=5, seed=29),
        record_retention=False,
    )

    assert agent.working_memory_cell is cell
    assert rollout.context.shape == (2, 8)
    assert cell.configuration()["read_order"] == (
        "read_old_state_then_append_current_row_v1"
    )


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


def test_appended_capability_can_use_and_grow_versioned_working_memory_cell() -> None:
    cell = ExternalWorkingMemoryCell(
        event_width=8,
        action_width=2,
        memory_capacity=4,
        context_width=8,
        hidden=16,
    )
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    slot = agent.add_adaptive_relation_capability(
        memory_capacity=4,
        seed=23,
        working_memory_cell=cell,
    )

    agent.expand_adaptive_relation_capability(slot, memory_capacity=5)

    extension = agent.extensions[slot - 1]
    assert isinstance(extension.reader, ExternalWorkingMemoryCell)
    assert extension.reader.memory_capacity == 5
    rollout = agent.rollout(
        NBackVerifier(batch_size=2, n_back=3, steps=6, seed=29),
        forced_slot=slot,
        record_retention=False,
    )
    assert torch.equal(rollout.selected_slots, torch.ones_like(rollout.selected_slots))


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


def test_context_route_failure_patience_prevents_single_noise_demotion() -> None:
    class FixedOutcomeVerifier:
        action_count = 2
        batch_size = 1
        device = torch.device("cpu")
        steps = 4

        def __init__(self) -> None:
            self._position = 0

        @property
        def position(self) -> int:
            return self._position

        @property
        def done(self) -> bool:
            return self._position >= self.steps

        def reset(self) -> None:
            self._position = 0

        def observation(self) -> torch.Tensor:
            return torch.tensor([4 if self._position == 0 else 0])

        def score(self, action: torch.Tensor) -> NBackVerifierStep:
            del action
            eligible = torch.tensor([self._position > 0])
            reward = torch.tensor(
                [0.0 if self._position in (1, 2) else 1.0]
            )
            self._position += 1
            return NBackVerifierStep(reward=reward, eligible=eligible)

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
    cue_event = agent.runtime.encoders["stimulus"](torch.tensor([4]))[0]
    for _ in range(agent.context_route_evidence.min_mastery_observations):
        agent.context_route_evidence.observe(cue_event, slot, 1.0)

    impatient = agent.rollout(
        FixedOutcomeVerifier(),
        sample=False,
        record_retention=False,
        context_route=True,
        context_route_failure_patience=1,
    )
    patient = agent.rollout(
        FixedOutcomeVerifier(),
        sample=False,
        record_retention=False,
        context_route=True,
        context_route_failure_patience=2,
    )

    assert impatient.selected_slots[0].tolist() == [slot, slot, 0, 0]
    assert patient.selected_slots[0].tolist() == [slot, slot, slot, 0]


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


def test_route_state_rejects_incompatible_learned_event_representation() -> None:
    agent = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=17,
    )
    agent.add_relation_capability(n_back=3, seed=23)
    payload = agent.route_state_payload()

    restored = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        reader_kind="relation",
        seed=18,
    )
    restored.add_relation_capability(n_back=3, seed=23)

    with pytest.raises(ValueError, match="learned event representation"):
        restored.load_route_state_payload(payload)


def test_canonical_intention_memory_is_external_and_reloadable() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        seed=17,
    )
    before = {
        name: value.detach().clone()
        for name, value in agent.controller.state_dict().items()
    }
    agent.observe_intention(
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        utility=torch.tensor([1.0, 0.0]),
        propensity=torch.ones(2),
        outcome_mask=torch.tensor([True, False]),
    )
    payload = agent.intention_state_payload()

    restored = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        seed=17,
    )
    restored.load_intention_state_payload(payload)

    assert (
        restored.intention_state_payload()["repertoire"]["sha256"]
        == payload["repertoire"]["sha256"]
    )
    assert restored.intention_repertoire.record_count == 2
    assert restored.propose_intentions(include_seed=False).source_indices == (0, 1)
    assert "intention_repertoire" not in dict(agent.named_parameters())
    for name, value in agent.controller.state_dict().items():
        assert torch.equal(value, before[name])


def test_rollout_can_record_only_present_verifier_outcomes_externally() -> None:
    agent = CanonicalBrainWorkshopAgent(
        n_back=2,
        event_width=8,
        intention_width=4,
        feedback_width=4,
        seed=17,
    )
    rollout = agent.rollout(
        NBackVerifier(batch_size=2, n_back=2, steps=4, seed=29),
        sample=False,
        record_intention_memory=True,
    )

    statistics = agent.intention_repertoire.statistics()
    assert int(statistics["attempts"].sum()) == rollout.actions.numel()
    assert int(statistics["outcome_counts"].sum()) == int(rollout.eligible.sum())
    assert int(statistics["outcome_counts"].sum()) < int(
        statistics["attempts"].sum()
    )


def test_cross_family_growth_smoke_keeps_route_and_core_boundaries(tmp_path) -> None:
    report = run_cross_family_rule_growth(
        argparse.Namespace(
            report_out=tmp_path / "cross-family-smoke.json",
            seed=17,
            source_updates=1,
            target_updates=1,
            batch_size=2,
            steps=6,
            calibration_lifetimes=1,
            discovery_lifetimes=1,
            retention_lifetimes=1,
            learning_rate=1e-2,
            target_family="triplet_parity",
            training_cue=7,
            heldout_cue=8,
            shuffled_cue=9,
            target_warmup_family="parity2",
            target_warmup_updates=1,
        )
    )

    assert report["schema"] == (
        "neural-computer.brainworkshop-cross-family-rule-growth.v1"
    )
    assert report["gates"]["controller_unchanged"] is True
    assert report["gates"]["encoder_unchanged"] is True
    assert report["gates"]["route_reload_exact"] is True
    assert report["gates"]["incompatible_route_representation_rejected"] is True
    assert report["gates"]["zero_replayed_examples"] is True
    assert report["training_rule"]["family"] == "triplet_parity"
    assert report["target_warmup"]["family"] == "parity2"
    assert report["accounting"]["optimizer_updates"] == 5
