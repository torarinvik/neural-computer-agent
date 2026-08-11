from __future__ import annotations

from experiments.brainworkshop_canonical.factored_active_disambiguation_pressure import (
    run_active_disambiguation_pressure,
)


def test_fresh_brainworkshop_active_probe_resolves_target_without_writes() -> None:
    result = run_active_disambiguation_pressure(
        seed=41,
        training_lifetimes=6,
        steps=9,
        random_feature_width=128,
    )

    assert result.status == "active_probe_resolved_fresh_target"
    assert result.active_probe_recovered_target
    assert not result.passive_control_recovered_target
    assert result.active_probe_read_only
    assert result.active_decoder_state_free
    assert result.controller_unchanged
    assert result.active_trial is not None
    assert result.active_trial.verifier_outcome_eligible
    assert result.active_trial.strict_route_slot_id == result.target_slot_id
    assert result.support_calibration_rows > 0
    assert result.active_trial.selected_probe_support
    assert result.optimizer_updates == 0
    assert result.replayed_examples == 0


def test_context_transfer_probe_memory_preserves_read_only_boundary() -> None:
    result = run_active_disambiguation_pressure(
        seed=41,
        training_lifetimes=6,
        steps=9,
        random_feature_width=128,
        utility_memory_kind="context_transfer",
        utility_calibration_repeats=4,
    )

    assert result.status == "active_probe_resolved_fresh_target"
    assert result.active_probe_recovered_target
    assert result.active_probe_read_only
    assert result.active_decoder_state_free
    assert result.controller_unchanged
    assert result.utility_memory_kind == "context_transfer"
    assert result.utility_calibration_repeats == 4
    assert result.utility_calibration_observations == 16
    assert result.optimizer_updates == 0
    assert result.replayed_examples == 0
