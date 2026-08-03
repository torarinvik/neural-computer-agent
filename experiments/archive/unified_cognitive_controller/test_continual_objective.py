"""Controls for the verifier-side continual acquisition objective."""

from .audit_continual_checkpoint import _chance_control_passed
from .continual_objective import score_continual_acquisition


def test_missing_evidence_controls_are_required_to_stay_near_chance() -> None:
    assert _chance_control_passed(0.50, 0.05)
    assert _chance_control_passed(0.54, 0.05)
    assert not _chance_control_passed(0.44, 0.05)
    assert not _chance_control_passed(0.56, 0.05)


def test_causal_gain_is_conservative_and_replay_bonus_is_gated() -> None:
    result = score_continual_acquisition(
        new_parent=0.60, new_child=0.70, new_causal_baseline=0.65,
        old_parent=(0.80, 0.75), old_child=(0.79, 0.75),
        replay_lifetimes=250, reference_replay_lifetimes=1_000)
    assert abs(result.parent_gain - 0.10) < 1e-9
    assert abs(result.causal_gain - 0.05) < 1e-9
    assert abs(result.new_gain - 0.05) < 1e-9
    assert result.retention_gate_passed
    assert result.replay_efficiency_bonus > 0


def test_replay_savings_cannot_pay_for_forgetting() -> None:
    result = score_continual_acquisition(
        new_parent=0.60, new_child=0.70, new_causal_baseline=0.60,
        old_parent=(0.80, 0.75), old_child=(0.70, 0.75),
        replay_lifetimes=0, reference_replay_lifetimes=1_000)
    assert not result.retention_gate_passed
    assert result.replay_savings == 1.0
    assert result.replay_efficiency_bonus == 0.0
    assert result.score < 0


def test_noncausal_improvement_receives_no_new_skill_reward() -> None:
    result = score_continual_acquisition(
        new_parent=0.60, new_child=0.70, new_causal_baseline=0.70,
        old_parent=(0.80,), old_child=(0.80,),
        replay_lifetimes=1_000, reference_replay_lifetimes=1_000)
    assert result.parent_gain > 0
    assert result.causal_gain == 0.0
    assert result.new_gain == 0.0
    assert not result.acquisition_gate_passed
    assert result.replay_efficiency_bonus == 0.0
    assert result.score == 0.0


def test_do_nothing_checkpoint_cannot_earn_replay_bonus() -> None:
    result = score_continual_acquisition(
        new_parent=0.60, new_child=0.60, new_causal_baseline=0.60,
        old_parent=(0.80, 0.75), old_child=(0.80, 0.75),
        replay_lifetimes=0, reference_replay_lifetimes=1_000)
    assert result.retention_gate_passed
    assert result.replay_savings == 1.0
    assert not result.acquisition_gate_passed
    assert result.replay_efficiency_bonus == 0.0
    assert result.score == 0.0
