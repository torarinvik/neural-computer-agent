"""Controls for the task-agnostic replay drift decision."""

from .audit_sequence_replay_drift import (
    relative_replay_stop_decision,
    replay_stop_decision,
)


def test_replay_stops_only_at_or_below_threshold() -> None:
    assert replay_stop_decision(0.79, 0.79)
    assert replay_stop_decision(0.70, 0.79)
    assert not replay_stop_decision(0.80, 0.79)


def test_negative_drift_is_rejected() -> None:
    try:
        replay_stop_decision(-0.1, 0.79)
    except ValueError:
        return
    raise AssertionError("negative KL was accepted")


def test_relative_policy_adapts_to_lineage_scale() -> None:
    assert relative_replay_stop_decision(0.89, 1.0, 0.10)
    assert relative_replay_stop_decision(0.44, 0.5, 0.10)
    assert not relative_replay_stop_decision(0.91, 1.0, 0.10)
