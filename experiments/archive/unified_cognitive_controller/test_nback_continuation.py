"""Tests for the verifier-side n-back continuation audit."""
from __future__ import annotations

from .audit_nback_continuation import audit_continuation, summarize_stage


def _stage(*, seed: int = 1, before: float = 0.50, after: float = 0.80,
           updates: int = 256) -> dict:
    return {
        "seed": seed,
        "updates": updates,
        "batch_size": 32,
        "unique_training_lifetimes": updates * 32,
        "target_modalities": ["text"],
        "config": {"trials": 8},
        "before": {"eligible_accuracy": before},
        "after": {"eligible_accuracy": after},
        "gate": {"accepted_for_longer_run": True},
        "history_reset_control": {"eligible_accuracy": 0.50},
        "time_shuffle_control": {"eligible_accuracy": 0.52},
    }


def test_verifier_bits_exclude_private_rehearsal_streams() -> None:
    summary = summarize_stage(_stage(updates=256))
    assert summary["unique_training_lifetimes"] == 8192
    assert summary["verifier_bits"] == 65_536


def test_gated_continuation_is_recommended_only_below_mastery() -> None:
    result = audit_continuation(_stage(after=0.68))
    assert result["continue_recommended"]
    assert result["bits_to_mastery"] is None

    mastered = audit_continuation(_stage(after=0.80))
    assert not mastered["continue_recommended"]
    assert mastered["bits_to_mastery"] == 65_536


def test_continuation_accumulates_bits_and_checks_controls() -> None:
    initial = _stage(seed=7, after=0.68)
    continuation = _stage(seed=7, before=0.68, after=0.77, updates=64)
    result = audit_continuation(initial, continuation)
    assert result["cumulative_verifier_bits"] == 81_920
    assert result["gates"]["mastery_reached"]
    assert result["gates"]["causal_controls_recorded_and_separated"]


def test_retention_gate_allows_small_improvement_but_rejects_forgetting() -> None:
    retained = audit_continuation(
        _stage(),
        retention={"before": {"eligible_accuracy": 0.94},
                   "after": {"eligible_accuracy": 0.93}})
    assert retained["gates"]["retention_within_gate"]

    forgotten = audit_continuation(
        _stage(),
        retention={"before": {"eligible_accuracy": 0.94},
                   "after": {"eligible_accuracy": 0.91}})
    assert not forgotten["gates"]["retention_within_gate"]
    assert not forgotten["capability_claim_accepted"]
